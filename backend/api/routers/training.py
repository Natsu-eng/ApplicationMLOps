"""Router training — lancement, suivi et résultat des entraînements ML.

Isolation : mêmes principes que `datasets.py` — filtrage systématique par
`organization_id`, accessible à toute l'équipe (pas réservé au owner).
L'entraînement lui-même s'exécute en tâche de fond (RQ, voir
`workers/training_worker.py`) — ce router ne fait qu'enfiler le job et lire
son état, jamais de calcul ML dans la requête HTTP.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.core.config import get_settings
from api.core.database import get_db
from api.core.job_queue import training_queue
from api.core.models import ClusteringJob, Dataset, DimensionalityJob, MLModel, ModelCandidate, TrainingJob, User
from api.routers.auth import get_current_user
from services.audit import log_action
from services.datasets import DatasetParsingError, read_dataframe
from services.feature_engineering import CURRENT_SPEC_VERSION
from services.job_quota import raise_if_quota_exceeded
from services.job_watchdog import reconcile_stale_jobs
from services.ml_inference import InferenceError, load_bundle, predict_one
from services.ml_registry import MODEL_REGISTRY
from services.ml_task import detect_task_type
from services.ml_training import selection_metric_label

_KNOWN_UPSTREAM_TYPES = {"datetime_decompose", "ratio", "numeric_coerce"}

# Mode expert (Lot E2) : modèles signalés comme lents dans le sélecteur —
# pas une propriété du registre (`ml_registry.ModelSpec`, purement une
# question de capacité), simple repère UX. SVC recalcule sa calibration de
# probabilité au fit final (voir `ml_registry._build_svm`) et KNN doit
# parcourir tout le jeu d'entraînement à chaque prédiction : les deux
# ralentissent nettement sur un gros dataset (mesuré au Lot 5).
_SLOW_MODEL_IDS = frozenset({"svm", "knn"})

router = APIRouter(prefix="/training", tags=["training"])
_settings = get_settings()


# ── Schémas ──────────────────────────────────────────────────────────────────

class FeatureEngineeringConfig(BaseModel):
    """Transformations de variables approuvées par l'utilisateur (Lot 4c) —
    chaque entrée reprend telle quelle le champ `transformation` d'une
    suggestion renvoyée par `GET /datasets/{id}/feature-engineering-suggestions`.
    `upstream` : déterministe (datetime, ratio), appliqué une fois avant le
    split. `pipeline` : appris (encodage de fréquence, imputation), fit dans
    chaque fold — voir `services/ml_preprocessing.build_preprocessor`."""
    upstream: List[dict[str, Any]] = []
    pipeline: dict[str, Any] = {}


class TrainingJobCreate(BaseModel):
    dataset_id: int
    target_column: str
    feature_columns: Optional[List[str]] = None
    task_type: Optional[str] = None  # "classification" | "regression" — auto-détecté si absent
    group_column: Optional[str] = None
    test_size: float = Field(0.2, gt=0.05, lt=0.5)
    optuna_trials: Optional[int] = Field(None, ge=3, le=100)
    cv_folds: Optional[int] = Field(None, ge=2, le=10)
    # Mode expert (Lot E2) — reproductibilité (graine aléatoire) et niveau de
    # confiance des intervalles CQR (régression uniquement, ignoré sinon).
    # Absents : comportement strictement inchangé (défauts serveur ci-dessous).
    seed: Optional[int] = Field(None, ge=0, le=99_999)
    cqr_alpha: Optional[float] = Field(None, gt=0.0, lt=1.0)
    # Mode expert (Lot E2) — sous-ensemble du catalogue à comparer (voir
    # `services/ml_registry.MODEL_REGISTRY`). `None` : sous-ensemble par
    # défaut (stratégie produit "B"), comportement strictement inchangé.
    model_ids: Optional[List[str]] = None
    feature_engineering: Optional[FeatureEngineeringConfig] = None
    # Rééquilibrage des classes (lot déséquilibre) — PROPOSÉ à l'utilisateur sur
    # la base du garde-fou Lot B (`desequilibre_classes`), jamais appliqué
    # d'office. Volontairement SIBLING de `feature_engineering`, pas imbriqué
    # dedans : contrairement à `feature_engineering_json`, ce choix n'est
    # jamais rejoué à l'inférence (voir `services/ml_inference.py`) — il ne
    # modifie que la pondération vue par le modèle pendant l'entraînement, pas
    # la forme du pipeline. Il vit dans `config_json` (comme `optuna_trials`,
    # `test_size`...), jamais dans `feature_engineering_json`. Ignoré si la
    # tâche est une régression (concept propre à la classification).
    class_rebalancing: bool = False


class ModelCatalogEntry(BaseModel):
    """Une entrée du catalogue de modèles (Lot 5) exposée au mode expert
    (Lot E2) — `label` régularise Ridge/LogisticRegression et SVR/SVC (même
    `id` de registre, nom différent selon la tâche) en un seul libellé
    lisible sans que le frontend ait besoin de connaître la tâche détectée."""
    id: str
    label: str
    family: str
    is_default: bool
    supports_rebalancing: bool
    supported_tasks: List[str]
    slow: bool


class ModelCatalogResponse(BaseModel):
    models: List[ModelCatalogEntry]


class TrainingJobSummary(BaseModel):
    id: int
    dataset_id: int
    dataset_name: Optional[str] = None
    task_type: str
    target_column: str
    status: str
    progress_step: Optional[str] = None
    progress_percent: int
    error_message: Optional[str] = None
    created_by: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    algorithm: Optional[str] = None
    headline_metric: Optional[dict[str, Any]] = None


class FeatureSchemaEntry(BaseModel):
    name: str
    dtype: str


class MLModelDetail(BaseModel):
    id: int
    training_job_id: int
    algorithm: str
    task_type: str
    target_column: str
    feature_columns: List[str]
    feature_schema: List[FeatureSchemaEntry] = []
    metrics: dict[str, Any]
    shap_summary: List[dict[str, Any]]
    cqr: Optional[dict[str, Any]] = None
    model_card: dict[str, Any]
    evaluation: dict[str, Any] = {}
    feature_engineering: Optional[dict[str, Any]] = None
    # Lot Explicabilité globale — absents ([]/{}/None) sur les modèles
    # entraînés avant ce lot (rétrocompat) : le frontend affiche "réentraînez
    # pour l'obtenir" plutôt que de planter, voir model_card.*_status.
    shap_beeswarm: dict[str, List[dict[str, Any]]] = {}
    permutation_importance: List[dict[str, Any]] = []
    calibration: Optional[dict[str, Any]] = None
    learning_curve: Optional[dict[str, Any]] = None
    # Lot 9 — registre de modèles versionné. `None` = jamais promu.
    stage: Optional[str] = None
    promoted_at: Optional[datetime] = None
    created_at: datetime


class ModelCandidateOut(BaseModel):
    """Un modèle comparé pendant le job — le gagnant ET tous les autres
    (Lot D, leaderboard). `selection_score` est LA métrique qui a
    départagé les candidats (voir `LeaderboardResponse.selection_metric_label`
    pour son libellé humain), jamais l'accuracy brute."""
    algorithm: str
    family: str
    selection_score: float
    is_winner: bool
    rank: int
    fold_scores: Optional[List[float]] = None
    secondary_metric: Optional[float] = None
    secondary_metric_label: Optional[str] = None


class LeaderboardResponse(BaseModel):
    selection_metric_label: str
    candidates: List[ModelCandidateOut]


# ── Lot 9 — registre de modèles versionné ───────────────────────────────────
#
# L'artefact (bundle joblib) existait déjà depuis le Lot 3 — ce lot ajoute ce
# qui manquait pour un vrai registre : savoir QUEL modèle fait autorité pour
# un problème donné (promotion), et pouvoir en RÉCUPÉRER l'artefact hors de
# la plateforme (export).

_VALID_STAGES = {"none", "staging", "production"}


class PromoteModelRequest(BaseModel):
    stage: str  # "none" | "staging" | "production"


class ModelRegistryEntry(BaseModel):
    """Une entrée du registre — un modèle PROMU (staging ou production),
    avec assez de contexte (dataset/cible/algorithme/métrique) pour
    identifier de quel problème il s'agit sans recharger le job complet."""
    job_id: int
    model_id: int
    dataset_id: int
    dataset_name: Optional[str] = None
    task_type: str
    target_column: str
    algorithm: str
    stage: str
    promoted_at: Optional[datetime] = None
    headline_metric: Optional[dict[str, Any]] = None


class ModelRegistryResponse(BaseModel):
    entries: List[ModelRegistryEntry]


# ── Lot D-bis — comparaison inter-jobs ──────────────────────────────────────
#
# Le Lot D a rendu visible la comparaison ENTRE LES MODÈLES D'UN MÊME JOB
# (leaderboard intra-job) — ce lot ajoute la comparaison ENTRE PLUSIEURS
# JOBS (config différente, dataset différent, ou simplement deux essais).

# Champs de configuration comparés — mêmes clés que `config_json` (voir
# `create_training_job`) ; `model_ids` traité à part (liste, comparaison par
# ensemble plutôt que par ordre — voir `_config_fields_differ`).
_COMPARABLE_SCALAR_CONFIG_FIELDS = ("test_size", "optuna_trials", "cv_folds", "seed", "cqr_alpha", "class_rebalancing")


class JobComparisonEntry(BaseModel):
    job_id: int
    dataset_id: int
    dataset_name: Optional[str] = None
    task_type: str
    target_column: str
    status: str
    algorithm: Optional[str] = None
    created_at: datetime
    headline_metric: Optional[dict[str, Any]] = None
    # Métriques complètes du modèle retenu (si le job est terminé) — mêmes
    # clés que MLModelDetail.metrics, pour que le frontend réutilise les
    # mêmes libellés/formatage que la page Résultats, sans dupliquer la
    # logique d'affichage par métrique.
    metrics: dict[str, Any] = {}
    config: dict[str, Any] = {}
    feature_engineering_active: bool = False


class JobComparisonResponse(BaseModel):
    entries: List[JobComparisonEntry]
    # Champs de `config` qui DIFFÈRENT entre au moins deux jobs comparés —
    # calculé ici plutôt que côté frontend, pour qu'un seul endroit décide
    # de la règle de comparaison (ex. model_ids par ensemble, pas par ordre).
    differing_config_fields: List[str]


class PredictionRequest(BaseModel):
    data: dict[str, Any]


class LocalContribution(BaseModel):
    feature: str
    value: float
    contribution: float


class LocalExplanation(BaseModel):
    """Explication locale (waterfall, Lot Explicabilité locale) — pourquoi
    CETTE prédiction précise, contribution signée par variable. `status`
    "degraded" (modèle antérieur à ce lot, ou calcul SHAP indisponible pour
    ce modèle) : `message` porte l'explication, les autres champs restent
    vides plutôt qu'une erreur HTTP — l'explication est un complément à la
    prédiction, jamais une condition pour l'obtenir."""
    status: str
    message: Optional[str] = None
    base_value: Optional[float] = None
    contributions: List[LocalContribution] = []
    other_contribution: Optional[float] = None


class PredictionResponse(BaseModel):
    prediction: Any
    probabilities: Optional[dict[str, float]] = None
    interval: Optional[dict[str, float]] = None
    explanation: Optional[LocalExplanation] = None


# ── Aides internes ───────────────────────────────────────────────────────────

def _headline_metric(task_type: str, metrics: dict[str, Any]) -> dict[str, Any]:
    """Métrique mise en avant sur la carte d'historique (`JobCard.tsx`).

    En classification, `accuracy` est trompeuse sur un dataset déséquilibré
    (un modèle qui ignore la classe rare peut afficher 95 % d'exactitude) —
    on affiche donc `cv_score`, la métrique RÉELLEMENT utilisée pour
    départager les candidats (ROC-AUC pondérée, voir
    `services/ml_training._classification_selection_score`), jamais
    l'accuracy brute (bug trouvé lors de l'audit leaderboard, Lot D)."""
    if task_type == "regression":
        return {"name": "r2_test", "value": metrics.get("r2_test")}
    return {"name": "cv_score", "value": metrics.get("cv_score")}


def _to_summary(job: TrainingJob) -> TrainingJobSummary:
    model = job.model
    metrics = json.loads(model.metrics_json) if model else None
    return TrainingJobSummary(
        id=job.id,
        dataset_id=job.dataset_id,
        dataset_name=job.dataset.name if job.dataset else None,
        task_type=job.task_type,
        target_column=job.target_column,
        status=job.status,
        progress_step=job.progress_step,
        progress_percent=job.progress_percent,
        error_message=job.error_message,
        created_by=job.created_by.nom if job.created_by else None,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        algorithm=model.algorithm if model else None,
        headline_metric=_headline_metric(job.task_type, metrics) if metrics else None,
    )


def _validate_and_serialize_feature_engineering(
    fe: Optional[FeatureEngineeringConfig], schema_columns: list[str]
) -> Optional[str]:
    """Valide les colonnes référencées par la spec approuvée (mêmes garanties
    que pour target_column/feature_columns/group_column ci-dessus), puis la
    sérialise avec son numéro de version (Lot 4c, precision 2) — c'est cette
    version, pas une saisie du frontend, qui fait foi à l'inférence."""
    if fe is None:
        return None

    known_columns = set(schema_columns)
    for transformation in fe.upstream:
        ttype = transformation.get("type")
        if ttype not in _KNOWN_UPSTREAM_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "TRANSFORMATION_INCONNUE", "message": f"Type de transformation amont inconnu : {ttype}"},
            )
        if ttype == "datetime_decompose":
            referenced = [transformation.get("source_column")]
        elif ttype == "numeric_coerce":
            referenced = [transformation.get("column")]
        else:  # "ratio"
            referenced = [transformation.get("numerator"), transformation.get("denominator")]
        unknown = {c for c in referenced if c not in known_columns}
        if unknown:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "COLONNES_INCONNUES", "message": f"Colonnes absentes du dataset : {', '.join(sorted(unknown))}"},
            )

    freq_cols = fe.pipeline.get("frequency_encoding") or []
    imputation_cols = list((fe.pipeline.get("imputation") or {}).keys())
    unknown_pipeline = (set(freq_cols) | set(imputation_cols)) - known_columns
    if unknown_pipeline:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "COLONNES_INCONNUES", "message": f"Colonnes absentes du dataset : {', '.join(sorted(unknown_pipeline))}"},
        )

    return json.dumps({"version": CURRENT_SPEC_VERSION, "upstream": fe.upstream, "pipeline": fe.pipeline})


def _get_org_job(job_id: int, current_user: User, db: Session) -> TrainingJob:
    job = (
        db.query(TrainingJob)
        .filter(TrainingJob.id == job_id, TrainingJob.organization_id == current_user.organization_id)
        .first()
    )
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "TRAINING_JOB_INTROUVABLE", "message": "Entraînement introuvable"},
        )
    return job


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/models-catalog", response_model=ModelCatalogResponse)
def get_models_catalog(current_user: User = Depends(get_current_user)):
    """Catalogue complet des 9 modèles du registre (Lot 5), pour le sélecteur
    du mode expert (Lot E2) — lecture pure du registre, aucun accès dataset :
    ce n'est pas la tâche (classification/régression) qui filtre ici, elle
    n'est connue qu'à la création du job (`supported_tasks` permet au
    frontend de signaler les modèles à usage restreint, ex. Naive Bayes)."""
    entries = []
    for spec in MODEL_REGISTRY.values():
        tasks = sorted(spec.supported_tasks)
        labels = list(dict.fromkeys(spec.label(t) for t in tasks))
        entries.append(
            ModelCatalogEntry(
                id=spec.id,
                label=" / ".join(labels),
                family=spec.family,
                is_default=spec.is_default,
                supports_rebalancing=spec.supports_rebalancing,
                supported_tasks=tasks,
                slow=spec.id in _SLOW_MODEL_IDS,
            )
        )
    return ModelCatalogResponse(models=entries)


@router.post("/jobs", response_model=TrainingJobSummary, status_code=status.HTTP_201_CREATED)
def create_training_job(
    body: TrainingJobCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Réconciliation des jobs orphelins (H2, AUDIT_ROADMAP.md) — AVANT le
    # comptage du quota ci-dessous : un job "running" dont le worker a
    # crashé ne doit jamais bloquer indéfiniment un slot de quota.
    reconcile_stale_jobs(db, current_user.organization_id, _settings.stale_job_timeout_minutes)

    # Garde-fou technique (Lot 10, quota partagé étendu au Lot 13/14 via
    # services/job_quota.py) — un seul worker RQ traite les jobs de TOUTES
    # les organisations ET de tous les types (supervisé/clustering/réduction
    # de dimension/anomalies) : sans compter tous les types ensemble, une
    # organisation pourrait saturer le worker en cumulant plusieurs types de
    # jobs actifs. Vérifié AVANT toute lecture du dataset (échec rapide).
    raise_if_quota_exceeded(
        db,
        current_user.organization_id,
        [TrainingJob, ClusteringJob, DimensionalityJob],
        _settings.max_concurrent_jobs_per_org,
    )

    dataset = (
        db.query(Dataset)
        .filter(Dataset.id == body.dataset_id, Dataset.organization_id == current_user.organization_id)
        .first()
    )
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "DATASET_INTROUVABLE", "message": "Dataset introuvable"},
        )
    if dataset.status != "ready":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "DATASET_NON_PRET", "message": "Ce dataset n'a pas pu être analysé"},
        )

    schema_columns = [c["name"] for c in json.loads(dataset.columns_json or "[]")]
    if body.target_column not in schema_columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "COLONNE_CIBLE_INTROUVABLE", "message": f"Colonne cible '{body.target_column}' absente du dataset"},
        )

    feature_columns = body.feature_columns or [c for c in schema_columns if c != body.target_column]
    unknown = set(feature_columns) - set(schema_columns)
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "COLONNES_INCONNUES", "message": f"Colonnes absentes du dataset : {', '.join(sorted(unknown))}"},
        )
    if body.group_column and body.group_column not in schema_columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "COLONNE_GROUPE_INTROUVABLE", "message": f"Colonne de groupe '{body.group_column}' absente du dataset"},
        )

    if body.model_ids is not None:
        unknown_models = set(body.model_ids) - set(MODEL_REGISTRY.keys())
        if unknown_models:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": "MODELES_INCONNUS",
                    "message": f"Modèles inconnus : {', '.join(sorted(unknown_models))}",
                },
            )

    feature_engineering_json = _validate_and_serialize_feature_engineering(body.feature_engineering, schema_columns)

    try:
        df = read_dataframe(Path(dataset.file_path), Path(dataset.file_path).suffix)
    except DatasetParsingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "DATASET_LECTURE_ECHEC", "message": str(exc)},
        )

    task_type = body.task_type or detect_task_type(df[body.target_column])
    if task_type not in ("classification", "regression"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "TACHE_NON_SUPPORTEE", "message": "Seuls classification et régression sont supportés (Lot 3)"},
        )

    # Mode expert (Lot E2) : intersection avec les modèles compatibles avec la
    # tâche détectée (ex. Naive Bayes est classification uniquement) — la
    # sélection est faite en amont dans l'UI mais le serveur ne fait jamais
    # confiance à cette cohérence côté client.
    selected_model_ids: Optional[List[str]] = None
    if body.model_ids is not None:
        selected_model_ids = [m for m in body.model_ids if task_type in MODEL_REGISTRY[m].supported_tasks]
        if not selected_model_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": "AUCUN_MODELE_COMPATIBLE",
                    "message": f"Aucun des modèles sélectionnés ne supporte cette tâche ({task_type})",
                },
            )

    config = {
        "test_size": body.test_size,
        "seed": body.seed if body.seed is not None else _settings.model_seed,
        "optuna_trials": body.optuna_trials or _settings.optuna_trials_default,
        "cv_folds": body.cv_folds or _settings.cv_folds_default,
        "cqr_alpha": body.cqr_alpha if body.cqr_alpha is not None else _settings.cqr_alpha,
        "cqr_n_strata": _settings.cqr_n_strata,
        "shap_sample_size": _settings.shap_sample_size,
        "class_rebalancing": body.class_rebalancing,
        "model_ids": selected_model_ids,
    }

    job = TrainingJob(
        organization_id=current_user.organization_id,
        dataset_id=dataset.id,
        created_by_id=current_user.id,
        task_type=task_type,
        target_column=body.target_column,
        feature_columns_json=json.dumps(feature_columns),
        group_column=body.group_column,
        config_json=json.dumps(config),
        feature_engineering_json=feature_engineering_json,
        status="queued",
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    from workers.training_worker import run_training_job

    rq_job = training_queue.enqueue(run_training_job, job.id, job_timeout=1800)
    job.rq_job_id = rq_job.id
    db.commit()
    db.refresh(job)

    return _to_summary(job)


@router.get("/jobs", response_model=List[TrainingJobSummary])
def list_training_jobs(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    jobs = (
        db.query(TrainingJob)
        .filter(TrainingJob.organization_id == current_user.organization_id)
        .order_by(TrainingJob.created_at.desc())
        .all()
    )
    return [_to_summary(j) for j in jobs]


def _config_field_value(config: dict[str, Any], field: str) -> Any:
    """Valeur normalisée d'un champ de config pour la comparaison —
    `model_ids` par ENSEMBLE (l'ordre de sélection n'a pas de sens produit),
    tout le reste tel quel."""
    if field == "model_ids":
        value = config.get("model_ids")
        return frozenset(value) if value else None
    return config.get(field)


def _differing_config_fields(configs: List[dict[str, Any]]) -> List[str]:
    """Champs de config qui diffèrent entre au moins deux des jobs comparés
    — un seul job (ou des configs identiques) : liste vide, jamais une
    erreur. Ordre de sortie stable (ordre des constantes), pour un affichage
    reproductible côté frontend."""
    differing: List[str] = []
    fields = list(_COMPARABLE_SCALAR_CONFIG_FIELDS) + ["model_ids"]
    for field in fields:
        values = {_config_field_value(c, field) for c in configs}
        # Un ensemble non hashable (ex. frozenset(None)) ne peut pas être
        # inséré tel quel — None est déjà géré par _config_field_value.
        if len(values) > 1:
            differing.append(field)
    return differing


@router.get("/jobs/compare", response_model=JobComparisonResponse)
def compare_training_jobs(
    job_ids: List[int] = Query(..., min_length=2, max_length=8),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Comparaison inter-jobs (Lot D-bis) — le Lot D comparait déjà les
    modèles D'UN MÊME job (leaderboard intra-job) ; ce endpoint compare
    PLUSIEURS jobs entre eux (config, métriques), quel que soit leur
    dataset/cible — le frontend affiche le dataset/cible de chacun pour que
    l'utilisateur juge lui-même la pertinence du rapprochement.

    Isolé par organisation comme le reste : un id de job d'une autre
    organisation dans `job_ids` est traité comme absent (jamais un indice
    d'existence croisée), 404 si au moins un id demandé n'est pas trouvé —
    plutôt qu'une comparaison partielle silencieuse sur un sous-ensemble
    différent de ce que l'utilisateur a demandé."""
    unique_ids = list(dict.fromkeys(job_ids))  # dédoublonné, ordre conservé
    jobs = (
        db.query(TrainingJob)
        .filter(TrainingJob.id.in_(unique_ids), TrainingJob.organization_id == current_user.organization_id)
        .all()
    )
    jobs_by_id = {j.id: j for j in jobs}
    missing = [jid for jid in unique_ids if jid not in jobs_by_id]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "TRAINING_JOB_INTROUVABLE", "message": f"Entraînement(s) introuvable(s) : {missing}"},
        )
    if len(unique_ids) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "COMPARAISON_INSUFFISANTE", "message": "Sélectionnez au moins deux entraînements à comparer"},
        )

    entries: List[JobComparisonEntry] = []
    configs: List[dict[str, Any]] = []
    for jid in unique_ids:  # ordre demandé par l'appelant, pas l'ordre SQL
        job = jobs_by_id[jid]
        model = job.model
        metrics = json.loads(model.metrics_json) if model else {}
        config = json.loads(job.config_json) if job.config_json else {}
        configs.append(config)
        entries.append(JobComparisonEntry(
            job_id=job.id,
            dataset_id=job.dataset_id,
            dataset_name=job.dataset.name if job.dataset else None,
            task_type=job.task_type,
            target_column=job.target_column,
            status=job.status,
            algorithm=model.algorithm if model else None,
            created_at=job.created_at,
            headline_metric=_headline_metric(job.task_type, metrics) if metrics else None,
            metrics=metrics,
            config=config,
            feature_engineering_active=bool(job.feature_engineering_json),
        ))

    return JobComparisonResponse(entries=entries, differing_config_fields=_differing_config_fields(configs))


@router.get("/jobs/{job_id}", response_model=TrainingJobSummary)
def get_training_job(job_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return _to_summary(_get_org_job(job_id, current_user, db))


def _to_model_detail(model: MLModel) -> MLModelDetail:
    return MLModelDetail(
        id=model.id,
        training_job_id=model.training_job_id,
        algorithm=model.algorithm,
        task_type=model.task_type,
        target_column=model.target_column,
        feature_columns=json.loads(model.feature_columns_json),
        feature_schema=json.loads(model.feature_schema_json) if model.feature_schema_json else [],
        metrics=json.loads(model.metrics_json),
        shap_summary=json.loads(model.shap_summary_json) if model.shap_summary_json else [],
        cqr=json.loads(model.cqr_json) if model.cqr_json else None,
        model_card=json.loads(model.model_card_json) if model.model_card_json else {},
        evaluation=json.loads(model.evaluation_json) if model.evaluation_json else {},
        feature_engineering=json.loads(model.feature_engineering_json) if model.feature_engineering_json else None,
        shap_beeswarm=json.loads(model.shap_beeswarm_json) if model.shap_beeswarm_json else {},
        permutation_importance=json.loads(model.permutation_importance_json) if model.permutation_importance_json else [],
        calibration=json.loads(model.calibration_json) if model.calibration_json else None,
        learning_curve=json.loads(model.learning_curve_json) if model.learning_curve_json else None,
        stage=model.stage,
        promoted_at=model.promoted_at,
        created_at=model.created_at,
    )


@router.get("/jobs/{job_id}/model", response_model=MLModelDetail)
def get_training_job_model(job_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    job = _get_org_job(job_id, current_user, db)
    if job.status != "completed" or job.model is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "MODELE_NON_DISPONIBLE", "message": "Cet entraînement n'a pas encore produit de modèle"},
        )
    return _to_model_detail(job.model)


@router.post("/jobs/{job_id}/model/promote", response_model=MLModelDetail)
def promote_model(
    job_id: int,
    body: PromoteModelRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Promotion d'un modèle (Lot 9) — "staging" (à valider), "production"
    (celui utilisé en confiance pour ce problème) ou "none" (retrait).

    Règle du registre : UN SEUL modèle "production" à la fois par couple
    (dataset, cible) au sein d'une organisation — promouvoir un nouveau
    modèle en production DÉMET automatiquement l'ancien (repasse à
    "staging", jamais supprimé ni écrasé, juste son statut qui change),
    pour qu'il n'y ait jamais d'ambiguïté sur "quel modèle fait autorité".
    Aucune limite en "staging" : plusieurs candidats peuvent y attendre
    validation en parallèle."""
    if body.stage not in _VALID_STAGES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "STAGE_INVALIDE", "message": f"Statut inconnu : {body.stage!r} (attendu : none/staging/production)"},
        )
    job = _get_org_job(job_id, current_user, db)
    if job.status != "completed" or job.model is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "MODELE_NON_DISPONIBLE", "message": "Cet entraînement n'a pas encore produit de modèle"},
        )
    model = job.model

    if body.stage == "production":
        # Démotion de l'éventuel modèle déjà en production pour LE MÊME
        # problème (dataset + cible) — jamais entre problèmes différents, où
        # "production" n'a pas de sens à départager.
        previously_promoted = (
            db.query(MLModel)
            .join(TrainingJob, MLModel.training_job_id == TrainingJob.id)
            .filter(
                MLModel.organization_id == current_user.organization_id,
                MLModel.stage == "production",
                MLModel.id != model.id,
                TrainingJob.dataset_id == job.dataset_id,
                TrainingJob.target_column == job.target_column,
            )
            .all()
        )
        for other in previously_promoted:
            other.stage = "staging"

    model.stage = None if body.stage == "none" else body.stage
    model.promoted_at = datetime.now(timezone.utc) if body.stage != "none" else None
    log_action(
        db, current_user.organization_id, current_user.id, "model.promoted",
        target_type="model", target_id=model.id, details={"stage": body.stage, "algorithm": model.algorithm},
    )
    db.commit()
    db.refresh(model)
    return _to_model_detail(model)


@router.get("/jobs/{job_id}/model/export")
def export_model(job_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Export de l'artefact (Lot 9) — le bundle joblib complet (modèle +
    préprocesseur + CQR le cas échéant), pour une utilisation hors de la
    plateforme (chargement via `joblib.load` dans un environnement Python
    équivalent — mêmes versions de scikit-learn/lightgbm/xgboost/catboost/
    shap que `backend/requirements.txt`, non garanties par cet export)."""
    job = _get_org_job(job_id, current_user, db)
    if job.status != "completed" or job.model is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "MODELE_NON_DISPONIBLE", "message": "Cet entraînement n'a pas encore produit de modèle"},
        )
    artifact_path = Path(job.model.file_path)
    if not artifact_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "ARTEFACT_INTROUVABLE", "message": "Artefact du modèle introuvable sur le serveur"},
        )
    filename = f"modele_{job.dataset.name.rsplit('.', 1)[0] if job.dataset else 'export'}_{job.target_column}_job{job.id}.joblib"
    return FileResponse(path=artifact_path, filename=filename, media_type="application/octet-stream")


@router.get("/models/registry", response_model=ModelRegistryResponse)
def list_model_registry(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Registre de modèles (Lot 9) — tous les modèles PROMUS (staging ou
    production) de l'organisation, tous datasets/cibles confondus. Un modèle
    jamais promu (`stage IS NULL`, comportement historique) n'y apparaît
    jamais — le registre n'est PAS un doublon de l'historique complet
    (`GET /training/jobs`), seulement ce qui a été explicitement retenu."""
    models = (
        db.query(MLModel)
        .join(TrainingJob, MLModel.training_job_id == TrainingJob.id)
        .filter(MLModel.organization_id == current_user.organization_id, MLModel.stage.isnot(None))
        .order_by(MLModel.promoted_at.desc())
        .all()
    )
    entries = []
    for model in models:
        job = model.training_job
        metrics = json.loads(model.metrics_json) if model.metrics_json else {}
        entries.append(ModelRegistryEntry(
            job_id=job.id,
            model_id=model.id,
            dataset_id=job.dataset_id,
            dataset_name=job.dataset.name if job.dataset else None,
            task_type=model.task_type,
            target_column=model.target_column,
            algorithm=model.algorithm,
            stage=model.stage,
            promoted_at=model.promoted_at,
            headline_metric=_headline_metric(model.task_type, metrics) if metrics else None,
        ))
    return ModelRegistryResponse(entries=entries)


@router.get("/jobs/{job_id}/candidates", response_model=LeaderboardResponse)
def get_job_candidates(job_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Leaderboard du job (Lot D) — TOUS les modèles comparés, pas seulement
    le gagnant (déjà exposé par `GET /jobs/{id}/model`).

    Rétrocompatible par absence, jamais par erreur : un job entraîné avant ce
    lot n'a aucune ligne `ModelCandidate` (jamais recalculée a posteriori,
    voir `services/ml_training.py`) — `candidates` renvoie alors `[]`, pas un
    404/409, pour que le frontend affiche proprement le seul gagnant déjà
    disponible via `GET /jobs/{id}/model`."""
    job = _get_org_job(job_id, current_user, db)
    rows = (
        db.query(ModelCandidate)
        .filter(ModelCandidate.training_job_id == job.id, ModelCandidate.organization_id == current_user.organization_id)
        .order_by(ModelCandidate.rank.asc())
        .all()
    )
    return LeaderboardResponse(
        selection_metric_label=selection_metric_label(job.task_type),
        candidates=[
            ModelCandidateOut(
                algorithm=row.algorithm,
                family=row.family,
                selection_score=row.selection_score,
                is_winner=row.is_winner,
                rank=row.rank,
                fold_scores=json.loads(row.fold_scores_json) if row.fold_scores_json else None,
                secondary_metric=row.secondary_metric,
                secondary_metric_label=row.secondary_metric_label,
            )
            for row in rows
        ],
    )


@router.post("/jobs/{job_id}/predict", response_model=PredictionResponse)
def predict_with_model(
    job_id: int,
    body: PredictionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Prédit sur une nouvelle observation avec le modèle produit par ce job.

    Referme la boucle ouverte au Lot 3 : entraîner un modèle ne servait à
    rien tant qu'il ne pouvait pas être réutilisé (voir workflow.md, Lot 4).
    """
    job = _get_org_job(job_id, current_user, db)
    if job.status != "completed" or job.model is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "MODELE_NON_DISPONIBLE", "message": "Cet entraînement n'a pas encore produit de modèle"},
        )
    model: MLModel = job.model
    feature_columns = json.loads(model.feature_columns_json)

    try:
        bundle = load_bundle(model.file_path)
        result = predict_one(bundle, feature_columns, body.data)
    except InferenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "PREDICTION_IMPOSSIBLE", "message": str(exc)},
        )
    return PredictionResponse(**result)


@router.delete("/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_training_job(job_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Supprime un entraînement (et le modèle associé, s'il existe).

    Si le job est encore `queued`, on tente d'annuler le job RQ correspondant
    (best-effort — s'il a déjà été pris en charge par un worker, ça ne
    l'interrompt pas, mais évite au worker de traiter un job orphelin ; de
    toute façon `training_worker.py` gère déjà l'absence du job en base sans
    planter, donc une annulation ratée n'est jamais dangereuse).
    """
    job = _get_org_job(job_id, current_user, db)

    if job.status in ("queued", "running") and job.rq_job_id:
        try:
            from rq.job import Job as RQJob

            rq_job = RQJob.fetch(job.rq_job_id, connection=training_queue.connection)
            rq_job.cancel()
            rq_job.delete()
        except Exception:
            pass  # best-effort — la suppression en base reste sûre dans tous les cas

    if job.model:
        # Supprimé explicitement ici (pas seulement via le ON DELETE CASCADE
        # de la contrainte FK) : `job.model` est déjà chargé en mémoire à ce
        # stade (ligne précédente), et SQLAlchemy tente alors de mettre à
        # NULL `ml_models.training_job_id` avant de supprimer `job` — colonne
        # NOT NULL, ça lève une IntegrityError. Le supprimer nous-mêmes ici
        # évite ce comportement, quelle que soit la base (Postgres/SQLite).
        if job.model.file_path:
            Path(job.model.file_path).unlink(missing_ok=True)
        db.delete(job.model)

    log_action(
        db, current_user.organization_id, current_user.id, "training_job.deleted",
        target_type="training_job", target_id=job.id,
        details={"target_column": job.target_column, "status": job.status},
    )
    db.delete(job)
    db.commit()
