"""Router training — lancement, suivi et résultat des entraînements ML.

Isolation : mêmes principes que `datasets.py` — filtrage systématique par
`organization_id`, accessible à toute l'équipe (pas réservé au owner).
L'entraînement lui-même s'exécute en tâche de fond (RQ, voir
`workers/training_worker.py`) — ce router ne fait qu'enfiler le job et lire
son état, jamais de calcul ML dans la requête HTTP.
"""
from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session, joinedload

from api.core.config import get_settings
from api.core.database import SessionLocal, get_db
from api.core.error_codes import ErrorCode
from api.core.job_queue import analysis_queue, redis_conn, training_queue
from api.core.models import (
    AuditLog,
    BatchPredictionJob,
    Dataset,
    MLModel,
    ModelCandidate,
    Prediction,
    TrainingJob,
    User,
)
from api.core.pagination import paginate_by_id
from api.core.rate_limit import rate_limit_dependency
from api.core.storage import batch_prediction_input_file_path
from domains.auth.router import get_current_user
from domains.shared.audit import log_action
from domains.shared.dataset_io import (
    DatasetParsingError,
    UnsupportedFileType,
    read_dataset_dataframe,
    validate_extension,
)
from domains.shared.drift import MAX_PREDICTIONS_FOR_DRIFT, MIN_CURRENT_ROWS_FOR_DRIFT, compute_drift_report
from domains.shared.feature_engineering import CURRENT_SPEC_VERSION
from domains.shared.job_creation import enqueue_or_mark_failed, remember_idempotent_job_id, resolve_idempotent_job_id
from domains.shared.job_events import stream_job_updates
from domains.shared.job_lifecycle import ACTIVE_STATUSES, CANCELLED_MESSAGE, try_cancel_rq_job
from domains.shared.job_quota import ALL_JOB_MODELS, raise_if_quota_exceeded
from domains.shared.job_watchdog import reconcile_stale_jobs
from domains.shared.ml_task import detect_task_type
from domains.shared.model_bundle import InferenceError, load_bundle
from domains.training.batch_prediction_worker import run_batch_prediction_job
from domains.training.services.deployment_export import generate_deployment_script
from domains.training.services.duration_estimate import estimate_training_duration
from domains.training.services.engine import selection_metric_label
from domains.training.services.inference import predict_one
from domains.training.services.prediction_retention import purge_old_predictions
from domains.training.services.registry import MODEL_REGISTRY, models_for_task
from domains.training.services.verdict import compute_verdict

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
    # Mode expert (retour utilisateur direct : "laisser le choix sur les
    # hyperparamètres, profondeur des arbres etc.") — clé = id du registre,
    # valeur = {nom_hyperparamètre: valeur fixée}. Validé contre
    # `ModelSpec.tunable_hyperparameters` avant d'enfiler le job (voir
    # `_validate_hyperparameter_overrides` ci-dessous) : jamais un nom ou
    # une valeur hors-catalogue transmis tel quel au moteur. `None` :
    # comportement strictement inchangé, recherche entièrement automatique.
    hyperparameter_overrides: Optional[dict[str, dict[str, Any]]] = None
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


class HyperparamMetaOut(BaseModel):
    """Un hyperparamètre réglable en mode expert (retour utilisateur direct :
    "laisser le choix sur les hyperparamètres, profondeur des arbres
    etc.") — décrit le type de contrôle à rendre côté frontend, jamais le
    catalogue de modèles codé en dur côté client."""
    name: str
    label: str
    kind: str  # "int" | "float" | "categorical"
    low: Optional[float] = None
    high: Optional[float] = None
    log: bool = False
    choices: Optional[List[str]] = None
    help: str = ""


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
    tunable_hyperparameters: List[HyperparamMetaOut] = []


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
    # Lot 3 (correctif I1, AUDIT_DATALAB_2026-08-16.md §E.3) — verdict en
    # langage clair, {"claims": [...], "next_actions": [{"code":,
    # "action":}, ...]}, voir services/verdict.py. Toujours présent
    # (calculé à la volée, jamais persisté — les règles peuvent évoluer
    # sans backfill).
    verdict: dict[str, Any]
    # Lot 9 — registre de modèles versionné. `None` = jamais promu.
    stage: Optional[str] = None
    promoted_at: Optional[datetime] = None
    # Lot 5 (correctif P1) — numéro de version au sein du problème
    # (dataset + cible), voir api/core/models.py::MLModel.version.
    version: int
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

_VALID_STAGES = {"none", "staging", "production", "archived"}


class PromoteModelRequest(BaseModel):
    stage: str  # "none" | "staging" | "production" | "archived"


class ModelRegistryEntry(BaseModel):
    """Une entrée du registre — un modèle PROMU (staging ou production),
    avec assez de contexte (dataset/cible/algorithme/métrique) pour
    identifier de quel problème il s'agit sans recharger le job complet.
    Un modèle "archived" (Lot 5, correctif P1) n'apparaît JAMAIS ici —
    explicitement retiré, voir list_model_registry."""
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


class ModelVersionEntry(BaseModel):
    """Une version du "problème" (dataset + cible) — Lot 5, correctif
    P1. Permet de voir tout l'historique d'un problème et d'identifier
    le job_id d'une version antérieure pour y revenir (rollback : voir
    promote_model, aucun endpoint dédié — repromouvoir une version
    antérieure DÉMET automatiquement la version courante, même mécanisme
    qu'une promotion normale)."""
    job_id: int
    model_id: int
    version: int
    algorithm: str
    stage: Optional[str] = None
    promoted_at: Optional[datetime] = None
    created_at: datetime
    headline_metric: Optional[dict[str, Any]] = None


class ModelVersionsResponse(BaseModel):
    entries: List[ModelVersionEntry]


class ModelTransitionEntry(BaseModel):
    """Une transition de stage passée (Lot 5, correctif P1) — lue depuis
    le journal d'audit existant (`AuditLog`, action "model.promoted",
    déjà écrit par promote_model depuis le Lot 9), jamais un second
    mécanisme de journalisation parallèle."""
    model_id: int
    version: int
    stage: str
    actor: Optional[str] = None
    created_at: datetime


class ModelHistoryResponse(BaseModel):
    entries: List[ModelTransitionEntry]


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


class PredictionHistoryEntry(BaseModel):
    """Une prédiction passée (Lot 5, correctif I2) — mêmes champs que
    `PredictionResponse`, jamais `explanation` (jamais persistée, voir
    `api/core/models.py::Prediction`).

    `dataset_id`/`training_job_id`/`model_version` (Phase 3,
    AUDIT_BACKEND_2026-08-23.md, Axe I) — lignage prédiction → dataset
    directement exposé plutôt que seulement reconstructible par une
    jointure manuelle (`Prediction.ml_model_id → MLModel.dataset_id`/
    `training_job_id`) : ce sont les MÊMES colonnes que celles déjà lues
    sur `job`/`job.model` pour répondre à cet endpoint, jamais dupliquées
    en base sur `Prediction` elle-même (le modèle documente déjà ce choix
    délibéré — un seul point de vérité, `MLModel`)."""
    id: int
    input: dict[str, Any]
    prediction: Any
    probabilities: Optional[dict[str, float]] = None
    interval: Optional[dict[str, float]] = None
    requested_by: Optional[str] = None
    dataset_id: int
    training_job_id: int
    model_version: int
    created_at: datetime


class PredictionHistoryResponse(BaseModel):
    entries: List[PredictionHistoryEntry]


class DriftFeatureOut(BaseModel):
    """Une variable du rapport de dérive — voir `domains/shared/drift.py`
    pour la méthode (PSI) et les seuils de `severity`."""
    feature: str
    psi: float
    severity: str


class DriftReportOut(BaseModel):
    n_predictions_analyzed: int
    insufficient_data: bool
    features: List[DriftFeatureOut]
    n_significant: int
    n_moderate: int
    min_predictions_required: int


class BatchPredictionJobSummary(BaseModel):
    """Prédiction en lot (retour utilisateur : "batch prediction" — upload
    d'un fichier, prédictions pour toutes les lignes) — même forme que
    `TrainingJobSummary`/`VisionClassificationJobSummary` (statut/
    progression/erreur), pour un traitement UI cohérent avec les autres
    types de job."""
    id: int
    training_job_id: int
    input_filename: str
    status: str
    progress_step: Optional[str] = None
    progress_percent: int
    error_message: Optional[str] = None
    n_rows: Optional[int] = None
    created_by: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


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


def to_summary(job: TrainingJob) -> TrainingJobSummary:
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
                detail={
                    "code": ErrorCode.COLONNES_INCONNUES,
                    "message": f"Colonnes absentes du dataset : {', '.join(sorted(unknown))}",
                },
            )

    freq_cols = fe.pipeline.get("frequency_encoding") or []
    imputation_cols = list((fe.pipeline.get("imputation") or {}).keys())
    unknown_pipeline = (set(freq_cols) | set(imputation_cols)) - known_columns
    if unknown_pipeline:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": ErrorCode.COLONNES_INCONNUES,
                "message": f"Colonnes absentes du dataset : {', '.join(sorted(unknown_pipeline))}",
            },
        )

    return json.dumps({"version": CURRENT_SPEC_VERSION, "upstream": fe.upstream, "pipeline": fe.pipeline})


def _validate_hyperparameter_overrides(
    overrides: Optional[dict[str, dict[str, Any]]], effective_model_ids: set[str], task_type: str
) -> None:
    """Mode expert hyperparamètres (retour utilisateur direct : "laisser le
    choix sur les hyperparamètres, profondeur des arbres etc.") — jamais
    fait confiance au client : rejette toute surcharge qui référence un
    modèle inconnu ou non sélectionné pour ce job, un nom d'hyperparamètre
    qui n'existe pas pour ce modèle, ou une valeur hors du type/des bornes
    déclarés dans `ModelSpec.tunable_hyperparameters` — avant d'enfiler le
    job, jamais découvert en pleine recherche Optuna."""
    if not overrides:
        return
    for model_id, params in overrides.items():
        spec = MODEL_REGISTRY.get(model_id)
        if spec is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "MODELE_INCONNU", "message": f"Modèle inconnu pour les hyperparamètres : {model_id}"},
            )
        if model_id not in effective_model_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": "MODELE_NON_SELECTIONNE",
                    "message": f"{spec.label(task_type)} : hyperparamètres fixés hors des modèles comparés",
                },
            )
        meta_by_name = {m.name: m for m in spec.tunable_hyperparameters}
        for name, value in params.items():
            meta = meta_by_name.get(name)
            if meta is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "code": "HYPERPARAMETRE_INCONNU",
                        "message": f"« {name} » n'est pas un hyperparamètre réglable de « {spec.label(task_type)} »",
                    },
                )
            if meta.kind == "categorical":
                if value not in (meta.choices or ()):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail={
                            "code": "HYPERPARAMETRE_INVALIDE",
                            "message": f"{name} doit être l'une de {list(meta.choices or ())}",
                        },
                    )
            else:
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail={
                            "code": "HYPERPARAMETRE_INVALIDE",
                            "message": f"« {name} » doit être un nombre pour « {spec.label(task_type)} »",
                        },
                    )
                if (meta.low is not None and value < meta.low) or (meta.high is not None and value > meta.high):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail={
                            "code": "HYPERPARAMETRE_HORS_BORNES",
                            "message": f"{name} doit être entre {meta.low} et {meta.high}",
                        },
                    )


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


def _get_org_batch_job(batch_job_id: int, current_user: User, db: Session) -> BatchPredictionJob:
    job = (
        db.query(BatchPredictionJob)
        .filter(
            BatchPredictionJob.id == batch_job_id,
            BatchPredictionJob.organization_id == current_user.organization_id,
        )
        .first()
    )
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "PREDICTION_LOT_INTROUVABLE", "message": "Prédiction en lot introuvable"},
        )
    return job


def _to_batch_summary(job: BatchPredictionJob) -> BatchPredictionJobSummary:
    return BatchPredictionJobSummary(
        id=job.id,
        training_job_id=job.training_job_id,
        input_filename=job.input_filename,
        status=job.status,
        progress_step=job.progress_step,
        progress_percent=job.progress_percent,
        error_message=job.error_message,
        n_rows=job.n_rows,
        created_by=job.created_by.nom if job.created_by else None,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


_batch_prediction_upload_rate_limit = rate_limit_dependency(
    "batch_prediction_upload", _settings.upload_rate_limit_max_attempts, _settings.upload_rate_limit_window_seconds,
    # Même raisonnement que `datasets.py::_upload_rate_limit` — échec fermé :
    # l'upload lit un fichier entier en mémoire, jamais laissé passer sans
    # limite parce que Redis est momentanément indisponible.
    fail_open=False,
)


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
                tunable_hyperparameters=[
                    HyperparamMetaOut(
                        name=m.name,
                        label=m.label,
                        kind=m.kind,
                        low=m.low,
                        high=m.high,
                        log=m.log,
                        choices=list(m.choices) if m.choices else None,
                        help=m.help,
                    )
                    for m in spec.tunable_hyperparameters
                ],
            )
        )
    return ModelCatalogResponse(models=entries)


class DurationEstimateOut(BaseModel):
    status: str  # "estimated" | "degraded"
    estimated_seconds: Optional[float] = None
    based_on_n_jobs: int
    message: Optional[str] = None


@router.get("/estimate-duration", response_model=DurationEstimateOut)
def get_duration_estimate(
    dataset_id: int,
    n_models: int = Query(4, ge=1, le=9),
    optuna_trials: int = Query(20, ge=3, le=100),
    cv_folds: int = Query(4, ge=2, le=10),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Estimation de la durée AVANT lancement (Lot 7, §J.1) — dérivée des
    entraînements RÉELLEMENT terminés par cette organisation, jamais d'une
    constante inventée. Voir `services/duration_estimate.py`."""
    dataset = (
        db.query(Dataset)
        .filter(Dataset.id == dataset_id, Dataset.organization_id == current_user.organization_id)
        .first()
    )
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": ErrorCode.DATASET_INTROUVABLE, "message": "Dataset introuvable"},
        )
    estimate = estimate_training_duration(
        db, current_user.organization_id, dataset.row_count or 0, n_models, optuna_trials, cv_folds
    )
    return DurationEstimateOut(
        status=estimate.status,
        estimated_seconds=estimate.estimated_seconds,
        based_on_n_jobs=estimate.based_on_n_jobs,
        message=estimate.message,
    )


@router.post("/jobs", response_model=TrainingJobSummary, status_code=status.HTTP_201_CREATED)
def create_training_job(
    body: TrainingJobCreate,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Idempotence (Phase 2, AUDIT_BACKEND_2026-08-23.md §F4) — un double-clic
    # ou une requête retentée après un timeout réseau (le serveur avait
    # pourtant déjà traité la première) créait deux jobs identiques,
    # consommant deux fois le quota. `Idempotency-Key` optionnel, fourni par
    # le client (frontend/src/api/client.ts) : si déjà vue pour cette
    # organisation, renvoie le job déjà créé plutôt que d'en créer un
    # nouveau.
    existing_job_id = resolve_idempotent_job_id(redis_conn, current_user.organization_id, request)
    if existing_job_id is not None:
        existing = (
            db.query(TrainingJob)
            .filter(TrainingJob.id == existing_job_id, TrainingJob.organization_id == current_user.organization_id)
            .first()
        )
        if existing is not None:
            return to_summary(existing)

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
        ALL_JOB_MODELS,
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
            detail={"code": ErrorCode.DATASET_INTROUVABLE, "message": "Dataset introuvable"},
        )
    if dataset.status != "ready":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": ErrorCode.DATASET_NON_PRET, "message": "Ce dataset n'a pas pu être analysé"},
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
            detail={
                "code": ErrorCode.COLONNES_INCONNUES,
                "message": f"Colonnes absentes du dataset : {', '.join(sorted(unknown))}",
            },
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
        df = read_dataset_dataframe(Path(dataset.file_path), Path(dataset.file_path).suffix)
    except DatasetParsingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": ErrorCode.DATASET_LECTURE_ECHEC, "message": str(exc)},
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

    # Mode expert hyperparamètres (retour utilisateur direct : "laisser le
    # choix sur les hyperparamètres, profondeur des arbres etc.") — jamais
    # fait confiance au client : un nom de modèle/hyperparamètre inconnu ou
    # une valeur hors bornes déclarées est rejetée ici, avant d'enfiler le
    # job (jamais découvert 20 essais Optuna plus tard).
    effective_model_ids = set(selected_model_ids) if selected_model_ids is not None else {
        s.id for s in models_for_task(task_type, subset="default")
    }
    _validate_hyperparameter_overrides(body.hyperparameter_overrides, effective_model_ids, task_type)

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
        "hyperparameter_overrides": body.hyperparameter_overrides,
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
        request_id=request.state.request_id,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    remember_idempotent_job_id(redis_conn, current_user.organization_id, request, job.id)
    # Phase 3 (AUDIT_BACKEND_2026-08-23.md, Axe I) — la création de job
    # n'était auditée dans AUCUN des 6 domaines avant ce correctif (seuls
    # `cancel`/`delete` l'étaient) : un owner ne pouvait pas répondre à
    # "qui a lancé cet entraînement, et quand" depuis le journal d'audit.
    # Committé par `enqueue_or_mark_failed` juste après (même transaction).
    log_action(
        db, current_user.organization_id, current_user.id, "training_job.created",
        target_type="training_job", target_id=job.id,
    )

    from domains.training.worker import run_training_job

    # F5 — n'enfile jamais un job qui resterait "queued" pour toujours si
    # Redis tombe à cet instant précis (voir domains/shared/job_creation.py).
    enqueue_or_mark_failed(db, job, training_queue, run_training_job, 1800)

    return to_summary(job)


@router.get("/jobs", response_model=List[TrainingJobSummary])
def list_training_jobs(
    response: Response,
    limit: Optional[int] = Query(None, ge=1, le=500),
    cursor: Optional[int] = Query(None, description="id de la dernière ligne de la page précédente"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # joinedload (Lot 4, correctif I3) : to_summary accède à job.dataset,
    # job.created_by et job.model — sans ça, 3 requêtes SQL PAR JOB
    # (N+1, AUDIT_DATALAB_2026-08-16.md §C.2.4). Un seul aller-retour
    # désormais, quel que soit le nombre de jobs.
    query = (
        db.query(TrainingJob)
        .options(joinedload(TrainingJob.dataset), joinedload(TrainingJob.created_by), joinedload(TrainingJob.model))
        .filter(TrainingJob.organization_id == current_user.organization_id)
        # id DESC, pas created_at DESC — voir anomalies.py::list_anomaly_jobs.
        .order_by(TrainingJob.id.desc())
    )
    jobs = paginate_by_id(query, TrainingJob.id, response, cursor, limit)
    return [to_summary(j) for j in jobs]


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
    return to_summary(_get_org_job(job_id, current_user, db))


@router.get("/jobs/{job_id}/events")
async def stream_training_job_events(job_id: int, current_user: User = Depends(get_current_user)):
    """Notifications de fin de job par SSE (Lot 7, §J.2) — remplace le
    polling `setInterval` côté frontend. Vérifie l'existence/l'appartenance
    du job une première fois avant d'ouvrir le flux (404 propre plutôt qu'un
    flux qui s'ouvre puis échoue)."""
    organization_id = current_user.organization_id
    db = SessionLocal()
    try:
        job = db.query(TrainingJob).filter(TrainingJob.id == job_id, TrainingJob.organization_id == organization_id).first()
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "TRAINING_JOB_INTROUVABLE", "message": "Entraînement introuvable"},
            )
    finally:
        db.close()

    def fetch_snapshot():
        session = SessionLocal()
        try:
            row = session.query(TrainingJob).filter(TrainingJob.id == job_id, TrainingJob.organization_id == organization_id).first()
            if row is None:
                return None
            return {
                "status": row.status,
                "progress_percent": row.progress_percent,
                "progress_step": row.progress_step,
                "error_message": row.error_message,
            }
        finally:
            session.close()

    return StreamingResponse(stream_job_updates(fetch_snapshot), media_type="text/event-stream")


def to_model_detail(model: MLModel, db: Session) -> MLModelDetail:
    metrics = json.loads(model.metrics_json)
    evaluation = json.loads(model.evaluation_json) if model.evaluation_json else {}
    calibration = json.loads(model.calibration_json) if model.calibration_json else None
    learning_curve = json.loads(model.learning_curve_json) if model.learning_curve_json else None
    cqr = json.loads(model.cqr_json) if model.cqr_json else None
    # Lot 3 (correctif I1) — même requête que GET /jobs/{id}/candidates,
    # nécessaire ici pour juger l'écart gagnant/2ᵉ (services/model_verdict.py) ;
    # [] pour un job antérieur au Lot D (jamais de ModelCandidate), le
    # verdict omet alors simplement cette affirmation, pas d'erreur.
    candidates = [
        {
            "algorithm": row.algorithm,
            "rank": row.rank,
            "selection_score": row.selection_score,
            "fold_scores": json.loads(row.fold_scores_json) if row.fold_scores_json else None,
        }
        for row in (
            db.query(ModelCandidate)
            .filter(ModelCandidate.training_job_id == model.training_job_id, ModelCandidate.organization_id == model.organization_id)
            .order_by(ModelCandidate.rank.asc())
            .all()
        )
    ]
    model_card = json.loads(model.model_card_json) if model.model_card_json else {}
    verdict = compute_verdict(
        model.task_type,
        metrics,
        evaluation,
        candidates,
        calibration,
        learning_curve,
        cqr,
        duplicates_removed=model_card.get("duplicates_removed"),
        anti_leak_grouping=model_card.get("anti_leak_grouping"),
    )

    return MLModelDetail(
        id=model.id,
        training_job_id=model.training_job_id,
        algorithm=model.algorithm,
        task_type=model.task_type,
        target_column=model.target_column,
        feature_columns=json.loads(model.feature_columns_json),
        feature_schema=json.loads(model.feature_schema_json) if model.feature_schema_json else [],
        metrics=metrics,
        shap_summary=json.loads(model.shap_summary_json) if model.shap_summary_json else [],
        cqr=cqr,
        model_card=model_card,
        evaluation=evaluation,
        feature_engineering=json.loads(model.feature_engineering_json) if model.feature_engineering_json else None,
        shap_beeswarm=json.loads(model.shap_beeswarm_json) if model.shap_beeswarm_json else {},
        permutation_importance=json.loads(model.permutation_importance_json) if model.permutation_importance_json else [],
        calibration=calibration,
        learning_curve=learning_curve,
        verdict=verdict,
        stage=model.stage,
        promoted_at=model.promoted_at,
        version=model.version,
        created_at=model.created_at,
    )


@router.get("/jobs/{job_id}/model", response_model=MLModelDetail)
def get_training_job_model(job_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    job = _get_org_job(job_id, current_user, db)
    if job.status != "completed" or job.model is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": ErrorCode.MODELE_NON_DISPONIBLE,
                "message": "Cet entraînement n'a pas encore produit de modèle",
            },
        )
    return to_model_detail(job.model, db)


@router.post("/jobs/{job_id}/model/promote", response_model=MLModelDetail)
def promote_model(
    job_id: int,
    body: PromoteModelRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Promotion d'un modèle (Lot 9) — "staging" (à valider), "production"
    (celui utilisé en confiance pour ce problème), "archived" (Lot 5,
    correctif P1 : retiré du registre actif sans être supprimé, pour
    désencombrer `GET /models/registry` d'anciennes versions non
    pertinentes) ou "none" (retrait, sans connotation d'archivage).

    Règle du registre : UN SEUL modèle "production" à la fois par couple
    (dataset, cible) au sein d'une organisation — promouvoir un nouveau
    modèle en production DÉMET automatiquement l'ancien (repasse à
    "staging", jamais supprimé ni écrasé, juste son statut qui change),
    pour qu'il n'y ait jamais d'ambiguïté sur "quel modèle fait autorité".
    Aucune limite en "staging" : plusieurs candidats peuvent y attendre
    validation en parallèle.

    Rollback (Lot 5, correctif P1) : aucun endpoint dédié — repromouvoir
    une version ANTÉRIEURE en "production" (via son propre job_id, voir
    `GET /jobs/{id}/model/versions` pour retrouver ce job_id) déclenche
    exactement le même mécanisme de démotion ci-dessus, donc revient bien
    en arrière sans code séparé à maintenir."""
    if body.stage not in _VALID_STAGES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "STAGE_INVALIDE",
                "message": f"Statut inconnu : {body.stage!r} (attendu : none/staging/production/archived)",
            },
        )
    job = _get_org_job(job_id, current_user, db)
    if job.status != "completed" or job.model is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": ErrorCode.MODELE_NON_DISPONIBLE,
                "message": "Cet entraînement n'a pas encore produit de modèle",
            },
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
    return to_model_detail(model, db)


@router.get("/jobs/{job_id}/model/versions", response_model=ModelVersionsResponse)
def list_model_versions(job_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Toutes les versions du "problème" (même dataset + même cible) que
    le modèle de ce job (Lot 5, correctif P1) — la plus récente d'abord.
    Permet de retrouver le job_id d'une version antérieure pour la
    repromouvoir (rollback, voir promote_model)."""
    job = _get_org_job(job_id, current_user, db)
    if job.model is None:
        return ModelVersionsResponse(entries=[])
    rows = (
        db.query(MLModel)
        .filter(
            MLModel.organization_id == current_user.organization_id,
            MLModel.dataset_id == job.dataset_id,
            MLModel.target_column == job.target_column,
        )
        .order_by(MLModel.version.desc())
        .all()
    )
    entries = []
    for row in rows:
        metrics = json.loads(row.metrics_json) if row.metrics_json else {}
        entries.append(ModelVersionEntry(
            job_id=row.training_job_id,
            model_id=row.id,
            version=row.version,
            algorithm=row.algorithm,
            stage=row.stage,
            promoted_at=row.promoted_at,
            created_at=row.created_at,
            headline_metric=_headline_metric(row.task_type, metrics) if metrics else None,
        ))
    return ModelVersionsResponse(entries=entries)


@router.get("/jobs/{job_id}/model/history", response_model=ModelHistoryResponse)
def get_model_history(job_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Historique des transitions de stage pour TOUTES les versions de ce
    problème (Lot 5, correctif P1) — lu depuis `AuditLog` (action
    "model.promoted", déjà écrite par promote_model depuis le Lot 9),
    jamais un second mécanisme de journalisation parallèle. Le plus
    récent d'abord."""
    job = _get_org_job(job_id, current_user, db)
    if job.model is None:
        return ModelHistoryResponse(entries=[])
    version_by_model_id = {
        row.id: row.version
        for row in db.query(MLModel.id, MLModel.version).filter(
            MLModel.organization_id == current_user.organization_id,
            MLModel.dataset_id == job.dataset_id,
            MLModel.target_column == job.target_column,
        )
    }
    if not version_by_model_id:
        return ModelHistoryResponse(entries=[])
    logs = (
        db.query(AuditLog)
        .filter(
            AuditLog.organization_id == current_user.organization_id,
            AuditLog.action == "model.promoted",
            AuditLog.target_id.in_(list(version_by_model_id.keys())),
        )
        .order_by(AuditLog.id.desc())
        .all()
    )
    entries = []
    for log in logs:
        details = json.loads(log.details_json) if log.details_json else {}
        entries.append(ModelTransitionEntry(
            model_id=log.target_id,
            version=version_by_model_id[log.target_id],
            stage=details.get("stage", "?"),
            actor=log.actor.nom if log.actor else None,
            created_at=log.created_at,
        ))
    return ModelHistoryResponse(entries=entries)


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
            detail={
                "code": ErrorCode.MODELE_NON_DISPONIBLE,
                "message": "Cet entraînement n'a pas encore produit de modèle",
            },
        )
    artifact_path = Path(job.model.file_path)
    if not artifact_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": ErrorCode.ARTEFACT_INTROUVABLE, "message": "Artefact du modèle introuvable sur le serveur"},
        )
    filename = f"modele_{job.dataset.name.rsplit('.', 1)[0] if job.dataset else 'export'}_{job.target_column}_job{job.id}.joblib"
    return FileResponse(path=artifact_path, filename=filename, media_type="application/octet-stream")


@router.get("/jobs/{job_id}/model/export-script")
def export_deployment_script(
    job_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """Script de déploiement autonome (retour utilisateur direct : "tous les
    modèles doivent pouvoir être déployés dans d'autres plateformes") — un
    fichier `.py` prêt à l'emploi à côté de l'artefact (`.../model/export`
    ci-dessus), aucune dépendance à ce projet. Voir
    `services/deployment_export.py` pour la génération complète."""
    job = _get_org_job(job_id, current_user, db)
    if job.status != "completed" or job.model is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": ErrorCode.MODELE_NON_DISPONIBLE,
                "message": "Cet entraînement n'a pas encore produit de modèle",
            },
        )
    model = job.model
    artifact_path = Path(model.file_path)
    if not artifact_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": ErrorCode.ARTEFACT_INTROUVABLE, "message": "Artefact du modèle introuvable sur le serveur"},
        )
    try:
        bundle = load_bundle(model.file_path)
    except InferenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": ErrorCode.ARTEFACT_ILLISIBLE, "message": str(exc)},
        ) from exc

    dataset_name = job.dataset.name.rsplit(".", 1)[0] if job.dataset else "export"
    base_name = f"modele_{dataset_name}_{job.target_column}_job{job.id}"
    artifact_filename = f"{base_name}.joblib"
    script_filename = f"{base_name}_deploiement.py"
    script = generate_deployment_script(
        bundle=bundle,
        feature_columns=json.loads(model.feature_columns_json),
        algorithm=model.algorithm,
        task_type=model.task_type,
        target_column=model.target_column,
        artifact_filename=artifact_filename,
        script_filename=script_filename,
    )
    return Response(
        content=script,
        media_type="text/x-python",
        headers={"Content-Disposition": f'attachment; filename="{script_filename}"'},
    )


@router.get("/models/registry", response_model=ModelRegistryResponse)
def list_model_registry(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Registre de modèles (Lot 9) — tous les modèles PROMUS (staging ou
    production) de l'organisation, tous datasets/cibles confondus. Un modèle
    jamais promu (`stage IS NULL`, comportement historique) n'y apparaît
    jamais — le registre n'est PAS un doublon de l'historique complet
    (`GET /training/jobs`), seulement ce qui a été explicitement retenu.

    Un modèle "archived" (Lot 5, correctif P1) n'y apparaît pas non plus —
    explicitement retiré du registre actif, mais toujours consultable via
    `GET /jobs/{id}/model/versions` (tout l'historique du problème)."""
    models = (
        db.query(MLModel)
        .join(TrainingJob, MLModel.training_job_id == TrainingJob.id)
        .filter(MLModel.organization_id == current_user.organization_id, MLModel.stage.in_(("staging", "production")))
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

    Lot 5 (correctif I2) — chaque prédiction réussie est persistée
    (`Prediction`, voir `api/core/models.py`) : entrée, sortie
    (prédiction + probabilités + intervalle, jamais `explanation` —
    recalculable, voir `PredictionHistoryEntry`), modèle, utilisateur,
    date. Une prédiction en échec (`InferenceError`) n'a rien à
    persister — il n'y a pas de sortie."""
    job = _get_org_job(job_id, current_user, db)
    if job.status != "completed" or job.model is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": ErrorCode.MODELE_NON_DISPONIBLE,
                "message": "Cet entraînement n'a pas encore produit de modèle",
            },
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

    purge_old_predictions(db, current_user.organization_id, _settings.prediction_retention_days)
    output = {k: v for k, v in result.items() if k != "explanation"}
    db.add(Prediction(
        organization_id=current_user.organization_id,
        ml_model_id=model.id,
        requested_by_id=current_user.id,
        input_json=json.dumps(body.data),
        output_json=json.dumps(output),
    ))
    db.commit()

    return PredictionResponse(**result)


@router.get("/jobs/{job_id}/predictions", response_model=PredictionHistoryResponse)
def list_job_predictions(
    job_id: int,
    limit: int = Query(50, ge=1, le=500),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Historique des prédictions faites avec le modèle de ce job (Lot 5,
    correctif I2) — traçabilité : qui a demandé quoi, avec quelle réponse,
    quand. Les plus récentes d'abord.

    Pas de curseur ici (contrairement aux listes de jobs, Lot 4/I3, sur
    une branche distincte au moment de ce lot) — un simple `limit` borné,
    suffisant tant que la rétention (`services/prediction_retention.py`)
    plafonne déjà la taille de cette table dans le temps ; à harmoniser
    avec `api/core/pagination.py` si les deux lots fusionnent."""
    job = _get_org_job(job_id, current_user, db)
    if job.model is None:
        return PredictionHistoryResponse(entries=[])
    rows = (
        db.query(Prediction)
        .filter(Prediction.ml_model_id == job.model.id, Prediction.organization_id == current_user.organization_id)
        .order_by(Prediction.id.desc())
        .limit(limit)
        .all()
    )
    entries = []
    for row in rows:
        output = json.loads(row.output_json)
        entries.append(PredictionHistoryEntry(
            id=row.id,
            input=json.loads(row.input_json),
            prediction=output.get("prediction"),
            probabilities=output.get("probabilities"),
            interval=output.get("interval"),
            requested_by=row.requested_by.nom if row.requested_by else None,
            dataset_id=job.dataset_id,
            training_job_id=job.id,
            model_version=job.model.version,
            created_at=row.created_at,
        ))
    return PredictionHistoryResponse(entries=entries)


@router.get("/jobs/{job_id}/model/drift", response_model=DriftReportOut)
def get_model_drift(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Dérive des données (data drift) — compare la distribution des
    variables d'entrée réellement envoyées en production
    (`Prediction.input_json`, les plus récentes d'abord) à celle du
    dataset d'entraînement, variable par variable (PSI, voir
    `domains/shared/drift.py`).

    Manque identifié en évaluant une maquette externe (dont le texte
    anticipe "le plus ancien n'a pas été revérifié — contrôler sa dérive")
    face à notre propre tableau de bord ("Fiabilité des modèles actifs",
    `domains/dashboard/router.py`), qui n'offrait jusqu'ici aucun moyen de
    le faire — première fonctionnalité de suivi POST-déploiement du
    produit (verdict/seuil/fiabilité s'arrêtaient tous à l'instant du
    déploiement)."""
    job = _get_org_job(job_id, current_user, db)
    if job.model is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": ErrorCode.MODELE_NON_DISPONIBLE,
                "message": "Cet entraînement n'a pas encore produit de modèle",
            },
        )
    model = job.model
    feature_columns = json.loads(model.feature_columns_json)

    rows = (
        db.query(Prediction)
        .filter(Prediction.ml_model_id == model.id, Prediction.organization_id == current_user.organization_id)
        .order_by(Prediction.id.desc())
        .limit(MAX_PREDICTIONS_FOR_DRIFT)
        .all()
    )
    if len(rows) < MIN_CURRENT_ROWS_FOR_DRIFT:
        return DriftReportOut(
            n_predictions_analyzed=len(rows),
            insufficient_data=True,
            features=[],
            n_significant=0,
            n_moderate=0,
            min_predictions_required=MIN_CURRENT_ROWS_FOR_DRIFT,
        )

    current_df = pd.DataFrame([json.loads(row.input_json) for row in rows])
    try:
        reference_df = read_dataset_dataframe(Path(job.dataset.file_path), Path(job.dataset.file_path).suffix)
    except (DatasetParsingError, UnsupportedFileType) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": ErrorCode.DATASET_LECTURE_ECHEC, "message": str(exc)},
        ) from exc

    report = compute_drift_report(reference_df, current_df, feature_columns)
    return DriftReportOut(**report, min_predictions_required=MIN_CURRENT_ROWS_FOR_DRIFT)


# ── Prédiction en lot (retour utilisateur : "batch prediction" — upload
# d'un fichier, prédictions pour toutes les lignes) ─────────────────────────


@router.post(
    "/jobs/{job_id}/predict-batch",
    response_model=BatchPredictionJobSummary,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(_batch_prediction_upload_rate_limit)],
)
async def create_batch_prediction_job(
    job_id: int,
    request: Request,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Lance une prédiction en lot avec le modèle de ce job — upload d'un
    fichier (csv/xlsx/xls/parquet/json, mêmes formats que l'upload de
    dataset), une prédiction par ligne, résultat téléchargeable une fois le
    job terminé (`GET /training/batch-predictions/{id}/download`).

    Tâche de fond (`analysis_queue`) — jamais de calcul dans la requête HTTP,
    la taille du fichier n'est pas bornée à l'avance (contrairement à
    `POST /jobs/{id}/predict`, une seule observation)."""
    job = _get_org_job(job_id, current_user, db)
    if job.status != "completed" or job.model is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": ErrorCode.MODELE_NON_DISPONIBLE,
                "message": "Cet entraînement n'a pas encore produit de modèle",
            },
        )

    existing_batch_id = resolve_idempotent_job_id(redis_conn, current_user.organization_id, request)
    if existing_batch_id is not None:
        existing = (
            db.query(BatchPredictionJob)
            .filter(
                BatchPredictionJob.id == existing_batch_id,
                BatchPredictionJob.organization_id == current_user.organization_id,
            )
            .first()
        )
        if existing is not None:
            return _to_batch_summary(existing)

    try:
        extension = validate_extension(file.filename or "")
    except UnsupportedFileType as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "DATASET_FORMAT_NON_SUPPORTE", "message": str(exc)},
        ) from exc

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "DATASET_FICHIER_VIDE", "message": "Le fichier est vide"},
        )
    max_upload_bytes = _settings.max_upload_size_mb * 1024 * 1024
    if len(content) > max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "code": "DATASET_TROP_VOLUMINEUX",
                "message": f"Fichier trop volumineux (max {_settings.max_upload_size_mb} Mo)",
            },
        )

    reconcile_stale_jobs(
        db, current_user.organization_id, _settings.stale_job_timeout_minutes, model=BatchPredictionJob
    )
    raise_if_quota_exceeded(db, current_user.organization_id, ALL_JOB_MODELS, _settings.max_concurrent_jobs_per_org)

    batch_job = BatchPredictionJob(
        organization_id=current_user.organization_id,
        training_job_id=job.id,
        created_by_id=current_user.id,
        input_filename=file.filename or "fichier",
        input_file_path="",
        status="queued",
        request_id=request.state.request_id,
    )
    db.add(batch_job)
    db.flush()

    target_path = batch_prediction_input_file_path(current_user.organization_id, batch_job.id, extension)
    target_path.write_bytes(content)
    batch_job.input_file_path = str(target_path)
    db.commit()
    db.refresh(batch_job)

    remember_idempotent_job_id(redis_conn, current_user.organization_id, request, batch_job.id)
    log_action(
        db, current_user.organization_id, current_user.id, "batch_prediction_job.created",
        target_type="batch_prediction_job", target_id=batch_job.id,
    )

    enqueue_or_mark_failed(db, batch_job, analysis_queue, run_batch_prediction_job, 600)

    return _to_batch_summary(batch_job)


@router.get("/batch-predictions", response_model=List[BatchPredictionJobSummary])
def list_batch_prediction_jobs(
    response: Response,
    limit: Optional[int] = Query(None, ge=1, le=500),
    cursor: Optional[int] = Query(None, description="id de la dernière ligne de la page précédente"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = (
        db.query(BatchPredictionJob)
        .options(joinedload(BatchPredictionJob.created_by))
        .filter(BatchPredictionJob.organization_id == current_user.organization_id)
        .order_by(BatchPredictionJob.id.desc())
    )
    jobs = paginate_by_id(query, BatchPredictionJob.id, response, cursor, limit)
    return [_to_batch_summary(j) for j in jobs]


@router.get("/batch-predictions/{batch_job_id}", response_model=BatchPredictionJobSummary)
def get_batch_prediction_job(
    batch_job_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    return _to_batch_summary(_get_org_batch_job(batch_job_id, current_user, db))


@router.get("/batch-predictions/{batch_job_id}/events")
async def stream_batch_prediction_job_events(batch_job_id: int, current_user: User = Depends(get_current_user)):
    """Notifications de progression par SSE — voir
    `training.py::stream_training_job_events` pour le raisonnement complet."""
    organization_id = current_user.organization_id
    db = SessionLocal()
    try:
        job = (
            db.query(BatchPredictionJob)
            .filter(BatchPredictionJob.id == batch_job_id, BatchPredictionJob.organization_id == organization_id)
            .first()
        )
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "PREDICTION_LOT_INTROUVABLE", "message": "Prédiction en lot introuvable"},
            )
    finally:
        db.close()

    def fetch_snapshot():
        session = SessionLocal()
        try:
            row = (
                session.query(BatchPredictionJob)
                .filter(BatchPredictionJob.id == batch_job_id, BatchPredictionJob.organization_id == organization_id)
                .first()
            )
            if row is None:
                return None
            return {
                "status": row.status,
                "progress_percent": row.progress_percent,
                "progress_step": row.progress_step,
                "error_message": row.error_message,
            }
        finally:
            session.close()

    return StreamingResponse(stream_job_updates(fetch_snapshot), media_type="text/event-stream")


@router.get("/batch-predictions/{batch_job_id}/download")
def download_batch_prediction_result(
    batch_job_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    job = _get_org_batch_job(batch_job_id, current_user, db)
    if job.status != "completed" or not job.output_file_path:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": ErrorCode.RESULTAT_INDISPONIBLE,
                "message": "Cette prédiction en lot n'a pas encore de résultat",
            },
        )
    output_path = Path(job.output_file_path)
    if not output_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "RESULTAT_INTROUVABLE", "message": "Résultat introuvable sur le serveur"},
        )
    filename = f"predictions_{Path(job.input_filename).stem}.csv"
    return FileResponse(path=output_path, filename=filename, media_type="text/csv")


@router.get("/batch-predictions/{batch_job_id}/download-excel")
def download_batch_prediction_result_excel(
    batch_job_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """Même résultat que `.../download`, au format Excel (retour utilisateur
    direct : "on doit télécharger aussi les prédictions en format excel pour
    voir directement") — généré à la volée depuis le CSV déjà stocké, jamais
    un second fichier persisté sur le serveur (même principe que le modèle
    de fichier vide côté frontend : un format d'affichage, pas une donnée
    supplémentaire à retenir)."""
    job = _get_org_batch_job(batch_job_id, current_user, db)
    if job.status != "completed" or not job.output_file_path:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": ErrorCode.RESULTAT_INDISPONIBLE,
                "message": "Cette prédiction en lot n'a pas encore de résultat",
            },
        )
    output_path = Path(job.output_file_path)
    if not output_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "RESULTAT_INTROUVABLE", "message": "Résultat introuvable sur le serveur"},
        )
    result_df = pd.read_csv(output_path)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        result_df.to_excel(writer, index=False, sheet_name="Prédictions")
    buffer.seek(0)
    filename = f"predictions_{Path(job.input_filename).stem}.xlsx"
    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/batch-predictions/{batch_job_id}/cancel", response_model=BatchPredictionJobSummary)
def cancel_batch_prediction_job(
    batch_job_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    job = _get_org_batch_job(batch_job_id, current_user, db)
    if job.status not in ACTIVE_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "PREDICTION_LOT_NON_ANNULABLE", "message": "Cette prédiction en lot n'est plus en cours"},
        )
    try_cancel_rq_job(job.rq_job_id, analysis_queue)
    job.status = "cancelled"
    job.error_message = CANCELLED_MESSAGE
    job.finished_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(job)
    return _to_batch_summary(job)


@router.delete("/batch-predictions/{batch_job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_batch_prediction_job(
    batch_job_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    job = _get_org_batch_job(batch_job_id, current_user, db)
    if job.status in ACTIVE_STATUSES:
        try_cancel_rq_job(job.rq_job_id, analysis_queue)
    if job.output_file_path:
        Path(job.output_file_path).unlink(missing_ok=True)
    Path(job.input_file_path).unlink(missing_ok=True)
    db.delete(job)
    db.commit()


@router.post("/jobs/{job_id}/cancel", response_model=TrainingJobSummary)
def cancel_training_job(job_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Annule un entraînement en attente ou en cours (Lot 7, §J.2) —
    contrairement à `DELETE /jobs/{id}`, garde une trace consultable
    (`status="cancelled"`) plutôt que de supprimer le job."""
    job = _get_org_job(job_id, current_user, db)
    if job.status not in ACTIVE_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": ErrorCode.JOB_NON_ANNULABLE,
                "message": "Cet entraînement n'est plus en attente ni en cours",
            },
        )
    try_cancel_rq_job(job.rq_job_id, training_queue)
    job.status = "cancelled"
    job.error_message = CANCELLED_MESSAGE
    job.finished_at = datetime.now(timezone.utc)
    log_action(
        db, current_user.organization_id, current_user.id, "training_job.cancelled",
        target_type="training_job", target_id=job.id, details={"target_column": job.target_column},
    )
    db.commit()
    db.refresh(job)
    return to_summary(job)


@router.post("/jobs/{job_id}/rerun", response_model=TrainingJobSummary, status_code=status.HTTP_201_CREATED)
def rerun_training_job(
    job_id: int, request: Request, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """Relance un entraînement avec EXACTEMENT la même configuration (Lot 7,
    §J.2) — le geste le plus fréquent en pratique, jusqu'ici impossible sans
    tout ressaisir. Reconstruit le corps de `POST /jobs` depuis le job
    d'origine et réutilise SA validation complète (dataset toujours prêt,
    colonnes toujours présentes...) — jamais une copie partielle de cette
    logique, qui divergerait avec le temps."""
    job = _get_org_job(job_id, current_user, db)
    config = json.loads(job.config_json)
    fe = json.loads(job.feature_engineering_json) if job.feature_engineering_json else None
    body = TrainingJobCreate(
        dataset_id=job.dataset_id,
        target_column=job.target_column,
        feature_columns=json.loads(job.feature_columns_json),
        task_type=job.task_type,
        group_column=job.group_column,
        test_size=config.get("test_size", 0.2),
        optuna_trials=config.get("optuna_trials"),
        cv_folds=config.get("cv_folds"),
        seed=config.get("seed"),
        cqr_alpha=config.get("cqr_alpha"),
        model_ids=config.get("model_ids"),
        hyperparameter_overrides=config.get("hyperparameter_overrides"),
        feature_engineering=FeatureEngineeringConfig(upstream=fe["upstream"], pipeline=fe["pipeline"]) if fe else None,
        class_rebalancing=config.get("class_rebalancing", False),
    )
    # Bug réel trouvé en testant (Phase 2, AUDIT_BACKEND_2026-08-23.md) —
    # `create_training_job` a gagné un paramètre `request` en Phase 2
    # (idempotence) ; cet appel direct (pas une vraie requête HTTP, un
    # simple appel Python réutilisant la validation complète) le construisait
    # positionnellement et s'est silencieusement désaligné. Arguments
    # nommés désormais, pour qu'un futur paramètre ajouté à
    # `create_training_job` ne puisse plus jamais se glisser au mauvais
    # endroit ici.
    return create_training_job(body=body, request=request, current_user=current_user, db=db)


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

    if job.status in ACTIVE_STATUSES:
        try_cancel_rq_job(job.rq_job_id, training_queue)

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
