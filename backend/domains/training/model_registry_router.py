"""Router registre de modèles — détail, promotion, versions, export.

Extrait de `router.py` lors du découpage du router training. Le registre
répond à une question différente de celle de l'entraînement : `router.py`
orchestre des JOBS (lancer, suivre, annuler, rejouer), ce module gère
l'ARTEFACT produit — quel modèle fait autorité pour un problème donné
(promotion/démotion, versions, historique des transitions) et comment le
récupérer hors de la plateforme (export du bundle, script de déploiement).

Les chemins exposés sont RIGOUREUSEMENT inchangés par l'extraction (même
préfixe `/training`, mêmes URL) — ce module est enregistré à part dans
`api/main.py`, à côté du router training.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.core.database import get_db
from api.core.error_codes import ErrorCode
from api.core.models import AuditLog, MLModel, ModelCandidate, TrainingJob, User
from domains.auth.router import get_current_user
from domains.shared.audit import log_action
from domains.shared.model_bundle import InferenceError, load_bundle
from domains.training.dependencies import get_org_training_job, headline_metric
from domains.training.services.deployment_export import generate_deployment_script
from domains.training.services.verdict import compute_verdict

router = APIRouter(prefix="/training", tags=["training"])


# ── Schémas ──────────────────────────────────────────────────────────────────

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


# ── Aides internes ───────────────────────────────────────────────────────────

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


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/jobs/{job_id}/model", response_model=MLModelDetail)
def get_training_job_model(job_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    job = get_org_training_job(job_id, current_user, db)
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
    job = get_org_training_job(job_id, current_user, db)
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
    job = get_org_training_job(job_id, current_user, db)
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
            headline_metric=headline_metric(row.task_type, metrics) if metrics else None,
        ))
    return ModelVersionsResponse(entries=entries)


@router.get("/jobs/{job_id}/model/history", response_model=ModelHistoryResponse)
def get_model_history(job_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Historique des transitions de stage pour TOUTES les versions de ce
    problème (Lot 5, correctif P1) — lu depuis `AuditLog` (action
    "model.promoted", déjà écrite par promote_model depuis le Lot 9),
    jamais un second mécanisme de journalisation parallèle. Le plus
    récent d'abord."""
    job = get_org_training_job(job_id, current_user, db)
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
    job = get_org_training_job(job_id, current_user, db)
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
    job = get_org_training_job(job_id, current_user, db)
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
            headline_metric=headline_metric(model.task_type, metrics) if metrics else None,
        ))
    return ModelRegistryResponse(entries=entries)
