"""Router détection d'anomalies visuelles (structure normal/défaut) — pilier Vision, Lot 15
sous-lot C.

Mêmes principes que `api/routers/vision_classification.py` : isolation
systématique par `organization_id`, tâche de fond obligatoire (RQ,
`vision_queue` — voir `api/core/job_queue.py`, correctif I6), jamais de
calcul ML dans la requête HTTP. Le dataset source doit être un
`VisionDataset` de structure "mvtec_ad" — vérifié ici ET dans le worker
(défense en profondeur)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session, joinedload

from api.core.config import get_settings
from api.core.database import SessionLocal, get_db
from api.core.job_queue import redis_conn, vision_queue
from api.core.models import User, VisionAnomalyExampleRecord, VisionAnomalyJob, VisionDataset
from api.core.pagination import paginate_by_id
from domains.auth.router import get_current_user
from domains.shared.audit import log_action
from domains.shared.job_creation import enqueue_or_mark_failed, remember_idempotent_job_id, resolve_idempotent_job_id
from domains.shared.job_events import stream_job_updates
from domains.shared.job_lifecycle import ACTIVE_STATUSES, CANCELLED_MESSAGE, try_cancel_rq_job
from domains.shared.job_quota import ALL_JOB_MODELS, raise_if_quota_exceeded
from domains.shared.job_watchdog import reconcile_stale_jobs
from domains.vision.anomalies.services.engine import IMAGE_SIZE, MAX_MODELS_PER_COMPARISON
from domains.vision.anomalies.services.registry import ANOMALY_MODEL_REGISTRY, DEFAULT_ANOMALY_MODEL_ID
from domains.vision.localization import DEFAULT_MASK_PERCENTILE
from domains.vision.shared import ALLOWED_IMAGE_SIZES, AUGMENTATION_PRESET_IDS

router = APIRouter(prefix="/vision/anomalies", tags=["vision"])
_settings = get_settings()

_VALID_MODEL_IDS = {s.id for s in ANOMALY_MODEL_REGISTRY}


# ── Schémas ──────────────────────────────────────────────────────────────


class VisionAnomalyJobCreate(BaseModel):
    vision_dataset_id: int
    model_id: str = DEFAULT_ANOMALY_MODEL_ID
    # Mode expert (retour utilisateur direct — parité avec `backbone_ids` de
    # la classification et `model_ids` du ML tabulaire) — quand fourni (≥ 2
    # entrées), remplace `model_id` : chaque architecture listée est
    # entraînée avec les mêmes autres réglages, la meilleure sur la
    # validation est automatiquement retenue (voir
    # `services/engine.py::train_and_compare_anomaly_models`). None (défaut) :
    # comportement historique inchangé, une seule architecture (`model_id`).
    model_ids: Optional[List[str]] = None
    # Mode expert (retour utilisateur direct : "vision n'offre pas de
    # réduire/augmenter la taille des images") — voir
    # domains/vision/shared.py::ALLOWED_IMAGE_SIZES.
    image_size: int = IMAGE_SIZE
    num_epochs: int = Field(default=15, ge=1, le=50)
    batch_size: int = Field(default=16, ge=1, le=128)
    learning_rate: float = Field(default=1e-3, gt=0, le=1)
    # Régularisation L2 (mode expert) — voir
    # services/engine.py::AnomalyVisionConfig.weight_decay.
    weight_decay: float = Field(default=0.0, ge=0, le=0.1)
    mask_percentile: float = Field(default=DEFAULT_MASK_PERCENTILE, gt=0, lt=1)
    seed: Optional[int] = None
    # Même système de presets que la classification (Lot 6A, parité des
    # étapes) — voir services/vision_anomaly_training.py::AnomalyVisionConfig.
    augmentation_preset: str = "aucune"
    # Part de train/good/ réservée à la validation (répartition, Lot 6A).
    val_ratio: float = Field(default=0.15, ge=0.05, le=0.5)


class AnomalyModelOut(BaseModel):
    id: str
    label: str


class VisionAnomalyJobSummary(BaseModel):
    id: int
    vision_dataset_id: int
    vision_dataset_name: Optional[str] = None
    model_id: str
    status: str
    progress_step: Optional[str] = None
    progress_percent: int
    error_message: Optional[str] = None
    created_by: Optional[str] = None
    created_at: Any
    started_at: Optional[Any] = None
    finished_at: Optional[Any] = None
    roc_auc: Optional[float] = None


class AnomalyEpochMetricsOut(BaseModel):
    epoch: int
    train_loss: float
    val_loss: float


class VisionAnomalyResultOut(BaseModel):
    model_id: str
    n_train: int
    n_val: int
    n_test: int
    # Lot 0.2 (correctif C2) — absents (None) sur les modèles entraînés
    # avant ce correctif, rétrocompatibilité par absence.
    n_calibration: Optional[int] = None
    n_evaluation: Optional[int] = None
    history: List[AnomalyEpochMetricsOut]
    threshold: float
    roc_auc: float
    test_accuracy: float
    test_precision: float
    test_recall: float
    test_f1: float
    confusion_matrix: List[List[int]]
    model_card: dict[str, Any]
    # Retour utilisateur : "d'autres fonctionnalités modernes que les autres
    # plateformes n'offrent pas" — None sur les modèles entraînés avant ce
    # correctif (rétrocompatibilité par absence, même motif que ci-dessus).
    roc_curves: Optional[dict[str, dict[str, List[float]]]] = None
    pr_curves: Optional[dict[str, dict[str, List[float]]]] = None
    score_histogram: Optional[dict[str, Any]] = None
    category_breakdown: Optional[List[dict[str, Any]]] = None


class VisionAnomalyExampleOut(BaseModel):
    relative_path: str
    defect_category: str
    true_label: int
    predicted_label: int
    anomaly_score: float
    heatmap_png: str
    mask_png: str


# ── Aides ────────────────────────────────────────────────────────────────


def _get_org_job(job_id: int, current_user: User, db: Session) -> VisionAnomalyJob:
    job = (
        db.query(VisionAnomalyJob)
        .filter(VisionAnomalyJob.id == job_id, VisionAnomalyJob.organization_id == current_user.organization_id)
        .first()
    )
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "VISION_ANOMALY_JOB_INTROUVABLE", "message": "Détection d'anomalies visuelles introuvable"},
        )
    return job


def to_summary(job: VisionAnomalyJob) -> VisionAnomalyJobSummary:
    config = json.loads(job.config_json)
    result = job.result
    return VisionAnomalyJobSummary(
        id=job.id,
        vision_dataset_id=job.vision_dataset_id,
        vision_dataset_name=job.vision_dataset.name if job.vision_dataset else None,
        model_id=config.get("model_id", DEFAULT_ANOMALY_MODEL_ID),
        status=job.status,
        progress_step=job.progress_step,
        progress_percent=job.progress_percent,
        error_message=job.error_message,
        created_by=job.created_by.nom if job.created_by else None,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        roc_auc=result.roc_auc if result else None,
    )


# ── Endpoints ────────────────────────────────────────────────────────────


@router.get("/models", response_model=List[AnomalyModelOut])
def list_anomaly_models():
    return [AnomalyModelOut(id=s.id, label=s.label) for s in ANOMALY_MODEL_REGISTRY]


@router.post("/jobs", response_model=VisionAnomalyJobSummary, status_code=status.HTTP_201_CREATED)
def create_vision_anomaly_job(
    body: VisionAnomalyJobCreate,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Idempotence (Phase 2, AUDIT_BACKEND_2026-08-23.md §F4) — voir
    # domains/training/router.py::create_training_job pour la justification.
    existing_job_id = resolve_idempotent_job_id(redis_conn, current_user.organization_id, request)
    if existing_job_id is not None:
        existing = (
            db.query(VisionAnomalyJob)
            .filter(
                VisionAnomalyJob.id == existing_job_id,
                VisionAnomalyJob.organization_id == current_user.organization_id,
            )
            .first()
        )
        if existing is not None:
            return to_summary(existing)

    if body.model_id not in _VALID_MODEL_IDS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "MODELE_INCONNU", "message": f"Modèle inconnu : {body.model_id}"},
        )
    if body.model_ids is not None:
        if len(body.model_ids) < 2 or len(body.model_ids) > MAX_MODELS_PER_COMPARISON:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": "COMPARATIF_MODELES_INVALIDE",
                    "message": f"Comparatif : entre 2 et {MAX_MODELS_PER_COMPARISON} architectures requises",
                },
            )
        if len(set(body.model_ids)) != len(body.model_ids):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "COMPARATIF_MODELES_INVALIDE", "message": "Architectures en double dans le comparatif"},
            )
        unknown = [m for m in body.model_ids if m not in _VALID_MODEL_IDS]
        if unknown:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "MODELE_INCONNU", "message": f"Modèle(s) inconnu(s) : {', '.join(unknown)}"},
            )
    if body.augmentation_preset not in AUGMENTATION_PRESET_IDS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "AUGMENTATION_PRESET_INCONNU",
                "message": f"Preset d'augmentation inconnu : {body.augmentation_preset!r}",
            },
        )
    if body.image_size not in ALLOWED_IMAGE_SIZES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "TAILLE_IMAGE_INCONNUE",
                "message": f"Taille d'image invalide : {body.image_size} (voir {ALLOWED_IMAGE_SIZES})",
            },
        )

    reconcile_stale_jobs(
        db, current_user.organization_id, _settings.stale_job_timeout_minutes, model=VisionAnomalyJob
    )
    raise_if_quota_exceeded(db, current_user.organization_id, ALL_JOB_MODELS, _settings.max_concurrent_jobs_per_org)

    dataset = (
        db.query(VisionDataset)
        .filter(VisionDataset.id == body.vision_dataset_id, VisionDataset.organization_id == current_user.organization_id)
        .first()
    )
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "VISION_DATASET_INTROUVABLE", "message": "Dataset d'images introuvable"},
        )
    if dataset.status != "ready":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "VISION_DATASET_NON_PRET", "message": "Ce dataset n'a pas pu être validé"},
        )
    if dataset.structure_type != "mvtec_ad":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "VISION_DATASET_STRUCTURE_INVALIDE",
                "message": "Ce dataset n'a pas une structure normal/défaut (train/good + test/good + test/<defaut>)",
            },
        )

    config = {
        "model_id": body.model_id,
        "image_size": body.image_size,
        "num_epochs": body.num_epochs,
        "batch_size": body.batch_size,
        "learning_rate": body.learning_rate,
        "weight_decay": body.weight_decay,
        "mask_percentile": body.mask_percentile,
        "seed": body.seed if body.seed is not None else _settings.model_seed,
        "augmentation_preset": body.augmentation_preset,
        "val_ratio": body.val_ratio,
    }
    if body.model_ids is not None:
        config["model_ids"] = body.model_ids

    job = VisionAnomalyJob(
        organization_id=current_user.organization_id,
        vision_dataset_id=dataset.id,
        created_by_id=current_user.id,
        config_json=json.dumps(config),
        status="queued",
        request_id=request.state.request_id,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    remember_idempotent_job_id(redis_conn, current_user.organization_id, request, job.id)
    log_action(
        db, current_user.organization_id, current_user.id, "vision_anomaly_job.created",
        target_type="vision_anomaly_job", target_id=job.id,
    )

    from domains.vision.anomalies.worker import run_vision_anomaly_job

    enqueue_or_mark_failed(db, job, vision_queue, run_vision_anomaly_job, 1800)

    return to_summary(job)


@router.get("/jobs", response_model=List[VisionAnomalyJobSummary])
def list_vision_anomaly_jobs(
    response: Response,
    limit: Optional[int] = Query(None, ge=1, le=500),
    cursor: Optional[int] = Query(None, description="id de la dernière ligne de la page précédente"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # joinedload (Lot 4, correctif I3) — voir training.py::list_training_jobs.
    query = (
        db.query(VisionAnomalyJob)
        .options(
            joinedload(VisionAnomalyJob.vision_dataset),
            joinedload(VisionAnomalyJob.created_by),
            joinedload(VisionAnomalyJob.result),
        )
        .filter(VisionAnomalyJob.organization_id == current_user.organization_id)
        # id DESC, pas created_at DESC — voir anomalies.py::list_anomaly_jobs.
        .order_by(VisionAnomalyJob.id.desc())
    )
    jobs = paginate_by_id(query, VisionAnomalyJob.id, response, cursor, limit)
    return [to_summary(j) for j in jobs]


@router.get("/jobs/{job_id}", response_model=VisionAnomalyJobSummary)
def get_vision_anomaly_job(job_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return to_summary(_get_org_job(job_id, current_user, db))


@router.get("/jobs/{job_id}/events")
async def stream_vision_anomaly_job_events(job_id: int, current_user: User = Depends(get_current_user)):
    """Notifications de fin de job par SSE (Lot 7, §J.2) — voir
    `training.py::stream_training_job_events` pour le raisonnement complet."""
    organization_id = current_user.organization_id
    db = SessionLocal()
    try:
        job = (
            db.query(VisionAnomalyJob)
            .filter(VisionAnomalyJob.id == job_id, VisionAnomalyJob.organization_id == organization_id)
            .first()
        )
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "VISION_ANOMALY_JOB_INTROUVABLE", "message": "Détection d'anomalies visuelles introuvable"},
            )
    finally:
        db.close()

    def fetch_snapshot():
        session = SessionLocal()
        try:
            row = (
                session.query(VisionAnomalyJob)
                .filter(VisionAnomalyJob.id == job_id, VisionAnomalyJob.organization_id == organization_id)
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


@router.get("/jobs/{job_id}/result", response_model=VisionAnomalyResultOut)
def get_vision_anomaly_result(job_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    job = _get_org_job(job_id, current_user, db)
    if job.result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "RESULTAT_INDISPONIBLE", "message": "Cet entraînement n'a pas encore de résultat"},
        )
    result = job.result
    return VisionAnomalyResultOut(
        model_id=result.model_id,
        n_train=result.n_train,
        n_val=result.n_val,
        n_test=result.n_test,
        n_calibration=result.n_calibration,
        n_evaluation=result.n_evaluation,
        history=[AnomalyEpochMetricsOut(**m) for m in json.loads(result.history_json)],
        threshold=result.threshold,
        roc_auc=result.roc_auc,
        test_accuracy=result.test_accuracy,
        test_precision=result.test_precision,
        test_recall=result.test_recall,
        test_f1=result.test_f1,
        confusion_matrix=json.loads(result.confusion_matrix_json),
        model_card=json.loads(result.model_card_json or "{}"),
        roc_curves=json.loads(result.roc_curves_json) if result.roc_curves_json else None,
        pr_curves=json.loads(result.pr_curves_json) if result.pr_curves_json else None,
        score_histogram=json.loads(result.score_histogram_json) if result.score_histogram_json else None,
        category_breakdown=json.loads(result.category_breakdown_json) if result.category_breakdown_json else None,
    )


@router.get("/jobs/{job_id}/model/export")
def export_vision_anomaly_model(job_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Export de l'artefact (Lot 10, retour utilisateur direct — parité avec
    `GET /training/jobs/{job_id}/model/export`)."""
    job = _get_org_job(job_id, current_user, db)
    if job.status != "completed" or job.result is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "MODELE_NON_DISPONIBLE", "message": "Cet entraînement n'a pas encore produit de modèle"},
        )
    artifact_path = Path(job.result.file_path)
    if not artifact_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "ARTEFACT_INTROUVABLE", "message": "Artefact du modèle introuvable sur le serveur"},
        )
    filename = f"vision_anomalies_job{job.id}.pt"
    return FileResponse(path=artifact_path, filename=filename, media_type="application/octet-stream")


@router.get("/jobs/{job_id}/examples", response_model=List[VisionAnomalyExampleOut])
def get_vision_anomaly_examples(job_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    job = _get_org_job(job_id, current_user, db)
    if job.result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "RESULTAT_INDISPONIBLE", "message": "Cet entraînement n'a pas encore de résultat"},
        )
    examples = (
        db.query(VisionAnomalyExampleRecord)
        .filter(VisionAnomalyExampleRecord.vision_anomaly_model_id == job.result.id)
        .order_by(VisionAnomalyExampleRecord.anomaly_score.desc())
        .all()
    )
    return [
        VisionAnomalyExampleOut(
            relative_path=e.relative_path,
            defect_category=e.defect_category,
            true_label=e.true_label,
            predicted_label=e.predicted_label,
            anomaly_score=e.anomaly_score,
            heatmap_png=e.heatmap_png,
            mask_png=e.mask_png,
        )
        for e in examples
    ]


@router.post("/jobs/{job_id}/cancel", response_model=VisionAnomalyJobSummary)
def cancel_vision_anomaly_job(job_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Annule un entraînement Vision (anomalies) en attente ou en cours
    (Lot 7, §J.2) — garde une trace consultable (`status="cancelled"`)."""
    job = _get_org_job(job_id, current_user, db)
    if job.status not in ACTIVE_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "JOB_NON_ANNULABLE", "message": "Cet entraînement n'est plus en attente ni en cours"},
        )
    try_cancel_rq_job(job.rq_job_id, vision_queue)
    job.status = "cancelled"
    job.error_message = CANCELLED_MESSAGE
    job.finished_at = datetime.now(timezone.utc)
    log_action(
        db, current_user.organization_id, current_user.id, "vision_anomaly_job.cancelled",
        target_type="vision_anomaly_job", target_id=job.id, details={"vision_dataset_id": job.vision_dataset_id},
    )
    db.commit()
    db.refresh(job)
    return to_summary(job)


@router.post("/jobs/{job_id}/rerun", response_model=VisionAnomalyJobSummary, status_code=status.HTTP_201_CREATED)
def rerun_vision_anomaly_job(
    job_id: int, request: Request, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """Relance un entraînement Vision (anomalies) avec EXACTEMENT la même
    configuration (Lot 7, §J.2) — `config_json` reprend ici tous les champs
    de `VisionAnomalyJobCreate` à l'identique, dépaquetage direct."""
    job = _get_org_job(job_id, current_user, db)
    config = json.loads(job.config_json)
    body = VisionAnomalyJobCreate(vision_dataset_id=job.vision_dataset_id, **config)
    # Arguments nommés (Phase 2, AUDIT_BACKEND_2026-08-23.md) — voir
    # domains/training/router.py::rerun_training_job pour le bug réel que
    # cela corrige.
    return create_vision_anomaly_job(body=body, request=request, current_user=current_user, db=db)


@router.delete("/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_vision_anomaly_job(job_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    job = _get_org_job(job_id, current_user, db)

    if job.status in ACTIVE_STATUSES:
        try_cancel_rq_job(job.rq_job_id, vision_queue)

    if job.result:
        if job.result.file_path:
            Path(job.result.file_path).unlink(missing_ok=True)
        db.delete(job.result)

    log_action(
        db, current_user.organization_id, current_user.id, "vision_anomaly_job.deleted",
        target_type="vision_anomaly_job", target_id=job.id,
        details={"vision_dataset_id": job.vision_dataset_id, "status": job.status},
    )
    db.delete(job)
    db.commit()
