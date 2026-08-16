"""Router détection d'anomalies visuelles MVTec AD — pilier Vision, Lot 15
sous-lot C.

Mêmes principes que `api/routers/vision_classification.py` : isolation
systématique par `organization_id`, tâche de fond obligatoire (RQ, réutilise
`training_queue`), jamais de calcul ML dans la requête HTTP. Le dataset
source doit être un `VisionDataset` de structure "mvtec_ad" — vérifié ici ET
dans le worker (défense en profondeur)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.core.config import get_settings
from api.core.database import get_db
from api.core.job_queue import training_queue
from api.core.models import User, VisionAnomalyExampleRecord, VisionAnomalyJob, VisionAnomalyModel, VisionDataset
from api.routers.auth import get_current_user
from services.audit import log_action
from services.job_quota import ALL_JOB_MODELS, raise_if_quota_exceeded
from services.job_watchdog import reconcile_stale_jobs
from services.vision_anomaly_registry import ANOMALY_MODEL_REGISTRY, DEFAULT_ANOMALY_MODEL_ID
from services.vision_localization import DEFAULT_MASK_PERCENTILE

router = APIRouter(prefix="/vision/anomalies", tags=["vision"])
_settings = get_settings()

_VALID_MODEL_IDS = {s.id for s in ANOMALY_MODEL_REGISTRY}


# ── Schémas ──────────────────────────────────────────────────────────────


class VisionAnomalyJobCreate(BaseModel):
    vision_dataset_id: int
    model_id: str = DEFAULT_ANOMALY_MODEL_ID
    num_epochs: int = Field(default=15, ge=1, le=50)
    batch_size: int = Field(default=16, ge=1, le=128)
    learning_rate: float = Field(default=1e-3, gt=0, le=1)
    mask_percentile: float = Field(default=DEFAULT_MASK_PERCENTILE, gt=0, lt=1)
    seed: Optional[int] = None


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
    history: List[AnomalyEpochMetricsOut]
    threshold: float
    roc_auc: float
    test_accuracy: float
    test_precision: float
    test_recall: float
    test_f1: float
    confusion_matrix: List[List[int]]
    model_card: dict[str, Any]


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


def _to_summary(job: VisionAnomalyJob) -> VisionAnomalyJobSummary:
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
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if body.model_id not in _VALID_MODEL_IDS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "MODELE_INCONNU", "message": f"Modèle inconnu : {body.model_id}"},
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
                "message": "Ce dataset n'a pas une structure MVTec AD (train/good + test/good + test/<defaut>)",
            },
        )

    config = {
        "model_id": body.model_id,
        "num_epochs": body.num_epochs,
        "batch_size": body.batch_size,
        "learning_rate": body.learning_rate,
        "mask_percentile": body.mask_percentile,
        "seed": body.seed if body.seed is not None else _settings.model_seed,
    }

    job = VisionAnomalyJob(
        organization_id=current_user.organization_id,
        vision_dataset_id=dataset.id,
        created_by_id=current_user.id,
        config_json=json.dumps(config),
        status="queued",
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    from workers.vision_anomaly_worker import run_vision_anomaly_job

    rq_job = training_queue.enqueue(run_vision_anomaly_job, job.id, job_timeout=1800)
    job.rq_job_id = rq_job.id
    db.commit()
    db.refresh(job)

    return _to_summary(job)


@router.get("/jobs", response_model=List[VisionAnomalyJobSummary])
def list_vision_anomaly_jobs(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    jobs = (
        db.query(VisionAnomalyJob)
        .filter(VisionAnomalyJob.organization_id == current_user.organization_id)
        .order_by(VisionAnomalyJob.created_at.desc())
        .all()
    )
    return [_to_summary(j) for j in jobs]


@router.get("/jobs/{job_id}", response_model=VisionAnomalyJobSummary)
def get_vision_anomaly_job(job_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return _to_summary(_get_org_job(job_id, current_user, db))


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
        history=[AnomalyEpochMetricsOut(**m) for m in json.loads(result.history_json)],
        threshold=result.threshold,
        roc_auc=result.roc_auc,
        test_accuracy=result.test_accuracy,
        test_precision=result.test_precision,
        test_recall=result.test_recall,
        test_f1=result.test_f1,
        confusion_matrix=json.loads(result.confusion_matrix_json),
        model_card=json.loads(result.model_card_json or "{}"),
    )


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


@router.delete("/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_vision_anomaly_job(job_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    job = _get_org_job(job_id, current_user, db)

    if job.status in ("queued", "running") and job.rq_job_id:
        try:
            from rq.job import Job as RQJob

            rq_job = RQJob.fetch(job.rq_job_id, connection=training_queue.connection)
            rq_job.cancel()
            rq_job.delete()
        except Exception:
            pass

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
