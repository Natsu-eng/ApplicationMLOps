"""Router training — lancement, suivi et résultat des entraînements ML.

Isolation : mêmes principes que `datasets.py` — filtrage systématique par
`organization_id`, accessible à toute l'équipe (pas réservé au owner).
L'entraînement lui-même s'exécute en tâche de fond (RQ, voir
`workers/training_worker.py`) — ce router ne fait qu'enfiler le job et lire
son état, jamais de calcul ML dans la requête HTTP.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.core.config import get_settings
from api.core.database import get_db
from api.core.job_queue import training_queue
from api.core.models import Dataset, MLModel, TrainingJob, User
from api.routers.auth import get_current_user
from services.datasets import DatasetParsingError, read_dataframe
from services.ml_inference import InferenceError, load_bundle, predict_one
from services.ml_task import detect_task_type

router = APIRouter(prefix="/training", tags=["training"])
_settings = get_settings()


# ── Schémas ──────────────────────────────────────────────────────────────────

class TrainingJobCreate(BaseModel):
    dataset_id: int
    target_column: str
    feature_columns: Optional[List[str]] = None
    task_type: Optional[str] = None  # "classification" | "regression" — auto-détecté si absent
    group_column: Optional[str] = None
    test_size: float = Field(0.2, gt=0.05, lt=0.5)
    optuna_trials: Optional[int] = Field(None, ge=3, le=100)
    cv_folds: Optional[int] = Field(None, ge=2, le=10)


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
    created_at: datetime


class PredictionRequest(BaseModel):
    data: dict[str, Any]


class PredictionResponse(BaseModel):
    prediction: Any
    probabilities: Optional[dict[str, float]] = None
    interval: Optional[dict[str, float]] = None


# ── Aides internes ───────────────────────────────────────────────────────────

def _headline_metric(task_type: str, metrics: dict[str, Any]) -> dict[str, Any]:
    if task_type == "regression":
        return {"name": "r2_test", "value": metrics.get("r2_test")}
    return {"name": "accuracy", "value": metrics.get("accuracy")}


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

@router.post("/jobs", response_model=TrainingJobSummary, status_code=status.HTTP_201_CREATED)
def create_training_job(
    body: TrainingJobCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
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

    config = {
        "test_size": body.test_size,
        "seed": _settings.model_seed,
        "optuna_trials": body.optuna_trials or _settings.optuna_trials_default,
        "cv_folds": body.cv_folds or _settings.cv_folds_default,
        "cqr_alpha": _settings.cqr_alpha,
        "cqr_n_strata": _settings.cqr_n_strata,
        "shap_sample_size": _settings.shap_sample_size,
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


@router.get("/jobs/{job_id}", response_model=TrainingJobSummary)
def get_training_job(job_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return _to_summary(_get_org_job(job_id, current_user, db))


@router.get("/jobs/{job_id}/model", response_model=MLModelDetail)
def get_training_job_model(job_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    job = _get_org_job(job_id, current_user, db)
    if job.status != "completed" or job.model is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "MODELE_NON_DISPONIBLE", "message": "Cet entraînement n'a pas encore produit de modèle"},
        )
    model: MLModel = job.model
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
        created_at=model.created_at,
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

    if job.model and job.model.file_path:
        Path(job.model.file_path).unlink(missing_ok=True)

    db.delete(job)  # cascade DB vers MLModel (ondelete="CASCADE")
    db.commit()
