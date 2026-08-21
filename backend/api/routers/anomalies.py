"""Router détection d'anomalies — Lot 14 (ML non supervisé).

Mêmes principes que `api/routers/clustering.py`/`dimensionality.py` :
isolation systématique par `organization_id`, tâche de fond obligatoire (RQ,
`analysis_queue` — voir `api/core/job_queue.py`, correctif I6), jamais de
calcul ML dans la requête HTTP.

Pas de `GET /algorithms-catalog` — contrairement au clustering et à la
réduction de dimension, il n'y a aucun choix d'algorithme à cataloguer :
Isolation Forest et LOF tournent systématiquement ensemble (voir
services/anomaly_registry.py)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import joinedload

from api.core.config import get_settings
from api.core.database import get_db
from api.core.job_queue import analysis_queue
from api.core.models import AnomalyJob, AnomalyObservationRecord, Dataset, User
from api.core.pagination import paginate_by_id
from api.routers.auth import get_current_user
from services.anomaly_training import DEFAULT_TOP_N, MAX_TOP_N
from services.audit import log_action
from services.datasets import DatasetParsingError, read_dataset_dataframe
from services.job_quota import ALL_JOB_MODELS, raise_if_quota_exceeded
from services.job_watchdog import reconcile_stale_jobs

router = APIRouter(prefix="/anomalies", tags=["anomalies"])
_settings = get_settings()


# ── Schémas ──────────────────────────────────────────────────────────────


class AnomalyJobCreate(BaseModel):
    dataset_id: int
    feature_columns: List[str]
    top_n: int = Field(default=DEFAULT_TOP_N, ge=1, le=MAX_TOP_N)
    seed: Optional[int] = None
    # `None` = "auto" (comportement par défaut, formule du papier original de
    # chaque algorithme). Fraction explicite sinon — bornée à (0, 0.5] : 0
    # n'aurait pas de sens (aucune anomalie), et sklearn rejette au-delà de
    # 0.5 (la "majorité" ne peut pas être l'anomalie par définition).
    contamination: Optional[float] = Field(default=None, gt=0.0, le=0.5)


class AnomalyJobSummary(BaseModel):
    id: int
    dataset_id: int
    dataset_name: Optional[str] = None
    feature_columns: List[str]
    status: str
    progress_step: Optional[str] = None
    progress_percent: int
    error_message: Optional[str] = None
    created_by: Optional[str] = None
    created_at: Any
    started_at: Optional[Any] = None
    finished_at: Optional[Any] = None
    n_anomalies_consensus: Optional[int] = None
    anomaly_rate_consensus: Optional[float] = None


class AnomalyResultOut(BaseModel):
    n_samples_total: int
    n_samples_used: int
    sampled: bool
    n_anomalies_isolation_forest: int
    n_anomalies_lof: int
    n_anomalies_consensus: int
    anomaly_rate_isolation_forest: float
    anomaly_rate_lof: float
    anomaly_rate_consensus: float
    score_histogram: dict[str, Any]
    model_card: dict[str, Any]


class AnomalyObservationOut(BaseModel):
    row_index: int
    rank: int
    consensus_score: float
    score_isolation_forest: float
    score_lof: float
    is_anomaly_isolation_forest: bool
    is_anomaly_lof: bool
    agreement: str
    numeric_deviations: dict[str, Any]
    categorical_flags: dict[str, Any]


# ── Aides ────────────────────────────────────────────────────────────────


def _get_org_job(job_id: int, current_user: User, db) -> AnomalyJob:
    job = (
        db.query(AnomalyJob)
        .filter(AnomalyJob.id == job_id, AnomalyJob.organization_id == current_user.organization_id)
        .first()
    )
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "ANOMALY_JOB_INTROUVABLE", "message": "Détection d'anomalies introuvable"},
        )
    return job


def _to_summary(job: AnomalyJob) -> AnomalyJobSummary:
    result = job.result
    return AnomalyJobSummary(
        id=job.id,
        dataset_id=job.dataset_id,
        dataset_name=job.dataset.name if job.dataset else None,
        feature_columns=json.loads(job.feature_columns_json),
        status=job.status,
        progress_step=job.progress_step,
        progress_percent=job.progress_percent,
        error_message=job.error_message,
        created_by=job.created_by.nom if job.created_by else None,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        n_anomalies_consensus=result.n_anomalies_consensus if result else None,
        anomaly_rate_consensus=result.anomaly_rate_consensus if result else None,
    )


# ── Endpoints ────────────────────────────────────────────────────────────


@router.post("/jobs", response_model=AnomalyJobSummary, status_code=status.HTTP_201_CREATED)
def create_anomaly_job(
    body: AnomalyJobCreate,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    reconcile_stale_jobs(db, current_user.organization_id, _settings.stale_job_timeout_minutes, model=AnomalyJob)

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
            detail={"code": "DATASET_INTROUVABLE", "message": "Dataset introuvable"},
        )
    if dataset.status != "ready":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "DATASET_NON_PRET", "message": "Ce dataset n'a pas pu être analysé"},
        )

    schema_columns = [c["name"] for c in json.loads(dataset.columns_json or "[]")]
    if not body.feature_columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "COLONNES_MANQUANTES", "message": "Sélectionnez au moins une variable"},
        )
    unknown = set(body.feature_columns) - set(schema_columns)
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "COLONNES_INCONNUES", "message": f"Colonnes absentes du dataset : {', '.join(sorted(unknown))}"},
        )

    try:
        read_dataset_dataframe(Path(dataset.file_path), Path(dataset.file_path).suffix)
    except DatasetParsingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "DATASET_LECTURE_ECHEC", "message": str(exc)},
        )

    config = {
        "top_n": body.top_n,
        "seed": body.seed if body.seed is not None else _settings.model_seed,
        "contamination": body.contamination if body.contamination is not None else "auto",
    }

    job = AnomalyJob(
        organization_id=current_user.organization_id,
        dataset_id=dataset.id,
        created_by_id=current_user.id,
        feature_columns_json=json.dumps(body.feature_columns),
        config_json=json.dumps(config),
        status="queued",
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    from workers.anomaly_worker import run_anomaly_job

    rq_job = analysis_queue.enqueue(run_anomaly_job, job.id, job_timeout=600)
    job.rq_job_id = rq_job.id
    db.commit()
    db.refresh(job)

    return _to_summary(job)


@router.get("/jobs", response_model=List[AnomalyJobSummary])
def list_anomaly_jobs(
    response: Response,
    limit: Optional[int] = Query(None, ge=1, le=500),
    cursor: Optional[int] = Query(None, description="id de la dernière ligne de la page précédente"),
    current_user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    # joinedload (Lot 4, correctif I3) — voir training.py::list_training_jobs.
    query = (
        db.query(AnomalyJob)
        .options(joinedload(AnomalyJob.dataset), joinedload(AnomalyJob.created_by), joinedload(AnomalyJob.result))
        .filter(AnomalyJob.organization_id == current_user.organization_id)
        # id DESC, pas created_at DESC (Lot 4, correctif I3) : SQLite stocke
        # `func.now()` avec une précision à la seconde — plusieurs jobs créés
        # rapidement (rafale d'appels API, tests) peuvent partager le même
        # `created_at`, rendant l'ordre non déterministe et cassant le
        # curseur de pagination (des lignes sautées ou dupliquées entre deux
        # pages). `id` auto-incrémenté encode l'ordre de création SANS
        # ambiguïté possible — équivalent en pratique, strictement fiable.
        .order_by(AnomalyJob.id.desc())
    )
    jobs = paginate_by_id(query, AnomalyJob.id, response, cursor, limit)
    return [_to_summary(j) for j in jobs]


@router.get("/jobs/{job_id}", response_model=AnomalyJobSummary)
def get_anomaly_job(job_id: int, current_user: User = Depends(get_current_user), db=Depends(get_db)):
    return _to_summary(_get_org_job(job_id, current_user, db))


@router.get("/jobs/{job_id}/result", response_model=AnomalyResultOut)
def get_anomaly_result(job_id: int, current_user: User = Depends(get_current_user), db=Depends(get_db)):
    job = _get_org_job(job_id, current_user, db)
    if job.result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "RESULTAT_INDISPONIBLE", "message": "Cette détection n'a pas encore de résultat"},
        )
    result = job.result
    return AnomalyResultOut(
        n_samples_total=result.n_samples_total,
        n_samples_used=result.n_samples_used,
        sampled=result.sampled,
        n_anomalies_isolation_forest=result.n_anomalies_isolation_forest,
        n_anomalies_lof=result.n_anomalies_lof,
        n_anomalies_consensus=result.n_anomalies_consensus,
        anomaly_rate_isolation_forest=result.anomaly_rate_isolation_forest,
        anomaly_rate_lof=result.anomaly_rate_lof,
        anomaly_rate_consensus=result.anomaly_rate_consensus,
        score_histogram=json.loads(result.score_histogram_json),
        model_card=json.loads(result.model_card_json or "{}"),
    )


@router.get("/jobs/{job_id}/observations", response_model=List[AnomalyObservationOut])
def get_anomaly_observations(job_id: int, current_user: User = Depends(get_current_user), db=Depends(get_db)):
    job = _get_org_job(job_id, current_user, db)
    observations = (
        db.query(AnomalyObservationRecord)
        .filter(AnomalyObservationRecord.anomaly_job_id == job.id)
        .order_by(AnomalyObservationRecord.rank)
        .all()
    )
    return [
        AnomalyObservationOut(
            row_index=o.row_index,
            rank=o.rank,
            consensus_score=o.consensus_score,
            score_isolation_forest=o.score_isolation_forest,
            score_lof=o.score_lof,
            is_anomaly_isolation_forest=o.is_anomaly_isolation_forest,
            is_anomaly_lof=o.is_anomaly_lof,
            agreement=o.agreement,
            numeric_deviations=json.loads(o.numeric_deviations_json),
            categorical_flags=json.loads(o.categorical_flags_json),
        )
        for o in observations
    ]


@router.delete("/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_anomaly_job(job_id: int, current_user: User = Depends(get_current_user), db=Depends(get_db)):
    job = _get_org_job(job_id, current_user, db)

    if job.status in ("queued", "running") and job.rq_job_id:
        try:
            from rq.job import Job as RQJob

            rq_job = RQJob.fetch(job.rq_job_id, connection=analysis_queue.connection)
            rq_job.cancel()
            rq_job.delete()
        except Exception:
            pass

    if job.result:
        if job.result.file_path:
            Path(job.result.file_path).unlink(missing_ok=True)
        db.delete(job.result)

    log_action(
        db, current_user.organization_id, current_user.id, "anomaly_job.deleted",
        target_type="anomaly_job", target_id=job.id,
        details={"dataset_id": job.dataset_id, "status": job.status},
    )
    db.delete(job)
    db.commit()
