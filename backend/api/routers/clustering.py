"""Router clustering — Lot 11+ (ML non supervisé).

Mêmes principes que `api/routers/training.py` : isolation systématique par
`organization_id`, tâche de fond obligatoire (RQ, réutilise `training_queue`
— même worker physique, voir `docker-compose.yml`), jamais de calcul ML dans
la requête HTTP. Router DÉDIÉ, jamais fusionné dans `training.py` — même
raisonnement que la séparation `clustering_registry.py`/`ml_registry.py`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from api.core.config import get_settings
from api.core.database import get_db
from api.core.job_queue import training_queue
from api.core.models import ClusterCandidateRecord, ClusterModel, ClusteringJob, Dataset, User
from api.routers.auth import get_current_user
from services.clustering_registry import CLUSTER_REGISTRY, DEFAULT_ALGORITHM_IDS
from services.datasets import DatasetParsingError, read_dataframe
from services.job_watchdog import reconcile_stale_jobs

router = APIRouter(prefix="/clustering", tags=["clustering"])
_settings = get_settings()


# ── Schémas ──────────────────────────────────────────────────────────────


class ClusteringJobCreate(BaseModel):
    dataset_id: int
    feature_columns: List[str]
    seed: Optional[int] = None
    # Mode expert — sous-ensemble explicite du catalogue (voir
    # `services/clustering_registry.CLUSTER_REGISTRY`). `None` : sous-ensemble
    # par défaut, même pattern que `TrainingJobCreate.model_ids`.
    algorithm_ids: Optional[List[str]] = None


class ClusteringJobSummary(BaseModel):
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
    algorithm: Optional[str] = None
    n_clusters: Optional[int] = None
    silhouette: Optional[float] = None


class ClusterProfileOut(BaseModel):
    cluster_id: int
    size: int
    size_pct: float
    numeric_summary: dict[str, dict[str, float]]
    categorical_summary: dict[str, dict[str, Any]]
    differentiating_variables: List[str]


class ClusteringResultOut(BaseModel):
    algorithm: str
    n_clusters: int
    metrics: dict[str, Any]
    profiles: List[ClusterProfileOut]
    model_card: dict[str, Any]


class ClusterCandidateOut(BaseModel):
    algorithm: str
    family: str
    params: dict[str, Any]
    n_clusters: int
    silhouette: Optional[float]
    davies_bouldin: Optional[float]
    calinski_harabasz: Optional[float]
    noise_ratio: float
    is_winner: bool
    rank: int


class AlgorithmCatalogEntry(BaseModel):
    id: str
    label: str
    family: str
    is_default: bool


class AlgorithmCatalogResponse(BaseModel):
    algorithms: List[AlgorithmCatalogEntry]


# ── Aides ────────────────────────────────────────────────────────────────


def _get_org_job(job_id: int, current_user: User, db) -> ClusteringJob:
    job = (
        db.query(ClusteringJob)
        .filter(ClusteringJob.id == job_id, ClusteringJob.organization_id == current_user.organization_id)
        .first()
    )
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "CLUSTERING_JOB_INTROUVABLE", "message": "Entraînement de clustering introuvable"},
        )
    return job


def _to_summary(job: ClusteringJob) -> ClusteringJobSummary:
    result = job.result
    metrics = json.loads(result.metrics_json) if result else None
    return ClusteringJobSummary(
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
        algorithm=result.algorithm if result else None,
        n_clusters=result.n_clusters if result else None,
        silhouette=metrics.get("silhouette") if metrics else None,
    )


# ── Endpoints ────────────────────────────────────────────────────────────


@router.get("/algorithms-catalog", response_model=AlgorithmCatalogResponse)
def get_algorithms_catalog(current_user: User = Depends(get_current_user)):
    """Lecture pure du registre — aucun accès dataset, même pattern que
    `GET /training/models-catalog`."""
    return AlgorithmCatalogResponse(
        algorithms=[
            AlgorithmCatalogEntry(
                id=spec.id, label=spec.label, family=spec.family, is_default=spec.id in DEFAULT_ALGORITHM_IDS
            )
            for spec in CLUSTER_REGISTRY
        ]
    )


@router.post("/jobs", response_model=ClusteringJobSummary, status_code=status.HTTP_201_CREATED)
def create_clustering_job(
    body: ClusteringJobCreate,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    # Réconciliation des jobs orphelins (H2) — AVANT le comptage du quota,
    # même mécanisme que le supervisé (services/job_watchdog.py, généralisé
    # pour ce lot).
    reconcile_stale_jobs(
        db, current_user.organization_id, _settings.stale_job_timeout_minutes, model=ClusteringJob
    )

    # Quota partagé avec le supervisé : un seul worker physique traite les
    # deux types de job (voir docker-compose.yml) — compter les deux
    # ensemble contre la même limite, pas une limite séparée qui laisserait
    # une organisation saturer le worker en cumulant les deux types.
    from api.core.models import TrainingJob

    active_supervised = (
        db.query(TrainingJob)
        .filter(TrainingJob.organization_id == current_user.organization_id, TrainingJob.status.in_(("queued", "running")))
        .count()
    )
    active_clustering = (
        db.query(ClusteringJob)
        .filter(ClusteringJob.organization_id == current_user.organization_id, ClusteringJob.status.in_(("queued", "running")))
        .count()
    )
    active_total = active_supervised + active_clustering
    if active_total >= _settings.max_concurrent_jobs_per_org:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "code": "QUOTA_ENTRAINEMENTS_ATTEINT",
                "message": (
                    f"Trop d'entraînements en cours ({active_total}/{_settings.max_concurrent_jobs_per_org}, "
                    "supervisé et clustering confondus) — attendez qu'un entraînement se termine, ou "
                    "supprimez-en un depuis l'historique."
                ),
            },
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
            detail={"code": "COLONNES_MANQUANTES", "message": "Sélectionnez au moins une variable pour le clustering"},
        )
    unknown = set(body.feature_columns) - set(schema_columns)
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "COLONNES_INCONNUES", "message": f"Colonnes absentes du dataset : {', '.join(sorted(unknown))}"},
        )

    known_algorithm_ids = {s.id for s in CLUSTER_REGISTRY}
    if body.algorithm_ids is not None:
        unknown_algorithms = set(body.algorithm_ids) - known_algorithm_ids
        if unknown_algorithms:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": "ALGORITHMES_INCONNUS",
                    "message": f"Algorithmes inconnus : {', '.join(sorted(unknown_algorithms))}",
                },
            )

    try:
        read_dataframe(Path(dataset.file_path), Path(dataset.file_path).suffix)
    except DatasetParsingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "DATASET_LECTURE_ECHEC", "message": str(exc)},
        )

    config = {
        "algorithm_ids": body.algorithm_ids,
        "seed": body.seed if body.seed is not None else _settings.model_seed,
    }

    job = ClusteringJob(
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

    from workers.clustering_worker import run_clustering_job

    rq_job = training_queue.enqueue(run_clustering_job, job.id, job_timeout=1800)
    job.rq_job_id = rq_job.id
    db.commit()
    db.refresh(job)

    return _to_summary(job)


@router.get("/jobs", response_model=List[ClusteringJobSummary])
def list_clustering_jobs(current_user: User = Depends(get_current_user), db=Depends(get_db)):
    jobs = (
        db.query(ClusteringJob)
        .filter(ClusteringJob.organization_id == current_user.organization_id)
        .order_by(ClusteringJob.created_at.desc())
        .all()
    )
    return [_to_summary(j) for j in jobs]


@router.get("/jobs/{job_id}", response_model=ClusteringJobSummary)
def get_clustering_job(job_id: int, current_user: User = Depends(get_current_user), db=Depends(get_db)):
    return _to_summary(_get_org_job(job_id, current_user, db))


@router.get("/jobs/{job_id}/result", response_model=ClusteringResultOut)
def get_clustering_result(job_id: int, current_user: User = Depends(get_current_user), db=Depends(get_db)):
    job = _get_org_job(job_id, current_user, db)
    if job.result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "RESULTAT_INDISPONIBLE", "message": "Ce clustering n'a pas encore de résultat"},
        )
    result = job.result
    return ClusteringResultOut(
        algorithm=result.algorithm,
        n_clusters=result.n_clusters,
        metrics=json.loads(result.metrics_json),
        profiles=[ClusterProfileOut(**p) for p in json.loads(result.profiles_json)],
        model_card=json.loads(result.model_card_json or "{}"),
    )


@router.get("/jobs/{job_id}/candidates", response_model=List[ClusterCandidateOut])
def get_clustering_candidates(job_id: int, current_user: User = Depends(get_current_user), db=Depends(get_db)):
    job = _get_org_job(job_id, current_user, db)
    candidates = (
        db.query(ClusterCandidateRecord)
        .filter(ClusterCandidateRecord.clustering_job_id == job.id)
        .order_by(ClusterCandidateRecord.rank)
        .all()
    )
    return [
        ClusterCandidateOut(
            algorithm=c.algorithm,
            family=c.family,
            params=json.loads(c.params_json),
            n_clusters=c.n_clusters,
            silhouette=c.silhouette,
            davies_bouldin=c.davies_bouldin,
            calinski_harabasz=c.calinski_harabasz,
            noise_ratio=c.noise_ratio,
            is_winner=c.is_winner,
            rank=c.rank,
        )
        for c in candidates
    ]


@router.delete("/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_clustering_job(job_id: int, current_user: User = Depends(get_current_user), db=Depends(get_db)):
    """Supprime un clustering (et son résultat, s'il existe) — même logique
    que `training.py::delete_training_job` (annulation RQ best-effort,
    suppression explicite du résultat déjà chargé avant celle du job pour
    éviter l'IntegrityError SQLAlchemy sur la colonne NOT NULL, voir le
    correctif historique du Lot 4a)."""
    job = _get_org_job(job_id, current_user, db)

    if job.status in ("queued", "running") and job.rq_job_id:
        try:
            from rq.job import Job as RQJob

            rq_job = RQJob.fetch(job.rq_job_id, connection=training_queue.connection)
            rq_job.cancel()
            rq_job.delete()
        except Exception:
            pass  # best-effort — la suppression en base reste sûre dans tous les cas

    if job.result:
        if job.result.file_path:
            Path(job.result.file_path).unlink(missing_ok=True)
        db.delete(job.result)

    db.delete(job)
    db.commit()
