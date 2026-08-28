"""Router clustering — Lot 11+ (ML non supervisé).

Mêmes principes que `api/routers/training.py` : isolation systématique par
`organization_id`, tâche de fond obligatoire (RQ, `analysis_queue` —
file dédiée aux jobs courts, séparée de `training_queue`/`vision_queue`
depuis le correctif I6, voir `api/core/job_queue.py`), jamais de calcul
ML dans la requête HTTP. Router DÉDIÉ, jamais fusionné dans
`training.py` — même raisonnement que la séparation
`clustering_registry.py`/`ml_registry.py`.
"""
from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import joinedload

from api.core.config import get_settings
from api.core.database import SessionLocal, get_db
from api.core.job_queue import analysis_queue, redis_conn
from api.core.models import ClusterCandidateRecord, ClusteringJob, Dataset, User
from api.core.pagination import paginate_by_id
from domains.auth.router import get_current_user
from domains.clustering.services.inference import ClusterInferenceError, assign_cluster, assign_clusters_batch
from domains.clustering.services.registry import CLUSTER_REGISTRY, DEFAULT_ALGORITHM_IDS
from domains.shared.audit import log_action
from domains.shared.dataset_io import DatasetParsingError, read_dataset_dataframe
from domains.shared.job_creation import enqueue_or_mark_failed, remember_idempotent_job_id, resolve_idempotent_job_id
from domains.shared.job_events import stream_job_updates
from domains.shared.job_lifecycle import ACTIVE_STATUSES, CANCELLED_MESSAGE, try_cancel_rq_job
from domains.shared.job_quota import ALL_JOB_MODELS, raise_if_quota_exceeded
from domains.shared.job_watchdog import reconcile_stale_jobs
from domains.shared.model_bundle import InferenceError, load_bundle

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
    # Rangs individuels + rang composite (transparence de la sélection,
    # jamais une boîte noire — voir engine.py::_attach_composite_rank).
    rank_silhouette: Optional[float] = None
    rank_davies_bouldin: Optional[float] = None
    rank_calinski_harabasz: Optional[float] = None
    composite_rank: Optional[float] = None
    # Résultats complets (top 3 seulement, voir engine.py — retour
    # utilisateur : comparer plusieurs modèles au lieu d'un seul).
    cluster_profiles: Optional[List[ClusterProfileOut]] = None
    noise_count: Optional[int] = None


class AlgorithmCatalogEntry(BaseModel):
    id: str
    label: str
    family: str
    is_default: bool


class AlgorithmCatalogResponse(BaseModel):
    algorithms: List[AlgorithmCatalogEntry]


class ClusterPredictionRequest(BaseModel):
    data: dict[str, Any]


class ClusterPredictionResponse(BaseModel):
    cluster_id: Optional[int]
    is_noise: bool
    # "exact" (KMeans/K-Means rapide) / "approximate_centroid" (hiérarchique)
    # / "approximate_nearest_core" (DBSCAN) / "unsupported" (clustering
    # entraîné avant ce lot, ou famille future non couverte) — voir
    # services/clustering_inference.py pour le détail de chaque méthode.
    assignment_method: str


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


def to_summary(job: ClusteringJob) -> ClusteringJobSummary:
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
    request: Request,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    # Idempotence (Phase 2, AUDIT_BACKEND_2026-08-23.md §F4) — voir
    # domains/training/router.py::create_training_job pour la justification
    # complète, appliquée à l'identique ici.
    existing_job_id = resolve_idempotent_job_id(redis_conn, current_user.organization_id, request)
    if existing_job_id is not None:
        existing = (
            db.query(ClusteringJob)
            .filter(ClusteringJob.id == existing_job_id, ClusteringJob.organization_id == current_user.organization_id)
            .first()
        )
        if existing is not None:
            return to_summary(existing)

    # Réconciliation des jobs orphelins (H2) — AVANT le comptage du quota,
    # même mécanisme que le supervisé (services/job_watchdog.py, généralisé
    # pour ce lot).
    reconcile_stale_jobs(
        db, current_user.organization_id, _settings.stale_job_timeout_minutes, model=ClusteringJob
    )

    # Quota partagé avec le supervisé (et les autres types de job non
    # supervisé) via services/job_quota.py — un seul worker physique traite
    # tous les types de job (voir docker-compose.yml), comptés ensemble
    # contre la même limite.
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
        read_dataset_dataframe(Path(dataset.file_path), Path(dataset.file_path).suffix)
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
        request_id=request.state.request_id,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    remember_idempotent_job_id(redis_conn, current_user.organization_id, request, job.id)
    log_action(
        db, current_user.organization_id, current_user.id, "clustering_job.created",
        target_type="clustering_job", target_id=job.id,
    )

    from domains.clustering.worker import run_clustering_job

    enqueue_or_mark_failed(db, job, analysis_queue, run_clustering_job, 600)

    return to_summary(job)


@router.get("/jobs", response_model=List[ClusteringJobSummary])
def list_clustering_jobs(
    response: Response,
    limit: Optional[int] = Query(None, ge=1, le=500),
    cursor: Optional[int] = Query(None, description="id de la dernière ligne de la page précédente"),
    current_user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    # joinedload (Lot 4, correctif I3) — voir training.py::list_training_jobs.
    query = (
        db.query(ClusteringJob)
        .options(joinedload(ClusteringJob.dataset), joinedload(ClusteringJob.created_by), joinedload(ClusteringJob.result))
        .filter(ClusteringJob.organization_id == current_user.organization_id)
        # id DESC, pas created_at DESC — voir anomalies.py::list_anomaly_jobs
        # (précision seconde de SQLite, curseur de pagination cassé par des
        # égalités de created_at).
        .order_by(ClusteringJob.id.desc())
    )
    jobs = paginate_by_id(query, ClusteringJob.id, response, cursor, limit)
    return [to_summary(j) for j in jobs]


@router.get("/jobs/{job_id}", response_model=ClusteringJobSummary)
def get_clustering_job(job_id: int, current_user: User = Depends(get_current_user), db=Depends(get_db)):
    return to_summary(_get_org_job(job_id, current_user, db))


@router.get("/jobs/{job_id}/events")
async def stream_clustering_job_events(job_id: int, current_user: User = Depends(get_current_user)):
    """Notifications de fin de job par SSE (Lot 7, §J.2) — voir
    `training.py::stream_training_job_events` pour le raisonnement complet."""
    organization_id = current_user.organization_id
    db = SessionLocal()
    try:
        job = db.query(ClusteringJob).filter(ClusteringJob.id == job_id, ClusteringJob.organization_id == organization_id).first()
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "CLUSTERING_JOB_INTROUVABLE", "message": "Entraînement de clustering introuvable"},
            )
    finally:
        db.close()

    def fetch_snapshot():
        session = SessionLocal()
        try:
            row = session.query(ClusteringJob).filter(ClusteringJob.id == job_id, ClusteringJob.organization_id == organization_id).first()
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


@router.get("/jobs/{job_id}/model/export")
def export_clustering_model(job_id: int, current_user: User = Depends(get_current_user), db=Depends(get_db)):
    """Export de l'artefact (Lot 10, retour utilisateur direct — parité avec
    `GET /training/jobs/{job_id}/model/export`) : le bundle joblib complet
    (modèle de clustering + préprocesseur), pour un déploiement hors de la
    plateforme (`joblib.load` dans un environnement Python équivalent)."""
    job = _get_org_job(job_id, current_user, db)
    if job.status != "completed" or job.result is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "MODELE_NON_DISPONIBLE", "message": "Ce clustering n'a pas encore produit de modèle"},
        )
    artifact_path = Path(job.result.file_path)
    if not artifact_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "ARTEFACT_INTROUVABLE", "message": "Artefact du modèle introuvable sur le serveur"},
        )
    filename = f"clustering_{job.dataset.name.rsplit('.', 1)[0] if job.dataset else 'export'}_job{job.id}.joblib"
    return FileResponse(path=artifact_path, filename=filename, media_type="application/octet-stream")


@router.post("/jobs/{job_id}/predict", response_model=ClusterPredictionResponse)
def predict_cluster(
    job_id: int,
    body: ClusterPredictionRequest,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """Assigne une nouvelle observation à un groupe déjà découvert par ce
    clustering (Lot 6B, §F.2) — referme la même boucle que
    `training.py::predict_with_model` côté supervisé. Voir
    `services/clustering_inference.py` pour le détail des 3 méthodes
    d'assignation selon la famille d'algorithme retenue."""
    job = _get_org_job(job_id, current_user, db)
    if job.result is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "RESULTAT_INDISPONIBLE", "message": "Ce clustering n'a pas encore de résultat"},
        )
    result = job.result
    feature_columns = json.loads(result.feature_columns_json)

    try:
        bundle = load_bundle(result.file_path)
        assignment = assign_cluster(bundle, feature_columns, body.data)
    except (InferenceError, ClusterInferenceError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "ASSIGNATION_IMPOSSIBLE", "message": str(exc)},
        )

    return ClusterPredictionResponse(**assignment)


@router.get("/jobs/{job_id}/assignments/export")
def export_cluster_assignments(job_id: int, current_user: User = Depends(get_current_user), db=Depends(get_db)):
    """Assigne un cluster à CHAQUE ligne du dataset d'origine, pas seulement
    aux lignes effectivement clusterisées à l'entraînement — retour
    utilisateur direct : "l'entreprise vient avec 50 000 lignes, seules
    5 000 sont clusterisées (plafond mémoire, voir `MAX_ROWS_FOR_CLUSTERING`
    dans `services/clustering_training.py`), comment couvrir le reste ?".
    Applique le modèle déjà entraîné à la totalité du fichier, vectorisé
    (voir `services/clustering_inference.py::assign_clusters_batch`), jamais
    un nouveau clustering. `assignment_method` reste visible par ligne dans
    l'export — jamais un chiffre affiché sans dire comment il a été obtenu."""
    job = _get_org_job(job_id, current_user, db)
    if job.result is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "RESULTAT_INDISPONIBLE", "message": "Ce clustering n'a pas encore de résultat"},
        )
    if job.dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "DATASET_INTROUVABLE", "message": "Dataset introuvable"},
        )
    result = job.result
    feature_columns = json.loads(result.feature_columns_json)

    try:
        full_df = read_dataset_dataframe(Path(job.dataset.file_path), Path(job.dataset.file_path).suffix)
    except DatasetParsingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "DATASET_LECTURE_ECHEC", "message": str(exc)},
        ) from exc

    try:
        bundle = load_bundle(result.file_path)
    except InferenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "ASSIGNATION_IMPOSSIBLE", "message": str(exc)},
        ) from exc
    assigned = assign_clusters_batch(bundle, feature_columns, full_df)

    buffer = io.StringIO()
    assigned.to_csv(buffer, index=False)
    buffer.seek(0)
    filename = f"clustering_assignations_{job.dataset.name.rsplit('.', 1)[0]}_job{job.id}.csv"
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
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
            rank_silhouette=c.rank_silhouette,
            rank_davies_bouldin=c.rank_davies_bouldin,
            rank_calinski_harabasz=c.rank_calinski_harabasz,
            composite_rank=c.composite_rank,
            cluster_profiles=(
                [ClusterProfileOut(**p) for p in json.loads(c.cluster_profiles_json)]
                if c.cluster_profiles_json
                else None
            ),
            noise_count=c.noise_count,
        )
        for c in candidates
    ]


@router.post("/jobs/{job_id}/cancel", response_model=ClusteringJobSummary)
def cancel_clustering_job(job_id: int, current_user: User = Depends(get_current_user), db=Depends(get_db)):
    """Annule un clustering en attente ou en cours (Lot 7, §J.2) — contrairement
    à `DELETE /jobs/{id}`, garde une trace consultable (`status="cancelled"`)
    plutôt que de supprimer le job."""
    job = _get_org_job(job_id, current_user, db)
    if job.status not in ACTIVE_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "JOB_NON_ANNULABLE", "message": "Ce clustering n'est plus en attente ni en cours"},
        )
    try_cancel_rq_job(job.rq_job_id, analysis_queue)
    job.status = "cancelled"
    job.error_message = CANCELLED_MESSAGE
    job.finished_at = datetime.now(timezone.utc)
    log_action(
        db, current_user.organization_id, current_user.id, "clustering_job.cancelled",
        target_type="clustering_job", target_id=job.id, details={"dataset_id": job.dataset_id},
    )
    db.commit()
    db.refresh(job)
    return to_summary(job)


@router.post("/jobs/{job_id}/rerun", response_model=ClusteringJobSummary, status_code=status.HTTP_201_CREATED)
def rerun_clustering_job(
    job_id: int, request: Request, current_user: User = Depends(get_current_user), db=Depends(get_db)
):
    """Relance un clustering avec EXACTEMENT la même configuration (Lot 7,
    §J.2) — réutilise la validation complète de `POST /jobs`."""
    job = _get_org_job(job_id, current_user, db)
    config = json.loads(job.config_json)
    body = ClusteringJobCreate(
        dataset_id=job.dataset_id,
        feature_columns=json.loads(job.feature_columns_json),
        seed=config.get("seed"),
        algorithm_ids=config.get("algorithm_ids"),
    )
    # Arguments nommés (Phase 2, AUDIT_BACKEND_2026-08-23.md) — voir
    # domains/training/router.py::rerun_training_job pour le bug réel que
    # cela corrige (désalignement positionnel silencieux).
    return create_clustering_job(body=body, request=request, current_user=current_user, db=db)


@router.delete("/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_clustering_job(job_id: int, current_user: User = Depends(get_current_user), db=Depends(get_db)):
    """Supprime un clustering (et son résultat, s'il existe) — même logique
    que `training.py::delete_training_job` (annulation RQ best-effort,
    suppression explicite du résultat déjà chargé avant celle du job pour
    éviter l'IntegrityError SQLAlchemy sur la colonne NOT NULL, voir le
    correctif historique du Lot 4a)."""
    job = _get_org_job(job_id, current_user, db)

    if job.status in ACTIVE_STATUSES:
        try_cancel_rq_job(job.rq_job_id, analysis_queue)

    if job.result:
        if job.result.file_path:
            Path(job.result.file_path).unlink(missing_ok=True)
        db.delete(job.result)

    log_action(
        db, current_user.organization_id, current_user.id, "clustering_job.deleted",
        target_type="clustering_job", target_id=job.id,
        details={"dataset_id": job.dataset_id, "status": job.status},
    )
    db.delete(job)
    db.commit()
