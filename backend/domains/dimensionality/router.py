"""Router réduction de dimension — Lot 13 (ML non supervisé).

Mêmes principes que `api/routers/clustering.py` : isolation systématique par
`organization_id`, tâche de fond obligatoire (RQ, `analysis_queue` —
voir `api/core/job_queue.py`, correctif I6), jamais de calcul ML dans la
requête HTTP. Router DÉDIÉ, jamais fusionné.
"""
from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import joinedload

from api.core.config import get_settings
from api.core.database import SessionLocal, get_db
from api.core.error_codes import ErrorCode
from api.core.job_queue import analysis_queue, redis_conn
from api.core.models import Dataset, DimensionalityJob, DimensionalityPoint, DimensionalityProjectionLog, User
from api.core.pagination import paginate_by_id
from domains.auth.router import get_current_user
from domains.dimensionality.services.deployment_export import generate_dimensionality_deployment_script
from domains.dimensionality.services.inference import (
    DimensionalityInferenceError,
    project_point,
    project_points_batch,
)
from domains.dimensionality.services.registry import DEFAULT_ALGORITHM_ID, DIMENSIONALITY_REGISTRY
from domains.shared.audit import log_action
from domains.shared.dataset_io import DatasetParsingError, UnsupportedFileType, read_dataset_dataframe
from domains.shared.drift import MAX_PREDICTIONS_FOR_DRIFT, MIN_CURRENT_ROWS_FOR_DRIFT, compute_drift_report
from domains.shared.job_creation import enqueue_or_mark_failed, remember_idempotent_job_id, resolve_idempotent_job_id
from domains.shared.job_events import stream_job_updates
from domains.shared.job_lifecycle import ACTIVE_STATUSES, CANCELLED_MESSAGE, try_cancel_rq_job
from domains.shared.job_quota import ALL_JOB_MODELS, raise_if_quota_exceeded
from domains.shared.job_watchdog import reconcile_stale_jobs
from domains.shared.model_bundle import InferenceError, load_bundle

router = APIRouter(prefix="/dimensionality", tags=["dimensionality"])
_settings = get_settings()


# ── Schémas ──────────────────────────────────────────────────────────────


class DimensionalityJobCreate(BaseModel):
    dataset_id: int
    feature_columns: List[str]
    algorithm_id: str = DEFAULT_ALGORITHM_ID
    seed: Optional[int] = None


class DimensionalityJobSummary(BaseModel):
    id: int
    dataset_id: int
    dataset_name: Optional[str] = None
    feature_columns: List[str]
    algorithm_id: str
    status: str
    progress_step: Optional[str] = None
    progress_percent: int
    error_message: Optional[str] = None
    created_by: Optional[str] = None
    created_at: Any
    started_at: Optional[Any] = None
    finished_at: Optional[Any] = None
    algorithm: Optional[str] = None
    total_variance_explained: Optional[float] = None


class LoadingOut(BaseModel):
    feature: str
    pc1: float
    pc2: float


class DimensionalityResultOut(BaseModel):
    algorithm: str
    algorithm_id: str
    n_samples_total: int
    n_samples_used: int
    sampled: bool
    trustworthiness_primary: float
    trustworthiness_pca: float
    variance_explained: List[float]
    total_variance_explained: float
    loadings: List[LoadingOut]
    distance_fidelity_note: str
    feature_columns: List[str]
    model_card: dict[str, Any]


class DimensionalityPointOut(BaseModel):
    row_index: int
    x: float
    y: float


class DimensionalityColorByOut(BaseModel):
    column: str
    kind: str  # "numeric" | "categorical"
    values: dict[str, Any]  # clé = row_index (string, compat JSON), valeur brute ou null


class AlgorithmCatalogEntry(BaseModel):
    id: str
    label: str
    family: str
    is_default: bool


class AlgorithmCatalogResponse(BaseModel):
    algorithms: List[AlgorithmCatalogEntry]


class DimensionalityProjectionRequest(BaseModel):
    data: dict[str, Any]


class DimensionalityProjectionResponse(BaseModel):
    x: Optional[float]
    y: Optional[float]
    # "exact" (PCA/UMAP, `.transform()` natif) / "unsupported" (t-SNE,
    # transductif par nature — voir services/inference.py).
    projection_method: str


# ── Aides ────────────────────────────────────────────────────────────────


def _get_org_job(job_id: int, current_user: User, db) -> DimensionalityJob:
    job = (
        db.query(DimensionalityJob)
        .filter(DimensionalityJob.id == job_id, DimensionalityJob.organization_id == current_user.organization_id)
        .first()
    )
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "DIMENSIONALITY_JOB_INTROUVABLE", "message": "Calcul de réduction de dimension introuvable"},
        )
    return job


def to_summary(job: DimensionalityJob) -> DimensionalityJobSummary:
    result = job.result
    config = json.loads(job.config_json)
    return DimensionalityJobSummary(
        id=job.id,
        dataset_id=job.dataset_id,
        dataset_name=job.dataset.name if job.dataset else None,
        feature_columns=json.loads(job.feature_columns_json),
        algorithm_id=config.get("algorithm_id", DEFAULT_ALGORITHM_ID),
        status=job.status,
        progress_step=job.progress_step,
        progress_percent=job.progress_percent,
        error_message=job.error_message,
        created_by=job.created_by.nom if job.created_by else None,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        algorithm=result.algorithm if result else None,
        total_variance_explained=result.total_variance_explained if result else None,
    )


# ── Endpoints ────────────────────────────────────────────────────────────


@router.get("/algorithms-catalog", response_model=AlgorithmCatalogResponse)
def get_algorithms_catalog(current_user: User = Depends(get_current_user)):
    return AlgorithmCatalogResponse(
        algorithms=[
            AlgorithmCatalogEntry(
                id=spec.id, label=spec.label, family=spec.family, is_default=spec.id == DEFAULT_ALGORITHM_ID
            )
            for spec in DIMENSIONALITY_REGISTRY
        ]
    )


@router.post("/jobs", response_model=DimensionalityJobSummary, status_code=status.HTTP_201_CREATED)
def create_dimensionality_job(
    body: DimensionalityJobCreate,
    request: Request,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    # Idempotence (Phase 2, AUDIT_BACKEND_2026-08-23.md §F4) — voir
    # domains/training/router.py::create_training_job pour la justification.
    existing_job_id = resolve_idempotent_job_id(redis_conn, current_user.organization_id, request)
    if existing_job_id is not None:
        existing = (
            db.query(DimensionalityJob)
            .filter(
                DimensionalityJob.id == existing_job_id,
                DimensionalityJob.organization_id == current_user.organization_id,
            )
            .first()
        )
        if existing is not None:
            return to_summary(existing)

    reconcile_stale_jobs(
        db, current_user.organization_id, _settings.stale_job_timeout_minutes, model=DimensionalityJob
    )

    raise_if_quota_exceeded(
        db,
        current_user.organization_id,
        ALL_JOB_MODELS,
        _settings.max_concurrent_jobs_per_org,
    )

    known_algorithm_ids = {s.id for s in DIMENSIONALITY_REGISTRY}
    if body.algorithm_id not in known_algorithm_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "ALGORITHME_INCONNU", "message": f"Méthode inconnue ou indisponible : {body.algorithm_id}"},
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
        "algorithm_id": body.algorithm_id,
        "seed": body.seed if body.seed is not None else _settings.model_seed,
    }

    job = DimensionalityJob(
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
        db, current_user.organization_id, current_user.id, "dimensionality_job.created",
        target_type="dimensionality_job", target_id=job.id,
    )

    from domains.dimensionality.worker import run_dimensionality_job

    enqueue_or_mark_failed(db, job, analysis_queue, run_dimensionality_job, 600)

    return to_summary(job)


@router.get("/jobs", response_model=List[DimensionalityJobSummary])
def list_dimensionality_jobs(
    response: Response,
    limit: Optional[int] = Query(None, ge=1, le=500),
    cursor: Optional[int] = Query(None, description="id de la dernière ligne de la page précédente"),
    current_user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    # joinedload (Lot 4, correctif I3) — voir training.py::list_training_jobs.
    query = (
        db.query(DimensionalityJob)
        .options(
            joinedload(DimensionalityJob.dataset),
            joinedload(DimensionalityJob.created_by),
            joinedload(DimensionalityJob.result),
        )
        .filter(DimensionalityJob.organization_id == current_user.organization_id)
        # id DESC, pas created_at DESC — voir anomalies.py::list_anomaly_jobs.
        .order_by(DimensionalityJob.id.desc())
    )
    jobs = paginate_by_id(query, DimensionalityJob.id, response, cursor, limit)
    return [to_summary(j) for j in jobs]


@router.get("/jobs/{job_id}", response_model=DimensionalityJobSummary)
def get_dimensionality_job(job_id: int, current_user: User = Depends(get_current_user), db=Depends(get_db)):
    return to_summary(_get_org_job(job_id, current_user, db))


@router.get("/jobs/{job_id}/events")
async def stream_dimensionality_job_events(job_id: int, current_user: User = Depends(get_current_user)):
    """Notifications de fin de job par SSE (Lot 7, §J.2) — voir
    `training.py::stream_training_job_events` pour le raisonnement complet."""
    organization_id = current_user.organization_id
    db = SessionLocal()
    try:
        job = db.query(DimensionalityJob).filter(DimensionalityJob.id == job_id, DimensionalityJob.organization_id == organization_id).first()
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "DIMENSIONALITY_JOB_INTROUVABLE", "message": "Calcul de réduction de dimension introuvable"},
            )
    finally:
        db.close()

    def fetch_snapshot():
        session = SessionLocal()
        try:
            row = session.query(DimensionalityJob).filter(DimensionalityJob.id == job_id, DimensionalityJob.organization_id == organization_id).first()
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


@router.get("/jobs/{job_id}/result", response_model=DimensionalityResultOut)
def get_dimensionality_result(job_id: int, current_user: User = Depends(get_current_user), db=Depends(get_db)):
    job = _get_org_job(job_id, current_user, db)
    if job.result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": ErrorCode.RESULTAT_INDISPONIBLE, "message": "Ce calcul n'a pas encore de résultat"},
        )
    result = job.result
    return DimensionalityResultOut(
        algorithm=result.algorithm,
        algorithm_id=result.algorithm_id,
        n_samples_total=result.n_samples_total,
        n_samples_used=result.n_samples_used,
        sampled=result.sampled,
        trustworthiness_primary=result.trustworthiness_primary,
        trustworthiness_pca=result.trustworthiness_pca,
        variance_explained=json.loads(result.variance_explained_json),
        total_variance_explained=result.total_variance_explained,
        loadings=[LoadingOut(**loading) for loading in json.loads(result.loadings_json)],
        distance_fidelity_note=result.distance_fidelity_note,
        feature_columns=json.loads(result.feature_columns_json),
        model_card=json.loads(result.model_card_json or "{}"),
    )


@router.get("/jobs/{job_id}/model/export")
def export_dimensionality_model(job_id: int, current_user: User = Depends(get_current_user), db=Depends(get_db)):
    """Export de l'artefact (Lot 10, retour utilisateur direct — parité avec
    `GET /training/jobs/{job_id}/model/export`)."""
    job = _get_org_job(job_id, current_user, db)
    if job.status != "completed" or job.result is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "MODELE_NON_DISPONIBLE", "message": "Ce calcul n'a pas encore produit de modèle"},
        )
    artifact_path = Path(job.result.file_path)
    if not artifact_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "ARTEFACT_INTROUVABLE", "message": "Artefact du modèle introuvable sur le serveur"},
        )
    filename = f"projection_{job.dataset.name.rsplit('.', 1)[0] if job.dataset else 'export'}_job{job.id}.joblib"
    return FileResponse(path=artifact_path, filename=filename, media_type="application/octet-stream")


@router.get("/jobs/{job_id}/model/export-script")
def export_dimensionality_deployment_script(
    job_id: int, current_user: User = Depends(get_current_user), db=Depends(get_db)
):
    """Script de déploiement autonome — même principe que
    `GET /clustering/jobs/{job_id}/model/export-script` : recharge le bundle
    joblib déjà exportable et reproduit la projection SANS jamais importer un
    module `domains.*` de ce projet. Honnête sur les limites : t-SNE n'a pas
    de méthode `.transform()` (transductif), le script généré le rappelle
    explicitement plutôt que d'échouer silencieusement."""
    job = _get_org_job(job_id, current_user, db)
    if job.status != "completed" or job.result is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "MODELE_NON_DISPONIBLE", "message": "Ce calcul n'a pas encore produit de modèle"},
        )
    result = job.result
    artifact_path = Path(result.file_path)
    if not artifact_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "ARTEFACT_INTROUVABLE", "message": "Artefact du modèle introuvable sur le serveur"},
        )
    try:
        load_bundle(result.file_path)
    except InferenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "ARTEFACT_ILLISIBLE", "message": str(exc)},
        ) from exc
    dataset_name = job.dataset.name.rsplit(".", 1)[0] if job.dataset else "export"
    base_name = f"projection_{dataset_name}_job{job.id}"
    artifact_filename = f"{base_name}.joblib"
    script_filename = f"{base_name}_deploiement.py"
    script = generate_dimensionality_deployment_script(
        feature_columns=json.loads(result.feature_columns_json),
        algorithm=result.algorithm,
        algorithm_id=result.algorithm_id,
        artifact_filename=artifact_filename,
        script_filename=script_filename,
    )
    return Response(
        content=script,
        media_type="text/x-python",
        headers={"Content-Disposition": f'attachment; filename="{script_filename}"'},
    )


@router.post("/jobs/{job_id}/project", response_model=DimensionalityProjectionResponse)
def project_new_point(
    job_id: int,
    body: DimensionalityProjectionRequest,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """Projette une NOUVELLE observation avec une réduction de dimension déjà
    entraînée (Lot 6B, §F.2) — referme la même boucle que
    `clustering.py::predict_cluster` côté clustering. Voir
    `services/inference.py` (exact pour PCA/UMAP, `"unsupported"` pour
    t-SNE — jamais d'approximation inventée pour un modèle transductif)."""
    job = _get_org_job(job_id, current_user, db)
    if job.result is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": ErrorCode.RESULTAT_INDISPONIBLE, "message": "Ce calcul n'a pas encore de résultat"},
        )
    result = job.result
    feature_columns = json.loads(result.feature_columns_json)

    try:
        bundle = load_bundle(result.file_path)
        projected = project_point(bundle, feature_columns, body.data)
    except (InferenceError, DimensionalityInferenceError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "PROJECTION_IMPOSSIBLE", "message": str(exc)},
        ) from exc

    # Lot Dérive — voir clustering.py::predict_cluster pour le raisonnement
    # complet (journalise seulement une projection réussie).
    db.add(DimensionalityProjectionLog(
        organization_id=current_user.organization_id,
        dimensionality_job_id=job.id,
        input_json=json.dumps(body.data),
    ))
    db.commit()

    return DimensionalityProjectionResponse(**projected)


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


@router.get("/jobs/{job_id}/drift", response_model=DriftReportOut)
def get_dimensionality_drift(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """Dérive des données — voir
    `domains/training/router.py::get_model_drift` /
    `domains/clustering/router.py::get_clustering_drift` pour le
    raisonnement complet, identique ici (variables réellement soumises à
    `/project` comparées au dataset d'entraînement, voir
    `DimensionalityProjectionLog`)."""
    job = _get_org_job(job_id, current_user, db)
    if job.result is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": ErrorCode.RESULTAT_INDISPONIBLE, "message": "Ce calcul n'a pas encore de résultat"},
        )
    result = job.result
    feature_columns = json.loads(result.feature_columns_json)

    rows = (
        db.query(DimensionalityProjectionLog)
        .filter(
            DimensionalityProjectionLog.dimensionality_job_id == job.id,
            DimensionalityProjectionLog.organization_id == current_user.organization_id,
        )
        .order_by(DimensionalityProjectionLog.id.desc())
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
            detail={"code": "DATASET_LECTURE_ECHEC", "message": str(exc)},
        ) from exc

    report = compute_drift_report(reference_df, current_df, feature_columns)
    return DriftReportOut(**report, min_predictions_required=MIN_CURRENT_ROWS_FOR_DRIFT)


@router.get("/jobs/{job_id}/points/export")
def export_dimensionality_points(job_id: int, current_user: User = Depends(get_current_user), db=Depends(get_db)):
    """Projette CHAQUE ligne du dataset d'origine, pas seulement l'échantillon
    effectivement utilisé à l'entraînement — même retour utilisateur que
    `clustering.py::export_cluster_assignments`. `projection_status` reste
    visible par ligne (inclut `"unsupported"` pour un job t-SNE entier)."""
    job = _get_org_job(job_id, current_user, db)
    if job.result is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": ErrorCode.RESULTAT_INDISPONIBLE, "message": "Ce calcul n'a pas encore de résultat"},
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
            detail={"code": "PROJECTION_IMPOSSIBLE", "message": str(exc)},
        ) from exc
    projected = project_points_batch(bundle, feature_columns, full_df)

    buffer = io.StringIO()
    projected.to_csv(buffer, index=False)
    buffer.seek(0)
    filename = f"projection_{job.dataset.name.rsplit('.', 1)[0]}_job{job.id}.csv"
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/jobs/{job_id}/points", response_model=List[DimensionalityPointOut])
def get_dimensionality_points(job_id: int, current_user: User = Depends(get_current_user), db=Depends(get_db)):
    job = _get_org_job(job_id, current_user, db)
    points = (
        db.query(DimensionalityPoint)
        .filter(DimensionalityPoint.dimensionality_job_id == job.id)
        .order_by(DimensionalityPoint.row_index)
        .all()
    )
    return [DimensionalityPointOut(row_index=p.row_index, x=p.x, y=p.y) for p in points]


@router.get("/jobs/{job_id}/color-by", response_model=DimensionalityColorByOut)
def get_dimensionality_color_by(
    job_id: int,
    column: str = Query(...),
    current_user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """Relit UNE colonne du dataset source pour les observations déjà
    projetées — calcul léger à la demande (comme `/datasets/{id}/histogram`),
    pas un nouveau job : changer la coloration du nuage de points devient
    instantané, sans relancer un calcul coûteux (t-SNE/UMAP)."""
    job = _get_org_job(job_id, current_user, db)
    if job.dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "DATASET_INTROUVABLE", "message": "Dataset source introuvable"},
        )

    try:
        df = read_dataset_dataframe(Path(job.dataset.file_path), Path(job.dataset.file_path).suffix)
    except DatasetParsingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "DATASET_LECTURE_ECHEC", "message": str(exc)},
        )
    if column not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "COLONNE_INCONNUE", "message": f"Colonne absente du dataset : {column}"},
        )

    row_indices = [
        p.row_index
        for p in db.query(DimensionalityPoint.row_index)
        .filter(DimensionalityPoint.dimensionality_job_id == job.id)
        .all()
    ]
    series = df[column]
    kind = "numeric" if pd.api.types.is_numeric_dtype(series) else "categorical"
    values: dict[str, Any] = {}
    for row_index in row_indices:
        if row_index < 0 or row_index >= len(series):
            continue
        raw_value = series.iloc[row_index]
        if pd.isna(raw_value):
            values[str(row_index)] = None
        elif kind == "numeric":
            values[str(row_index)] = float(raw_value)
        else:
            values[str(row_index)] = str(raw_value)

    return DimensionalityColorByOut(column=column, kind=kind, values=values)


@router.post("/jobs/{job_id}/cancel", response_model=DimensionalityJobSummary)
def cancel_dimensionality_job(job_id: int, current_user: User = Depends(get_current_user), db=Depends(get_db)):
    """Annule une réduction de dimension en attente ou en cours (Lot 7,
    §J.2) — garde une trace consultable (`status="cancelled"`)."""
    job = _get_org_job(job_id, current_user, db)
    if job.status not in ACTIVE_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "JOB_NON_ANNULABLE", "message": "Cette réduction de dimension n'est plus en attente ni en cours"},
        )
    try_cancel_rq_job(job.rq_job_id, analysis_queue)
    job.status = "cancelled"
    job.error_message = CANCELLED_MESSAGE
    job.finished_at = datetime.now(timezone.utc)
    log_action(
        db, current_user.organization_id, current_user.id, "dimensionality_job.cancelled",
        target_type="dimensionality_job", target_id=job.id, details={"dataset_id": job.dataset_id},
    )
    db.commit()
    db.refresh(job)
    return to_summary(job)


@router.post("/jobs/{job_id}/rerun", response_model=DimensionalityJobSummary, status_code=status.HTTP_201_CREATED)
def rerun_dimensionality_job(
    job_id: int, request: Request, current_user: User = Depends(get_current_user), db=Depends(get_db)
):
    """Relance une réduction de dimension avec EXACTEMENT la même
    configuration (Lot 7, §J.2) — réutilise la validation complète de
    `POST /jobs`."""
    job = _get_org_job(job_id, current_user, db)
    config = json.loads(job.config_json)
    body = DimensionalityJobCreate(
        dataset_id=job.dataset_id,
        feature_columns=json.loads(job.feature_columns_json),
        algorithm_id=config.get("algorithm_id", DEFAULT_ALGORITHM_ID),
        seed=config.get("seed"),
    )
    # Arguments nommés (Phase 2, AUDIT_BACKEND_2026-08-23.md) — voir
    # domains/training/router.py::rerun_training_job pour le bug réel que
    # cela corrige.
    return create_dimensionality_job(body=body, request=request, current_user=current_user, db=db)


@router.delete("/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dimensionality_job(job_id: int, current_user: User = Depends(get_current_user), db=Depends(get_db)):
    job = _get_org_job(job_id, current_user, db)

    if job.status in ACTIVE_STATUSES:
        try_cancel_rq_job(job.rq_job_id, analysis_queue)

    if job.result:
        if job.result.file_path:
            Path(job.result.file_path).unlink(missing_ok=True)
        db.delete(job.result)

    log_action(
        db, current_user.organization_id, current_user.id, "dimensionality_job.deleted",
        target_type="dimensionality_job", target_id=job.id,
        details={"dataset_id": job.dataset_id, "status": job.status},
    )
    db.delete(job)
    db.commit()
