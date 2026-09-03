"""Router prédiction en lot — upload d'un fichier, une prédiction par ligne.

Extrait de `router.py` lors du découpage du router training (qui dépassait
1 900 lignes) : la prédiction en lot est une fonctionnalité distincte de
l'entraînement lui-même — son propre modèle (`BatchPredictionJob`), sa
propre file (`analysis_queue`, jamais `training_queue` : un job court), son
propre worker (`batch_prediction_worker.py`). Mêmes principes que
`router.py` : isolation systématique par `organization_id`, tâche de fond
obligatoire, jamais de calcul dans la requête HTTP.

Les chemins exposés sont RIGOUREUSEMENT inchangés par l'extraction (même
préfixe `/training`, mêmes URL) — ce module est enregistré à part dans
`api/main.py`, à côté du router training.
"""
from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session, joinedload

from api.core.config import get_settings
from api.core.database import SessionLocal, get_db
from api.core.error_codes import ErrorCode
from api.core.job_queue import analysis_queue, redis_conn
from api.core.models import BatchPredictionJob, User
from api.core.pagination import paginate_by_id
from api.core.rate_limit import rate_limit_dependency
from api.core.storage import batch_prediction_input_file_path
from domains.auth.router import get_current_user
from domains.shared.audit import log_action
from domains.shared.dataset_io import UnsupportedFileType, validate_extension
from domains.shared.job_creation import enqueue_or_mark_failed, remember_idempotent_job_id, resolve_idempotent_job_id
from domains.shared.job_events import stream_job_updates
from domains.shared.job_lifecycle import ACTIVE_STATUSES, CANCELLED_MESSAGE, try_cancel_rq_job
from domains.shared.job_quota import ALL_JOB_MODELS, raise_if_quota_exceeded
from domains.shared.job_watchdog import reconcile_stale_jobs
from domains.training.batch_prediction_worker import run_batch_prediction_job
from domains.training.dependencies import get_org_training_job

router = APIRouter(prefix="/training", tags=["training"])
_settings = get_settings()


# ── Schémas ──────────────────────────────────────────────────────────────────

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
            detail={"code": ErrorCode.PREDICTION_LOT_INTROUVABLE, "message": "Prédiction en lot introuvable"},
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
    job = get_org_training_job(job_id, current_user, db)
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
            detail={"code": ErrorCode.DATASET_FORMAT_NON_SUPPORTE, "message": str(exc)},
        ) from exc

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": ErrorCode.DATASET_FICHIER_VIDE, "message": "Le fichier est vide"},
        )
    max_upload_bytes = _settings.max_upload_size_mb * 1024 * 1024
    if len(content) > max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "code": ErrorCode.DATASET_TROP_VOLUMINEUX,
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
                detail={"code": ErrorCode.PREDICTION_LOT_INTROUVABLE, "message": "Prédiction en lot introuvable"},
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
            detail={"code": ErrorCode.RESULTAT_INTROUVABLE, "message": "Résultat introuvable sur le serveur"},
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
            detail={"code": ErrorCode.RESULTAT_INTROUVABLE, "message": "Résultat introuvable sur le serveur"},
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
