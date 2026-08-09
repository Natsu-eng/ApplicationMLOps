"""Router datasets — upload, liste, aperçu, suppression.

Isolation : toute opération est filtrée par
`Dataset.organization_id == current_user.organization_id` — jamais par un
identifiant fourni par le client (même principe que
`api/routers/auth.py::list_team_members`). Accessible à tout membre de
l'organisation, pas réservé au owner : un dataset appartient à l'équipe.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.core.config import get_settings
from api.core.database import get_db
from api.core.models import Dataset, User
from api.core.storage import dataset_file_path, delete_dataset_file
from api.routers.auth import get_current_user
from services.datasets import (
    DatasetParsingError,
    UnsupportedFileType,
    extract_schema,
    read_dataframe,
    sample_rows,
    validate_extension,
)

logger = logging.getLogger("datalab.datasets")
router = APIRouter(prefix="/datasets", tags=["datasets"])

_settings = get_settings()
_MAX_UPLOAD_BYTES = _settings.max_upload_size_mb * 1024 * 1024


# ── Schémas ──────────────────────────────────────────────────────────────────

class DatasetSummary(BaseModel):
    id: int
    name: str
    file_size_bytes: int
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    status: str
    error_message: Optional[str] = None
    uploaded_by: Optional[str] = None
    created_at: datetime


class ColumnSchema(BaseModel):
    name: str
    dtype: str


class DatasetDetail(DatasetSummary):
    columns: List[ColumnSchema] = []


class PreviewResponse(BaseModel):
    columns: List[str]
    rows: List[dict]
    sample_size: int
    row_count: Optional[int] = None


def _to_summary(dataset: Dataset) -> DatasetSummary:
    # Construction explicite plutôt que model_validate(dataset, from_attributes=True) :
    # le champ Pydantic `uploaded_by` (str) entre en collision de nom avec la relation
    # SQLAlchemy `Dataset.uploaded_by` (objet User) — from_attributes essaierait de
    # valider l'objet User comme une string et lèverait une ValidationError.
    return DatasetSummary(
        id=dataset.id,
        name=dataset.name,
        file_size_bytes=dataset.file_size_bytes,
        row_count=dataset.row_count,
        column_count=dataset.column_count,
        status=dataset.status,
        error_message=dataset.error_message,
        uploaded_by=dataset.uploaded_by.nom if dataset.uploaded_by else None,
        created_at=dataset.created_at,
    )


def _to_detail(dataset: Dataset) -> DatasetDetail:
    columns = [ColumnSchema(**c) for c in json.loads(dataset.columns_json)] if dataset.columns_json else []
    return DatasetDetail(**_to_summary(dataset).model_dump(), columns=columns)


def _get_org_dataset(dataset_id: int, current_user: User, db: Session) -> Dataset:
    dataset = (
        db.query(Dataset)
        .filter(Dataset.id == dataset_id, Dataset.organization_id == current_user.organization_id)
        .first()
    )
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "DATASET_INTROUVABLE", "message": "Dataset introuvable"},
        )
    return dataset


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("", response_model=DatasetDetail, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Upload d'un dataset tabulaire (csv/xlsx/xls/parquet/json) — visible par toute l'organisation."""
    try:
        extension = validate_extension(file.filename or "")
    except UnsupportedFileType as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "DATASET_FORMAT_NON_SUPPORTE", "message": str(exc)},
        )

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "DATASET_FICHIER_VIDE", "message": "Le fichier est vide"},
        )
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "code": "DATASET_TROP_VOLUMINEUX",
                "message": f"Fichier trop volumineux (max {_settings.max_upload_size_mb} Mo)",
            },
        )

    # La ligne DB est créée avant l'écriture sur disque pour obtenir un id
    # (utilisé dans le nom du fichier stocké — voir api/core/storage.py).
    dataset = Dataset(
        organization_id=current_user.organization_id,
        uploaded_by_id=current_user.id,
        name=file.filename or "dataset",
        file_path="",
        file_size_bytes=len(content),
        status="processing",
    )
    db.add(dataset)
    db.flush()

    target_path = dataset_file_path(current_user.organization_id, dataset.id, extension)
    target_path.write_bytes(content)
    dataset.file_path = str(target_path)

    try:
        df = read_dataframe(target_path, extension)
        dataset.row_count = int(len(df))
        dataset.column_count = int(len(df.columns))
        dataset.columns_json = json.dumps(extract_schema(df))
        dataset.status = "ready"
    except DatasetParsingError as exc:
        dataset.status = "error"
        dataset.error_message = str(exc)
        logger.warning("[Datasets] Échec de parsing pour %s : %s", file.filename, exc)

    db.commit()
    db.refresh(dataset)
    return _to_detail(dataset)


@router.get("", response_model=List[DatasetSummary])
def list_datasets(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    datasets = (
        db.query(Dataset)
        .filter(Dataset.organization_id == current_user.organization_id)
        .order_by(Dataset.created_at.desc())
        .all()
    )
    return [_to_summary(d) for d in datasets]


@router.get("/{dataset_id}", response_model=DatasetDetail)
def get_dataset(dataset_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return _to_detail(_get_org_dataset(dataset_id, current_user, db))


@router.get("/{dataset_id}/preview", response_model=PreviewResponse)
def preview_dataset(
    dataset_id: int,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    dataset = _get_org_dataset(dataset_id, current_user, db)
    if dataset.status != "ready":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "DATASET_NON_PRET", "message": "Ce dataset n'a pas pu être analysé"},
        )
    extension = Path(dataset.file_path).suffix
    try:
        df = read_dataframe(Path(dataset.file_path), extension)
    except DatasetParsingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "DATASET_LECTURE_ECHEC", "message": str(exc)},
        )
    limit = max(1, min(limit, 500))
    rows = sample_rows(df, limit)
    return PreviewResponse(
        columns=list(df.columns.astype(str)),
        rows=rows,
        sample_size=len(rows),
        row_count=dataset.row_count,
    )


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(dataset_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    dataset = _get_org_dataset(dataset_id, current_user, db)
    if dataset.file_path:
        delete_dataset_file(Path(dataset.file_path))
    db.delete(dataset)
    db.commit()
