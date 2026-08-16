"""Router dataset vision — upload ZIP, liste, détail, suppression (pilier
Vision, Lot 15 sous-lot A).

Même principe d'isolation que `api/routers/datasets.py` : toute opération
filtrée par `organization_id`. Upload synchrone (comme les datasets
tabulaires) — pas de tâche de fond ici, la validation d'images est bornée
par `max_vision_dataset_images` et reste de l'ordre de la seconde/dizaine de
secondes, pas un entraînement.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.core.config import get_settings
from api.core.database import get_db
from api.core.models import User, VisionDataset
from api.core.storage import delete_vision_dataset_dir, vision_dataset_dir
from api.routers.auth import get_current_user
from services.audit import log_action
from services.vision_datasets import (
    UnsupportedFileType,
    VisionDatasetError,
    analyze_and_extract_vision_zip,
    validate_zip_extension,
)

logger = logging.getLogger("datalab.vision_datasets")
router = APIRouter(prefix="/vision/datasets", tags=["vision"])

_settings = get_settings()
_MAX_UPLOAD_BYTES = _settings.max_vision_upload_size_mb * 1024 * 1024


# ── Schémas ──────────────────────────────────────────────────────────────


class VisionDatasetSummary(BaseModel):
    id: int
    name: str
    structure_type: str
    n_images: int
    n_classes: Optional[int] = None
    status: str
    error_message: Optional[str] = None
    uploaded_by: Optional[str] = None
    created_at: datetime


class VisionDatasetDetail(VisionDatasetSummary):
    class_distribution: dict[str, int] = {}
    validation_report: dict = {}


def _to_summary(dataset: VisionDataset) -> VisionDatasetSummary:
    return VisionDatasetSummary(
        id=dataset.id,
        name=dataset.name,
        structure_type=dataset.structure_type,
        n_images=dataset.n_images,
        n_classes=dataset.n_classes,
        status=dataset.status,
        error_message=dataset.error_message,
        uploaded_by=dataset.uploaded_by.nom if dataset.uploaded_by else None,
        created_at=dataset.created_at,
    )


def _to_detail(dataset: VisionDataset) -> VisionDatasetDetail:
    return VisionDatasetDetail(
        **_to_summary(dataset).model_dump(),
        class_distribution=json.loads(dataset.class_distribution_json) if dataset.class_distribution_json else {},
        validation_report=json.loads(dataset.validation_report_json) if dataset.validation_report_json else {},
    )


def _get_org_dataset(dataset_id: int, current_user: User, db: Session) -> VisionDataset:
    dataset = (
        db.query(VisionDataset)
        .filter(VisionDataset.id == dataset_id, VisionDataset.organization_id == current_user.organization_id)
        .first()
    )
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "VISION_DATASET_INTROUVABLE", "message": "Dataset d'images introuvable"},
        )
    return dataset


# ── Endpoints ────────────────────────────────────────────────────────────


@router.post("", response_model=VisionDatasetDetail, status_code=status.HTTP_201_CREATED)
async def upload_vision_dataset(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Upload d'un dataset d'images (ZIP) — dossiers de classes
    (classification) ou structure MVTec AD (train/good + test/good +
    test/<defaut>), détectée automatiquement et validée strictement (jamais
    devinée en silence — correctif du bug #1,
    docs/legacy/ANALYSE_COMPLETE_COMPUTER_VISION.md)."""
    try:
        validate_zip_extension(file.filename or "")
    except UnsupportedFileType as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "VISION_DATASET_FORMAT_NON_SUPPORTE", "message": str(exc)},
        )

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "VISION_DATASET_FICHIER_VIDE", "message": "L'archive est vide"},
        )
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "code": "VISION_DATASET_TROP_VOLUMINEUX",
                "message": f"Archive trop volumineuse (max {_settings.max_vision_upload_size_mb} Mo)",
            },
        )

    dataset = VisionDataset(
        organization_id=current_user.organization_id,
        uploaded_by_id=current_user.id,
        name=file.filename or "dataset.zip",
        structure_type="",
        storage_dir="",
        status="processing",
    )
    db.add(dataset)
    db.flush()

    target_dir = vision_dataset_dir(current_user.organization_id, dataset.id)
    dataset.storage_dir = str(target_dir)

    try:
        report = analyze_and_extract_vision_zip(
            content,
            target_dir,
            max_images=_settings.max_vision_dataset_images,
            max_uncompressed_bytes=_MAX_UPLOAD_BYTES * 4,
        )
        dataset.structure_type = report.structure_type
        dataset.n_images = report.n_images
        dataset.n_classes = report.n_classes
        dataset.class_distribution_json = json.dumps(report.class_distribution)
        dataset.validation_report_json = json.dumps({
            "n_corrupted": report.n_corrupted,
            "corrupted_files": report.corrupted_files,
            "n_duplicates": report.n_duplicates,
            "duplicate_groups": report.duplicate_groups,
            "n_undersized": report.n_undersized,
            "undersized_files": report.undersized_files,
            "warnings": report.warnings,
        })
        dataset.status = "ready"
    except VisionDatasetError as exc:
        dataset.status = "error"
        dataset.error_message = str(exc)
        logger.warning("[VisionDatasets] Échec de validation pour %s : %s", file.filename, exc)

    db.commit()
    db.refresh(dataset)
    return _to_detail(dataset)


@router.get("", response_model=List[VisionDatasetSummary])
def list_vision_datasets(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    datasets = (
        db.query(VisionDataset)
        .filter(VisionDataset.organization_id == current_user.organization_id)
        .order_by(VisionDataset.created_at.desc())
        .all()
    )
    return [_to_summary(d) for d in datasets]


@router.get("/{dataset_id}", response_model=VisionDatasetDetail)
def get_vision_dataset(dataset_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return _to_detail(_get_org_dataset(dataset_id, current_user, db))


@router.get("/{dataset_id}/image")
def get_vision_dataset_image(
    dataset_id: int,
    path: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Sert une image individuelle du dataset — nécessaire pour que le
    frontend affiche les exemples de prédiction (classification, `relative_path`)
    et les images sources des exemples d'anomalies visuelles (sous-lots B/C).
    `path` est le chemin relatif renvoyé par ces endpoints (ex.
    "classe_a/img_3.png" ou "test/scratch/2.png"), jamais un chemin absolu
    fourni librement par le client — protection contre la traversée de
    répertoire (`..`) en vérifiant que le chemin résolu reste bien SOUS le
    dossier du dataset avant de le servir."""
    dataset = _get_org_dataset(dataset_id, current_user, db)
    base_dir = Path(dataset.storage_dir).resolve()
    target = (base_dir / path).resolve()
    if base_dir not in target.parents or not target.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "IMAGE_INTROUVABLE", "message": "Image introuvable dans ce dataset"},
        )
    return FileResponse(target)


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_vision_dataset(dataset_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    dataset = _get_org_dataset(dataset_id, current_user, db)
    if dataset.storage_dir:
        delete_vision_dataset_dir(Path(dataset.storage_dir))
    log_action(
        db, current_user.organization_id, current_user.id, "vision_dataset.deleted",
        target_type="vision_dataset", target_id=dataset.id, details={"name": dataset.name},
    )
    db.delete(dataset)
    db.commit()
