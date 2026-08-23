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

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.core.config import get_settings
from api.core.database import get_db
from api.core.models import User, VisionDataset
from api.core.rate_limit import rate_limit_dependency
from api.core.storage import delete_vision_dataset_dir, vision_dataset_dir
from domains.auth.router import get_current_user
from domains.shared.audit import log_action
from domains.vision.shared import (
    ANOMALY_IMAGE_SIZE,
    AUGMENTATION_PRESET_IDS,
    IMAGE_SIZE as CLASSIFICATION_IMAGE_SIZE,
    augmentation_transforms,
    recommend_augmentation_preset,
)
from domains.vision.datasets.service import (
    UnsupportedFileType,
    VisionDatasetError,
    analyze_and_extract_vision_archive,
    analyze_and_extract_vision_folder,
    validate_archive_extension,
)
from domains.vision.localization import encode_image_png

logger = logging.getLogger("datalab.vision_datasets")
router = APIRouter(prefix="/vision/datasets", tags=["vision"])

_settings = get_settings()
_MAX_UPLOAD_BYTES = _settings.max_vision_upload_size_mb * 1024 * 1024

# Exploration (Lot 16C) — galerie de miniatures par classe, plafonnée pour
# ne jamais renvoyer des milliers de chemins d'un coup (une classe MVTec AD
# peut en compter plusieurs centaines).
MAX_GALLERY_IMAGES_PER_CLASS = 60


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
    # Lot 6A (correctif I9, AUDIT_DATALAB_2026-08-16.md §I9) — indicative
    # uniquement (jamais appliquée automatiquement), absente (None) pour
    # un dataset "mvtec_ad" (l'augmentation d'images ne concerne que
    # l'entraînement de classification).
    recommended_augmentation_preset: Optional[str] = None


class VisionDatasetImageList(BaseModel):
    class_name: str
    total: int
    # Chemins relatifs (`relative_path`) prêts à être passés tels quels à
    # GET /{dataset_id}/image?path=... — plafonné à MAX_GALLERY_IMAGES_PER_CLASS,
    # `total` reste le compte réel pour que le frontend puisse afficher
    # "60 sur 340 images affichées" plutôt qu'un compte tronqué silencieux.
    paths: List[str]


class AugmentationPreviewPair(BaseModel):
    original_png: str
    augmented_png: str


class AugmentationPreviewOut(BaseModel):
    preset: str
    pairs: List[AugmentationPreviewPair]


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
    class_distribution = json.loads(dataset.class_distribution_json) if dataset.class_distribution_json else {}
    # Lot 6A (correctif I9) — fondée sur la classe la plus PETITE (le
    # goulot d'étranglement réel), jamais un dataset "mvtec_ad" (recommandation
    # sans objet — l'augmentation d'images ne concerne que la classification).
    recommended_preset = (
        recommend_augmentation_preset(min(class_distribution.values()))
        if dataset.structure_type == "classification" and class_distribution
        else None
    )
    return VisionDatasetDetail(
        **_to_summary(dataset).model_dump(),
        class_distribution=class_distribution,
        validation_report=json.loads(dataset.validation_report_json) if dataset.validation_report_json else {},
        recommended_augmentation_preset=recommended_preset,
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


_upload_rate_limit = rate_limit_dependency(
    "vision_dataset_upload", _settings.upload_rate_limit_max_attempts, _settings.upload_rate_limit_window_seconds,
    # Échec FERMÉ (Phase 1, AUDIT_BACKEND_2026-08-23.md §4) — extraction
    # ZIP synchrone dans la requête HTTP, encore plus coûteuse que l'upload
    # tabulaire. Voir domains/datasets/router.py::_upload_rate_limit.
    fail_open=False,
)


def _derive_folder_dataset_name(filenames: list[str]) -> str:
    """Nom du dataset pour un import de dossier (Lot 6A) — le dossier
    commun porté par `webkitRelativePath` sur chaque fichier (ex.
    "MonDataset/chat/1.png" → "MonDataset"), jamais un nom de fichier
    individuel qui n'aurait aucun sens comme nom de dataset."""
    tops = {Path(f).parts[0] for f in filenames if Path(f).parts}
    if len(tops) == 1:
        return next(iter(tops))
    return "dataset (dossier)"


@router.post(
    "", response_model=VisionDatasetDetail, status_code=status.HTTP_201_CREATED, dependencies=[Depends(_upload_rate_limit)]
)
async def upload_vision_dataset(
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Upload d'un dataset d'images — dossiers de classes (classification)
    ou structure MVTec AD (train/good + test/good + test/<defaut>),
    détectée automatiquement et validée strictement (jamais devinée en
    silence — correctif du bug #1,
    docs/legacy/ANALYSE_COMPLETE_COMPUTER_VISION.md).

    Deux formes (Lot 6A) : un SEUL fichier archive (`.zip`/`.tar`/
    `.tar.gz`/`.tgz`, sniffée par contenu réel, jamais par extension
    déclarée — voir `services/vision_datasets.py::_extract_archive_members`)
    ou PLUSIEURS fichiers représentant un dossier déjà déplié (chaque
    `UploadFile.filename` porte son chemin relatif complet, envoyé par le
    frontend via `webkitRelativePath` — voir `VisionDatasets.tsx`)."""
    is_folder_upload = len(files) > 1
    if not is_folder_upload:
        try:
            validate_archive_extension(files[0].filename or "")
        except UnsupportedFileType as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "VISION_DATASET_FORMAT_NON_SUPPORTE", "message": str(exc)},
            )

    read_files: list[tuple[str, bytes]] = [(f.filename or "", await f.read()) for f in files]
    total_size = sum(len(content) for _, content in read_files)
    if total_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "VISION_DATASET_FICHIER_VIDE", "message": "L'archive est vide"},
        )
    if total_size > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "code": "VISION_DATASET_TROP_VOLUMINEUX",
                "message": f"Archive trop volumineuse (max {_settings.max_vision_upload_size_mb} Mo)",
            },
        )

    dataset_name = _derive_folder_dataset_name([f for f, _ in read_files]) if is_folder_upload else (
        read_files[0][0] or "dataset"
    )
    dataset = VisionDataset(
        organization_id=current_user.organization_id,
        uploaded_by_id=current_user.id,
        name=dataset_name,
        structure_type="",
        storage_dir="",
        status="processing",
    )
    db.add(dataset)
    db.flush()

    target_dir = vision_dataset_dir(current_user.organization_id, dataset.id)
    dataset.storage_dir = str(target_dir)

    try:
        if is_folder_upload:
            report = analyze_and_extract_vision_folder(
                read_files,
                target_dir,
                max_images=_settings.max_vision_dataset_images,
                max_uncompressed_bytes=_MAX_UPLOAD_BYTES * 4,
            )
        else:
            report = analyze_and_extract_vision_archive(
                read_files[0][1],
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
            # Lot 0.1 (correctif C1) — renommés depuis n_duplicates/duplicate_groups :
            # les champs ci-dessous décrivent désormais des exclusions réelles, plus
            # des doublons "conservés". Les datasets uploadés avant ce correctif
            # gardent l'ancienne forme dans leur validation_report_json existant
            # (rétrocompatibilité par absence — le frontend traite ces champs comme
            # optionnels).
            "n_duplicates_removed": report.n_duplicates_removed,
            "duplicate_removed_files": report.duplicate_removed_files,
            "label_conflicts": report.label_conflicts,
            "duplicate_detection_note": report.duplicate_detection_note,
            "n_undersized": report.n_undersized,
            "undersized_files": report.undersized_files,
            "warnings": report.warnings,
            # Lot 6A (§G.3/§G.4/§G.5) — distribution résolutions/formats/modes
            # colorimétriques (voir _compute_image_eda) ; absent (clé
            # manquante, jamais un dict vide trompeur) sur les datasets
            # uploadés avant ce correctif — le frontend traite déjà
            # `validation_report` comme un objet à champs optionnels.
            "image_eda": report.image_eda,
        })
        dataset.status = "ready"
    except VisionDatasetError as exc:
        dataset.status = "error"
        dataset.error_message = str(exc)
        logger.warning("[VisionDatasets] Échec de validation pour %s : %s", dataset_name, exc)

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


@router.get("/{dataset_id}/images", response_model=VisionDatasetImageList)
def list_vision_dataset_images(
    dataset_id: int,
    class_name: str = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Liste les images d'UNE classe (ou d'un bucket MVTec AD, ex.
    "train/good") pour la galerie d'exploration (Lot 16C) — parcourt le
    disque à la demande, aucune nouvelle table : les fichiers sont déjà
    extraits sur `storage_dir` au moment de l'upload
    (`services/vision_datasets.py::analyze_and_extract_vision_zip`). Même
    protection contre la traversée de répertoire que `GET .../image`."""
    dataset = _get_org_dataset(dataset_id, current_user, db)
    base_dir = Path(dataset.storage_dir).resolve()
    bucket_dir = (base_dir / class_name).resolve()
    if base_dir not in bucket_dir.parents or not bucket_dir.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "CLASSE_INTROUVABLE", "message": f"Classe introuvable dans ce dataset : {class_name}"},
        )
    files = sorted(p for p in bucket_dir.iterdir() if p.is_file())
    paths = [p.relative_to(base_dir).as_posix() for p in files[:MAX_GALLERY_IMAGES_PER_CLASS]]
    return VisionDatasetImageList(class_name=class_name, total=len(files), paths=paths)


_PREVIEW_SAMPLE_COUNT = 3


@router.get("/{dataset_id}/augmentation-preview", response_model=AugmentationPreviewOut)
def get_augmentation_preview(
    dataset_id: int,
    preset: str = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Aperçu réel avant/après (Lot 6A) — applique le MÊME
    `augmentation_transforms(preset)` que l'entraînement (jamais une
    approximation côté frontend) à quelques images du dataset, pour que
    l'utilisateur voie concrètement l'effet d'un preset avant de lancer un
    entraînement. Utilisé par les deux wizards Vision (classification ET
    anomalies, même endpoint — le dossier échantillonné dépend uniquement
    de `structure_type`)."""
    if preset not in AUGMENTATION_PRESET_IDS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "AUGMENTATION_PRESET_INCONNU", "message": f"Preset d'augmentation inconnu : {preset!r}"},
        )
    dataset = _get_org_dataset(dataset_id, current_user, db)
    if dataset.status != "ready":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "VISION_DATASET_NON_PRET", "message": "Ce dataset n'a pas pu être validé"},
        )

    base_dir = Path(dataset.storage_dir).resolve()
    if dataset.structure_type == "mvtec_ad":
        # Seul train/good/ est augmenté à l'entraînement (voir
        # vision_anomaly_training.py) — l'aperçu doit montrer EXACTEMENT
        # les images qui seront transformées, jamais test/.
        sample_dir = base_dir / "train" / "good"
        image_size = ANOMALY_IMAGE_SIZE
    else:
        # Classification — première classe trouvée (ordre alphabétique,
        # déterministe) : suffisant pour juger l'effet visuel d'un preset,
        # pas besoin de couvrir toutes les classes.
        first_class_dir = next((p for p in sorted(base_dir.iterdir()) if p.is_dir()), None)
        sample_dir = first_class_dir if first_class_dir else base_dir
        image_size = CLASSIFICATION_IMAGE_SIZE

    sample_paths = sorted(p for p in sample_dir.iterdir() if p.is_file())[:_PREVIEW_SAMPLE_COUNT] if sample_dir.is_dir() else []
    if not sample_paths:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "AUCUNE_IMAGE_POUR_APERCU", "message": "Aucune image disponible pour générer un aperçu"},
        )

    resize = Image.Resampling.BILINEAR
    transform_pipeline = augmentation_transforms(preset)
    pairs: list[AugmentationPreviewPair] = []
    for path in sample_paths:
        with Image.open(path) as raw:
            original = raw.convert("RGB").resize((image_size, image_size), resample=resize)
        augmented = original
        for t in transform_pipeline:
            augmented = t(augmented)
        pairs.append(
            AugmentationPreviewPair(
                original_png=encode_image_png(original),
                augmented_png=encode_image_png(augmented),
            )
        )
    return AugmentationPreviewOut(preset=preset, pairs=pairs)


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
