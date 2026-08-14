"""Stockage des fichiers uploadés — disque local pour l'instant.

Voir le diagnostic de migration (section D2) : migration vers un stockage
objet compatible S3 prévue quand le volume de clients l'impose. En
attendant, organisation du disque : storage/datasets/{organization_id}/{id}{ext}
— l'organization_id dans le chemin est une isolation supplémentaire
("défense en profondeur"), en plus du filtrage systématique par
organization_id en base (voir api/routers/datasets.py).
"""
from __future__ import annotations

from pathlib import Path

STORAGE_ROOT = Path(__file__).resolve().parent.parent.parent / "storage"
DATASETS_DIR = STORAGE_ROOT / "datasets"
MODELS_DIR = STORAGE_ROOT / "models"


def dataset_file_path(organization_id: int, dataset_id: int, extension: str) -> Path:
    org_dir = DATASETS_DIR / str(organization_id)
    org_dir.mkdir(parents=True, exist_ok=True)
    return org_dir / f"{dataset_id}{extension}"


def delete_dataset_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def model_file_path(organization_id: int, training_job_id: int) -> Path:
    org_dir = MODELS_DIR / str(organization_id)
    org_dir.mkdir(parents=True, exist_ok=True)
    return org_dir / f"{training_job_id}.joblib"


def cluster_model_file_path(organization_id: int, clustering_job_id: int) -> Path:
    """Même isolation que `model_file_path`, sous-dossier dédié — Lot 11+
    (ML non supervisé)."""
    org_dir = MODELS_DIR / str(organization_id) / "clustering"
    org_dir.mkdir(parents=True, exist_ok=True)
    return org_dir / f"{clustering_job_id}.joblib"
