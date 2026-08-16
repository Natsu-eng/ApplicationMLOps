"""Stockage des fichiers uploadés — disque local pour l'instant.

Voir le diagnostic de migration (section D2) : migration vers un stockage
objet compatible S3 prévue quand le volume de clients l'impose. En
attendant, organisation du disque : storage/datasets/{organization_id}/{id}{ext}
— l'organization_id dans le chemin est une isolation supplémentaire
("défense en profondeur"), en plus du filtrage systématique par
organization_id en base (voir api/routers/datasets.py).
"""
from __future__ import annotations

import shutil
from pathlib import Path

STORAGE_ROOT = Path(__file__).resolve().parent.parent.parent / "storage"
DATASETS_DIR = STORAGE_ROOT / "datasets"
MODELS_DIR = STORAGE_ROOT / "models"
# Sous-dossier dédié : un dataset vision est un dossier de nombreuses petites
# images extraites du ZIP uploadé, pas un fichier unique comme DATASETS_DIR
# — pilier Vision, Lot 15 sous-lot A.
VISION_DATASETS_DIR = STORAGE_ROOT / "datasets_vision"


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


def dimensionality_model_file_path(organization_id: int, dimensionality_job_id: int) -> Path:
    """Même isolation que `cluster_model_file_path` — Lot 13 (réduction de
    dimension)."""
    org_dir = MODELS_DIR / str(organization_id) / "dimensionality"
    org_dir.mkdir(parents=True, exist_ok=True)
    return org_dir / f"{dimensionality_job_id}.joblib"


def anomaly_model_file_path(organization_id: int, anomaly_job_id: int) -> Path:
    """Même isolation que `cluster_model_file_path` — Lot 14 (détection
    d'anomalies)."""
    org_dir = MODELS_DIR / str(organization_id) / "anomalies"
    org_dir.mkdir(parents=True, exist_ok=True)
    return org_dir / f"{anomaly_job_id}.joblib"


def vision_dataset_dir(organization_id: int, dataset_id: int) -> Path:
    """Dossier racine — TOUJOURS VIDE au retour — des images extraites d'un
    `VisionDataset`. Même isolation disque que `dataset_file_path`
    (organization_id dans le chemin, en plus du filtre systématique par
    organization_id en base). Pilier Vision, Lot 15 sous-lot A.

    Nettoie tout contenu préexistant avant de le recréer : un id de dataset
    ne devrait jamais être réutilisé en production (séquence Postgres
    strictement croissante), mais un bug réel a été trouvé en test (SQLite,
    séquence qui redémarre à 1 après `drop_all`/`create_all` entre deux
    tests) où un id réutilisé héritait silencieusement des fichiers d'un
    dataset précédent (classes MVTec AD mélangées à une classification)."""
    target_dir = VISION_DATASETS_DIR / str(organization_id) / str(dataset_id)
    shutil.rmtree(target_dir, ignore_errors=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def delete_vision_dataset_dir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def vision_classification_model_file_path(organization_id: int, job_id: int) -> Path:
    """Même isolation que `cluster_model_file_path` — Lot 15 sous-lot B
    (classification d'images). Extension `.pt` (torch.save), pas `.joblib` :
    l'artefact contient un state_dict PyTorch, pas un pipeline scikit-learn."""
    org_dir = MODELS_DIR / str(organization_id) / "vision_classification"
    org_dir.mkdir(parents=True, exist_ok=True)
    return org_dir / f"{job_id}.pt"


def vision_anomaly_model_file_path(organization_id: int, job_id: int) -> Path:
    """Même isolation que `vision_classification_model_file_path` — Lot 15
    sous-lot C (détection d'anomalies visuelles MVTec AD)."""
    org_dir = MODELS_DIR / str(organization_id) / "vision_anomaly"
    org_dir.mkdir(parents=True, exist_ok=True)
    return org_dir / f"{job_id}.pt"
