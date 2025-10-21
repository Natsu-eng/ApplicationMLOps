"""
Chargement d'images flexible pour datasets variés.
Version unifiée avec toutes les fonctions.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from typing import Tuple, List, Dict, Optional
from src.shared.logging import get_logger

logger = get_logger(__name__)

# === FONCTIONS DE DÉTECTION DE STRUCTURE ===

def detect_dataset_structure(data_dir: str) -> Dict:
    """
    Détecte automatiquement la structure du dataset d'images.
    Version corrigée avec nomenclature cohérente.
    """
    if not os.path.exists(data_dir):
        return {"type": "invalid", "error": "Dossier introuvable"}
    
    items = os.listdir(data_dir)
    
    # Structure MVTec AD standard - CORRIGÉ
    if "train" in items and "test" in items:
        train_path = os.path.join(data_dir, "train")
        if os.path.exists(train_path):
            train_items = os.listdir(train_path)
            if "good" in train_items:
                return {"type": "mvtec_ad", "categories": train_items}  # ← minuscules
    
    # Structure avec sous-dossiers = catégories
    subdirs = [d for d in items if os.path.isdir(os.path.join(data_dir, d))]
    image_files = [f for f in items if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if subdirs and not image_files:
        return {"type": "categorical_folders", "categories": subdirs}
    elif image_files and not subdirs:
        return {"type": "flat_directory", "image_count": len(image_files)}
    elif subdirs and image_files:
        return {"type": "mixed", "categories": subdirs, "root_images": len(image_files)}
    else:
        return {"type": "unknown", "items": items}

def _get_image_files(folder_path: str) -> List[str]:
    """Liste les fichiers images valides."""
    if not os.path.exists(folder_path):
        return []
    
    return [f for f in os.listdir(folder_path) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

def get_dataset_stats(data_dir: str) -> pd.DataFrame:
    """
    Calcule les statistiques du dataset (compatibilité ancienne version).
    """
    try:
        stats = []
        structure = detect_dataset_structure(data_dir)
        
        if structure["type"] == "mvtec_ad":
            # Structure MVTec AD
            categories = ['train/good', 'test/good']
            test_path = os.path.join(data_dir, 'test')
            if os.path.exists(test_path):
                categories.extend([f'test/{d}' for d in os.listdir(test_path) if d != 'good'])
            
            for category in categories:
                folder_path = os.path.join(data_dir, *category.split('/'))
                if os.path.exists(folder_path):
                    num_images = len(_get_image_files(folder_path))
                    stats.append({"Catégorie": category, "Nombre d'images": num_images})
        
        elif structure["type"] == "categorical_folders":
            # Dossiers = catégories
            for category in structure["categories"]:
                folder_path = os.path.join(data_dir, category)
                num_images = len(_get_image_files(folder_path))
                stats.append({"Catégorie": category, "Nombre d'images": num_images})
        
        else:
            # Structure plate
            num_images = len(_get_image_files(data_dir))
            stats.append({"Catégorie": "racine", "Nombre d'images": num_images})
        
        return pd.DataFrame(stats)
        
    except Exception as e:
        logger.error(f"Erreur calcul stats: {e}")
        return pd.DataFrame()

def get_dataset_info(data_dir: str) -> Dict:
    """
    Retourne des infos simples sur le dataset.
    """
    structure = detect_dataset_structure(data_dir)
    info = {"structure": structure["type"]}
    
    if structure["type"] == "mvtec_ad":
        normal = len(_get_image_files(os.path.join(data_dir, "train", "good"))) + \
                len(_get_image_files(os.path.join(data_dir, "test", "good")))
        anomaly = 0
        test_path = os.path.join(data_dir, "test")
        if os.path.exists(test_path):
            for cat in os.listdir(test_path):
                if cat != "good":
                    anomaly += len(_get_image_files(os.path.join(test_path, cat)))
        info.update({"normal": normal, "anomaly": anomaly, "total": normal + anomaly})
    
    elif structure["type"] == "categorical_folders":
        categories = {}
        for cat in structure["categories"]:
            cat_path = os.path.join(data_dir, cat)
            if os.path.isdir(cat_path):
                categories[cat] = len(_get_image_files(cat_path))
        info["categories"] = categories
        info["total"] = sum(categories.values())
    
    elif structure["type"] == "flat_directory":
        info["total"] = len(_get_image_files(data_dir))
    
    return info

# === FONCTIONS DE CHARGEMENT ===

def load_images_flexible(data_dir: str, target_size: Tuple[int, int] = (256, 256)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Charge les images depuis n'importe quelle structure de dossier.
    """
    structure = detect_dataset_structure(data_dir)
    logger.info(f"Structure détectée: {structure['type']}")
    
    if structure["type"] == "mvtec_ad":
        return _load_mvtec_structure(data_dir, target_size)
    elif structure["type"] == "categorical_folders":
        return _load_categorical_folders(data_dir, target_size)
    elif structure["type"] == "flat_directory":
        return _load_flat_directory(data_dir, target_size)
    else:
        raise ValueError(f"Structure non supportée: {structure}")

def load_images_from_folder(data_dir: str, target_size: Tuple[int, int] = (128, 128), normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Version compatible ancien code - utilise la nouvelle fonction flexible.
    """
    X, y = load_images_flexible(data_dir, target_size)
    
    # Appliquer la normalisation si demandée
    if normalize and len(X) > 0:
        X = X / 255.0
    
    return X, y

def _load_mvtec_structure(data_dir: str, target_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Charge la structure MVTec AD standard."""
    images, labels = [], []
    
    # Images normales (label 0)
    normal_paths = [
        os.path.join(data_dir, "train", "good"),
        os.path.join(data_dir, "test", "good")
    ]
    
    for path in normal_paths:
        if os.path.exists(path):
            for img_file in _get_image_files(path):
                img = _load_single_image(os.path.join(path, img_file), target_size)
                if img is not None:
                    images.append(img)
                    labels.append(0)  # Normal
    
    # Images anormales (label 1)
    test_path = os.path.join(data_dir, "test")
    if os.path.exists(test_path):
        for category in os.listdir(test_path):
            if category != "good":
                anomaly_path = os.path.join(test_path, category)
                if os.path.isdir(anomaly_path):
                    for img_file in _get_image_files(anomaly_path):
                        img = _load_single_image(os.path.join(anomaly_path, img_file), target_size)
                        if img is not None:
                            images.append(img)
                            labels.append(1)  # Anomalie
    
    return np.array(images), np.array(labels)

def _load_categorical_folders(data_dir: str, target_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Charge depuis des dossiers = catégories."""
    images, labels = [], []
    
    for label, category in enumerate(os.listdir(data_dir)):
        category_path = os.path.join(data_dir, category)
        if os.path.isdir(category_path):
            for img_file in _get_image_files(category_path):
                img = _load_single_image(os.path.join(category_path, img_file), target_size)
                if img is not None:
                    images.append(img)
                    labels.append(label)
    
    return np.array(images), np.array(labels)

def _load_flat_directory(data_dir: str, target_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Charge depuis un dossier plat (toutes images mélangées)."""
    images = []
    
    for img_file in _get_image_files(data_dir):
        img = _load_single_image(os.path.join(data_dir, img_file), target_size)
        if img is not None:
            images.append(img)
    
    # Toutes étiquetées comme normales (0) par défaut
    labels = np.zeros(len(images), dtype=int)
    
    return np.array(images), labels

def _load_single_image(image_path: str, target_size: Tuple[int, int]) -> Optional[np.ndarray]:
    """Charge et prétraite une seule image."""
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            img = img.resize(target_size)
            return np.array(img)
    except Exception as e:
        logger.warning(f"Erreur chargement {image_path}: {e}")
        return None
    
