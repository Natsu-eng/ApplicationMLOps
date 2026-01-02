"""
🖼️ Module de traitement d'images pour DataLab Pro
Version:1 - PRODUCTION READY avec gestion labels cohérente
Ordre des classes et détection de tâches

"""

import numpy as np
import pandas as pd
from PIL import Image
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from src.shared.logging import get_logger

logger = get_logger(__name__)

# ===================================
# CONSTANTES
# ===================================

class DatasetType(Enum):
    MVTEC_AD = "mvtec_ad"
    CATEGORICAL = "categorical_folders"
    FLAT = "flat_directory"
    MIXED = "mixed"
    INVALID = "invalid"
    UNKNOWN = "unknown"

@dataclass
class ImageConfig:
    target_size: Tuple[int, int] = (256, 256)
    normalize: bool = False
    color_mode: str = 'RGB'
    max_images: Optional[int] = None
    
@dataclass
class QualityThresholds:
    dark_threshold: float = 50.0
    bright_threshold: float = 200.0
    low_contrast_threshold: float = 20.0
    min_sharpness: float = 0.01

SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')

# ===================================
# DÉTECTION DE STRUCTURE
# ===================================

def detect_dataset_structure(data_dir: str) -> Dict[str, Any]:
    """
    Détection améliorée avec prise en charge des structures :
    - MVTec AD classique
    - Catégorielle plate (classes à la racine)
    - Split + catégorielle (train/val/test contenant des classes) ← NOUVEAU
    """
    try:
        if not data_dir or not isinstance(data_dir, str):
            return {"type": DatasetType.INVALID.value, "error": "Chemin invalide"}
        
        data_path = Path(data_dir)
        
        if not data_path.exists():
            return {"type": DatasetType.INVALID.value, "error": f"Dossier introuvable: {data_dir}"}
        
        if not data_path.is_dir():
            return {"type": DatasetType.INVALID.value, "error": f"Pas un dossier: {data_dir}"}
        
        items = [item.name for item in data_path.iterdir()]
        
        if not items:
            return {"type": DatasetType.INVALID.value, "error": "Dossier vide"}
        
        subdirs = []
        image_files = []
        
        for item in data_path.iterdir():
            try:
                if item.is_dir():
                    subdirs.append(item.name)
                elif item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
                    image_files.append(item.name)
            except:
                continue
        
        subdirs_lower = [s.lower() for s in subdirs]

        # ✅ 1. Détection MVTec AD stricte
        if "train" in items and "test" in items:
            train_path = data_path / "train"
            if train_path.exists() and train_path.is_dir():
                train_items = [item.name for item in train_path.iterdir() if item.is_dir()]
                if "good" in train_items:
                    logger.info(f"✅ Structure MVTec AD détectée: {data_dir}")
                    return {
                        "type": DatasetType.MVTEC_AD.value,
                        "categories": train_items,
                        "description": "MVTec AD - Unsupervised Anomaly Detection",
                        "is_mvtec": True
                    }

        # ✅ 2. NOUVEAU : Détection structure split + catégorielle (train/val/test avec classes dedans)
        splits_present = {"train", "val", "test"}
        if len(set(subdirs_lower) & splits_present) >= 2:  # Au moins 2 splits parmi train/val/test
            train_path = data_path / "train"
            if train_path.exists() and train_path.is_dir():
                potential_classes = [p.name for p in train_path.iterdir() if p.is_dir()]
                if potential_classes:
                    # Vérifie qu'il y a des images dans les sous-dossiers
                    has_images = any(len(_get_image_files(train_path / cls)) > 0 for cls in potential_classes)
                    if has_images:
                        sorted_classes = sorted(potential_classes)
                        logger.info(f"✅ Structure split + catégorielle détectée: {len(sorted_classes)} classes ({', '.join(sorted_classes)})")
                        return {
                            "type": DatasetType.CATEGORICAL.value,
                            "categories": sorted_classes,
                            "n_categories": len(sorted_classes),
                            "description": f"Classification multiclasse avec splits ({len(sorted_classes)} classes)",
                            "is_anomaly_supervised": False,
                            "class_to_idx": {cls: idx for idx, cls in enumerate(sorted_classes)},
                            "has_splits": True  # Flag important pour le chargement
                        }

        # ✅ 3. Cas anomaly supervisée binaire (normal/defect, good/bad, etc.)
        if subdirs and not image_files and len(subdirs) == 2:
            is_anomaly_supervised = (
                ("normal" in subdirs_lower and "defect" in subdirs_lower) or
                ("normal" in subdirs_lower and "anomaly" in subdirs_lower) or
                ("good" in subdirs_lower and "bad" in subdirs_lower)
            )
            if is_anomaly_supervised:
                ordered_subdirs = sorted(subdirs, key=lambda x: 0 if 'normal' in x.lower() or 'good' in x.lower() else 1)
                logger.info(f"✅ Anomaly Detection Supervisée détectée: {subdirs}")
                return {
                    "type": DatasetType.CATEGORICAL.value,
                    "categories": ordered_subdirs,
                    "n_categories": 2,
                    "description": "Anomaly Detection - Supervised",
                    "is_anomaly_supervised": True,
                    "class_to_idx": {ordered_subdirs[0]: 0, ordered_subdirs[1]: 1}
                }

        # ✅ 4. Classification classique multiclass (classes directement à la racine)
        if subdirs and not image_files:
            sorted_subdirs = sorted(subdirs)
            logger.info(f"✅ Structure catégorielle classique: {len(sorted_subdirs)} classes")
            return {
                "type": DatasetType.CATEGORICAL.value,
                "categories": sorted_subdirs,
                "n_categories": len(sorted_subdirs),
                "description": f"Classification ({len(sorted_subdirs)} classes)",
                "is_anomaly_supervised": False,
                "class_to_idx": {cls: idx for idx, cls in enumerate(sorted_subdirs)}
            }

        # Structure plate
        elif image_files and not subdirs:
            logger.info(f"✅ Structure plate: {len(image_files)} images")
            return {
                "type": DatasetType.FLAT.value,
                "image_count": len(image_files),
                "description": f"Dataset plat ({len(image_files)} images)"
            }

        # Mixte
        elif subdirs and image_files:
            logger.warning(f"⚠️ Structure mixte: {len(subdirs)} dossiers + {len(image_files)} images")
            return {
                "type": DatasetType.MIXED.value,
                "categories": subdirs,
                "root_images": len(image_files),
                "description": "Structure mixte"
            }

        return {"type": DatasetType.UNKNOWN.value, "items": items[:10]}
    
    except Exception as e:
        logger.error(f"❌ Erreur detect_dataset_structure: {e}", exc_info=True)
        return {"type": DatasetType.INVALID.value, "error": str(e)}

def get_dataset_info(data_dir: str) -> Dict[str, Any]:
    """
    Informations enrichies avec métadonnées de classes
    """
    try:
        structure = detect_dataset_structure(data_dir)
        
        info = {
            "structure_type": structure.get("type"),
            "data_dir": data_dir,
            "is_valid": structure.get("type") not in [DatasetType.INVALID.value, DatasetType.UNKNOWN.value]
        }
        
        if not info["is_valid"]:
            info["error"] = structure.get("error", "Structure invalide")
            return info
        
        data_path = Path(data_dir)
        
        # MVTec AD
        if structure["type"] == DatasetType.MVTEC_AD.value:
            normal_count = 0
            anomaly_count = 0
            
            for folder in ["train/good", "test/good"]:
                folder_path = data_path / folder
                if folder_path.exists():
                    normal_count += len(_get_image_files(folder_path))
            
            test_path = data_path / "test"
            if test_path.exists():
                for category in test_path.iterdir():
                    if category.is_dir() and category.name != "good":
                        anomaly_count += len(_get_image_files(category))
            
            info.update({
                "normal": normal_count,
                "anomaly": anomaly_count,
                "total": normal_count + anomaly_count,
                "balance_ratio": anomaly_count / normal_count if normal_count > 0 else 0,
                "task_type": "anomaly_detection_unsupervised",
                "is_mvtec": True,
                "class_names": ["Normal", "Anomaly"]  # ✅ Noms explicites
            })
            
            logger.info(f"MVTec AD: {normal_count} normal, {anomaly_count} anomalies")
        
        # Catégoriel (Supervised ou Classification)
        elif structure["type"] == DatasetType.CATEGORICAL.value:
            categories = {}
            class_names = []
            
            # ✅ UTILISER L'ORDRE DE class_to_idx
            if 'class_to_idx' in structure:
                sorted_categories = sorted(structure['class_to_idx'].items(), key=lambda x: x[1])
                for category, idx in sorted_categories:
                    cat_path = data_path / category
                    if cat_path.is_dir():
                        count = len(_get_image_files(cat_path))
                        categories[category] = count
                        class_names.append(category)
            else:
                # Fallback tri alphabétique
                for category in sorted(structure.get("categories", [])):
                    cat_path = data_path / category
                    if cat_path.is_dir():
                        count = len(_get_image_files(cat_path))
                        categories[category] = count
                        class_names.append(category)
            
            total = sum(categories.values())
            
            # ✅ Détection anomaly supervised
            is_anomaly = structure.get("is_anomaly_supervised", False)
            
            info.update({
                "categories": categories,
                "n_categories": len(categories),
                "total": total,
                "task_type": "anomaly_detection" if is_anomaly else "classification",
                "is_anomaly_supervised": is_anomaly,
                "class_names": class_names,  # ✅ Ordre garanti
                "balance_info": _compute_balance_stats(categories)
            })
            
            logger.info(f"Catégoriel: {len(categories)} classes, {total} images")
        
        # Plat
        elif structure["type"] == DatasetType.FLAT.value:
            total = len(_get_image_files(data_path))
            info.update({
                "total": total,
                "task_type": "unsupervised"
            })
        
        return info
    
    except Exception as e:
        logger.error(f"Erreur get_dataset_info: {e}", exc_info=True)
        return {"structure_type": DatasetType.INVALID.value, "error": str(e), "is_valid": False}

def get_dataset_stats(data_dir: str) -> pd.DataFrame:
    """
    Stats avec noms de classes cohérents
    """
    try:
        stats = []
        structure = detect_dataset_structure(data_dir)
        data_path = Path(data_dir)
        
        if structure["type"] == DatasetType.MVTEC_AD.value:
            categories = [
                ('train/good', 'Normal (Train)'),
                ('test/good', 'Normal (Test)')
            ]
            
            test_path = data_path / 'test'
            if test_path.exists():
                for item in test_path.iterdir():
                    if item.is_dir() and item.name != 'good':
                        categories.append((f'test/{item.name}', f'Anomalie ({item.name})'))
            
            for category_path, display_name in categories:
                folder_path = data_path / category_path
                if folder_path.exists():
                    image_files = _get_image_files(folder_path)
                    if image_files:
                        stats.append({
                            "Catégorie": display_name,
                            "Chemin": str(category_path),
                            "Nombre d'images": len(image_files),
                            "Type": "Normal" if "good" in category_path else "Anomalie"
                        })
        
        elif structure["type"] == DatasetType.CATEGORICAL.value:
            # ✅ Utiliser class_to_idx pour ordre garanti
            if 'class_to_idx' in structure:
                sorted_categories = sorted(structure['class_to_idx'].items(), key=lambda x: x[1])
                for category, idx in sorted_categories:
                    folder_path = data_path / category
                    if folder_path.exists():
                        image_files = _get_image_files(folder_path)
                        stats.append({
                            "Catégorie": category,
                            "Label": idx,
                            "Chemin": category,
                            "Nombre d'images": len(image_files),
                            "Type": "Classe"
                        })
            else:
                for category in structure.get("categories", []):
                    folder_path = data_path / category
                    if folder_path.exists():
                        image_files = _get_image_files(folder_path)
                        stats.append({
                            "Catégorie": category,
                            "Chemin": category,
                            "Nombre d'images": len(image_files),
                            "Type": "Classe"
                        })
        
        else:
            image_files = _get_image_files(data_path)
            stats.append({
                "Catégorie": "Toutes images",
                "Chemin": "racine",
                "Nombre d'images": len(image_files),
                "Type": "Non catégorisé"
            })
        
        df_stats = pd.DataFrame(stats)
        
        if not df_stats.empty and "Nombre d'images" in df_stats.columns:
            df_stats['Pourcentage'] = (df_stats['Nombre d\'images'] / df_stats['Nombre d\'images'].sum() * 100).round(2)
        
        return df_stats
        
    except Exception as e:
        logger.error(f"Erreur get_dataset_stats: {e}", exc_info=True)
        return pd.DataFrame()

# ===================================
# CHARGEMENT D'IMAGES
# ===================================

def _load_mvtec_train_labels(data_dir: str) -> np.ndarray:
    """
    Charge UNIQUEMENT les labels train/good (toujours 0)
    """
    try:
        train_good_path = Path(data_dir) / "train" / "good"
        if not train_good_path.exists():
            return np.array([], dtype=int)
        
        image_files = _get_image_files(train_good_path)
        logger.debug(f"{len(image_files)} images train/good → y_train = [0]")
        return np.zeros(len(image_files), dtype=int)
    
    except Exception as e:
        logger.error(f"Erreur _load_mvtec_train_labels: {e}")
        return np.array([], dtype=int)

def load_images_flexible(
    data_dir: str,
    target_size: Tuple[int, int] = (256, 256),
    config: Optional[ImageConfig] = None
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Chargement avec labels cohérents et y_train pour unsupervised
    Retourne: X, X_norm, y (complet ou None), y_train (train only pour MVTec ou None)
    
    ✅ Support complet non supervisé: y_full et y_train peuvent être None
    """
    try:
        if config is None:
            config = ImageConfig(target_size=target_size)
        
        structure = detect_dataset_structure(data_dir)
        structure_type = structure.get("type")
        
        logger.info(f"🔍 Structure détectée: {structure_type}")
        
        if structure_type == DatasetType.INVALID.value:
            raise ValueError(f"Structure invalide: {structure.get('error')}")

        y_train = None
        
        # ✅ CHARGEMENT SELON TYPE
        if structure_type == DatasetType.MVTEC_AD.value:
            X, y_full = _load_mvtec_structure(data_dir, config)
            y_train = _load_mvtec_train_labels(data_dir)
            logger.info(f"MVTec AD: {len(X)} images | y_train: {len(y_train)} normales")
        
        elif structure_type == DatasetType.CATEGORICAL.value:
            X, y_full = _load_categorical_folders(data_dir, config, structure)
            logger.info(f"Categorical: {len(X)} images | {len(np.unique(y_full))} classes")
        
        elif structure_type == DatasetType.FLAT.value:
            X, y_full = _load_flat_directory(data_dir, config)
            # ✅ y_full peut être None pour non supervisé
            if y_full is None:
                logger.info(f"Flat (non supervisé): {len(X)} images | y=None")
            else:
                logger.info(f"Flat: {len(X)} images")
        
        else:
            raise ValueError(f"Structure non supportée: {structure_type}")

        if len(X) == 0:
            raise RuntimeError("Aucune image valide")

        # ✅ VALIDATION y_train pour MVTec AD (seulement si présent)
        if y_train is not None and len(y_train) > 0:
            unique_train = np.unique(y_train)
            if len(unique_train) > 1 or (len(unique_train) == 1 and unique_train[0] != 0):
                raise ValueError(
                    f"❌ y_train invalide pour unsupervised: {unique_train}. "
                    f"Attendu uniquement [0]"
                )
            logger.info(f"✅ y_train validé: {len(y_train)} images normales (unsupervised)")

        # Normalisation
        X_norm = X.astype(np.float32) / 255.0

        logger.info(
            f"✅ Chargement terminé: {len(X)} images | "
            f"y_full={'présent' if y_full is not None else 'None'} | "
            f"y_train={'présent' if y_train is not None else 'None'}"
        )
        return X.astype(np.uint8), X_norm, y_full, y_train

    except Exception as e:
        logger.error(f"❌ Erreur load_images_flexible: {e}", exc_info=True)
        raise

def load_images_from_folder(
    data_dir: str,
    target_size: Tuple[int, int] = (128, 128),
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Ancienne API pour compatibilité"""
    X, X_norm, _, _ = load_images_flexible(
        data_dir, 
        target_size=target_size, 
        config=ImageConfig(normalize=normalize)
    )
    return X_norm if normalize else X, None

# ===================================
# CHARGEMENT PRIVÉ
# ===================================

def _load_mvtec_structure(data_dir: str, config: ImageConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    MVTec AD: normal=0, anomaly=1 (garanti)
    """
    images, labels = [], []
    data_path = Path(data_dir)
    
    # Normal (label 0)
    normal_paths = [
        data_path / "train" / "good",
        data_path / "test" / "good"
    ]
    
    loaded_normal = 0
    for path in normal_paths:
        if path.exists():
            for img_file in _get_image_files(path):
                img = _load_single_image(path / img_file, config)
                if img is not None:
                    images.append(img)
                    labels.append(0)  # ✅ Normal = 0
                    loaded_normal += 1
    
    logger.info(f"✅ {loaded_normal} images normales (label 0)")
    
    # Anomaly (label 1)
    test_path = data_path / "test"
    if test_path.exists():
        loaded_anomaly = 0
        for category in test_path.iterdir():
            if category.is_dir() and category.name != "good":
                for img_file in _get_image_files(category):
                    img = _load_single_image(category / img_file, config)
                    if img is not None:
                        images.append(img)
                        labels.append(1)  # ✅ Anomaly = 1
                        loaded_anomaly += 1
        
        logger.info(f"✅ {loaded_anomaly} anomalies (label 1)")
    
    return np.array(images), np.array(labels)

def _load_categorical_folders(
    data_dir: str, 
    config: ImageConfig,
    structure: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chargement des images catégorielles avec support :
    - Classes à la racine
    - Classes dans train/val/test (has_splits=True)
    Ordre des labels garanti via class_to_idx
    """
    images, labels = [], []
    data_path = Path(data_dir)
    
    # Récupération de l'ordre garanti des classes
    if 'class_to_idx' in structure:
        sorted_categories = sorted(structure['class_to_idx'].items(), key=lambda x: x[1])
        logger.info(f"📋 Ordre des classes (garanti): {[cat for cat, idx in sorted_categories]}")
    else:
        # Fallback (ne devrait pas arriver)
        sorted_categories = [(cls, idx) for idx, cls in enumerate(sorted(structure.get("categories", [])))]

    # Déterminer les dossiers à explorer
    if structure.get('has_splits', False):
        split_dirs = ['train', 'val', 'test']
        logger.info("🔄 Chargement multiclass avec splits (train/val/test)")
    else:
        split_dirs = ['']  # Racine uniquement
        logger.info("🔄 Chargement multiclass classique (classes à la racine)")

    for split in split_dirs:
        base_path = data_path if not split else data_path / split
        if not base_path.exists():
            continue

        for category, label in sorted_categories:
            cat_path = base_path / category
            if not cat_path.is_dir():
                continue
            
            image_files = _get_image_files(cat_path)
            if not image_files:
                continue

            logger.debug(f"Chargement {len(image_files)} images depuis {cat_path} → label {label}")

            for img_file in image_files:
                img = _load_single_image(cat_path / img_file, config)
                if img is not None:
                    images.append(img)
                    labels.append(label)

                    # Limite optionnelle
                    if config.max_images and len(images) >= config.max_images:
                        logger.info(f"✅ Limite max_images atteinte ({config.max_images})")
                        return np.array(images), np.array(labels)

    if len(images) == 0:
        logger.warning("Aucune image chargée dans les dossiers catégoriels")
    
    logger.info(f"✅ {len(images)} images chargées avec labels cohérents (multiclass)")
    return np.array(images), np.array(labels)

def _load_flat_directory(data_dir: str, config: ImageConfig) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Dossier plat: images sans labels (non supervisé)
    Retourne y=None pour support non supervisé
    """
    images = []
    data_path = Path(data_dir)
    
    for img_file in _get_image_files(data_path):
        img = _load_single_image(data_path / img_file, config)
        if img is not None:
            images.append(img)
            
            if config.max_images and len(images) >= config.max_images:
                break
    
    # ✅ CORRECTION: Retourner None pour non supervisé
    logger.info(f"{len(images)} images chargées (non supervisé, y=None)")
    return np.array(images), None

def _load_single_image(image_path: Path, config: ImageConfig) -> Optional[np.ndarray]:
    """
    Charge une image avec prétraitement
    """
    try:
        with Image.open(image_path) as img:
            if config.color_mode == 'RGB' and img.mode != 'RGB':
                img = img.convert('RGB')
            elif config.color_mode == 'L' and img.mode != 'L':
                img = img.convert('L')
            
            if img.size != config.target_size:
                img = img.resize(config.target_size, Image.Resampling.LANCZOS)
            
            img_array = np.array(img)
            
            if img_array.size == 0:
                logger.warning(f"Image vide: {image_path}")
                return None
            
            return img_array
            
    except Exception as e:
        logger.warning(f"Erreur {image_path.name}: {e}")
        return None

def _get_image_files(folder_path: Path) -> List[str]:
    """
    Liste fichiers images triés
    """
    try:
        if not folder_path.exists() or not folder_path.is_dir():
            return []
        
        files = [
            item.name for item in folder_path.iterdir()
            if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        
        return sorted(files)
        
    except Exception as e:
        logger.warning(f"Erreur lecture {folder_path}: {e}")
        return []

# ===================================
# ANALYSE QUALITÉ (inchangé)
# ===================================

def analyze_image_quality(
    images: np.ndarray,
    sample_size: int = 200,
    thresholds: Optional[QualityThresholds] = None
) -> Dict[str, Any]:
    """Analyse qualité (code inchangé)"""
    try:
        if thresholds is None:
            thresholds = QualityThresholds()
        
        if images is None or len(images) == 0:
            return {'error': 'Tableau vide'}
        
        try:
            from scipy import ndimage
            scipy_available = True
        except ImportError:
            scipy_available = False
        
        actual_sample_size = min(sample_size, len(images))
        if actual_sample_size < len(images):
            indices = np.random.choice(len(images), actual_sample_size, replace=False)
            sample_images = images[indices]
        else:
            sample_images = images
        
        if sample_images.max() > 1.0:
            sample_normalized = sample_images / 255.0
        else:
            sample_normalized = sample_images.copy()
        
        if len(sample_normalized.shape) == 4:
            brightness = np.mean(sample_normalized, axis=(1, 2, 3)) * 255
        else:
            brightness = np.mean(sample_normalized, axis=(1, 2)) * 255
        
        if len(sample_normalized.shape) == 4:
            contrast = np.std(sample_normalized, axis=(1, 2, 3)) * 255
        else:
            contrast = np.std(sample_normalized, axis=(1, 2)) * 255
        
        dark_images = np.sum(brightness < thresholds.dark_threshold)
        bright_images = np.sum(brightness > thresholds.bright_threshold)
        low_contrast_images = np.sum(contrast < thresholds.low_contrast_threshold)
        
        result = {
            'brightness': {
                'values': brightness.tolist(),
                'mean': float(np.mean(brightness)),
                'std': float(np.std(brightness)),
                'min': float(np.min(brightness)),
                'max': float(np.max(brightness)),
                'problematic': {
                    'dark': int(dark_images),
                    'bright': int(bright_images)
                },
                'thresholds': {
                    'dark': thresholds.dark_threshold,
                    'bright': thresholds.bright_threshold
                }
            },
            'contrast': {
                'values': contrast.tolist(),
                'mean': float(np.mean(contrast)),
                'std': float(np.std(contrast)),
                'min': float(np.min(contrast)),
                'max': float(np.max(contrast)),
                'problematic': {
                    'low_contrast': int(low_contrast_images)
                },
                'thresholds': {
                    'low_contrast': thresholds.low_contrast_threshold
                }
            },
            'sample_size': len(sample_images),
            'total_images': len(images),
            'problematic_summary': {
                'total_dark': int(dark_images),
                'total_bright': int(bright_images),
                'total_low_contrast': int(low_contrast_images),
                'total_problematic': int(dark_images + bright_images + low_contrast_images),
                'percentage_problematic': float((dark_images + bright_images + low_contrast_images) / len(sample_images) * 100)
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Erreur analyze_image_quality: {e}", exc_info=True)
        return {'error': str(e)}

def analyze_image_distribution(
    images: np.ndarray,
    sample_size: int = 100
) -> Dict[str, Any]:
    """
    Analyse des distributions de couleur dans les images.
    Version production optimisée.
    
    Args:
        images: Tableau numpy d'images
        sample_size: Nombre d'images à échantillonner
        
    Returns:
        Dictionnaire avec les statistiques de distribution par canal
    """
    try:
        # Validation
        if images is None or len(images) == 0:
            return {'error': 'Tableau d\'images vide'}
        
        if not isinstance(images, np.ndarray):
            return {'error': 'Format invalide'}
        
        # Échantillonnage pour performance
        actual_sample_size = min(sample_size, len(images))
        if actual_sample_size < len(images):
            indices = np.random.choice(len(images), actual_sample_size, replace=False)
            sample_images = images[indices]
        else:
            sample_images = images
        
        logger.info(f"Analyse distribution sur {len(sample_images)} images")
        
        # Normalisation
        if sample_images.max() > 1.0:
            sample_images = sample_images / 255.0
        
        # Détection du nombre de canaux
        if len(sample_images.shape) == 4:
            n_channels = sample_images.shape[-1]
        elif len(sample_images.shape) == 3:
            n_channels = 1
        else:
            return {'error': f'Shape invalide: {sample_images.shape}'}
        
        channels_data = {}
        
        if n_channels == 3:
            # Images couleur RGB
            channel_names = ['Rouge', 'Vert', 'Bleu']
            
            for i, channel_name in enumerate(channel_names):
                channel_data = sample_images[:, :, :, i].flatten()
                
                channels_data[channel_name] = {
                    'data': channel_data.tolist(),  # Pour JSON serialization
                    'mean': float(np.mean(channel_data)),
                    'std': float(np.std(channel_data)),
                    'min': float(np.min(channel_data)),
                    'max': float(np.max(channel_data)),
                    'median': float(np.median(channel_data)),
                    'q25': float(np.percentile(channel_data, 25)),
                    'q75': float(np.percentile(channel_data, 75))
                }
        else:
            # Images en niveau de gris
            channel_data = sample_images.flatten()
            
            channels_data['Niveau de Gris'] = {
                'data': channel_data.tolist(),
                'mean': float(np.mean(channel_data)),
                'std': float(np.std(channel_data)),
                'min': float(np.min(channel_data)),
                'max': float(np.max(channel_data)),
                'median': float(np.median(channel_data)),
                'q25': float(np.percentile(channel_data, 25)),
                'q75': float(np.percentile(channel_data, 75))
            }
        
        result = {
            'channels': channels_data,
            'sample_size': len(sample_images),
            'total_images': len(images),
            'n_channels': n_channels,
            'color_mode': 'RGB' if n_channels == 3 else 'Grayscale'
        }
        
        logger.info(f"Analyse distribution terminée: {n_channels} canaux")
        
        return result
        
    except Exception as e:
        logger.error(f"Erreur dans analyze_image_distribution: {e}", exc_info=True)
        return {'error': str(e)}

# ===================================
# FONCTIONS UTILITAIRES
# ===================================

def _compute_balance_stats(categories: Dict[str, int]) -> Dict[str, Any]:
    """
    Calcule les statistiques d'équilibre entre catégories.
    
    Args:
        categories: Dictionnaire {nom_catégorie: nombre_images}
        
    Returns:
        Statistiques d'équilibre
    """
    if not categories:
        return {}
    
    counts = list(categories.values())
    total = sum(counts)
    
    if total == 0:
        return {'error': 'Aucune image'}
    
    max_count = max(counts)
    min_count = min(counts)
    
    return {
        'is_balanced': max_count / min_count < 3 if min_count > 0 else False,
        'imbalance_ratio': float(max_count / min_count) if min_count > 0 else float('inf'),
        'majority_class': max(categories, key=categories.get),
        'majority_count': max_count,
        'minority_class': min(categories, key=categories.get),
        'minority_count': min_count,
        'total_images': total
    }

def validate_image_array(images: np.ndarray) -> Tuple[bool, Optional[str]]:
    """
    Valide un tableau d'images.
    
    Args:
        images: Tableau numpy à valider
        
    Returns:
        Tuple (is_valid, error_message)
    """
    try:
        if images is None:
            return False, "Tableau None"
        
        if not isinstance(images, np.ndarray):
            return False, f"Type invalide: {type(images)}"
        
        if images.size == 0:
            return False, "Tableau vide"
        
        if len(images.shape) not in [3, 4]:
            return False, f"Shape invalide: {images.shape}. Attendu (N,H,W) ou (N,H,W,C)"
        
        if images.dtype not in [np.uint8, np.float32, np.float64]:
            return False, f"Dtype invalide: {images.dtype}"
        
        return True, None
        
    except Exception as e:
        return False, str(e)

def compute_image_statistics(images: np.ndarray) -> Dict[str, Any]:
    """
    Calcule des statistiques globales sur un ensemble d'images.
    
    Args:
        images: Tableau numpy d'images
        
    Returns:
        Dictionnaire avec les statistiques
    """
    try:
        is_valid, error = validate_image_array(images)
        if not is_valid:
            return {'error': error}
        
        stats = {
            'n_images': len(images),
            'shape': images.shape,
            'dtype': str(images.dtype),
            'memory_mb': images.nbytes / (1024**2),
            'mean_pixel_value': float(np.mean(images)),
            'std_pixel_value': float(np.std(images)),
            'min_pixel_value': float(np.min(images)),
            'max_pixel_value': float(np.max(images))
        }
        
        # Détection du range de valeurs
        if images.max() <= 1.0:
            stats['value_range'] = 'normalized'
        elif images.max() <= 255:
            stats['value_range'] = 'uint8'
        else:
            stats['value_range'] = 'custom'
        
        return stats
        
    except Exception as e:
        logger.error(f"Erreur compute_image_statistics: {e}")
        return {'error': str(e)}

def resize_images_batch(
    images: np.ndarray,
    target_size: Tuple[int, int],
    method: str = 'lanczos'
) -> np.ndarray:
    """
    Redimensionne un batch d'images.
    
    Args:
        images: Tableau numpy d'images
        target_size: Taille cible (height, width)
        method: Méthode de redimensionnement ('lanczos', 'bilinear', 'nearest')
        
    Returns:
        Tableau d'images redimensionnées
    """
    try:
        is_valid, error = validate_image_array(images)
        if not is_valid:
            raise ValueError(f"Images invalides: {error}")
        
        # Mapping des méthodes
        methods_map = {
            'lanczos': Image.Resampling.LANCZOS,
            'bilinear': Image.Resampling.BILINEAR,
            'nearest': Image.Resampling.NEAREST
        }
        
        resample_method = methods_map.get(method.lower(), Image.Resampling.LANCZOS)
        
        resized = []
        for img_array in images:
            img = Image.fromarray(img_array.astype(np.uint8) if img_array.max() > 1 else (img_array * 255).astype(np.uint8))
            img_resized = img.resize(target_size, resample_method)
            resized.append(np.array(img_resized))
        
        return np.array(resized)
        
    except Exception as e:
        logger.error(f"Erreur resize_images_batch: {e}")
        raise

def normalize_images(images: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalise un ensemble d'images.
    
    Args:
        images: Tableau numpy d'images
        method: Méthode de normalisation ('minmax', 'standard', 'scale')
        
    Returns:
        Images normalisées
    """
    try:
        is_valid, error = validate_image_array(images)
        if not is_valid:
            raise ValueError(f"Images invalides: {error}")
        
        if method == 'minmax':
            # Normalisation 0-1
            if images.max() > 1.0:
                return images / 255.0
            else:
                return images
                
        elif method == 'standard':
            # Standardisation (mean=0, std=1)
            mean = np.mean(images)
            std = np.std(images)
            return (images - mean) / (std + 1e-7)
            
        elif method == 'scale':
            # Mise à l'échelle 0-255
            if images.max() <= 1.0:
                return (images * 255).astype(np.uint8)
            else:
                return images.astype(np.uint8)
        else:
            raise ValueError(f"Méthode inconnue: {method}")
            
    except Exception as e:
        logger.error(f"Erreur normalize_images: {e}")
        raise

def augment_images_simple(
    images: np.ndarray,
    labels: np.ndarray,
    augmentation_factor: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augmentation simple des images (rotation, flip).
    
    Args:
        images: Tableau numpy d'images
        labels: Labels correspondants
        augmentation_factor: Facteur de multiplication du dataset
        
    Returns:
        Tuple (images_augmentées, labels_augmentés)
    """
    try:
        is_valid, error = validate_image_array(images)
        if not is_valid:
            raise ValueError(f"Images invalides: {error}")
        
        augmented_images = [images]
        augmented_labels = [labels]
        
        for _ in range(augmentation_factor - 1):
            # Rotation aléatoire
            rotated = np.array([np.rot90(img, k=np.random.randint(0, 4)) for img in images])
            augmented_images.append(rotated)
            augmented_labels.append(labels.copy())
            
            # Flip horizontal
            flipped = np.array([np.fliplr(img) for img in images])
            augmented_images.append(flipped)
            augmented_labels.append(labels.copy())
        
        # Concatenation
        final_images = np.concatenate(augmented_images, axis=0)
        final_labels = np.concatenate(augmented_labels, axis=0)
        
        logger.info(f"Augmentation: {len(images)} → {len(final_images)} images")
        
        return final_images, final_labels
        
    except Exception as e:
        logger.error(f"Erreur augment_images_simple: {e}")
        raise

def save_images_to_folder(
    images: np.ndarray,
    labels: np.ndarray,
    output_dir: str,
    prefix: str = 'img'
) -> int:
    """
    Sauvegarde des images dans un dossier organisé par classe.
    
    Args:
        images: Tableau numpy d'images
        labels: Labels correspondants
        output_dir: Dossier de sortie
        prefix: Préfixe pour les noms de fichiers
        
    Returns:
        Nombre d'images sauvegardées
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_count = 0
        
        for idx, (img, label) in enumerate(zip(images, labels)):
            # Créer sous-dossier par classe
            class_dir = output_path / f"class_{label}"
            class_dir.mkdir(exist_ok=True)
            
            # Sauvegarder l'image
            filename = class_dir / f"{prefix}_{idx:06d}.png"
            
            # Conversion pour sauvegarde
            if img.max() <= 1.0:
                img_save = (img * 255).astype(np.uint8)
            else:
                img_save = img.astype(np.uint8)
            
            Image.fromarray(img_save).save(filename)
            saved_count += 1
        
        logger.info(f"{saved_count} images sauvegardées dans {output_dir}")
        return saved_count
        
    except Exception as e:
        logger.error(f"Erreur save_images_to_folder: {e}")
        raise

# ===================================
# EXPORT DES FONCTIONS PUBLIQUES
# ===================================

__all__ = [
    # Détection et info
    'detect_dataset_structure',
    'get_dataset_info',
    'get_dataset_stats',
    
    # Chargement
    'load_images_flexible',
    'load_images_from_folder',
    
    # Analyse qualité
    'analyze_image_quality',
    'analyze_image_distribution',
    
    # Utilitaires
    'validate_image_array',
    'compute_image_statistics',
    'resize_images_batch',
    'normalize_images',
    'augment_images_simple',
    'save_images_to_folder',
    
    # Classes et enums
    'ImageConfig',
    'QualityThresholds',
    'DatasetType'
]