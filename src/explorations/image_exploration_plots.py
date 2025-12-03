"""
🖼️ Module de traitement d'images pour DataLab Pro
Optimisé pour la production avec gestion d'erreurs robuste
Version: 2.0.1 - Production Ready

Fonctionnalités:
- Détection automatique de structure de dataset
- Chargement flexible multi-formats
- Analyse de qualité d'images avancée
- Statistiques et distributions
- Support MVTec AD, catégoriel, plat

"""

import os
import numpy as np
import pandas as pd # type: ignore
from PIL import Image
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Import du système de logging centralisé
from src.shared.logging import get_logger

# Configuration logging
logger = get_logger(__name__)

# ===================================
# CONSTANTES ET CONFIGURATIONS
# ===================================

class DatasetType(Enum):
    """Types de datasets supportés"""
    MVTEC_AD = "mvtec_ad"
    CATEGORICAL = "categorical_folders"
    FLAT = "flat_directory"
    MIXED = "mixed"
    INVALID = "invalid"
    UNKNOWN = "unknown"

@dataclass
class ImageConfig:
    """Configuration pour le chargement d'images"""
    target_size: Tuple[int, int] = (256, 256)
    normalize: bool = False
    color_mode: str = 'RGB'  # RGB, L (grayscale)
    max_images: Optional[int] = None
    
@dataclass
class QualityThresholds:
    """Seuils pour l'analyse de qualité"""
    dark_threshold: float = 50.0
    bright_threshold: float = 200.0
    low_contrast_threshold: float = 20.0
    min_sharpness: float = 0.01

# Extensions d'images supportées
SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')

# ===================================
# FONCTIONS DE DÉTECTION DE STRUCTURE
# ===================================

def detect_dataset_structure(data_dir: str) -> Dict[str, Any]:
    """
    Détecte automatiquement la structure du dataset d'images.
    Version robuste avec validation complète.
    
    Args:
        data_dir: Chemin vers le dossier du dataset
        
    Returns:
        Dictionnaire avec le type et les métadonnées
        
    Raises:
        ValueError: Si le chemin est invalide
    """
    try:
        # Validation du chemin
        if not data_dir or not isinstance(data_dir, str):
            return {
                "type": DatasetType.INVALID.value,
                "error": "Chemin invalide ou vide"
            }
        
        data_path = Path(data_dir)
        
        if not data_path.exists():
            return {
                "type": DatasetType.INVALID.value,
                "error": f"Dossier introuvable: {data_dir}"
            }
        
        if not data_path.is_dir():
            return {
                "type": DatasetType.INVALID.value,
                "error": f"Le chemin n'est pas un dossier: {data_dir}"
            }
        
        # Lister les éléments
        try:
            items = [item.name for item in data_path.iterdir()]
        except PermissionError:
            return {
                "type": DatasetType.INVALID.value,
                "error": "Permission refusée pour lire le dossier"
            }
        
        if not items:
            return {
                "type": DatasetType.INVALID.value,
                "error": "Dossier vide"
            }
        
        # Structure MVTec AD standard
        if "train" in items and "test" in items:
            train_path = data_path / "train"
            if train_path.exists() and train_path.is_dir():
                try:
                    train_items = [item.name for item in train_path.iterdir() if item.is_dir()]
                    if "good" in train_items:
                        logger.info(f"Structure MVTec AD détectée: {data_dir}")
                        return {
                            "type": DatasetType.MVTEC_AD.value,
                            "categories": train_items,
                            "description": "Dataset MVTec AD avec train/test et good/defect"
                        }
                except Exception as e:
                    logger.warning(f"Erreur lecture train folder: {e}")
        
        # Analyse des sous-dossiers et fichiers
        subdirs = []
        image_files = []
        
        for item in data_path.iterdir():
            try:
                if item.is_dir():
                    subdirs.append(item.name)
                elif item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
                    image_files.append(item.name)
            except Exception as e:
                logger.debug(f"Erreur lecture item {item}: {e}")
                continue
        
        # Structure avec sous-dossiers = catégories
        if subdirs and not image_files:
            logger.info(f"Structure catégorielle détectée: {len(subdirs)} catégories")
            return {
                "type": DatasetType.CATEGORICAL.value,
                "categories": subdirs,
                "n_categories": len(subdirs),
                "description": f"Dataset catégoriel avec {len(subdirs)} classes"
            }
        
        # Structure plate (toutes images à la racine)
        elif image_files and not subdirs:
            logger.info(f"Structure plate détectée: {len(image_files)} images")
            return {
                "type": DatasetType.FLAT.value,
                "image_count": len(image_files),
                "description": f"Dataset plat avec {len(image_files)} images"
            }
        
        # Structure mixte
        elif subdirs and image_files:
            logger.warning(f"Structure mixte détectée: {len(subdirs)} dossiers + {len(image_files)} images racine")
            return {
                "type": DatasetType.MIXED.value,
                "categories": subdirs,
                "root_images": len(image_files),
                "description": "Structure mixte (dossiers + images racine)"
            }
        
        # Structure inconnue
        else:
            return {
                "type": DatasetType.UNKNOWN.value,
                "items": items[:10],  # Limiter pour éviter trop de données
                "description": "Structure non reconnue"
            }
    
    except Exception as e:
        logger.error(f"❌ Erreur critique dans detect_dataset_structure: {e}", exc_info=True)
        return {
            "type": DatasetType.INVALID.value,
            "error": f"Erreur inattendue: {str(e)}"
        }

def get_dataset_info(data_dir: str) -> Dict[str, Any]:
    """
    Retourne des informations détaillées sur le dataset.
    Version optimisée avec calculs parallélisables.
    
    Args:
        data_dir: Chemin vers le dossier du dataset
        
    Returns:
        Dictionnaire avec les statistiques complètes
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
            
            # Compter images normales
            for folder in ["train/good", "test/good"]:
                folder_path = data_path / folder
                if folder_path.exists():
                    normal_count += len(_get_image_files(folder_path))
            
            # Compter images anormales
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
                "task_type": "anomaly_detection"
            })
            
            logger.info(f"MVTec AD: {normal_count} normal, {anomaly_count} anomalies")
        
        # Catégoriel
        elif structure["type"] == DatasetType.CATEGORICAL.value:
            categories = {}
            total = 0
            
            for category in structure.get("categories", []):
                cat_path = data_path / category
                if cat_path.is_dir():
                    count = len(_get_image_files(cat_path))
                    categories[category] = count
                    total += count
            
            info.update({
                "categories": categories,
                "n_categories": len(categories),
                "total": total,
                "task_type": "classification",
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
            
            logger.info(f"Plat: {total} images")
        
        return info
    
    except Exception as e:
        logger.error(f"Erreur dans get_dataset_info: {e}", exc_info=True)
        return {
            "structure_type": DatasetType.INVALID.value,
            "error": str(e),
            "is_valid": False
        }

def get_dataset_stats(data_dir: str) -> pd.DataFrame:
    """
    Calcule les statistiques détaillées du dataset.
    Version compatible production avec gestion d'erreurs.
    
    Args:
        data_dir: Chemin vers le dossier du dataset
        
    Returns:
        DataFrame avec les statistiques par catégorie
    """
    try:
        stats = []
        structure = detect_dataset_structure(data_dir)
        data_path = Path(data_dir)
        
        if structure["type"] == DatasetType.MVTEC_AD.value:
            # Structure MVTec AD
            categories = [
                ('train/good', 'Normal (Train)'),
                ('test/good', 'Normal (Test)')
            ]
            
            # Ajouter les catégories d'anomalies
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
            # Dossiers = catégories
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
            # Structure plate
            image_files = _get_image_files(data_path)
            stats.append({
                "Catégorie": "Toutes images",
                "Chemin": "racine",
                "Nombre d'images": len(image_files),
                "Type": "Non catégorisé"
            })
        
        df_stats = pd.DataFrame(stats)
        
        # Ajouter des statistiques agrégées
        if not df_stats.empty and "Nombre d'images" in df_stats.columns:
            df_stats['Pourcentage'] = (df_stats['Nombre d\'images'] / df_stats['Nombre d\'images'].sum() * 100).round(2)
        
        return df_stats
        
    except Exception as e:
        logger.error(f"Erreur calcul stats: {e}", exc_info=True)
        return pd.DataFrame()

# ===================================
# FONCTIONS DE CHARGEMENT D'IMAGES
# ===================================
def _load_mvtec_train_labels(data_dir: str) -> np.ndarray:
    """
    Charge UNIQUEMENT les labels du dossier train/good pour MVTec AD.
    Retourne toujours un tableau de 0 (images normales).
    Utilisé pour détecter correctement la tâche UNSUPERVISED.
    """
    try:
        train_good_path = Path(data_dir) / "train" / "good"
        if not train_good_path.exists() or not train_good_path.is_dir():
            logger.warning("Dossier train/good introuvable → fallback y_train vide")
            return np.array([], dtype=int)
        
        image_files = _get_image_files(train_good_path)
        logger.debug(f"{len(image_files)} images normales dans train/good → y_train = [0]")
        return np.zeros(len(image_files), dtype=int)
    
    except Exception as e:
        logger.error(f"Erreur chargement y_train MVTec: {e}")
        return np.array([], dtype=int)
    

def load_images_flexible(
    data_dir: str,
    target_size: Tuple[int, int] = (256, 256),
    config: Optional[ImageConfig] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Retourne: X, X_norm, y (complet), y_train (seulement train/good)
    """
    try:
        if config is None:
            config = ImageConfig(target_size=target_size)
        
        if config.max_images:
            logger.warning(f"max_images={config.max_images} ignoré pour MVTec AD")

        structure = detect_dataset_structure(data_dir)
        structure_type = structure.get("type")
        
        logger.info(f"Structure détectée: {structure_type}")
        
        if structure_type == DatasetType.INVALID.value:
            raise ValueError(f"Structure invalide: {structure.get('error')}")

        y_train = None

        if structure_type == DatasetType.MVTEC_AD.value:
            X, y_full = _load_mvtec_structure(data_dir, config)        # ← y complet
            y_train = _load_mvtec_train_labels(data_dir)               # ← seulement train/good
        elif structure_type == DatasetType.CATEGORICAL.value:
            X, y_full = _load_categorical_folders(data_dir, config)
        elif structure_type == DatasetType.FLAT.value:
            X, y_full = _load_flat_directory(data_dir, config)
        else:
            raise ValueError(f"Structure non supportée: {structure_type}")

        if len(X) == 0:
            raise RuntimeError("Aucune image valide trouvée")

        logger.info(f"Chargement terminé: {len(X)} images, {len(np.unique(y_full))} classes au total")
        
        # ✅ CORRECTION #1: Validation stricte y_train pour MVTec AD
        if y_train is not None and structure_type == DatasetType.MVTEC_AD.value:
            # Validation: y_train doit contenir uniquement des 0
            if len(y_train) > 0:
                unique_labels_train = np.unique(y_train)
                if len(unique_labels_train) > 1:
                    error_msg = (
                        f"❌ ERREUR CRITIQUE: y_train contient {len(unique_labels_train)} classes différentes: {unique_labels_train}. "
                        f"Pour MVTec AD (unsupervised), y_train doit contenir UNIQUEMENT des 0 (images normales). "
                        f"Vérifiez que _load_mvtec_train_labels() charge uniquement train/good."
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                if len(unique_labels_train) == 1 and unique_labels_train[0] != 0:
                    error_msg = (
                        f"❌ ERREUR CRITIQUE: y_train contient uniquement le label {unique_labels_train[0]}, "
                        f"attendu 0 (images normales) pour MVTec AD."
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                # Validation shape: y_train doit être 1D
                if y_train.ndim != 1:
                    error_msg = (
                        f"❌ ERREUR CRITIQUE: y_train a une shape incorrecte {y_train.shape}, "
                        f"attendu 1D array (n_samples,). Veuillez vérifier le chargement des labels."
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                # Validation cohérence: y_train doit correspondre uniquement aux images train/good
                train_good_path = Path(data_dir) / "train" / "good"
                if train_good_path.exists():
                    train_good_files = _get_image_files(train_good_path)
                    if len(y_train) != len(train_good_files):
                        logger.warning(
                            f"⚠️ Incohérence: y_train contient {len(y_train)} labels mais "
                            f"{len(train_good_files)} images dans train/good"
                        )
                
                logger.info(f"✅ Validation y_train OK: {len(y_train)} images normales (label 0 uniquement) → UNSUPERVISED")
            else:
                logger.warning("⚠️ y_train vide - impossible de valider pour MVTec AD")
        elif y_train is not None:
            logger.info(f"→ y_train: {len(y_train)} images → mode SUPERVISED")

        # Normalisation
        X_norm = X.astype(np.float32) / 255.0

        return X.astype(np.uint8), X_norm, y_full, y_train

    except Exception as e:
        logger.error(f"Erreur load_images_flexible: {e}", exc_info=True)
        raise

def load_images_from_folder(
    data_dir: str,
    target_size: Tuple[int, int] = (128, 128),
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ancienne API conservée pour compatibilité.
    """
    X, X_norm, _ = load_images_flexible(data_dir, target_size=target_size, config=ImageConfig(normalize=normalize))
    return X_norm if normalize else X, None  

# ===================================
# FONCTIONS PRIVÉES DE CHARGEMENT
# ===================================

def _load_mvtec_structure(data_dir: str, config: ImageConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Charge la structure MVTec AD standard.
    Version optimisée avec gestion d'erreurs.
    """
    images, labels = [], []
    data_path = Path(data_dir)
    
    # Images normales (label 0)
    normal_paths = [
        data_path / "train" / "good",
        data_path / "test" / "good"
    ]
    
    loaded_normal = 0
    for path in normal_paths:
        if path.exists():
            image_files = _get_image_files(path)
            logger.debug(f"Chargement {len(image_files)} images normales depuis {path}")
            
            for img_file in image_files:
                img = _load_single_image(path / img_file, config)
                if img is not None:
                    images.append(img)
                    labels.append(0)  # Normal
                    loaded_normal += 1
                    
                    if config.max_images and len(images) >= config.max_images:
                        break
    
    logger.info(f"{loaded_normal} images normales chargées")
    
    # Images anormales (label 1)
    test_path = data_path / "test"
    if test_path.exists():
        loaded_anomaly = 0
        for category in test_path.iterdir():
            if category.is_dir() and category.name != "good":
                image_files = _get_image_files(category)
                logger.debug(f"Chargement {len(image_files)} anomalies depuis {category.name}")
                
                for img_file in image_files:
                    img = _load_single_image(category / img_file, config)
                    if img is not None:
                        images.append(img)
                        labels.append(1)  # Anomalie
                        loaded_anomaly += 1
                        
                        if config.max_images and len(images) >= config.max_images:
                            break
        
        logger.info(f"{loaded_anomaly} anomalies chargées")
    
    return np.array(images), np.array(labels)

def _load_categorical_folders(data_dir: str, config: ImageConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Charge depuis des dossiers = catégories.
    Version optimisée avec tri alphabétique pour cohérence.
    """
    images, labels = [], []
    data_path = Path(data_dir)
    
    # Trier les catégories pour cohérence
    categories = sorted([item for item in data_path.iterdir() if item.is_dir()])
    
    for label, category in enumerate(categories):
        image_files = _get_image_files(category)
        logger.debug(f"Chargement {len(image_files)} images pour classe {label} ({category.name})")
        
        for img_file in image_files:
            img = _load_single_image(category / img_file, config)
            if img is not None:
                images.append(img)
                labels.append(label)
                
                if config.max_images and len(images) >= config.max_images:
                    break
    
    logger.info(f"{len(categories)} classes chargées")
    return np.array(images), np.array(labels)

def _load_flat_directory(data_dir: str, config: ImageConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Charge depuis un dossier plat (toutes images mélangées).
    Version optimisée.
    """
    images = []
    data_path = Path(data_dir)
    
    image_files = _get_image_files(data_path)
    logger.debug(f"Chargement {len(image_files)} images depuis dossier plat")
    
    for img_file in image_files:
        img = _load_single_image(data_path / img_file, config)
        if img is not None:
            images.append(img)
            
            if config.max_images and len(images) >= config.max_images:
                break
    
    # Toutes étiquetées comme normales (0) par défaut
    labels = np.zeros(len(images), dtype=int)
    
    logger.info(f"{len(images)} images chargées")
    return np.array(images), labels

def _load_single_image(image_path: Path, config: ImageConfig) -> Optional[np.ndarray]:
    """
    Charge et prétraite une seule image.
    Version robuste avec gestion d'erreurs détaillée.
    """
    try:
        with Image.open(image_path) as img:
            # Conversion mode couleur
            if config.color_mode == 'RGB' and img.mode != 'RGB':
                img = img.convert('RGB')
            elif config.color_mode == 'L' and img.mode != 'L':
                img = img.convert('L')
            
            # Redimensionnement
            if img.size != config.target_size:
                img = img.resize(config.target_size, Image.Resampling.LANCZOS)
            
            # Conversion numpy
            img_array = np.array(img)
            
            # Validation
            if img_array.size == 0:
                logger.warning(f"Image vide: {image_path}")
                return None
            
            return img_array
            
    except Exception as e:
        logger.warning(f"Erreur chargement {image_path.name}: {e}")
        return None

def _get_image_files(folder_path: Path) -> List[str]:
    """
    Liste les fichiers images valides dans un dossier.
    Version optimisée avec tri.
    """
    try:
        if not folder_path.exists() or not folder_path.is_dir():
            return []
        
        files = [
            item.name for item in folder_path.iterdir()
            if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        
        return sorted(files)  # Tri pour cohérence
        
    except Exception as e:
        logger.warning(f"Erreur lecture dossier {folder_path}: {e}")
        return []

# ===================================
# FONCTIONS D'ANALYSE DE QUALITÉ
# ===================================

def analyze_image_quality(
    images: np.ndarray,
    sample_size: int = 200,
    thresholds: Optional[QualityThresholds] = None
) -> Dict[str, Any]:
    """
    Analyse complète de la qualité des images.
    Version production avec validation et gestion d'erreurs.
    
    Args:
        images: Tableau numpy d'images (N, H, W, C) ou (N, H, W)
        sample_size: Nombre d'images à échantillonner
        thresholds: Seuils personnalisés optionnels
        
    Returns:
        Dictionnaire avec toutes les métriques de qualité
    """
    try:
        if thresholds is None:
            thresholds = QualityThresholds()
        
        # Validation
        if images is None or len(images) == 0:
            return {'error': 'Tableau d\'images vide'}
        
        if not isinstance(images, np.ndarray):
            return {'error': 'Format d\'images invalide (doit être numpy.ndarray)'}
        
        # Import conditionnel de scipy
        try:
            from scipy import ndimage
            scipy_available = True
        except ImportError:
            scipy_available = False
            logger.warning("SciPy non disponible, calcul de netteté désactivé")
        
        # Échantillonnage intelligent
        actual_sample_size = min(sample_size, len(images))
        if actual_sample_size < len(images):
            indices = np.random.choice(len(images), actual_sample_size, replace=False)
            sample_images = images[indices]
        else:
            sample_images = images
        
        logger.info(f"Analyse qualité sur {len(sample_images)} images")
        
        # Normalisation pour analyse
        if sample_images.max() > 1.0:
            sample_normalized = sample_images / 255.0
        else:
            sample_normalized = sample_images.copy()
        
        # Calcul luminosité (brightness)
        if len(sample_normalized.shape) == 4:  # (N, H, W, C)
            brightness = np.mean(sample_normalized, axis=(1, 2, 3)) * 255
        else:  # (N, H, W)
            brightness = np.mean(sample_normalized, axis=(1, 2)) * 255
        
        # Calcul contraste (contrast)
        if len(sample_normalized.shape) == 4:
            contrast = np.std(sample_normalized, axis=(1, 2, 3)) * 255
        else:
            contrast = np.std(sample_normalized, axis=(1, 2)) * 255
        
        # Calcul netteté (sharpness) avec Laplacian
        sharpness_values = []
        if scipy_available:
            # Limiter pour performance
            sharpness_sample_size = min(100, len(sample_normalized))
            sharpness_sample = sample_normalized[:sharpness_sample_size]
            
            for img in sharpness_sample:
                try:
                    # Convertir en niveau de gris si couleur
                    if len(img.shape) == 3:
                        gray = np.mean(img, axis=2)
                    else:
                        gray = img
                    
                    # Laplacian pour détecter les contours
                    laplacian = ndimage.laplace(gray)
                    sharpness_values.append(float(laplacian.var()))
                    
                except Exception as e:
                    logger.debug(f"Erreur calcul netteté: {e}")
                    continue
        
        sharpness_array = np.array(sharpness_values) if sharpness_values else np.array([])
        
        # Détection des problèmes
        dark_images = np.sum(brightness < thresholds.dark_threshold)
        bright_images = np.sum(brightness > thresholds.bright_threshold)
        low_contrast_images = np.sum(contrast < thresholds.low_contrast_threshold)
        
        # Construction du résultat
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
        
        # Ajouter netteté si disponible
        if len(sharpness_array) > 0:
            result['sharpness'] = {
                'values': sharpness_array.tolist(),
                'mean': float(np.mean(sharpness_array)),
                'std': float(np.std(sharpness_array)),
                'min': float(np.min(sharpness_array)),
                'max': float(np.max(sharpness_array)),
                'available': True
            }
        else:
            result['sharpness'] = {
                'available': False,
                'reason': 'SciPy non disponible' if not scipy_available else 'Erreur de calcul'
            }
        
        logger.info(f"Analyse qualité terminée: {result['problematic_summary']['percentage_problematic']:.1f}% images problématiques")
        
        return result
        
    except Exception as e:
        logger.error(f"Erreur dans analyze_image_quality: {e}", exc_info=True)
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