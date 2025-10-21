"""
Fonctions de prétraitement pour computer vision : normalisation, resize, augmentation.
Réutilisables dans dashboard, training, evaluation.
"""
from collections import Counter
from dataclasses import dataclass, field
from sys import platform
import numpy as np
from PIL import Image
import albumentations as A
from typing import Dict, Any, Optional

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.shared.logging import StructuredLogger

def apply_normalization(image: np.ndarray, method: str) -> np.ndarray:
    """Applique la normalisation choisie."""
    if method == "0-1 (MinMax)":
        return image / 255.0
    elif method == "-1-1":
        return (image / 127.5) - 1.0
    elif method == "Standard (ImageNet)":
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        return (image / 255.0 - mean) / std
    return image

def apply_resize(image: np.ndarray, size: str) -> np.ndarray:
    """Redimensionne l'image."""
    if size == "Conserver original":
        return image
    new_size = int(size.split("×")[0])
    pil_img = Image.fromarray(image.astype(np.uint8))
    resized = pil_img.resize((new_size, new_size))
    return np.array(resized)

def apply_augmentation(image: np.ndarray, config: str) -> np.ndarray:
    """Applique l'augmentation (utilise Albumentations)."""
    if not config:
        return image
    
    intensity_map = {
        "Légère": A.Compose([A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast(p=0.5)]),
        "Moyenne": A.Compose([A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast(p=0.5), A.GaussNoise(p=0.5)]),
        "Forte": A.Compose([A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast(p=0.5), A.GaussNoise(p=0.5), A.Rotate(limit=30, p=0.5)])
    }
    transform = intensity_map.get(config, A.Compose([]))
    return transform(image=image)['image']

def apply_preprocessing(X: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """Applique tout le pipeline à un batch d'images."""
    processed = []
    for img in X:
        img = apply_normalization(img, config.get("normalization", "Aucune"))
        img = apply_resize(img, config.get("resize", "Conserver original"))
        if config.get("augmentation", False):
            img = apply_augmentation(img, config.get("augmentation_config"))
        processed.append(img)
    return np.array(processed)

def generate_preview(original: np.ndarray, config: Dict[str, Any]) -> list:
    """Génère une liste d'images transformées pour preview."""
    previews = [original]
    titles = ["Original"]
    
    norm_img = apply_normalization(original.copy(), config.get("normalization", "Aucune"))
    previews.append(norm_img)
    titles.append(f"Normalisé {config.get('normalization', 'Aucune')}")
    
    res_img = apply_resize(original.copy(), config.get("resize", "Conserver original"))
    previews.append(res_img)
    titles.append(f"Redimensionné {config.get('resize', 'Original')}")
    
    if config.get("augmentation", False):
        aug_img = apply_augmentation(original.copy(), config.get("augmentation_config"))
        previews.append(aug_img)
        titles.append(f"Augmenté ({config.get('augmentation_config', 'N/A')})")
    
    return previews, titles


# ===================================
# PREPROCESSING PIPELINE (SANS FUITE)
# ===================================

logger = StructuredLogger(__name__)
class DataPreprocessor:
    """
    Pipeline de preprocessing production-ready avec gestion automatique des formats.
    
    Features:
    - Détection automatique du format (channels_first/last)
    - Conversion transparente vers format PyTorch
    - Gestion des edge cases
    - Serialisation pour MLOps
    - Logging complet
    """
    
    def __init__(self, strategy: str = "standardize", auto_detect_format: bool = True):
        """
        Args:
            strategy: 'standardize', 'normalize', 'none'
            auto_detect_format: Détection automatique du format des données
        """
        self.strategy = strategy
        self.auto_detect_format = auto_detect_format
        self.fitted = False
        self.mean_ = None
        self.std_ = None
        self.min_ = None
        self.max_ = None
        self.data_format_ = None  # 'channels_first' ou 'channels_last'
        self.original_shape_ = None
        
        logger.info(
            "DataPreprocessor initialisé",
            strategy=strategy,
            auto_detect_format=auto_detect_format
        )
    
    def _detect_data_format(self, X: np.ndarray) -> str:
        """
        Détecte automatiquement le format des données.
        
        Returns:
            'channels_first' ou 'channels_last'
        """
        if X.ndim != 4:
            return 'channels_last'  # Par défaut pour les autres cas
            
        # Règles de détection
        n_samples, dim1, dim2, dim3 = X.shape
        
        # Cas channels_last: (N, H, W, C) où C est petit (1,3,4)
        if dim3 in [1, 3, 4]:
            return 'channels_last'
        # Cas channels_first: (N, C, H, W) où C est petit (1,3,4)
        elif dim1 in [1, 3, 4]:
            return 'channels_first'
        else:
            # Heuristique: si la première dimension est plus petite, c'est probablement channels
            if dim1 < dim2 and dim1 < dim3:
                return 'channels_first'
            else:
                return 'channels_last'
    
    def _ensure_channels_first(self, X: np.ndarray) -> np.ndarray:
        """Convertit vers le format PyTorch (N, C, H, W) si nécessaire."""
        if self.data_format_ == 'channels_last' and X.ndim == 4:
            return np.transpose(X, (0, 3, 1, 2))
        return X
    
    def _ensure_channels_last(self, X: np.ndarray) -> np.ndarray:
        """Convertit vers le format standard (N, H, W, C) si nécessaire."""
        if self.data_format_ == 'channels_first' and X.ndim == 4:
            return np.transpose(X, (0, 2, 3, 1))
        return X
    
    def _calculate_statistics(self, X: np.ndarray):
        """Calcule les statistiques sur le format approprié."""
        # Pour le calcul des stats, on utilise channels_last qui est plus standard
        if self.data_format_ == 'channels_first':
            X_for_stats = self._ensure_channels_last(X)
        else:
            X_for_stats = X
        
        if self.strategy == "standardize":
            if X_for_stats.ndim == 4:
                # Moyenne/std par canal: axes (N, H, W)
                axes = (0, 1, 2)
                self.mean_ = X_for_stats.mean(axis=axes, keepdims=True)
                self.std_ = X_for_stats.std(axis=axes, keepdims=True) + 1e-8
            else:
                self.mean_ = X_for_stats.mean()
                self.std_ = X_for_stats.std() + 1e-8
                
        elif self.strategy == "normalize":
            self.min_ = X_for_stats.min()
            self.max_ = X_for_stats.max()
    
    def fit(self, X: np.ndarray) -> 'DataPreprocessor':
        """
        Calcule les statistiques sur le training set UNIQUEMENT.
        
        Args:
            X: Training data (N, H, W, C) ou (N, C, H, W)
        """
        # Validation des données
        if X.ndim not in [3, 4]:
            raise ValueError(f"Dimensions invalides: {X.ndim}. Attendu 3D ou 4D.")
        
        if np.isnan(X).any() or np.isinf(X).any():
            raise ValueError("Les données contiennent des valeurs NaN ou Inf.")
        
        self.original_shape_ = X.shape
        
        # Détection du format
        if self.auto_detect_format:
            self.data_format_ = self._detect_data_format(X)
        else:
            self.data_format_ = 'channels_last'  # Par défaut
        
        # Calcul des statistiques
        self._calculate_statistics(X)
        
        self.fitted = True
        
        logger.info(
            "Preprocessing fitted",
            original_shape=self.original_shape_,
            data_format=self.data_format_,
            strategy=self.strategy,
            mean_shape=getattr(self.mean_, 'shape', None),
            std_shape=getattr(self.std_, 'shape', None)
        )
        
        return self
    
    def transform(self, X: np.ndarray, output_format: str = "channels_first") -> np.ndarray:
        """
        Applique la transformation avec gestion du format de sortie.
        
        Args:
            X: Data à transformer
            output_format: 'channels_first' (PyTorch) ou 'channels_last'
            
        Returns:
            Données transformées dans le format demandé
        """
        if not self.fitted and self.strategy != "none":
            raise ValueError("DataPreprocessor doit être fitted avant transform()")
        
        # Validation cohérence format
        current_format = self._detect_data_format(X) if self.auto_detect_format else self.data_format_
        if current_format != self.data_format_:
            logger.warning(
                f"Incohérence de format: attendu {self.data_format_}, reçu {current_format}"
            )
        
        # Conversion vers channels_last pour appliquer les stats
        if self.data_format_ == 'channels_first':
            X_conv = self._ensure_channels_last(X)
        else:
            X_conv = X
        
        # Application de la transformation
        if self.strategy == "standardize":
            X_norm = (X_conv - self.mean_) / self.std_
        elif self.strategy == "normalize":
            X_norm = (X_conv - self.min_) / (self.max_ - self.min_ + 1e-8)
        else:
            X_norm = X_conv
        
        # Conversion vers le format de sortie demandé
        if output_format == "channels_first":
            X_out = self._ensure_channels_first(X_norm)
        elif output_format == "channels_last":
            X_out = X_norm
        else:
            raise ValueError(f"Format de sortie non supporté: {output_format}")
        
        logger.debug(
            "Transformation appliquée",
            input_shape=X.shape,
            output_shape=X_out.shape,
            output_format=output_format
        )
        
        return X_out
    
    def fit_transform(self, X: np.ndarray, output_format: str = "channels_first") -> np.ndarray:
        """Fit puis transform (pour train uniquement)."""
        return self.fit(X).transform(X, output_format)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transformation (pour visualisation)."""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before inverse_transform")
        
        # Détection du format d'entrée
        input_format = self._detect_data_format(X)
        
        # Conversion vers channels_last pour l'inverse
        if input_format == 'channels_first':
            X_conv = self._ensure_channels_last(X)
        else:
            X_conv = X
        
        # Application inverse
        if self.strategy == "standardize":
            X_orig = X_conv * self.std_ + self.mean_
        elif self.strategy == "normalize":
            X_orig = X_conv * (self.max_ - self.min_) + self.min_
        else:
            X_orig = X_conv
        
        # Remise dans le format d'origine
        if input_format == 'channels_first':
            return self._ensure_channels_first(X_orig)
        else:
            return X_orig
    
    def get_config(self) -> Dict[str, Any]:
        """Retourne la configuration pour la serialisation."""
        return {
            "strategy": self.strategy,
            "auto_detect_format": self.auto_detect_format,
            "fitted": self.fitted,
            "data_format": self.data_format_,
            "original_shape": self.original_shape_,
            "mean_shape": getattr(self.mean_, 'shape', None),
            "std_shape": getattr(self.std_, 'shape', None)
        }
    
    def save(self, filepath: str):
        """Sauvegarde le preprocessor."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Preprocessor sauvegardé: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'DataPreprocessor':
        """Charge un preprocessor sauvegardé."""
        import pickle
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        logger.info(f"Preprocessor chargé: {filepath}")
        return preprocessor
    
@dataclass
class Result:
    """Type de retour standardisé"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def ok(cls, data: Any, **metadata) -> 'Result':
        """Succès"""
        return cls(success=True, data=data, metadata=metadata)
    
    @classmethod
    def err(cls, error: str, **metadata) -> 'Result':
        """Échec"""
        return cls(success=False, error=error, metadata=metadata)
    
# ======================
# VALIDATION DES DONNÉES
# ======================

class DataValidator:
    """Validation robuste des données d'entrée"""
    
    @staticmethod
    def validate_input_data(
        X: np.ndarray, 
        y: np.ndarray, 
        name: str = "data"
    ) -> Result:
        """
        Validation complète d'un dataset.
        
        Returns:
            Result avec success=True si OK, sinon error
        """
        try:
            # Vérification dimensions
            if X.ndim not in [3, 4]:
                return Result.err(
                    f"{name}: dimensions invalides {X.shape}, attendu 3D ou 4D"
                )
            
            # Vérification cohérence tailles
            if len(X) != len(y):
                return Result.err(
                    f"{name}: tailles incohérentes X={len(X)}, y={len(y)}"
                )
            
            # Vérification valeurs
            if np.isnan(X).any():
                return Result.err(f"{name}: contient des NaN")
            
            if np.isinf(X).any():
                return Result.err(f"{name}: contient des Inf")
            
            # Vérification labels
            if np.isnan(y).any() or np.isinf(y).any():
                return Result.err(f"{name}: labels contiennent NaN/Inf")
            
            # Vérification nombre d'échantillons minimum
            min_samples = 20
            if len(X) < min_samples:
                return Result.err(
                    f"{name}: échantillons insuffisants {len(X)} < {min_samples}"
                )
            
            # Informations sur les classes
            unique_classes = np.unique(y)
            class_counts = Counter(y)
            
            logger.info(
                f"Validation {name} OK",
                shape=X.shape,
                n_samples=len(X),
                n_classes=len(unique_classes),
                class_distribution=dict(class_counts)
            )
            
            return Result.ok(
                {"shape": X.shape, "n_classes": len(unique_classes)},
                class_counts=dict(class_counts)
            )
            
        except Exception as e:
            return Result.err(f"Erreur validation {name}: {str(e)}")
    
    @staticmethod
    def check_class_imbalance(y: np.ndarray) -> Dict[str, Any]:
        """Analyse le déséquilibre des classes"""
        class_counts = Counter(y)
        total = len(y)
        
        if len(class_counts) < 2:
            return {
                "imbalanced": False,
                "ratio": 1.0,
                "severity": "none",
                "counts": class_counts
            }
        
        min_count = min(class_counts.values())
        max_count = max(class_counts.values())
        ratio = max_count / max_count if min_count > 0 else float('inf')
        
        # Classification du déséquilibre
        if ratio >= 10:
            severity = "critical"
        elif ratio >= 5:
            severity = "high"
        elif ratio >= 2:
            severity = "moderate"
        else:
            severity = "low"
        
        return {
            "imbalanced": ratio >= 2,
            "ratio": ratio,
            "severity": severity,
            "counts": dict(class_counts),
            "percentages": {k: v/total*100 for k, v in class_counts.items()}
        }
    

# ===========
# DATALOADERS
# ===========

class DataLoaderFactory:
    """Factory production-ready pour DataLoaders."""
    
    @staticmethod
    def create(
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,    # Production: 0 pour éviter les problèmes
        pin_memory: bool = False  # Production: False sur CPU
    ) -> DataLoader:
        """
        Crée un DataLoader robuste avec validation.       
        Args:
            X: Features déjà dans le format channels_first (N, C, H, W)
            y: Labels
        """
        try:
            # Validation des entrées
            if X is None or len(X) == 0:
                raise ValueError("Les features X ne peuvent pas être vides")
            
            if y is None or len(y) == 0:
                raise ValueError("Les labels y ne peuvent pas être vides")
            
            if len(X) != len(y):
                raise ValueError(f"X et y ont des longueurs différentes: {len(X)} vs {len(y)}")
            
            # Conversion en tensors (suppose déjà format PyTorch)
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.long)
            
            # Validation finale des shapes
            if X_tensor.dim() != 4:
                logger.warning(f"Dimensions du tensor X inattendues: {X_tensor.shape}")
            
            # Dataset
            dataset = TensorDataset(X_tensor, y_tensor)
            
            # DataLoader
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=False
            )
            
            logger.debug(
                "DataLoader créé avec succès",
                n_samples=len(dataset),
                input_shape=X.shape,
                tensor_shape=X_tensor.shape,
                n_batches=len(loader),
                batch_size=batch_size
            )
            
            return loader
            
        except Exception as e:
            logger.error(
                f"Erreur création DataLoader: {str(e)}",
                exc_info=True,
                X_shape=getattr(X, 'shape', None),
                y_shape=getattr(y, 'shape', None),
                batch_size=batch_size
            )
            raise