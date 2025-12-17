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
from typing import Dict, Any, Optional, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.shared.logging import get_logger

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

logger = get_logger(__name__)
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
    
    def __init__(
        self,
        strategy: str = "standardize",
        auto_detect_format: bool = True,
        target_size: Optional[Tuple[int, int]] = None 
    ):
        """
        Args:
            strategy: Stratégie de normalisation
            auto_detect_format: Détection automatique du format
            target_size: Taille cible (H, W) pour resize. Si None, pas de resize.
        """
        self.strategy = strategy
        self.auto_detect_format = auto_detect_format
        self.target_size = target_size  # 🆕 NOUVEAU
        
        # État après fit
        self.fitted = False
        self.mean_ = None
        self.std_ = None
        self.data_format_ = None
        self.original_shape_ = None
        self.resized_ = False 
        
        logger.info(
            f"Initialisation DataPreprocessor - "
            f"strategy: {strategy}, "
            f"auto_detect_format: {auto_detect_format}, "
            f"target_size: {target_size}"
        )
    def _detect_data_format(self, X: np.ndarray) -> str:
        """
        Détecte automatiquement le format des données.     
        Amélioration robustesse détection format    
        Returns:
            'channels_first' ou 'channels_last'      
        Raises:
            ValueError: Si le format est ambigu ou invalide
        """
        if X.ndim != 4:
            logger.warning(f"⚠️ Dimensions invalides pour détection format: {X.ndim}D, fallback channels_last")
            return 'channels_last'
            
        # Règles de détection robustes
        n_samples, dim1, dim2, dim3 = X.shape
        
        # Validation: dimensions doivent être cohérentes (pas toutes identiques)
        if dim1 == dim2 == dim3:
            logger.warning(
                f"⚠️ Format ambigu: toutes dimensions identiques ({dim1}). "
                f"Impossible de déterminer channels_first/last. Fallback channels_last."
            )
            return 'channels_last'
        
        # Cas channels_last: (N, H, W, C) où C est petit (1,3,4)
        # Validation: C doit être significativement plus petit que H et W
        if dim3 in [1, 3, 4] and dim3 < dim1 and dim3 < dim2:
            # Double validation: vérifier que dim1 et dim2 sont similaires (hauteur/largeur)
            if abs(dim1 - dim2) / max(dim1, dim2) < 0.5:  # Ratio H/W acceptable
                logger.debug(f"✅ Format détecté: channels_last (shape={X.shape})")
                return 'channels_last'
        
        # Cas channels_first: (N, C, H, W) où C est petit (1,3,4)
        # Validation: C doit être significativement plus petit que H et W
        if dim1 in [1, 3, 4] and dim1 < dim2 and dim1 < dim3:
            # Double validation: vérifier que dim2 et dim3 sont similaires (hauteur/largeur)
            if abs(dim2 - dim3) / max(dim2, dim3) < 0.5:  # Ratio H/W acceptable
                logger.debug(f"✅ Format détecté: channels_first (shape={X.shape})")
                return 'channels_first'
        
        # Heuristique finale: si la première dimension est significativement plus petite
        if dim1 < min(dim2, dim3) * 0.1:  # dim1 < 10% de min(dim2, dim3)
            logger.debug(f"✅ Format détecté (heuristique): channels_first (dim1={dim1} << {min(dim2, dim3)})")
            return 'channels_first'
        elif dim3 < min(dim1, dim2) * 0.1:  # dim3 < 10% de min(dim1, dim2)
            logger.debug(f"✅ Format détecté (heuristique): channels_last (dim3={dim3} << {min(dim1, dim2)})")
            return 'channels_last'
        else:
            # Format ambigu: utiliser la dimension la plus petite
            if dim1 < dim3:
                logger.warning(
                    f"⚠️ Format ambigu (shape={X.shape}), utilisation heuristique: channels_first "
                    f"(dim1={dim1} < dim3={dim3})"
                )
                return 'channels_first'
            else:
                logger.warning(
                    f"⚠️ Format ambigu (shape={X.shape}), utilisation heuristique: channels_last "
                    f"(dim3={dim3} < dim1={dim1})"
                )
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
            f"Preprocessing fitted - "
            f"original_shape: {self.original_shape_}, "
            f"data_format: {self.data_format_}, "
            f"strategy: {self.strategy}, "
            f"mean_shape: {getattr(self.mean_, 'shape', None)}, "
            f"std_shape: {getattr(self.std_, 'shape', None)}"
        )
        
        return self
    
    def transform(
        self,
        X: np.ndarray,
        output_format: str = "channels_first"
    ) -> np.ndarray:
        """
        Transform cohérent avec fit_transform.    
        IMPORTANT: Applique le MÊME pipeline que fit_transform
        """
        if not self.fitted:
            raise ValueError("Preprocessor non fitted. Appelez fit() ou fit_transform() d'abord.")
        
        if X is None or len(X) == 0:
            raise ValueError("X est None ou vide")
        
        logger.debug(f"🔄 Transform: input_shape={X.shape}, target_size={self.target_size}")
        
        # 1. RESIZE (si appliqué pendant fit)
        # _resize_images() détecte automatiquement le format de X
        if self.target_size is not None:
            X = self._resize_images(X)
            logger.debug(f"✅ Après resize: {X.shape}")
        
        # 2. Normalisation
        X_normalized = self._normalize(X, fit=False)
        
        # 3. Conversion format
        # Détection format ACTUEL de X (peut différer de self.data_format_ si données différentes)
        current_format = self._detect_data_format(X_normalized)
        
        if output_format == "channels_first" and current_format == "channels_last":
            logger.debug("🔄 Conversion channels_last → channels_first")
            X_normalized = np.transpose(X_normalized, (0, 3, 1, 2))
        elif output_format == "channels_last" and current_format == "channels_first":
            logger.debug("🔄 Conversion channels_first → channels_last")
            X_normalized = np.transpose(X_normalized, (0, 2, 3, 1))
        
        logger.debug(f"✅ Transform terminé: output_shape={X_normalized.shape}")
        
        return X_normalized
    
    def _resize_images(self, X: np.ndarray) -> np.ndarray:
        """
        Resize images avec détection format LOCALE    
        Args:
            X: Images (N, H, W, C) ou (N, C, H, W)     
        Returns:
            Images resized (même format que input)     
        Raises:
            ValueError: Si format invalide
        """
        if self.target_size is None:
            return X  # Pas de resize
        
        if X.ndim != 4:
            raise ValueError(f"_resize_images attend 4D, reçu: {X.shape}")
        
        target_h, target_w = self.target_size
        
        # Détection format local
        n_samples, dim1, dim2, dim3 = X.shape
        
        # Détection robuste
        if dim3 in [1, 3, 4] and dim3 < dim1 and dim3 < dim2:
            # Format: (N, H, W, C) - channels_last
            current_format = "channels_last"
            current_h, current_w = dim1, dim2
        elif dim1 in [1, 3, 4] and dim1 < dim2 and dim1 < dim3:
            # Format: (N, C, H, W) - channels_first
            current_format = "channels_first"
            current_h, current_w = dim2, dim3
        else:
            # Format ambigu: utiliser heuristique (plus petite dimension = channels)
            if dim1 < dim3:
                current_format = "channels_first"
                current_h, current_w = dim2, dim3
            else:
                current_format = "channels_last"
                current_h, current_w = dim1, dim2
            
            logger.warning(
                f"⚠️ Format ambigu dans _resize_images: {X.shape}, "
                f"assume {current_format}"
            )
        
        # Si déjà à la bonne taille, skip
        if current_h == target_h and current_w == target_w:
            logger.debug(f"Images déjà à la taille cible {self.target_size}, skip resize")
            return X
        
        logger.info(
            f"🔄 Resize images: ({current_h}, {current_w}) → ({target_h}, {target_w}) "
            f"[format détecté: {current_format}]"
        )
        
        try:
            from skimage.transform import resize as sk_resize
            
            resized_images = []
            
            for i in range(len(X)):
                img = X[i]
                
                # Conversion temporaire en channels_last pour skimage (attend H, W, C)
                if current_format == "channels_first":
                    # (C, H, W) → (H, W, C)
                    img = np.transpose(img, (1, 2, 0))
                
                # Resize avec preservation du range
                img_resized = sk_resize(
                    img,
                    (target_h, target_w),
                    mode='reflect',
                    anti_aliasing=True,
                    preserve_range=True
                )
                
                # Reconversion dans le format d'origine
                if current_format == "channels_first":
                    # (H, W, C) → (C, H, W)
                    img_resized = np.transpose(img_resized, (2, 0, 1))
                
                resized_images.append(img_resized)
            
            X_resized = np.array(resized_images, dtype=X.dtype)
            
            logger.info(
                f"✅ Resize complété: {X.shape} → {X_resized.shape}"
            )
            
            self.resized_ = True
            return X_resized
        
        except ImportError as e:
            logger.error(f"❌ skimage non disponible: {e}")
            raise ImportError(
                "scikit-image requis pour resize. Installez avec: pip install scikit-image"
            ) from e
        
        except Exception as e:
            logger.error(f"❌ Erreur resize: {e}", exc_info=True)
            raise ValueError(f"Resize échoué: {str(e)}") from e

    def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Normalise les données selon la stratégie.     
        Args:
            X: Données à normaliser
            fit: Si True, calcule les statistiques     
        Returns:
            Données normalisées
        """
        if self.strategy == "standardize":
            if fit:
                # Calcule mean/std sur format channels_last pour cohérence
                if self.data_format_ == 'channels_first':
                    X_for_stats = self._ensure_channels_last(X)
                else:
                    X_for_stats = X
                
                if X_for_stats.ndim == 4:
                    axes = (0, 1, 2)
                    self.mean_ = X_for_stats.mean(axis=axes, keepdims=True)
                    self.std_ = X_for_stats.std(axis=axes, keepdims=True) + 1e-8
                else:
                    self.mean_ = X_for_stats.mean()
                    self.std_ = X_for_stats.std() + 1e-8
            
            # Application de la normalisation
            if self.data_format_ == 'channels_first':
                X_norm = self._ensure_channels_last(X)
                X_norm = (X_norm - self.mean_) / self.std_
                return self._ensure_channels_first(X_norm)
            else:
                return (X - self.mean_) / self.std_
        
        elif self.strategy == "normalize":
            if fit:
                if self.data_format_ == 'channels_first':
                    X_for_stats = self._ensure_channels_last(X)
                else:
                    X_for_stats = X
                
                self.min_ = X_for_stats.min()
                self.max_ = X_for_stats.max()
            
            # Application normalisation [0, 1]
            if self.data_format_ == 'channels_first':
                X_norm = self._ensure_channels_last(X)
                X_norm = (X_norm - self.min_) / (self.max_ - self.min_ + 1e-8)
                return self._ensure_channels_first(X_norm)
            else:
                return (X - self.min_) / (self.max_ - self.min_ + 1e-8)      
        else:  # "none"
            return X.copy()
    
    def fit_transform(
        self,
        X: np.ndarray,
        output_format: str = "channels_first"
    ) -> np.ndarray:
        """
        Ordre des opérations clarifié.      
        Pipeline:
        1. Détection format d'origine
        2. Resize (si target_size) AVANT normalisation
        3. Calcul statistiques (mean/std) sur données resizées
        4. Normalisation
        5. Conversion vers output_format
        """
        if X is None or len(X) == 0:
            raise ValueError("X est None ou vide")
        
        # 1. Détection format AVANT tout traitement
        if self.auto_detect_format:
            self.data_format_ = self._detect_data_format(X)
            logger.info(f"✅ Format détecté: {self.data_format_} (shape={X.shape})")
        else:
            self.data_format_ = "channels_last"
            logger.info(f"⚙️ Format forcé: {self.data_format_}")
        
        self.original_shape_ = X.shape
        
        # 2. RESIZE AVANT normalisation
        # _resize_images() utilise maintenant détection LOCALE indépendante
        if self.target_size is not None:
            logger.info(f"🔧 Application resize: target_size={self.target_size}")
            X = self._resize_images(X)
            logger.info(f"✅ Après resize: {X.shape}")
        
        # 3-4. Normalisation (avec statistiques calculées sur données RESIZÉES)
        X_normalized = self._normalize(X, fit=True)
        
        # 5. Conversion vers output_format
        if output_format == "channels_first" and self.data_format_ == "channels_last":
            logger.debug("🔄 Conversion channels_last → channels_first")
            X_normalized = np.transpose(X_normalized, (0, 3, 1, 2))
        elif output_format == "channels_last" and self.data_format_ == "channels_first":
            logger.debug("🔄 Conversion channels_first → channels_last")
            X_normalized = np.transpose(X_normalized, (0, 2, 3, 1))
        
        self.fitted = True
        
        logger.info(
            f"✅ Preprocessing fitted - "
            f"original_shape: {self.original_shape_}, "
            f"resized: {self.resized_}, "
            f"target_size: {self.target_size}, "
            f"data_format_detected: {self.data_format_}, "
            f"output_shape: {X_normalized.shape}, "
            f"strategy: {self.strategy}"
        )
        
        return X_normalized
    
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
                f"Validation {name} OK - "
                f"shape: {X.shape}, "
                f"n_samples: {len(X)}, "
                f"n_classes: {len(unique_classes)}, "
                f"class_distribution: {dict(class_counts)}"
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
                f"DataLoader créé avec succès - "
                f"n_samples: {len(dataset)}, "
                f"input_shape: {X.shape}, "
                f"tensor_shape: {X_tensor.shape}, "
                f"n_batches: {len(loader)}, "
                f"batch_size: {batch_size}"
            )
            
            return loader
            
        except Exception as e:
            logger.error(
                f"Erreur création DataLoader: {str(e)} - "
                f"X_shape: {getattr(X, 'shape', None)}, "
                f"y_shape: {getattr(y, 'shape', None)}, "
                f"batch_size: {batch_size}",
                exc_info=True
            )
            raise