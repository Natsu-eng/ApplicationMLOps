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
from typing import Dict, Any, List, Literal, Optional, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.shared.logging import get_logger
from utils.device_manager import DeviceManager

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
    Preprocessor unifié avec support complet normalize/standardize/imagenet.
    
    PRINCIPE CLÉ:
    - strategy="normalize" : [0,255] → [0,1]
    - strategy="standardize" : [0,255] → z-score (propres stats)
    - strategy="imagenet" : [0,255] → [0,1] (ImageNet norm DANS le modèle)
    
    RÈGLE D'OR:
    - Transfer Learning → strategy="imagenet" (pas de norm ici, sera dans forward())
    - Autoencoders → strategy="normalize" (FORCÉ)
    - CNN Custom → strategy="standardize" OU "normalize"
    """
    
    def __init__(
        self,
        strategy: Literal["normalize", "standardize", "imagenet"] = "normalize",
        auto_detect_format: bool = True,
        target_size: Optional[Tuple[int, int]] = None
    ):
        """
        Args:
            strategy: Stratégie de normalisation
                - "normalize": [0,1] simple
                - "standardize": z-score avec stats propres
                - "imagenet": [0,1] + flag pour ImageNet norm dans modèle
            target_size: Taille cible pour resize
        """
        # Validation strategy
        valid_strategies = ["normalize", "standardize", "imagenet"]
        if strategy not in valid_strategies:
            raise ValueError(
                f"strategy '{strategy}' invalide. "
                f"Valeurs acceptées: {valid_strategies}"
            )
        
        self.strategy = strategy
        self.auto_detect_format = auto_detect_format
        self.target_size = target_size
        
        # État après fit
        self.fitted = False
        self.mean_ = None
        self.std_ = None
        self.min_ = None
        self.max_ = None
        self.data_format_ = None
        self.original_shape_ = None
        self.resized_ = False
        
        logger.info(
            f"✅ DataPreprocessor initialisé - "
            f"strategy: {strategy}, "
            f"target_size: {target_size}"
        )
    
    def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Normalise selon la stratégie.
        Args:
            X: Données à normaliser
            fit: Si True, calcule les stats (mean/std ou min/max)
        Returns:
            X normalisé
        """
        if self.strategy == "normalize":
            # [0, 255] → [0, 1] avec min/max du dataset
            if fit:
                self.min_ = float(X.min())
                self.max_ = float(X.max())
            
            denominator = self.max_ - self.min_
            if denominator < 1e-8:
                logger.warning("⚠️ Range trop petit, pas de normalisation")
                return X.astype(np.float32)
            
            X_norm = (X - self.min_) / (denominator + 1e-8)
            X_norm = np.clip(X_norm, 0, 1)
            
            logger.debug(f"✅ Normalize: [{X.min():.1f}, {X.max():.1f}] → [0, 1]")
            return X_norm.astype(np.float32)
        
        elif self.strategy == "standardize":
            # Z-score avec stats propres
            if fit:
                X_for_stats = X
                if self.data_format_ == 'channels_first':
                    X_for_stats = np.transpose(X, (0, 2, 3, 1))
                
                if X_for_stats.ndim == 4:
                    axes = (0, 1, 2)
                    self.mean_ = X_for_stats.mean(axis=axes, keepdims=True)
                    self.std_ = X_for_stats.std(axis=axes, keepdims=True) + 1e-8
                else:
                    self.mean_ = X_for_stats.mean()
                    self.std_ = X_for_stats.std() + 1e-8
            
            X_for_norm = X
            if self.data_format_ == 'channels_first':
                X_for_norm = np.transpose(X, (0, 2, 3, 1))
            
            X_norm = (X_for_norm - self.mean_) / self.std_
            
            if self.data_format_ == 'channels_first':
                X_norm = np.transpose(X_norm, (0, 3, 1, 2))
            
            logger.debug(f"✅ Standardize: mean={self.mean_.mean():.3f}, std={self.std_.mean():.3f}")
            return X_norm.astype(np.float32)
        
        elif self.strategy == "imagenet":
            """
            PRÉPARATION ImageNet: [0, 255] → [0, 1] UNIQUEMENT
            La normalisation (mean=0.485/0.456/0.406, std=0.229/0.224/0.225)
            est appliquée DANS le modèle (TransferLearningModel.forward())
            """
            # Conversion [0, 255] → [0, 1]
            if X.max() > 1.0:
                X_norm = X.astype(np.float32) / 255.0
            else:
                X_norm = X.astype(np.float32)
            
            X_norm = np.clip(X_norm, 0, 1)
            
            logger.debug(
                f"✅ ImageNet prep: [{X.min():.1f}, {X.max():.1f}] → "
                f"[{X_norm.min():.3f}, {X_norm.max():.3f}] "
                f"(normalisation ImageNet dans le modèle)"
            )
            return X_norm
        
        else:
            # Fallback
            return X.astype(np.float32)
    
    def fit_transform(
        self,
        X: np.ndarray,
        output_format: str = "channels_first"
    ) -> np.ndarray:
        """
        Pipeline complet avec normalisation adaptée.
        
        GARANTIT:
        - Détection format initial
        - Resize si activé
        - Normalisation selon strategy
        - Conversion vers output_format
        - Validation finale
        """
        if X is None or len(X) == 0:
            raise ValueError("X est None ou vide")
        
        logger.info(
            f"fit_transform START - "
            f"input_shape: {X.shape}, "
            f"strategy: {self.strategy}, "
            f"output_format: {output_format}"
        )
        
        # 1. Détection format
        if self.auto_detect_format:
            self.data_format_ = self._detect_data_format(X)
        else:
            self.data_format_ = "channels_last"
        
        self.original_shape_ = X.shape
        
        # 2. Resize si activé
        if self.target_size is not None:
            X = self._resize_images(X)
            # Re-détecter format après resize
            self.data_format_ = self._detect_data_format(X)
        
        # 3. Normalisation selon strategy
        X_normalized = self._normalize(X, fit=True)
        
        # 4. Conversion format
        current_format = self._detect_data_format(X_normalized)
        
        if output_format == "channels_first" and current_format == "channels_last":
            X_output = np.transpose(X_normalized, (0, 3, 1, 2))
        elif output_format == "channels_last" and current_format == "channels_first":
            X_output = np.transpose(X_normalized, (0, 2, 3, 1))
        else:
            X_output = X_normalized
        
        # 5. Validation finale
        final_format = self._detect_data_format(X_output)
        if final_format != output_format:
            raise ValueError(
                f"❌ FORMAT FINAL INCORRECT: {final_format} != {output_format}"
            )
        
        self.fitted = True
        
        logger.info(
            f"✅ fit_transform TERMINÉ - "
            f"output_shape: {X_output.shape}, "
            f"strategy: {self.strategy}, "
            f"format: {final_format}, "
            f"range: [{X_output.min():.3f}, {X_output.max():.3f}]"
        )
        
        return X_output
    
    def transform(
        self,
        X: np.ndarray,
        output_format: str = "channels_first"
    ) -> np.ndarray:
        """Transform avec même pipeline que fit_transform"""
        if not self.fitted:
            raise ValueError("Preprocessor non fitted")
        
        # 1. Resize si activé
        if self.target_size is not None:
            X = self._resize_images(X)
        
        # 2. Normalisation (utilise stats de fit)
        X_normalized = self._normalize(X, fit=False)
        
        # 3. Conversion format
        current_format = self._detect_data_format(X_normalized)
        
        if output_format == "channels_first" and current_format == "channels_last":
            X_output = np.transpose(X_normalized, (0, 3, 1, 2))
        elif output_format == "channels_last" and current_format == "channels_first":
            X_output = np.transpose(X_normalized, (0, 2, 3, 1))
        else:
            X_output = X_normalized
        
        return X_output
    
    def _detect_data_format(self, X: np.ndarray) -> str:
        """Détection format robuste"""
        if X.ndim != 4:
            return 'channels_last'
        
        n, dim1, dim2, dim3 = X.shape
        
        # channels_last: (N, H, W, C)
        if dim3 in [1, 3, 4] and dim3 < dim1 and dim3 < dim2:
            return 'channels_last'
        
        # channels_first: (N, C, H, W)
        if dim1 in [1, 3, 4] and dim1 < dim2 and dim1 < dim3:
            return 'channels_first'
        
        # Heuristique
        if dim1 < min(dim2, dim3) * 0.1:
            return 'channels_first'
        elif dim3 < min(dim1, dim2) * 0.1:
            return 'channels_last'
        else:
            return 'channels_last'
    
    def _resize_images(self, X: np.ndarray) -> np.ndarray:
        """Resize avec détection format locale"""
        from skimage.transform import resize as sk_resize
        
        if self.target_size is None:
            return X
        
        target_h, target_w = self.target_size
        
        # Détection format local
        shape = X.shape
        if shape[-1] in [1, 3, 4]:
            current_format = "channels_last"
            current_h, current_w = shape[1], shape[2]
        else:
            current_format = "channels_first"
            current_h, current_w = shape[2], shape[3]
        
        if current_h == target_h and current_w == target_w:
            return X
        
        logger.info(f"🔄 Resize: ({current_h}, {current_w}) → ({target_h}, {target_w})")
        
        resized = []
        for img in X:
            # Conversion temporaire channels_last
            if current_format == "channels_first":
                img = np.transpose(img, (1, 2, 0))
            
            img_resized = sk_resize(
                img, (target_h, target_w),
                mode='reflect',
                anti_aliasing=True,
                preserve_range=True
            )
            
            # Reconversion format original
            if current_format == "channels_first":
                img_resized = np.transpose(img_resized, (2, 0, 1))
            
            resized.append(img_resized)
        
        self.resized_ = True
        return np.array(resized, dtype=X.dtype)
    
    def get_config(self) -> dict:
        """Config pour serialisation"""
        return {
            "strategy": self.strategy,
            "target_size": self.target_size,
            "fitted": self.fitted,
            "data_format": self.data_format_,
            "resized": self.resized_
        }
    
# ===================================
# AUTOENCODER PREPROCESSOR
# ===================================
class AutoencoderPreprocessor:
    """
    Preprocessing spécialisé pour autoencoders.
    
    PRINCIPES CRITIQUES:
    1. Normalisation [0,1] globale UNIQUEMENT
    2. PAS de Z-score (évite compression anomalies)
    3. PAS de normalisation ImageNet
    4. Préserve distribution relative
    """
    
    def __init__(self, target_size: Optional[Tuple[int, int]] = None):
        self.target_size = target_size
        self.fitted = False
        self.global_min = None
        self.global_max = None
        self.data_format_ = None
        self.original_shape_ = None
        
        logger.info(
            f"AutoencoderPreprocessor initialisé - "
            f"target_size: {target_size}"
        )
    
    def fit(self, X: np.ndarray) -> 'AutoencoderPreprocessor':
        """Calcule min/max globaux (pas par canal)"""
        if X.ndim not in [3, 4]:
            raise ValueError(f"Dimensions invalides: {X.ndim}")
        
        self.original_shape_ = X.shape
        
        # Détection format
        self.data_format_ = self._detect_format(X)
        
        # Stats GLOBALES (préserve contrastes)
        self.global_min = float(X.min())
        self.global_max = float(X.max())
        
        self.fitted = True
        
        logger.info(
            f"✅ AutoencoderPreprocessor fitted:\n"
            f"   • Global min: {self.global_min}\n"
            f"   • Global max: {self.global_max}\n"
            f"   • Format: {self.data_format_}\n"
            f"   • Shape: {self.original_shape_}"
        )
        
        return self
    
    def transform(self, X: np.ndarray, output_format: str = "channels_first") -> np.ndarray:
        """Normalisation [0,1] + resize"""
        if not self.fitted:
            raise ValueError("Preprocessor non fitted")
        
        # 1. Resize si nécessaire
        if self.target_size:
            X = self._resize(X)
        
        # 2. Normalisation [0,1] simple
        X_norm = (X - self.global_min) / (self.global_max - self.global_min + 1e-8)
        X_norm = np.clip(X_norm, 0, 1)
        
        # 3. Conversion format
        if output_format == "channels_first":
            current_format = self._detect_format(X_norm)
            if current_format == "channels_last":
                X_norm = np.transpose(X_norm, (0, 3, 1, 2))
        
        logger.debug(
            f"✅ Transform AE: "
            f"range=[{X_norm.min():.3f}, {X_norm.max():.3f}], "
            f"shape={X_norm.shape}"
        )
        
        return X_norm.astype(np.float32)
    
    def fit_transform(self, X: np.ndarray, output_format: str = "channels_first") -> np.ndarray:
        return self.fit(X).transform(X, output_format)
    
    def _resize(self, X: np.ndarray) -> np.ndarray:
        """Resize avec skimage"""
        from skimage.transform import resize as sk_resize
        
        resized = []
        for img in X:
            # Conversion temporaire channels_last si nécessaire
            if img.shape[0] in [1, 3, 4]:  # channels_first
                img = np.transpose(img, (1, 2, 0))
            
            img_resized = sk_resize(
                img, self.target_size,
                mode='reflect',
                anti_aliasing=True,
                preserve_range=True
            )
            
            # Reconversion si nécessaire
            if self.data_format_ == "channels_first":
                img_resized = np.transpose(img_resized, (2, 0, 1))
            
            resized.append(img_resized)
        
        return np.array(resized, dtype=X.dtype)
    
    def _detect_format(self, X: np.ndarray) -> str:
        """Détection format (réutilise logique DataPreprocessor)"""
        if X.ndim != 4:
            return 'channels_last'
        
        _, dim1, dim2, dim3 = X.shape
        
        if dim3 in [1, 3, 4] and dim3 < dim1 and dim3 < dim2:
            return 'channels_last'
        elif dim1 in [1, 3, 4] and dim1 < dim2 and dim1 < dim3:
            return 'channels_first'
        else:
            return 'channels_last'
    
    def get_config(self) -> Dict[str, Any]:
        """Config pour serialisation"""
        return {
            "preprocessor_type": "AutoencoderPreprocessor",
            "target_size": self.target_size,
            "global_min": self.global_min,
            "global_max": self.global_max,
            "fitted": self.fitted
        }
    
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
    """
    Validateur centralisé pour toutes les données.
    
    ✅ VALIDATION STRICTE:
    - Shapes cohérentes
    - Types corrects
    - Labels [0, n_classes-1]
    - Ranges valides
    """
    
    @staticmethod
    def validate_input_data(
        X: np.ndarray,
        y: Optional[np.ndarray],
        split_name: str
    ) -> Dict[str, Any]:
        """
        Valide les données d'entrée.
        
        Returns:
            Dict avec {'success': bool, 'error': Optional[str]}
        """
        try:
            # Validation X
            if X is None or len(X) == 0:
                return {
                    'success': False,
                    'error': f"{split_name}: X est vide"
                }
            
            if X.ndim not in [3, 4]:
                return {
                    'success': False,
                    'error': f"{split_name}: X doit être 3D ou 4D, reçu: {X.ndim}D"
                }
            
            # Validation y (si fourni)
            if y is not None:
                if len(y) != len(X):
                    return {
                        'success': False,
                        'error': f"{split_name}: len(X)={len(X)} != len(y)={len(y)}"
                    }
                
                # ✅ VALIDATION CRITIQUE: Labels commencent à 0
                label_min = int(y.min())
                label_max = int(y.max())
                
                if label_min != 0:
                    return {
                        'success': False,
                        'error': (
                            f"{split_name}: Labels invalides - min={label_min}\n"
                            f"Les labels doivent commencer à 0.\n"
                            f"Labels présents: {np.unique(y).tolist()}"
                        )
                    }
                
                n_classes = len(np.unique(y))
                
                if label_max != (n_classes - 1):
                    return {
                        'success': False,
                        'error': (
                            f"{split_name}: Labels non continus - "
                            f"max={label_max}, n_classes={n_classes}"
                        )
                    }
                
                logger.info(
                    f"✅ {split_name} validé: "
                    f"n_samples={len(X)}, n_classes={n_classes}, "
                    f"labels=[{label_min}, {label_max}]"
                )
            
            return {'success': True, 'error': None}
            
        except Exception as e:
            return {
                'success': False,
                'error': f"{split_name}: Erreur validation - {str(e)}"
            }
    
    @staticmethod
    def check_class_imbalance(y: np.ndarray) -> Dict[str, Any]:
        """
        Analyse le déséquilibre des classes.
        
        Returns:
            Dict avec severity, ratio, distribution
        """
        try:
            unique, counts = np.unique(y, return_counts=True)
            distribution = dict(zip(unique.tolist(), counts.tolist()))
            
            max_count = counts.max()
            min_count = counts.min()
            ratio = max_count / min_count if min_count > 0 else float('inf')
            
            # Classification sévérité
            if ratio < 1.5:
                severity = "low"
            elif ratio < 3:
                severity = "medium"
            elif ratio < 10:
                severity = "high"
            else:
                severity = "critical"
            
            return {
                'severity': severity,
                'ratio': float(ratio),
                'distribution': distribution,
                'min_count': int(min_count),
                'max_count': int(max_count)
            }
            
        except Exception as e:
            logger.error(f"Erreur analyse déséquilibre: {e}")
            return {
                'severity': 'unknown',
                'ratio': 1.0,
                'distribution': {},
                'error': str(e)
            }


import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class AugmentedImageDataset(Dataset):
    """
    Dataset avec augmentation Albumentations ROBUSTE.
    
    ✅ CORRECTIONS APPLIQUÉES:
    - Validation stricte des labels (min=0)
    - Préservation dtype int64
    - Logs détaillés pour debugging
    - Gestion d'erreurs avec fallback
    
    🎯 GARANTIES:
    - Labels JAMAIS modifiés (int64 préservé)
    - Augmentation UNIQUEMENT sur images
    - Synchronisation image/label parfaite
    - Pas de biais introduit
    """
    
    def __init__(
        self,
        images: np.ndarray,
        labels: Optional[np.ndarray],
        transform: Optional[A.Compose] = None,
        preprocessor: Optional['DataPreprocessor'] = None,
        output_format: str = "channels_first"
    ):
        """
        Args:
            images: Images BRUTES uint8 [0, 255] - shape (N, H, W, C)
            labels: Labels (N,) - DOIT être np.ndarray avec dtype int64
            transform: Pipeline Albumentations
            preprocessor: Pour normalisation POST-augmentation
            output_format: Format de sortie (channels_first / channels_last)
        """
        # ========================================================================
        # VALIDATION CRITIQUE DES LABELS
        # ========================================================================
        if labels is not None:
            # Type check
            if not isinstance(labels, np.ndarray):
                raise TypeError(
                    f"❌ labels doit être np.ndarray, reçu: {type(labels)}"
                )
            
            # Dtype check et conversion si nécessaire
            if labels.dtype != np.int64:
                logger.warning(
                    f"⚠️ Labels dtype={labels.dtype} → conversion int64 "
                    f"(requis pour PyTorch CrossEntropyLoss)"
                )
                labels = labels.astype(np.int64)
            
            # Range validation
            label_min = int(labels.min())
            label_max = int(labels.max())
            unique_labels = np.unique(labels)
            
            # ❌ ERREUR CRITIQUE: Labels ne commencent pas à 0
            if label_min != 0:
                raise ValueError(
                    f"❌ LABELS INVALIDES: min(labels)={label_min}\n"
                    f"   Les labels DOIVENT commencer à 0 pour CrossEntropyLoss.\n"
                    f"   Labels présents: {unique_labels}\n"
                    f"   \n"
                    f"   💡 SOLUTION:\n"
                    f"   1. Vérifier l'encodage des labels en amont\n"
                    f"   2. Utiliser LabelEncoder().fit_transform()\n"
                    f"   3. Réencoder manuellement: labels = labels - labels.min()"
                )
            
            # Log validation OK
            logger.info(
                f"✅ Labels validés dans AugmentedImageDataset:\n"
                f"   • dtype: {labels.dtype}\n"
                f"   • range: [{label_min}, {label_max}]\n"
                f"   • unique: {unique_labels.tolist()}\n"
                f"   • shape: {labels.shape}"
            )
        
        # ========================================================================
        # VALIDATION DES IMAGES
        # ========================================================================
        if images.dtype != np.uint8:
            raise TypeError(
                f"❌ Images doivent être uint8 [0,255], reçu: {images.dtype}\n"
                f"   Range actuel: [{images.min()}, {images.max()}]"
            )
        
        if images.ndim != 4:
            raise ValueError(
                f"❌ Images doivent être 4D (N,H,W,C), reçu: {images.shape}"
            )
        
        # ========================================================================
        # STOCKAGE
        # ========================================================================
        self.images = images
        self.labels = labels  # ✅ Stockage DIRECT sans conversion
        self.transform = transform
        self.preprocessor = preprocessor
        self.output_format = output_format
        
        # Stats pour logging
        self.n_samples = len(images)
        self.has_augmentation = transform is not None
        self.has_preprocessor = preprocessor is not None
        
        logger.info(
            f"✅ AugmentedImageDataset créé:\n"
            f"   • Samples: {self.n_samples}\n"
            f"   • Image shape: {images.shape[1:]}\n"
            f"   • Has labels: {labels is not None}\n"
            f"   • Has augmentation: {self.has_augmentation}\n"
            f"   • Has preprocessor: {self.has_preprocessor}\n"
            f"   • Output format: {output_format}"
        )
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx: int):
        """
        Récupère un échantillon avec augmentation.
        
        ✅ GARANTIES:
        - Labels JAMAIS modifiés
        - dtype int64 préservé
        - Augmentation UNIQUEMENT sur image
        - Synchronisation parfaite
        """
        # ========================================================================
        # 1. RÉCUPÉRATION IMAGE
        # ========================================================================
        image = self.images[idx]  # uint8 [0, 255]
        
        # ========================================================================
        # 2. AUGMENTATION (si activée)
        # ========================================================================
        if self.transform is not None:
            try:
                # Albumentations applique les transformations
                augmented = self.transform(image=image)
                image = augmented['image']
                
                # Validation post-augmentation
                if image.dtype != np.uint8:
                    logger.warning(
                        f"⚠️ Augmentation a changé dtype: {image.dtype}. "
                        f"Reconversion uint8."
                    )
                    image = image.astype(np.uint8)
                
            except Exception as e:
                logger.error(
                    f"❌ Erreur augmentation idx={idx}: {e}\n"
                    f"   Fallback: image originale utilisée"
                )
                # Fallback: image originale
                image = self.images[idx]
        
        # ========================================================================
        # 3. CONVERSION FLOAT32 [0, 1]
        # ========================================================================
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # ========================================================================
        # 4. NORMALISATION (si preprocessor fourni)
        # ========================================================================
        if self.preprocessor is not None:
            try:
                # Ajouter dimension batch temporaire
                image_batch = np.expand_dims(image, axis=0)
                
                # Transform (normalisation)
                image_normalized = self.preprocessor.transform(
                    image_batch,
                    output_format=self.output_format
                )
                
                # Retirer dimension batch
                image = image_normalized[0]
                
            except Exception as e:
                logger.error(
                    f"❌ Erreur normalisation idx={idx}: {e}\n"
                    f"   Fallback: conversion manuelle"
                )
                # Fallback: conversion manuelle
                if self.output_format == "channels_first":
                    image = np.transpose(image, (2, 0, 1))
        else:
            # Pas de preprocessor: conversion manuelle
            if self.output_format == "channels_first":
                image = np.transpose(image, (2, 0, 1))
        
        # ========================================================================
        # 5. CONVERSION TENSOR
        # ========================================================================
        image_tensor = torch.from_numpy(image).float()
        
        # ========================================================================
        # 6. RÉCUPÉRATION LABEL (CRITIQUE)
        # ========================================================================
        if self.labels is not None:
            label = self.labels[idx]
            
            # ✅ VALIDATION RUNTIME (debug mode)
            if not isinstance(label, (int, np.integer)):
                logger.error(
                    f"❌ Label idx={idx} type invalide: {type(label)}\n"
                    f"   Valeur: {label}\n"
                    f"   Labels dtype: {self.labels.dtype}"
                )
                raise TypeError(
                    f"Label doit être int/np.integer, reçu: {type(label)}"
                )
            
            # Vérification range (debug)
            label_value = int(label)
            if label_value < 0:
                raise ValueError(
                    f"❌ Label négatif détecté: idx={idx}, label={label_value}"
                )
            
            # ✅ CONVERSION TORCH.LONG (int64) SANS MODIFICATION
            label_tensor = torch.tensor(label_value, dtype=torch.long)
            
            return image_tensor, label_tensor
        
        else:
            # Mode non supervisé: dummy label
            return image_tensor, torch.tensor(0, dtype=torch.long)


# ============================================================================
#  DataLoaderFactory avec target_size 
# ============================================================================

class DataLoaderFactory:
    """
    Factory pour créer des DataLoaders robustes.
    
    ✅ CORRECTIONS APPLIQUÉES:
    - Validation labels AVANT création dataset
    - Forçage dtype int64
    - Logs détaillés
    - Gestion erreurs
    """
    
    @staticmethod
    def create(
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        augmentation_config: Optional[Dict[str, Any]] = None,
        is_training: bool = True,
        target_size: Optional[Tuple[int, int]] = None,
        output_format: str = "channels_first",
        device_manager: Optional[Any] = None,
        preprocessor: Optional['DataPreprocessor'] = None
    ) -> DataLoader:
        """
        Crée un DataLoader avec validation complète.
        
        ✅ VALIDATION STRICTE:
        - Labels min=0 vérifiés
        - dtype int64 garanti
        - Augmentation UNIQUEMENT si is_training=True
        - Logs détaillés pour debugging
        
        Args:
            X: Images (N, H, W, C) ou (N, C, H, W)
            y: Labels (N,) - DOIT commencer à 0
            batch_size: Taille des batchs
            shuffle: Mélanger les données
            num_workers: Workers DataLoader
            pin_memory: Pin memory GPU
            augmentation_config: Config augmentation
            is_training: Mode training (active augmentation)
            target_size: Taille cible (H, W)
            output_format: Format sortie
            device_manager: Gestionnaire device
            preprocessor: Preprocessor optionnel
        
        Returns:
            DataLoader configuré
        
        Raises:
            ValueError: Si labels invalides (min != 0)
        """
        logger.info(
            f"🔧 Création DataLoader:\n"
            f"   • X shape: {X.shape}\n"
            f"   • y shape: {y.shape if y is not None else 'None'}\n"
            f"   • Batch size: {batch_size}\n"
            f"   • Augmentation: {augmentation_config is not None and is_training}\n"
            f"   • Target size: {target_size}\n"
            f"   • Is training: {is_training}"
        )
        
        # ========================================================================
        # VALIDATION CRITIQUE DES LABELS
        # ========================================================================
        if y is not None:
            # Forcer dtype int64
            if y.dtype != np.int64:
                logger.warning(
                    f"⚠️ Labels dtype={y.dtype} → conversion int64 "
                    f"(requis pour CrossEntropyLoss)"
                )
                y = y.astype(np.int64)
            
            # Validation range
            label_min = int(y.min())
            label_max = int(y.max())
            unique_labels = np.unique(y)
            n_classes = len(unique_labels)
            
            # ❌ ERREUR CRITIQUE: Labels ne commencent pas à 0
            if label_min != 0:
                raise ValueError(
                    f"❌ LABELS INVALIDES AVANT DATALOADER:\n"
                    f"   • min(y) = {label_min} (DOIT être 0)\n"
                    f"   • max(y) = {label_max}\n"
                    f"   • Labels présents: {unique_labels.tolist()}\n"
                    f"   • Nombre de classes: {n_classes}\n"
                    f"   \n"
                    f"   💡 SOLUTION:\n"
                    f"   Vérifier l'encodage des labels EN AMONT du DataLoader.\n"
                    f"   Les labels doivent être [0, 1, 2, ..., n_classes-1]"
                )
            
            # ✅ VALIDATION OK
            logger.info(
                f"✅ Labels validés avant DataLoader:\n"
                f"   • dtype: {y.dtype}\n"
                f"   • range: [{label_min}, {label_max}]\n"
                f"   • unique: {unique_labels.tolist()}\n"
                f"   • n_classes: {n_classes}"
            )
        
        # ========================================================================
        # DÉTECTION AUGMENTATION
        # ========================================================================
        augmentation_enabled = False
        augmentation_pipeline = None
        
        if augmentation_config and is_training:
            augmentation_enabled = augmentation_config.get('augmentation_enabled', False)
            
            if augmentation_enabled:
                logger.info("🎨 Configuration augmentation de données")
                
                # ✅ PIPELINE STANDARD SANS BIAIS
                augmentation_pipeline = A.Compose([
                    # Flips (réalistes)
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.3),
                    
                    # Rotation limitée (pas de 180°)
                    A.Rotate(limit=15, p=0.5),
                    
                    # Ajustements luminosité/contraste (modérés)
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=0.2,
                        p=0.3
                    ),
                    
                    # Bruit gaussien léger
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                    
                    # Flou léger
                    A.Blur(blur_limit=3, p=0.2)
                ])
                
                logger.info(
                    f"✅ Pipeline augmentation créé:\n"
                    f"   • HorizontalFlip: p=0.5\n"
                    f"   • VerticalFlip: p=0.3\n"
                    f"   • Rotate: ±15°, p=0.5\n"
                    f"   • BrightnessContrast: ±20%, p=0.3\n"
                    f"   • GaussNoise: p=0.2\n"
                    f"   • Blur: p=0.2"
                )
        
        # ========================================================================
        # CRÉATION DATASET
        # ========================================================================
        if augmentation_enabled:
            logger.info("🎨 Création AugmentedImageDataset")
            
            # ✅ DATASET AVEC AUGMENTATION
            dataset = AugmentedImageDataset(
                images=X,
                labels=y,  # ✅ Déjà validé ci-dessus
                transform=augmentation_pipeline,
                preprocessor=preprocessor,
                output_format=output_format
            )
            
        else:
            logger.info("📦 Création TensorDataset standard")
            
            # ✅ DATASET STANDARD (pas d'augmentation)
            X_tensor = torch.from_numpy(X).float()
            
            if y is not None:
                y_tensor = torch.from_numpy(y).long()  # ✅ int64
                dataset = TensorDataset(X_tensor, y_tensor)
            else:
                # Mode non supervisé
                dummy_labels = torch.zeros(len(X), dtype=torch.long)
                dataset = TensorDataset(X_tensor, dummy_labels)
        
        # ========================================================================
        # CRÉATION DATALOADER
        # ========================================================================
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=num_workers > 0
        )
        
        logger.info(
            f"✅ DataLoader créé avec succès:\n"
            f"   • Batches: {len(dataloader)}\n"
            f"   • Batch size: {batch_size}\n"
            f"   • Augmentation: {augmentation_enabled}\n"
            f"   • Shuffle: {shuffle}\n"
            f"   • Workers: {num_workers}"
        )
        
        return dataloader
