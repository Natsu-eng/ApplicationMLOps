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
from typing import Dict, Any, List, Optional, Tuple

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
        Détection robuste du format avec heuristiques améliorées.     
        Returns:
            'channels_first' ou 'channels_last'
        """
        if X.ndim != 4:
            return 'channels_last'  # Fallback pour 2D/3D
        
        n_samples, dim1, dim2, dim3 = X.shape
        
        # Règle 1: Dernière dimension < les deux autres → channels_last
        if dim3 < dim1 and dim3 < dim2 and dim3 in [1, 3, 4]:
            # Vérifier que dim1 et dim2 sont proches (hauteur/largeur)
            if abs(dim1 - dim2) / max(dim1, dim2) < 0.3:  # Ratio H/W < 30%
                logger.debug(f"✅ channels_last détecté: dim3={dim3} (C)")
                return 'channels_last'
        
        # Règle 2: Première dimension < les deux autres → channels_first
        if dim1 < dim2 and dim1 < dim3 and dim1 in [1, 3, 4]:
            # Vérifier que dim2 et dim3 sont proches (hauteur/largeur)
            if abs(dim2 - dim3) / max(dim2, dim3) < 0.3:  # Ratio H/W < 30%
                logger.debug(f"✅ channels_first détecté: dim1={dim1} (C)")
                return 'channels_first'
        
        # Règle 3: Heuristique par différence relative
        ratio_dim1 = min(dim2, dim3) / max(dim2, dim3)
        ratio_dim3 = min(dim1, dim2) / max(dim1, dim2)
        
        if ratio_dim1 > 0.9 and dim1 in [1, 3, 4]:  # dim1 est petit et dim2≈dim3
            logger.debug(f"✅ channels_first (heuristique): dim1={dim1}, ratio={ratio_dim1:.2f}")
            return 'channels_first'
        
        if ratio_dim3 > 0.9 and dim3 in [1, 3, 4]:  # dim3 est petit et dim1≈dim2
            logger.debug(f"✅ channels_last (heuristique): dim3={dim3}, ratio={ratio_dim3:.2f}")
            return 'channels_last'
        
        # Règle 4: Fallback basé sur convention commune
        logger.warning(
            f"⚠️ Format ambigu: shape={X.shape}. "
            f"Fallback sur 'channels_last' (convention PIL/OpenCV)"
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
        Transform avec EXACTEMENT le même pipeline que fit_transform().
        
        Pipeline identique:
        1. Détection format actuel
        2. Resize (si activé pendant fit)
        3. Recalcul format après resize
        4. Normalisation (utilise stats de fit)
        5. Conversion vers output_format
        6. Validation finale
        
        Args:
            X: Images à transformer
            output_format: Format de sortie
            
        Returns:
            Images transformées avec format GARANTI
            
        Raises:
            ValueError: Si preprocessor non fitted ou conversion échoue
        """
        if not self.fitted:
            raise ValueError(
                "Preprocessor non fitted. "
                "Appelez fit() ou fit_transform() d'abord."
            )
        
        if X is None or len(X) == 0:
            raise ValueError("X est None ou vide")
        
        logger.debug(f"🔄 Transform START - input_shape: {X.shape}, output_format: {output_format}")
        
        # ========================================================================
        # ÉTAPE 1 : DÉTECTION FORMAT ACTUEL
        # ========================================================================
        current_format = self._detect_data_format(X)
        logger.debug(f"🔍 Format INITIAL: {current_format} (shape={X.shape})")
        
        # ========================================================================
        # ÉTAPE 2 : RESIZE (SI APPLIQUÉ PENDANT FIT)
        # ========================================================================
        if self.target_size is not None:
            logger.debug(f"🔧 Application resize: {X.shape} → target_size={self.target_size}")
            X_resized = self._resize_images(X)
            
            # ⚠️ RE-DÉTECTER le format après resize
            format_after_resize = self._detect_data_format(X_resized)
            
            if format_after_resize != current_format:
                logger.debug(
                    f"⚠️ Format changé après resize: "
                    f"{current_format} → {format_after_resize}"
                )
                current_format = format_after_resize
            
            X = X_resized
            logger.debug(f"✅ Après resize: shape={X.shape}, format={current_format}")
        
        # ========================================================================
        # ÉTAPE 3 : NORMALISATION (UTILISE STATS DE FIT)
        # ========================================================================
        X_normalized = self._normalize(X, fit=False)
        logger.debug(f"✅ Après normalisation: shape={X_normalized.shape}")
        
        # ========================================================================
        # ÉTAPE 4 : CONVERSION VERS output_format
        # ========================================================================
        current_format = self._detect_data_format(X_normalized)
        logger.debug(f"🔍 Format AVANT conversion: {current_format}")
        
        if output_format == "channels_first" and current_format == "channels_last":
            logger.debug("🔄 Conversion channels_last → channels_first")
            X_converted = np.transpose(X_normalized, (0, 3, 1, 2))
            
            # VALIDATION
            expected_shape = (X_normalized.shape[0], X_normalized.shape[3], 
                            X_normalized.shape[1], X_normalized.shape[2])
            
            if X_converted.shape != expected_shape:
                raise ValueError(
                    f"❌ CONVERSION ÉCHOUÉE:\n"
                    f"   Attendu: {expected_shape}\n"
                    f"   Obtenu:  {X_converted.shape}"
                )
            
            X_output = X_converted
            
        elif output_format == "channels_last" and current_format == "channels_first":
            logger.debug("🔄 Conversion channels_first → channels_last")
            X_converted = np.transpose(X_normalized, (0, 2, 3, 1))
            
            # VALIDATION
            expected_shape = (X_normalized.shape[0], X_normalized.shape[2], 
                            X_normalized.shape[3], X_normalized.shape[1])
            
            if X_converted.shape != expected_shape:
                raise ValueError(
                    f"❌ CONVERSION ÉCHOUÉE:\n"
                    f"   Attendu: {expected_shape}\n"
                    f"   Obtenu:  {X_converted.shape}"
                )
            
            X_output = X_converted
        
        else:
            logger.debug(f"✅ Pas de conversion nécessaire (format: {current_format})")
            X_output = X_normalized
        
        # ========================================================================
        # ÉTAPE 5 : VALIDATION FINALE
        # ========================================================================
        final_format = self._detect_data_format(X_output)
        
        if final_format != output_format:
            raise ValueError(
                f"❌ FORMAT FINAL INCORRECT:\n"
                f"   Demandé:  {output_format}\n"
                f"   Obtenu:   {final_format}\n"
                f"   Shape:    {X_output.shape}"
            )
        
        # ASSERTION FORMAT
        if output_format == "channels_first":
            if X_output.shape[1] not in [1, 3, 4]:
                raise ValueError(
                    f"❌ FORMAT channels_first INVALIDE: "
                    f"shape={X_output.shape}, canaux={X_output.shape[1]}"
                )
            logger.debug(
                f"✅ VALIDATION - channels_first: "
                f"shape={X_output.shape}, canaux={X_output.shape[1]}"
            )
        
        elif output_format == "channels_last":
            if X_output.shape[3] not in [1, 3, 4]:
                raise ValueError(
                    f"❌ FORMAT channels_last INVALIDE: "
                    f"shape={X_output.shape}, canaux={X_output.shape[3]}"
                )
            logger.debug(
                f"✅ VALIDATION - channels_last: "
                f"shape={X_output.shape}, canaux={X_output.shape[3]}"
            )
        
        logger.debug(f"✅ transform TERMINÉ - output_shape: {X_output.shape}")
        
        return X_output
    
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
        Pipeline de preprocessing avec VALIDATION STRICTE du format.
        
        Pipeline garanti:
        1. Détection format initial
        2. Resize (si activé)
        3. Recalcul format après resize (CRITIQUE)
        4. Normalisation
        5. Conversion vers output_format avec VALIDATION
        6. Assertion finale shape
        
        Args:
            X: Images (N, H, W, C) ou (N, C, H, W)
            output_format: Format de sortie ("channels_first" ou "channels_last")
            
        Returns:
            Images preprocessées avec format GARANTI
            
        Raises:
            ValueError: Si conversion échoue
        """
        if X is None or len(X) == 0:
            raise ValueError("X est None ou vide")
        
        logger.info(f"fit_transform START - input_shape: {X.shape}, output_format: {output_format}")
        
        # ========================================================================
        # ÉTAPE 1 : DÉTECTION FORMAT INITIAL
        # ========================================================================
        if self.auto_detect_format:
            self.data_format_ = self._detect_data_format(X)
            logger.info(f"Format INITIAL détecté: {self.data_format_} (shape={X.shape})")
        else:
            self.data_format_ = "channels_last"
            logger.info(f"⚙️ Format INITIAL forcé: {self.data_format_}")
        
        self.original_shape_ = X.shape
        
        # ========================================================================
        # ÉTAPE 2 : RESIZE (SI ACTIVÉ)
        # ========================================================================
        if self.target_size is not None:
            logger.info(f"🔧 Application resize: {X.shape} → target_size={self.target_size}")
            X_resized = self._resize_images(X)
            
            # ⚠️ CRITIQUE : RE-DÉTECTER le format APRÈS resize
            # Car _resize_images() peut changer le format en interne
            format_after_resize = self._detect_data_format(X_resized)
            
            if format_after_resize != self.data_format_:
                logger.warning(
                    f"⚠️ Format CHANGÉ après resize: "
                    f"{self.data_format_} → {format_after_resize}"
                )
                self.data_format_ = format_after_resize
            
            X = X_resized
            logger.info(f"Après resize: shape={X.shape}, format={self.data_format_}")
        
        # ========================================================================
        # ÉTAPE 3 : NORMALISATION
        # ========================================================================
        X_normalized = self._normalize(X, fit=True)
        logger.debug(f"Après normalisation: shape={X_normalized.shape}")
        
        # ========================================================================
        # ÉTAPE 4 : CONVERSION VERS output_format AVEC VALIDATION
        # ========================================================================
        current_format = self._detect_data_format(X_normalized)
        logger.debug(f"🔍 Format AVANT conversion: {current_format}")
        
        if output_format == "channels_first" and current_format == "channels_last":
            logger.info("🔄 Conversion channels_last → channels_first")
            X_converted = np.transpose(X_normalized, (0, 3, 1, 2))
            
            # VALIDATION CRITIQUE
            expected_shape = (X_normalized.shape[0], X_normalized.shape[3], 
                            X_normalized.shape[1], X_normalized.shape[2])
            
            if X_converted.shape != expected_shape:
                raise ValueError(
                    f"❌ CONVERSION ÉCHOUÉE:\n"
                    f"   Attendu: {expected_shape}\n"
                    f"   Obtenu:  {X_converted.shape}"
                )
            
            X_output = X_converted
            
        elif output_format == "channels_last" and current_format == "channels_first":
            logger.info("🔄 Conversion channels_first → channels_last")
            X_converted = np.transpose(X_normalized, (0, 2, 3, 1))
            
            # VALIDATION CRITIQUE
            expected_shape = (X_normalized.shape[0], X_normalized.shape[2], 
                            X_normalized.shape[3], X_normalized.shape[1])
            
            if X_converted.shape != expected_shape:
                raise ValueError(
                    f"❌ CONVERSION ÉCHOUÉE:\n"
                    f"   Attendu: {expected_shape}\n"
                    f"   Obtenu:  {X_converted.shape}"
                )
            
            X_output = X_converted
        
        else:
            # Pas de conversion nécessaire
            logger.debug(f"Pas de conversion (déjà au bon format: {current_format})")
            X_output = X_normalized
        
        # ========================================================================
        # ÉTAPE 5 : VALIDATION FINALE DU FORMAT
        # ========================================================================
        final_format = self._detect_data_format(X_output)
        
        if final_format != output_format:
            raise ValueError(
                f"❌ FORMAT FINAL INCORRECT:\n"
                f"   Demandé:  {output_format}\n"
                f"   Obtenu:   {final_format}\n"
                f"   Shape:    {X_output.shape}"
            )
        
        # ========================================================================
        # ÉTAPE 6 : ASSERTIONS FINALES
        # ========================================================================
        if output_format == "channels_first":
            # Format attendu: (N, C, H, W) avec C petit (1, 3, 4)
            if X_output.shape[1] not in [1, 3, 4]:
                raise ValueError(
                    f"❌ FORMAT channels_first INVALIDE:\n"
                    f"   Shape: {X_output.shape}\n"
                    f"   Dim[1] (canaux) devrait être 1, 3 ou 4, reçu: {X_output.shape[1]}"
                )
            
            logger.info(
                f"VALIDATION RÉUSSIE - channels_first: "
                f"shape={X_output.shape}, canaux={X_output.shape[1]}"
            )
        
        elif output_format == "channels_last":
            # Format attendu: (N, H, W, C) avec C petit (1, 3, 4)
            if X_output.shape[3] not in [1, 3, 4]:
                raise ValueError(
                    f"❌ FORMAT channels_last INVALIDE:\n"
                    f"   Shape: {X_output.shape}\n"
                    f"   Dim[3] (canaux) devrait être 1, 3 ou 4, reçu: {X_output.shape[3]}"
                )
            
            logger.info(
                f"✅ VALIDATION RÉUSSIE - channels_last: "
                f"shape={X_output.shape}, canaux={X_output.shape[3]}"
            )
        
        # ========================================================================
        # FINALISATION
        # ========================================================================
        self.fitted = True
        
        logger.info(
            f"✅ fit_transform TERMINÉ - "
            f"original_shape: {self.original_shape_}, "
            f"output_shape: {X_output.shape}, "
            f"resized: {self.resized_}, "
            f"target_size: {self.target_size}, "
            f"format_initial: {self.data_format_}, "
            f"format_final: {final_format}, "
            f"strategy: {self.strategy}"
        )
        
        return X_output
        
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


import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class AugmentedImageDataset(Dataset):
    """
    Dataset PyTorch avec augmentation RÉELLE + RESIZE GARANTI + NORMALISATION.
    
    Pipeline CORRIGÉ:
    1. Input: Données BRUTES uint8 [0, 255]
    2. Augmentation (Albumentations)
    3. Resize vers target_size
    4. Normalisation (via preprocessor)
    5. Conversion tensor channels_first
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_size: tuple,
        augmentation_config: Optional[Dict[str, Any]] = None,
        is_training: bool = True,
        preprocessor: Optional[Any] = None 
    ):
        """
        Args:
            preprocessor: DataPreprocessor pour normalisation POST-augmentation
        """
        if X.ndim != 4:
            raise ValueError(f"X doit être 4D (N,H,W,C), reçu: {X.shape}")
        
        if X.shape[-1] not in [1, 3, 4]:
            raise ValueError(f"Format channels_last attendu, reçu: {X.shape}")
        
        if not target_size or len(target_size) != 2:
            raise ValueError(f"target_size obligatoire (H,W), reçu: {target_size}")
        
        self.X = X
        self.y = y
        self.target_size = target_size
        self.is_training = is_training
        self.preprocessor = preprocessor  
        
        # Parse config
        self.augment = False
        self.transform = None
        
        if augmentation_config and is_training:
            self.augment = augmentation_config.get('augmentation_enabled', False)
            methods = augmentation_config.get('methods', [])
            
            if self.augment:
                self.transform = self._build_augmentation_pipeline(methods)
                logger.info(
                    f"✅ Augmentation activée - "
                    f"target_size: {target_size}, methods: {methods}"
                )
    
    def _build_augmentation_pipeline(self, methods: List[str]) -> A.Compose:
        """Pipeline Albumentations AVEC RESIZE FINAL FORCÉ."""
        transforms = []
        
        if 'flip' in methods:
            transforms.append(A.HorizontalFlip(p=0.5))
        
        if 'rotate' in methods:
            transforms.append(A.Rotate(limit=15, p=0.5, border_mode=0))
        
        if 'zoom' in methods:
            transforms.append(
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=0,
                    border_mode=0,
                    p=0.5
                )
            )
        
        if 'brightness' in methods:
            transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                )
            )
        
        # ✅ RESIZE GARANTI en dernière position
        transforms.append(
            A.Resize(
                height=self.target_size[0],
                width=self.target_size[1],
                interpolation=1,
                always_apply=True
            )
        )
        
        return A.Compose(transforms)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        img = self.X[idx]  # (H, W, C)
        label = self.y[idx]
        
        # STEP 1: S'assurer uint8 [0, 255] pour Albumentations
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        
        # STEP 2: Augmentation + Resize
        if self.augment and self.transform:
            try:
                augmented = self.transform(image=img)
                img = augmented['image']
                
                # Validation taille
                if img.shape[:2] != self.target_size:
                    logger.error(
                        f"❌ Taille incorrecte après augmentation: "
                        f"{img.shape[:2]} != {self.target_size}"
                    )
                    from skimage.transform import resize
                    img = resize(
                        img, self.target_size,
                        mode='reflect', anti_aliasing=True,
                        preserve_range=True
                    ).astype(np.uint8)
            
            except Exception as e:
                logger.error(f"❌ Erreur augmentation: {e}")
                from skimage.transform import resize
                img = resize(
                    img, self.target_size,
                    mode='reflect', anti_aliasing=True,
                    preserve_range=True
                ).astype(np.uint8)
        
        else:
            # Pas d'augmentation: resize direct si nécessaire
            if img.shape[:2] != self.target_size:
                from skimage.transform import resize
                img = resize(
                    img, self.target_size,
                    mode='reflect', anti_aliasing=True,
                    preserve_range=True
                ).astype(np.uint8)
        
        # STEP 3: NORMALISATION via preprocessor (POST-augmentation)
        if self.preprocessor is not None:
            # Reconversion float32 [0, 1] avant normalisation
            img = img.astype(np.float32) / 255.0
            
            # Appliquer la normalisation du preprocessor
            # (standardize: (x - mean) / std)
            img_batch = np.expand_dims(img, axis=0)  # (1, H, W, C)
            
            # ⚠️ Le preprocessor attend channels_last, applique sa logique, retourne channels_first
            try:
                img_normalized = self.preprocessor.transform(
                    img_batch,
                    output_format="channels_first"
                )
                img = img_normalized[0]  # (C, H, W)
            except Exception as e:
                logger.warning(f"⚠️ Normalisation preprocessor échouée: {e}, fallback /255")
                # Fallback: normalisation simple
                img = torch.from_numpy(img).float()
                if img.ndim == 3:
                    img = img.permute(2, 0, 1)  # HWC → CHW
        
        else:
            # Pas de preprocessor: normalisation simple [0, 1]
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img).float()
            if img.ndim == 3:
                img = img.permute(2, 0, 1)  # HWC → CHW
        
        label = torch.tensor(label, dtype=torch.long)
        
        # ASSERTION FINALE
        expected_shape = (3, self.target_size[0], self.target_size[1])
        if img.shape != expected_shape:
            raise RuntimeError(
                f"❌ Shape finale incorrecte: {img.shape} != {expected_shape}"
            )
        
        return img, label


# ============================================================================
#  DataLoaderFactory avec target_size 
# ============================================================================

class DataLoaderFactory:
    """
    Version CORRIGÉE avec target_size obligatoire pour augmentation
    """
    
    @staticmethod
    def create(
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
        num_workers: Optional[int] = None,
        pin_memory: bool = False,
        augmentation_config: Optional[Dict[str, Any]] = None,
        is_training: bool = True,
        preprocessor: Optional[Any] = None,
        target_size: Optional[tuple] = None,
        output_format: str = "channels_first",
        device_manager: Optional[DeviceManager] = None
    ) -> DataLoader:
        """
        Crée un DataLoader avec gestion ROBUSTE et conversion FORMAT UNIFIÉ.  
        PRINCIPE: Toutes les données sortent dans `output_format` (garanti).   
        Args:
            output_format: Format de sortie GARANTI ("channels_first" ou "channels_last")
            device_manager: Pour détection device (CPU/GPU/MPS)
        """
        try:
            from torch.utils.data import DataLoader, TensorDataset
            import os
            
            # Validation
            if X is None or len(X) == 0:
                raise ValueError("X vide")
            if y is None or len(y) == 0:
                raise ValueError("y vide")
            if len(X) != len(y):
                raise ValueError(f"X et y incompatibles: {len(X)} vs {len(y)}")
            
            logger.info(f"🔍 DataLoaderFactory - input_shape: {X.shape}, output_format: {output_format}")
            
            # ====================================================================
            # 1. DÉTECTION FORMAT INITIAL
            # ====================================================================
            def detect_format_static(arr: np.ndarray) -> str:
                """Détection format sans dépendre de DataPreprocessor."""
                if arr.ndim != 4:
                    return 'channels_last'
                
                n, dim1, dim2, dim3 = arr.shape
                
                # channels_last: (N, H, W, C) où C est petit
                if dim3 in [1, 3, 4] and dim3 < dim1 and dim3 < dim2:
                    return 'channels_last'
                
                # channels_first: (N, C, H, W) où C est petit
                if dim1 in [1, 3, 4] and dim1 < dim2 and dim1 < dim3:
                    return 'channels_first'
                
                # Heuristique finale
                if dim1 < min(dim2, dim3) * 0.1:
                    return 'channels_first'
                elif dim3 < min(dim1, dim2) * 0.1:
                    return 'channels_last'
                else:
                    return 'channels_last'  # fallback
            
            current_format = detect_format_static(X)
            logger.info(f"✅ Format détecté: {current_format} (shape={X.shape})")
            
            # ====================================================================
            # 2. NORMALISATION DU FORMAT (convertir vers output_format)
            # ====================================================================
            X_converted = X.copy()
            
            if current_format != output_format:
                if current_format == "channels_last" and output_format == "channels_first":
                    # (N, H, W, C) → (N, C, H, W)
                    X_converted = np.transpose(X, (0, 3, 1, 2))
                    logger.info(f"🔄 Conversion: channels_last → channels_first")
                    
                elif current_format == "channels_first" and output_format == "channels_last":
                    # (N, C, H, W) → (N, H, W, C)
                    X_converted = np.transpose(X, (0, 2, 3, 1))
                    logger.info(f"🔄 Conversion: channels_first → channels_last")
            
            # VALIDATION format après conversion
            if output_format == "channels_first":
                if X_converted.shape[1] not in [1, 3, 4]:
                    logger.warning(f"⚠️ Canaux suspects en channels_first: {X_converted.shape[1]}")
            else:  # channels_last
                if X_converted.shape[3] not in [1, 3, 4]:
                    logger.warning(f"⚠️ Canaux suspects en channels_last: {X_converted.shape[3]}")
            
            # ====================================================================
            # 3. CONFIGURATION AUGMENTATION
            # ====================================================================
            use_augmentation = (
                augmentation_config and 
                augmentation_config.get('augmentation_enabled', False) and 
                is_training
            )
            
            if use_augmentation:
                # VALIDATION CRITIQUE: target_size REQUIRED pour augmentation
                if not target_size:
                    raise ValueError(
                        "❌ ERREUR CRITIQUE: target_size REQUIS quand augmentation activée\n"
                        "Raison: Les augmentations (rotate, zoom) changent la taille des images,\n"
                        "        ce qui brise le batching PyTorch sans resize final.\n"
                        "Solution: Spécifiez target_size dans preprocessing_config ou model_config."
                    )
                
                # Augmentation nécessite channels_last pour Albumentations
                if output_format == "channels_first":
                    # Conversion temporaire pour augmentation
                    X_aug = np.transpose(X_converted, (0, 2, 3, 1))
                    logger.info(f"🔄 Conversion temporaire pour augmentation: channels_first → channels_last")
                else:
                    X_aug = X_converted
                
                dataset = AugmentedImageDataset(
                    X_aug, y,
                    target_size=target_size,
                    augmentation_config=augmentation_config,
                    is_training=True
                )
                
                logger.info(
                    f"✅ AugmentedImageDataset créé - "
                    f"samples: {len(dataset)}, "
                    f"augmentation: {use_augmentation}, "
                    f"target_size: {target_size}"
                )
            
            else:
                # Pas d'augmentation: utiliser TensorDataset simple
                X_tensor = torch.tensor(X_converted, dtype=torch.float32)
                y_tensor = torch.tensor(y, dtype=torch.long)
                
                # Si device_manager fourni, déplacer sur device
                if device_manager and device_manager.device != torch.device('cpu'):
                    X_tensor = X_tensor.to(device_manager.device)
                    y_tensor = y_tensor.to(device_manager.device)
                
                dataset = TensorDataset(X_tensor, y_tensor)
                
                logger.info(
                    f"✅ TensorDataset créé - "
                    f"samples: {len(dataset)}, "
                    f"shape: {X_tensor.shape}, "
                    f"format: {output_format}, "
                    f"dtype: {X_tensor.dtype}"
                )
            
            # ====================================================================
            # 4. CONFIGURATION DATALOADER (optimisée production)
            # ====================================================================
            # num_workers dynamique basé sur système
            if num_workers is None:
                import os
                if os.cpu_count():
                    num_workers = min(4, os.cpu_count() // 2)
                else:
                    num_workers = 0
                logger.info(f"⚙️ num_workers auto-configuré: {num_workers}")
            
            # pin_memory seulement si CUDA disponible
            if pin_memory and not torch.cuda.is_available():
                pin_memory = False
                logger.info("ℹ️ pin_memory désactivé (CUDA non disponible)")
            
            loader = DataLoader(
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
                f"   • Batches: {len(loader)}\n"
                f"   • Batch size: {batch_size}\n"
                f"   • Augmentation: {use_augmentation}\n"
                f"   • Format initial: {current_format}\n"
                f"   • Format final: {output_format}\n"
                f"   • num_workers: {num_workers}\n"
                f"   • pin_memory: {pin_memory}"
            )
            
            return loader
            
        except Exception as e:
            logger.error(f"❌ Erreur création DataLoader: {e}", exc_info=True)
            raise