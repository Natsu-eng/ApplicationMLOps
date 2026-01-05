"""
Trainer principal pour Computer Vision Supervisé.
Version 1.0 - Séparé pour éviter les imports circulaires
"""

import time
import warnings
import copy
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from sklearn.utils.class_weight import compute_class_weight

from src.config.model_config import ModelConfig, ModelType
from src.config.training_config import TrainingConfig, OptimizerType, SchedulerType
from src.data.computer_vision_preprocessing import AutoencoderPreprocessor, DataLoaderFactory, DataPreprocessor, DataValidator, Result
from src.shared.logging import get_logger
from utils.callbacks import TrainingCallback
from utils.device_manager import DeviceManager

warnings.filterwarnings('ignore', category=UserWarning)
logger = get_logger(__name__)


# =============================
# OPTIMIZER & SCHEDULER FACTORY
# =============================

class OptimizerFactory:
    """Factory pour créer des optimiseurs avec configuration robuste"""
    
    @staticmethod
    def create(
        model: nn.Module,
        config: TrainingConfig
    ) -> optim.Optimizer:
        """
        Crée un optimiseur selon la config.
        """
        if config.optimizer == OptimizerType.ADAMW:
            return optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif config.optimizer == OptimizerType.ADAM:
            return optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif config.optimizer == OptimizerType.SGD:
            return optim.SGD(
                model.parameters(),
                lr=config.learning_rate,
                momentum=0.9,
                weight_decay=config.weight_decay,
                nesterov=True
            )
        elif config.optimizer == OptimizerType.RMSPROP:
            return optim.RMSprop(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                alpha=0.99,
                eps=1e-8
            )
        else:
            raise ValueError(f"Optimiseur non supporté: {config.optimizer}")


class SchedulerFactory:
    """Factory pour créer des schedulers de learning rate"""
    
    @staticmethod
    def create(
        optimizer: optim.Optimizer,
        config: TrainingConfig
    ) -> Optional[optim.lr_scheduler._LRScheduler]:
        """
        Crée un scheduler selon la config.
        """
        if config.scheduler == SchedulerType.NONE:
            return None
        
        elif config.scheduler == SchedulerType.REDUCE_ON_PLATEAU:
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=config.reduce_lr_patience,
                min_lr=config.min_lr,
                verbose=True,
                threshold=1e-4,
                threshold_mode='rel'
            )
        
        elif config.scheduler == SchedulerType.COSINE:
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.epochs,
                eta_min=config.min_lr
            )
        
        elif config.scheduler == SchedulerType.STEP:
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=max(config.epochs // 3, 1),
                gamma=0.1
            )
        
        else:
            raise ValueError(f"Scheduler non supporté: {config.scheduler}")


# ==================
# TRAINER PRINCIPAL
# ==================

class ComputerVisionTrainer:
    """
    Trainer principal pour Computer Vision Supervisé.
    
    Architecture garantissant:
    - Pas de fuite de données (fit sur train uniquement)
    - Gestion robuste des formats (channels_first/last)
    - Métriques métier fiables
    - Extensibilité via callbacks
    - Observabilité complète via logging
    
    Usage:
        trainer = ComputerVisionTrainer(model_config, training_config)
        result = trainer.fit(X_train, y_train, X_val, y_val)
        
        if result.success:
            model = result.data['model']
            history = result.data['history']
            
            # Prédictions
            pred_result = trainer.predict(X_test)
            predictions = pred_result.data['predictions']
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        callbacks: Optional[List[TrainingCallback]] = None,
        device_manager: Optional[DeviceManager] = None
    ):
        """
        Initialise le trainer.
        
        Args:
            model_config: Configuration du modèle
            training_config: Configuration d'entraînement
            callbacks: Liste de callbacks optionnels
            device_manager: Gestionnaire de device (CPU/GPU)
        """
        self.model_config = model_config
        self.training_config = training_config
        self.callbacks = callbacks or []
        self.device_manager = device_manager or DeviceManager()
        
        # État interne
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        self.preprocessor: Optional[DataPreprocessor] = None
        self.train_criterion: Optional[nn.Module] = None
        self.val_criterion: Optional[nn.Module] = None
        
        # Historique (structure propre garantie)
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': [],
            'learning_rates': []
        }
        
        # Métadonnées d'entraînement
        self._training_metadata: Dict[str, Any] = {}
        
        # Setup déterminisme si demandé
        if training_config.deterministic:
            self._set_deterministic(training_config.seed)
    
    def _set_deterministic(self, seed: int) -> None:
        """
        Active le mode déterministe pour reproductibilité.
        
        Args:
            seed: Graine aléatoire
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        logger.info(f"Mode déterministe activé: seed={seed}")
    
    
    def fit(
            self,
            X_train: np.ndarray,
            y_train: Optional[np.ndarray],
            X_val: np.ndarray,
            y_val: Optional[np.ndarray],
            preprocessing_config: Optional[Dict[str, Any]] = None
        ) -> Result:
            """
            Entraîne le modèle sur les données fournies.
            AUGMENTATION:
            - Stockage X_train/X_val BRUTS (uint8 [0,255]) pour augmentation
            - Preprocessing standard sur validation (pas d'augmentation)
            - Preprocessor utilisé DANS AugmentedImageDataset pour normalisation POST-augmentation
            """
            try:
                logger.info("=== DÉBUT ENTRAÎNEMENT ===")
                start_total = time.time()
            
                if preprocessing_config:
                    # Fusionne avec config existante (priorité au paramètre)
                    if hasattr(self, 'preprocessing_config') and self.preprocessing_config:
                        self.preprocessing_config.update(preprocessing_config)
                    else:
                        self.preprocessing_config = preprocessing_config
                elif not hasattr(self, 'preprocessing_config') or not self.preprocessing_config:
                    # Fallback seulement si aucune config
                    self.preprocessing_config = {}
            
                logger.info(
                    f"Preprocessing config chargée: "
                    f"augmentation_enabled={self.preprocessing_config.get('augmentation_enabled', False)}, "
                    f"target_size={self.preprocessing_config.get('target_size')}"
                )
            
                # 1. VALIDATION DES DONNÉES
                val_result = self._validate_data(X_train, y_train, X_val, y_val)
                if not val_result.success:
                    return val_result
            
                # 2. DÉTERMINATION PARAMÈTRES DYNAMIQUES
                if hasattr(self.model_config, 'input_channels'):
                    expected_channels = self.model_config.input_channels
                else:
                    expected_channels = X_train.shape[-1] if X_train.shape[-1] in [1, 3, 4] else 3
                    logger.info(f"ℹ️ input_channels auto-détecté: {expected_channels}")
            
                target_size = self._determine_target_size(
                    X_train,
                    preprocessing_config,
                    expected_channels
                )
            
                # ====================================================================
                # 3. VÉRIFICATION AUGMENTATION ET GESTION DONNÉES BRUTES
                # ====================================================================
                augmentation_enabled = self.preprocessing_config.get('augmentation_enabled', False)
            
                if augmentation_enabled:
                    logger.info("🎨 Augmentation ACTIVÉE - Conservation données BRUTES")
                
                    # Vérifier que X_train est bien uint8 [0, 255]
                    if X_train.dtype != np.uint8:
                        if X_train.max() <= 1.0:
                            logger.warning("⚠️ X_train normalisé détecté, reconversion uint8")
                            X_train_brut = (X_train * 255).clip(0, 255).astype(np.uint8)
                        elif X_train.min() < 0:
                            logger.warning("⚠️ X_train contient valeurs négatives, normalisation forcée")
                            X_train_brut = ((X_train - X_train.min()) / (X_train.max() - X_train.min() + 1e-8) * 255).astype(np.uint8)
                        else:
                            X_train_brut = X_train.clip(0, 255).astype(np.uint8)
                    else:
                        X_train_brut = X_train.copy()
                
                    logger.info(
                        f"✅ Données train brutes préparées: "
                        f"shape={X_train_brut.shape}, dtype={X_train_brut.dtype}, "
                        f"range=[{X_train_brut.min()}, {X_train_brut.max()}]"
                    )
                
                    # Validation AUSSI nécessaire en uint8
                    if X_val.dtype != np.uint8:
                        if X_val.max() <= 1.0:
                            X_val_brut = (X_val * 255).clip(0, 255).astype(np.uint8)
                        elif X_val.min() < 0:
                            X_val_brut = ((X_val - X_val.min()) / (X_val.max() - X_val.min() + 1e-8) * 255).astype(np.uint8)
                        else:
                            X_val_brut = X_val.clip(0, 255).astype(np.uint8)
                    else:
                        X_val_brut = X_val.copy()

                    # ====================================================================
                    # 🆕 BLOC DE DEBUG CRITIQUE + VÉRIFICATION TARGET_SIZE
                    # ====================================================================
                    logger.info(
                        f"🔍 VALIDATION AUGMENTATION:\n"
                        f" • preprocessing_config: {self.preprocessing_config}\n"
                        f" • augmentation_enabled: {augmentation_enabled}\n"
                        f" • target_size: {target_size}\n"
                        f" • X_train_brut dtype: {X_train_brut.dtype}\n"
                        f" • X_train_brut range: [{X_train_brut.min()}, {X_train_brut.max()}]"
                    )

                    if not target_size:
                        # ✅ ERREUR EXPLICITE au lieu de silence
                        raise ValueError(
                            "❌ ERREUR CRITIQUE: augmentation_enabled=True mais target_size=None\n"
                            "Cause probable: preprocessing_config mal propagé depuis UI.\n"
                            "Vérifier orchestrateur._execute_training() et trainer.fit() signature."
                        )
                    # ====================================================================

                else:
                    logger.info("ℹ️ Augmentation désactivée - Preprocessing standard")
                    X_train_brut = X_train
                    X_val_brut = X_val
            
                # ====================================================================
                # 4. PREPROCESSING STANDARD (uniquement pour données normalisées)
                # ====================================================================
                prep_result = self._setup_preprocessing(
                    X_train_brut, y_train, X_val_brut, y_val,
                    target_size=target_size,
                    expected_channels=expected_channels
                )
                if not prep_result.success:
                    return prep_result
            
                X_train_norm, y_train, X_val_norm, y_val = prep_result.data
            
                logger.info(
                    f"✅ Preprocessing terminé:\n"
                    f" • X_train_norm: {X_train_norm.shape}, dtype={X_train_norm.dtype}\n"
                    f" • X_val_norm: {X_val_norm.shape}, dtype={X_val_norm.dtype}\n"
                    f" • Preprocessor fitted sur données brutes"
                )
            
                # 5. CONSTRUCTION DU MODÈLE
                model_result = self._build_model()
                if not model_result.success:
                    return model_result
            
                # 6. SETUP TRAINING
                if y_train is not None:
                    setup_result = self._setup_training(y_train)
                    if not setup_result.success:
                        return setup_result
                else:
                    setup_result = self._setup_training_unsupervised()
                    if not setup_result.success:
                        return setup_result
            
                # ====================================================================
                # 7. CRÉATION DATALOADERS AVEC GESTION CORRECTE AUGMENTATION
                # ====================================================================
                if augmentation_enabled:
                    logger.info(
                        f"🎨 Création DataLoader AVEC augmentation:\n"
                        f" • Input: X_train_brut (uint8 [0,255])\n"
                        f" • Augmentation via Albumentations\n"
                        f" • Normalisation POST-augmentation dans dataset\n"
                        f" • Preprocessor: {self.preprocessor is not None}"
                    )
                
                    # TRAIN: Données BRUTES + Augmentation + Preprocessor interne
                    train_loader = DataLoaderFactory.create(
                        X_train_brut, # Données BRUTES uint8
                        y_train,
                        batch_size=self.training_config.batch_size,
                        shuffle=True,
                        num_workers=self.training_config.num_workers,
                        pin_memory=self.training_config.pin_memory,
                        augmentation_config=self.preprocessing_config,
                        is_training=True,
                        target_size=target_size,
                        output_format="channels_first",
                        device_manager=self.device_manager,
                        preprocessor=self.preprocessor
                    )
                
                    # VALIDATION: Données normalisées SANS augmentation
                    val_loader = DataLoaderFactory.create(
                        X_val_norm,
                        y_val,
                        batch_size=self.training_config.batch_size,
                        shuffle=False,
                        num_workers=self.training_config.num_workers,
                        pin_memory=self.training_config.pin_memory,
                        augmentation_config=None,
                        is_training=False,
                        target_size=target_size,
                        output_format="channels_first",
                        device_manager=self.device_manager
                    )
                else:
                    train_loader = DataLoaderFactory.create(
                        X_train_norm,
                        y_train,
                        batch_size=self.training_config.batch_size,
                        shuffle=True,
                        num_workers=self.training_config.num_workers,
                        pin_memory=self.training_config.pin_memory,
                        augmentation_config=None,
                        is_training=True,
                        target_size=target_size,
                        output_format="channels_first",
                        device_manager=self.device_manager
                    )
                
                    val_loader = DataLoaderFactory.create(
                        X_val_norm,
                        y_val,
                        batch_size=self.training_config.batch_size,
                        shuffle=False,
                        num_workers=self.training_config.num_workers,
                        pin_memory=self.training_config.pin_memory,
                        augmentation_config=None,
                        is_training=False,
                        target_size=target_size,
                        output_format="channels_first",
                        device_manager=self.device_manager
                    )
            
                logger.info(
                    f"DataLoaders créés:\n"
                    f" • Train: {len(train_loader)} batches, augmentation={augmentation_enabled}\n"
                    f" • Val: {len(val_loader)} batches\n"
                    f" • Target size: {target_size}\n"
                    f" • Format: channels_first"
                )
            
                # Le reste de la fonction reste inchangé...
                # (validation premier batch, boucle d'entraînement, etc.)

                # 8. VALIDATION PREMIER BATCH
                validation_result = self._validate_first_batch(
                    train_loader,
                    expected_channels=expected_channels,
                    target_size=target_size
                )
            
                if not validation_result['success']:
                    logger.error(f"❌ Validation premier batch échouée: {validation_result.error}")
                    return Result.err(
                        f"Problème format détecté:\n{validation_result.error}"
                    )
            
                # 9. BOUCLE D'ENTRAÎNEMENT
                train_result = self._training_loop(train_loader, val_loader, y_val)
                if not train_result.success:
                    return train_result
            
                # 10. MÉTADONNÉES FINALES
                total_time = time.time() - start_total
                self._training_metadata.update({
                    'total_training_time': total_time,
                    'samples_per_second': len(X_train) / total_time if total_time > 0 else 0,
                    'final_model_params': sum(p.numel() for p in self.model.parameters()),
                    'device': str(self.device_manager.device),
                    'augmentation_enabled': augmentation_enabled,
                    'target_size': target_size,
                    'input_channels': expected_channels,
                    'original_shapes': {
                        'train': X_train.shape,
                        'val': X_val.shape
                    }
                })
            
                # 11. RETOUR STRUCTURÉ
                result_data = self._build_training_result(train_result)
            
                logger.info(
                    f"🎯 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS:\n"
                    f" • Durée: {total_time:.1f}s\n"
                    f" • Époques: {len(self.history['train_loss'])}\n"
                    f" • Best epoch: {result_data['history']['best_epoch']}\n"
                    f" • Best val loss: {result_data['history']['best_val_loss']:.4f}\n"
                    f" • Augmentation: {augmentation_enabled}\n"
                    f" • Canaux: {expected_channels}"
                )
            
                return Result.ok(result_data, training_time=total_time)
            
            except Exception as e:
                logger.error(f"❌ Fit échoué: {e}", exc_info=True)
                return Result.err(f"Entraînement échoué: {str(e)}")

    def _determine_target_size(
        self,
        X_train: np.ndarray,
        preprocessing_config: Dict[str, Any],
        expected_channels: int
    ) -> Optional[Tuple[int, int]]:
        """
        Détermine dynamiquement la target_size à partir de multiples sources.
        
        Priorités:
        1. preprocessing_config.target_size (explicite)
        2. model_config.input_size (si différent des données)
        3. Déduction depuis données
        4. None (pas de resize)
        """
        target_size = None
        
        # Source 1: preprocessing_config explicite
        if preprocessing_config and 'target_size' in preprocessing_config:
            target_size = preprocessing_config['target_size']
            if target_size and len(target_size) == 2:
                logger.info(f"✅ Target_size depuis preprocessing_config: {target_size}")
                return tuple(target_size)
        
        # Source 2: model_config.input_size
        if hasattr(self.model_config, 'input_size') and self.model_config.input_size:
            model_h, model_w = self.model_config.input_size
            
            # Détecter taille actuelle des données
            data_format = self._detect_data_format_static(X_train)
            
            if data_format == "channels_last":
                data_h, data_w = X_train.shape[1], X_train.shape[2]
            else:  # channels_first
                data_h, data_w = X_train.shape[2], X_train.shape[3]
            
            # Si taille différente, utiliser model_config
            if (data_h, data_w) != (model_h, model_w):
                target_size = (model_h, model_w)
                logger.info(
                    f"✅ Target_size depuis model_config: {target_size} "
                    f"(data: {data_h}x{data_w} → model: {model_h}x{model_w})"
                )
                return target_size
            else:
                logger.info(
                    f"ℹ️ Données déjà à la taille modèle ({data_h}x{data_w}), "
                    f"pas de resize nécessaire"
                )
                return None
        
        # Source 3: Déduction depuis données (pour augmentation)
        augmentation_enabled = preprocessing_config.get('augmentation_enabled', False)
        
        if augmentation_enabled:
            # Pour augmentation, on DOIT avoir une target_size fixe
            data_format = self._detect_data_format_static(X_train)
            
            if data_format == "channels_last":
                data_h, data_w = X_train.shape[1], X_train.shape[2]
            else:  # channels_first
                data_h, data_w = X_train.shape[2], X_train.shape[3]
            
            # Utiliser la taille actuelle comme target (pas de resize)
            target_size = (data_h, data_w)
            logger.info(
                f"⚠️ Target_size déduit pour augmentation: {target_size} "
                f"(même que données originales)"
            )
            return target_size
        
        # Source 4: Pas de resize
        logger.info("ℹ️ Pas de target_size spécifié, conservation taille originale")
        return None


    def _validate_first_batch(
        self,
        train_loader: DataLoader,
        expected_channels: int,
        target_size: Optional[Tuple[int, int]] = None
    ) -> Dict[str, Any]:
        """
        Validation ULTRA-STRICTE du premier batch.      
        VALIDATIONS:
        1. Dimensions tensor (4D)
        2. Nombre de canaux
        3. Taille d'image (si target_size fourni)
        4. Range des valeurs
        5. Labels (CRITIQUE: min=0, max<num_classes)
        
        Returns:
            Dict avec {'success': bool, 'data': dict, 'error': Optional[str]}
        """
        try:
            # Récupérer premier batch
            first_batch = next(iter(train_loader))
            
            if len(first_batch) != 2:
                return {
                    'success': False,
                    'error': f"Batch invalide: {len(first_batch)} éléments au lieu de 2",
                    'data': None
                }
            
            data, labels = first_batch
            
            logger.info(
                f"🔍 Validation premier batch:\n"
                f"   • Data shape: {data.shape}\n"
                f"   • Labels shape: {labels.shape}\n"
                f"   • Data dtype: {data.dtype}\n"
                f"   • Labels dtype: {labels.dtype}"
            )
            
            # ====================================================================
            # VALIDATION #1 : DIMENSIONS TENSOR
            # ====================================================================
            if data.dim() != 4:
                return {
                    'success': False,
                    'error': (
                        f"❌ DIMENSIONS INVALIDES:\n"
                        f"   Attendu: 4D (B, C, H, W)\n"
                        f"   Reçu: {data.dim()}D\n"
                        f"   Shape: {data.shape}\n"
                        f"\n"
                        f"   💡 CAUSE PROBABLE:\n"
                        f"   - Format channels_first/last incorrect\n"
                        f"   - Preprocessing a échoué\n"
                        f"   - DataLoader mal configuré"
                    ),
                    'data': None
                }
            
            batch_size, channels, height, width = data.shape
            
            # ====================================================================
            # VALIDATION #2 : NOMBRE DE CANAUX
            # ====================================================================
            if channels != expected_channels:
                return {
                    'success': False,
                    'error': (
                        f"❌ CANAUX INCORRECTS:\n"
                        f"   Attendu: {expected_channels}\n"
                        f"   Reçu: {channels}\n"
                        f"   Shape: {data.shape}\n"
                        f"\n"
                        f"   💡 CAUSE PROBABLE:\n"
                        f"   - input_channels mal configuré dans model_config\n"
                        f"   - Conversion channels_first/last échouée\n"
                        f"   - Modèle incompatible avec ces données\n"
                        f"\n"
                        f"   🛠️ SOLUTION:\n"
                        f"   Vérifier model_config.input_channels et preprocessor.transform()"
                    ),
                    'data': None
                }
            
            # ====================================================================
            # VALIDATION #3 : TAILLE D'IMAGE
            # ====================================================================
            if target_size:
                expected_h, expected_w = target_size
                if (height, width) != (expected_h, expected_w):
                    return {
                        'success': False,
                        'error': (
                            f"❌ TAILLE D'IMAGE INCORRECTE:\n"
                            f"   Attendu: {expected_h}x{expected_w}\n"
                            f"   Reçu: {height}x{width}\n"
                            f"\n"
                            f"   💡 CAUSE PROBABLE:\n"
                            f"   - Resize non appliqué correctement\n"
                            f"   - Augmentation sans resize final\n"
                            f"   - Incohérence dans target_size\n"
                            f"\n"
                            f"   🛠️ SOLUTION:\n"
                            f"   Vérifier preprocessing_config.target_size"
                        ),
                        'data': None
                    }
            
            # ====================================================================
            # VALIDATION #4 : RANGE DES VALEURS
            # ====================================================================
            data_min = float(data.min())
            data_max = float(data.max())
            data_mean = float(data.mean())
            data_std = float(data.std())
            
            # Warning si range anormal
            if data_max > 10 or data_min < -10:
                logger.warning(
                    f"⚠️ Range de valeurs large: [{data_min:.3f}, {data_max:.3f}]\n"
                    f"   Normalisation peut ne pas être appliquée.\n"
                    f"   Mean: {data_mean:.3f}, Std: {data_std:.3f}"
                )
            
            # ====================================================================
            # VALIDATION #5 : LABELS (CRITIQUE)
            # ====================================================================
            if labels.dim() != 1:
                return {
                    'success': False,
                    'error': f"❌ Labels 1D attendus, reçu: {labels.dim()}D",
                    'data': None
                }
            
            if len(labels) != batch_size:
                return {
                    'success': False,
                    'error': (
                        f"❌ Incohérence batch/labels:\n"
                        f"   Batch size: {batch_size}\n"
                        f"   Labels length: {len(labels)}"
                    ),
                    'data': None
                }
            
            # Conversion labels pour analyse
            labels_np = labels.cpu().numpy()
            label_min = int(labels_np.min())
            label_max = int(labels_np.max())
            unique_labels = np.unique(labels_np).tolist()
            
            # ❌ ERREUR CRITIQUE: Labels ne commencent pas à 0
            if label_min != 0:
                return {
                    'success': False,
                    'error': (
                        f"❌ LABELS INVALIDES DANS LE BATCH:\n"
                        f"   • min(labels) = {label_min} (DOIT être 0)\n"
                        f"   • max(labels) = {label_max}\n"
                        f"   • Labels présents: {unique_labels}\n"
                        f"   • Labels dtype: {labels.dtype}\n"
                        f"\n"
                        f"   💡 CAUSE PROBABLE:\n"
                        f"   1. Encoding incorrect des labels en amont\n"
                        f"   2. Bug dans AugmentedImageDataset.__getitem__()\n"
                        f"   3. Problème de conversion dans DataLoader\n"
                        f"   4. Labels non réencodés après filtrage\n"
                        f"\n"
                        f"   🛠️ SOLUTION:\n"
                        f"   - Vérifier que y_train contient bien [0, 1, 2, ...]\n"
                        f"   - Utiliser LabelEncoder().fit_transform(y_train)\n"
                        f"   - Débugger AugmentedImageDataset ligne par ligne\n"
                        f"   - Ajouter logging dans __getitem__()\n"
                        f"\n"
                        f"   🔍 DEBUG:\n"
                        f"   Exécuter le script diagnostic_labels.py pour identifier\n"
                        f"   l'étape exacte où les labels sont corrompus"
                    ),
                    'data': None
                }
            
            # ❌ Labels hors limites du modèle
            if hasattr(self, 'model_config') and hasattr(self.model_config, 'num_classes'):
                model_num_classes = self.model_config.num_classes
                
                if label_max >= model_num_classes:
                    return {
                        'success': False,
                        'error': (
                            f"❌ LABEL HORS LIMITES:\n"
                            f"   • max(labels) = {label_max}\n"
                            f"   • model.num_classes = {model_num_classes}\n"
                            f"   • CrossEntropyLoss attend labels ∈ [0, {model_num_classes-1}]\n"
                            f"\n"
                            f"   💡 CAUSE PROBABLE:\n"
                            f"   - Incohérence entre num_classes du modèle et données\n"
                            f"   - Labels mal encodés (skip de certaines classes)\n"
                            f"\n"
                            f"   🛠️ SOLUTION:\n"
                            f"   Vérifier que model_config.num_classes = len(np.unique(y_train))"
                        ),
                        'data': None
                    }
            
            # ====================================================================
            # ✅ VALIDATION RÉUSSIE - LOG DÉTAILLÉ
            # ====================================================================
            logger.info(
                f"✅ PREMIER BATCH VALIDÉ:\n"
                f"   • Shape: {tuple(data.shape)}\n"
                f"   • Canaux: {channels} (attendus: {expected_channels}) ✓\n"
                f"   • Taille: {height}x{width}" + 
                (f" (cible: {target_size})" if target_size else "") + " ✓\n"
                f"   • Range: [{data_min:.3f}, {data_max:.3f}]\n"
                f"   • Mean: {data_mean:.3f}, Std: {data_std:.3f}\n"
                f"   • Labels: {labels.shape}\n"
                f"   • Labels range: [{label_min}, {label_max}] ✓\n"
                f"   • Labels uniques: {unique_labels}\n"
                f"   • Labels dtype: {labels.dtype}\n"
                f"   • Device: {data.device}"
            )
            
            return {
                'success': True,
                'data': {
                    "batch_shape": tuple(data.shape),
                    "channels": channels,
                    "height": height,
                    "width": width,
                    "data_range": (data_min, data_max),
                    "data_mean": data_mean,
                    "data_std": data_std,
                    "labels_shape": tuple(labels.shape),
                    "label_min": label_min,
                    "label_max": label_max,
                    "unique_labels": unique_labels,
                    "labels_dtype": str(labels.dtype),
                    "device": str(data.device),
                    "labels_valid": True
                },
                'error': None
            }
            
        except StopIteration:
            return {
                'success': False,
                'error': "❌ DataLoader vide - Aucun batch disponible",
                'data': None
            }
        except Exception as e:
            logger.error(f"❌ Erreur validation batch: {e}", exc_info=True)
            return {
                'success': False,
                'error': f"Validation batch échouée: {str(e)}",
                'data': None
            }

    
    def _build_training_result(self, train_result: Result) -> Dict[str, Any]:
        """
        Construit le résultat final avec structure garantie.
        
        CRITIQUE: Assure qu'aucun booléen n'est dans l'historique.
        """
        is_autoencoder = self.model_config.model_type in [
            ModelType.CONV_AUTOENCODER, ModelType.VAE, ModelType.DENOISING_AE
        ]
        
        return {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'history': {
                # Méta-informations
                'success': True,
                'model_type': self.model_config.model_type.value,
                'is_autoencoder': is_autoencoder,
                
                # Métriques d'entraînement (LISTES de float garanties)
                'train_loss': [float(x) for x in self.history['train_loss']],
                'val_loss': [float(x) for x in self.history['val_loss']],
                'val_accuracy': [float(x) for x in self.history['val_accuracy']] if not is_autoencoder else [],
                'val_f1': [float(x) for x in self.history['val_f1']] if not is_autoencoder else [],
                'learning_rates': [float(x) for x in self.history['learning_rates']],
                
                # Résumé d'entraînement
                'best_epoch': int(train_result.metadata.get('best_epoch', 0)),
                'best_val_loss': float(min(self.history['val_loss'])) if self.history['val_loss'] else float('inf'),
                'final_train_loss': float(self.history['train_loss'][-1]) if self.history['train_loss'] else float('inf'),
                'training_time': float(train_result.metadata.get('training_time', 0)),
                'total_epochs_trained': len(self.history['train_loss']),
                'early_stopping_triggered': len(self.history['train_loss']) < self.training_config.epochs,
                
                # Shape et configuration
                'input_shape': tuple(self.preprocessor.original_shape_[1:]) if hasattr(self.preprocessor, 'original_shape_') else None,
                'output_format': 'channels_first',
                
                # Configuration d'entraînement
                'training_config': {
                    'learning_rate': float(self.training_config.learning_rate),
                    'batch_size': int(self.training_config.batch_size),
                    'optimizer': self.training_config.optimizer.value,
                    'scheduler': self.training_config.scheduler.value,
                    'epochs_requested': int(self.training_config.epochs),
                    'early_stopping_patience': int(self.training_config.early_stopping_patience),
                    'use_class_weights': bool(self.training_config.use_class_weights),
                    'gradient_clip': float(self.training_config.gradient_clip)
                },
                
                # Métadonnées additionnelles
                'metadata': self._training_metadata
            }
        }
    
    def _validate_data(
        self,
        X_train: np.ndarray,
        y_train: Optional[np.ndarray],
        X_val: np.ndarray,
        y_val: Optional[np.ndarray]
    ) -> Result:
        """
        Validation complète des données d'entrée.
        ✅ Support y=None pour non supervisé
        """
        # ✅ CORRECTION: Gérer y=None pour non supervisé
        is_unsupervised = (y_train is None) or (y_val is None)
        
        if is_unsupervised:
            # Mode non supervisé: validation simplifiée
            if X_train is None or len(X_train) == 0:
                return Result.err("X_train vide")
            if X_val is None or len(X_val) == 0:
                return Result.err("X_val vide")
            
            # Vérification cohérence des shapes
            if X_train.shape[1:] != X_val.shape[1:]:
                return Result.err(
                    f"Shapes incompatibles: train={X_train.shape}, val={X_val.shape}"
                )
            
            logger.info("✅ Validation données non supervisées OK")
            return Result.ok(None, imbalance=None, is_unsupervised=True)
        
        # Mode supervisé: validation complète
        train_val = DataValidator.validate_input_data(X_train, y_train, "train")
        if not train_val['success']:  # ✅ CORRECTION: Accès par clé ['success']
            return Result.err(train_val['error'])
        
        val_val = DataValidator.validate_input_data(X_val, y_val, "validation")
        if not val_val['success']:  # ✅ CORRECTION: Accès par clé ['success']
            return Result.err(val_val['error'])
        
        # Vérification cohérence des shapes
        if X_train.shape[1:] != X_val.shape[1:]:
            return Result.err(
                f"Shapes incompatibles: train={X_train.shape}, val={X_val.shape}"
            )
        
        # Analyse déséquilibre
        imbalance = DataValidator.check_class_imbalance(y_train)
        
        # CORRECTION: Logging sans kwargs arbitraires
        logger.info(
            f"Analyse déséquilibre classes - "
            f"severity: {imbalance.get('severity', 'unknown')}, "
            f"ratio: {imbalance.get('ratio', 0):.2f}"
        )
        
        if imbalance['severity'] in ['critical', 'high']:
            logger.warning(
                f"Déséquilibre {imbalance['severity']} détecté (ratio={imbalance['ratio']:.2f}). "
                f"Considérez d'activer use_class_weights=True"
            )
        
        return Result.ok(None, imbalance=imbalance, is_unsupervised=False)
    
    def _setup_preprocessing(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None,
        expected_channels: Optional[int] = None
    ) -> Result:
        """
        Setup preprocessing avec VALIDATION STRICTE du format de sortie.    
        Pipeline:
        1. Création DataPreprocessor avec target_size correct
        2. fit_transform sur train → channels_first
        3. transform sur val → channels_first
        4. VALIDATION CRITIQUE des shapes finales
        5. Vérification cohérence avec model_config.input_size
        
        Args:
            X_train, y_train: Données d'entraînement
            X_val, y_val: Données de validation
            target_size: Taille cible pour resize (H, W) - optionnel
            expected_channels: Nombre de canaux attendus - optionnel
            
        Returns:
            Result avec données preprocessées VALIDÉES
            
        Raises:
            ValueError: Si format incorrect ou incohérence détectée
        """
        try:
            if X_train is None or len(X_train) == 0:
                return Result.err("Données d'entraînement vides")
            if X_val is None or len(X_val) == 0:
                return Result.err("Données de validation vides")
            
            logger.info(
                f"🔧 Début setup preprocessing - "
                f"train_shape: {X_train.shape}, "
                f"val_shape: {X_val.shape}, "
                f"train_dtype: {X_train.dtype}, "
                f"val_dtype: {X_val.dtype}"
            )
            
            # ====================================================================
            # DÉTECTION DES CANAUX
            # ====================================================================
            if expected_channels is None:
                # Auto-détection depuis les données
                if X_train.shape[-1] in [1, 3, 4]:
                    expected_channels = X_train.shape[-1]
                else:
                    expected_channels = 3  # Fallback RGB
                logger.info(f"🔍 Canaux auto-détectés: {expected_channels}")
            else:
                logger.info(f"🔍 Canaux attendus: {expected_channels} (paramètre)")
            
            # ====================================================================
            # DÉTERMINATION TARGET_SIZE (si non fourni)
            # ====================================================================
            final_target_size = target_size
            
            if final_target_size is None:
                logger.info("ℹ️ target_size non fourni, recherche automatique...")
                
                # Priorité 1: preprocessing_config explicite
                if hasattr(self, 'preprocessing_config') and self.preprocessing_config:
                    final_target_size = self.preprocessing_config.get('target_size', None)
                    if final_target_size:
                        logger.info(f"✅ Target_size depuis preprocessing_config: {final_target_size}")
                
                # Priorité 2: model_config.input_size (si différent des données)
                if final_target_size is None and hasattr(self, 'model_config'):
                    if hasattr(self.model_config, 'input_size') and self.model_config.input_size:
                        # Détecter taille actuelle des données
                        data_format = self._detect_data_format_static(X_train)
                        
                        if data_format == "channels_last":
                            data_h, data_w = X_train.shape[1], X_train.shape[2]
                        else:  # channels_first
                            data_h, data_w = X_train.shape[2], X_train.shape[3]
                        
                        model_h, model_w = self.model_config.input_size
                        
                        if (data_h, data_w) != (model_h, model_w):
                            final_target_size = self.model_config.input_size
                            logger.info(
                                f"✅ Target_size déduit depuis model_config: {final_target_size} "
                                f"(data: {data_h}x{data_w} → model: {model_h}x{model_w})"
                            )
                        else:
                            logger.info(
                                f"ℹ️ Données déjà à la taille modèle ({data_h}x{data_w}), "
                                f"pas de resize"
                            )
                            final_target_size = None
            else:
                logger.info(f"✅ Target_size fourni en paramètre: {final_target_size}")
            
            # Log final
            if final_target_size:
                logger.info(f"🔄 Resize activé: target_size={final_target_size}")
            else:
                logger.info("ℹ️ Pas de resize (taille originale conservée)")
            
            # ====================================================================
            # SÉLECTION PREPROCESSOR SELON MODEL_TYPE
            # ====================================================================
            is_autoencoder = self.model_config.model_type in [
                ModelType.CONV_AUTOENCODER,
                ModelType.VAE,
                ModelType.DENOISING_AE,
                ModelType.PATCH_CORE
            ]
            
            is_transfer_learning = self.model_config.model_type == ModelType.TRANSFER_LEARNING
            
            if is_autoencoder:
                # AutoencoderPreprocessor pour AE
                self.preprocessor = AutoencoderPreprocessor(
                    target_size=final_target_size
                )
                logger.info("✅ AutoencoderPreprocessor sélectionné")
                
            elif is_transfer_learning:
                # Transfer Learning → strategy="imagenet"
                self.preprocessor = DataPreprocessor(
                    strategy="imagenet",  
                    auto_detect_format=True,
                    target_size=final_target_size
                )
                logger.info("✅ DataPreprocessor sélectionné - strategy: imagenet (normalisation dans modèle)")
                
            else:
                # Autres modèles
                strategy = self.preprocessing_config.get("strategy", "normalize")
                self.preprocessor = DataPreprocessor(
                    strategy=strategy,
                    auto_detect_format=True,
                    target_size=final_target_size
                )
                logger.info(f"✅ DataPreprocessor sélectionné - strategy: {strategy}")
            
            # ====================================================================
            # FIT_TRANSFORM SUR TRAIN
            # ====================================================================
            try:
                X_train_norm = self.preprocessor.fit_transform(
                    X_train,
                    output_format="channels_first"
                )
            except Exception as e:
                logger.error(f"❌ Erreur fit_transform: {e}", exc_info=True)
                return Result.err(f"Erreur preprocessing train: {str(e)}")
            
            # VALIDATION CRITIQUE #1 : Format train
            if X_train_norm.ndim != 4:
                return Result.err(
                    f"Format train invalide après preprocessing: "
                    f"{X_train_norm.ndim}D au lieu de 4D"
                )
            
            # Validation canaux
            actual_channels = X_train_norm.shape[1]
            if actual_channels != expected_channels:
                logger.warning(
                    f"⚠️ Incohérence canaux: attendu={expected_channels}, reçu={actual_channels}. "
                    f"Le modèle s'adaptera automatiquement."
                )
            
            if actual_channels not in [1, 3, 4]:
                return Result.err(
                    f"❌ ERREUR FORMAT TRAIN:\n"
                    f"   Shape: {X_train_norm.shape}\n"
                    f"   Format attendu: (N, C, H, W) avec C ∈ [1, 3, 4]\n"
                    f"   Reçu: C = {actual_channels} (INVALIDE)"
                )
            
            logger.info(
                f"✅ Train preprocessé - "
                f"shape: {X_train_norm.shape}, "
                f"format: channels_first, "
                f"canaux: {actual_channels} (attendus: {expected_channels})"
            )
            
            # ====================================================================
            # TRANSFORM SUR VAL
            # ====================================================================
            try:
                X_val_norm = self.preprocessor.transform(
                    X_val,
                    output_format="channels_first"
                )
            except Exception as e:
                logger.error(f"❌ Erreur transform val: {e}", exc_info=True)
                return Result.err(f"Erreur preprocessing val: {str(e)}")
            
            # VALIDATION CRITIQUE #2 : Format val
            if X_val_norm.ndim != 4:
                return Result.err(
                    f"Format val invalide après preprocessing: "
                    f"{X_val_norm.ndim}D au lieu de 4D"
                )
            
            val_channels = X_val_norm.shape[1]
            if val_channels != actual_channels:
                return Result.err(
                    f"❌ INCOHÉRENCE CANAUX TRAIN/VAL:\n"
                    f"   Train canaux: {actual_channels}\n"
                    f"   Val canaux: {val_channels}"
                )
            
            logger.info(
                f"✅ Val preprocessé - "
                f"shape: {X_val_norm.shape}, "
                f"format: channels_first, "
                f"canaux: {val_channels}"
            )
            
            # ====================================================================
            # VALIDATION CRITIQUE #3 : Vérifications NaN/Inf
            # ====================================================================
            if np.any(np.isnan(X_train_norm)) or np.any(np.isinf(X_train_norm)):
                return Result.err("Données train contiennent NaN ou Inf après preprocessing")
            
            if np.any(np.isnan(X_val_norm)) or np.any(np.isinf(X_val_norm)):
                return Result.err("Données val contiennent NaN ou Inf après preprocessing")
            
            # ====================================================================
            # VALIDATION CRITIQUE #4 : Cohérence avec model_config.input_size
            # ====================================================================
            if hasattr(self, 'model_config') and hasattr(self.model_config, 'input_size'):
                expected_h, expected_w = self.model_config.input_size
                actual_h, actual_w = X_train_norm.shape[2], X_train_norm.shape[3]
                
                if (actual_h, actual_w) != (expected_h, expected_w):
                    logger.error(
                        f"❌ INCOHÉRENCE TAILLE APRÈS PREPROCESSING:\n"
                        f"   model_config.input_size: {expected_h}x{expected_w}\n"
                        f"   Données preprocessées:   {actual_h}x{actual_w}\n"
                        f"   Shape train: {X_train_norm.shape}\n"
                        f"   target_size utilisé: {final_target_size}"
                    )
                    
                    # Correction: Mettre à jour model_config.input_size
                    if hasattr(self.model_config, 'input_size'):
                        self.model_config.input_size = (actual_h, actual_w)
                        logger.warning(
                            f"⚠️ Correction automatique: model_config.input_size mis à jour "
                            f"→ {self.model_config.input_size}"
                        )
            
            # ====================================================================
            # LOG RÉCAPITULATIF
            # ====================================================================
            input_format = getattr(self.preprocessor, 'data_format_', 'unknown')
            resized = getattr(self.preprocessor, 'resized_', False)
            
            logger.info(
                f"✅ Preprocessing configuré avec succès:\n"
                f"   • Strategy: {getattr(self.preprocessor, 'strategy', 'unknown')}\n"
                f"   • Input format: {input_format}\n"
                f"   • Output format: channels_first\n"
                f"   • Resized: {resized}\n"
                f"   • Target size: {final_target_size}\n"
                f"   • Train original: {X_train.shape}\n"
                f"   • Train processed: {X_train_norm.shape}\n"
                f"   • Val processed: {X_val_norm.shape}\n"
                f"   • Canaux train: {actual_channels} (attendus: {expected_channels})\n"
                f"   • Canaux val: {val_channels}"
            )
            
            # Stocker les métadonnées pour usage ultérieur
            self._preprocessing_metadata = {
                'input_channels': actual_channels,
                'target_size': final_target_size,
                'original_shape': X_train.shape,
                'processed_shape': X_train_norm.shape,
                'resized': resized
            }
            
            return Result.ok((X_train_norm, y_train, X_val_norm, y_val))
            
        except ValueError as e:
            logger.error(f"Erreur validation preprocessing: {str(e)}")
            return Result.err(f"Données invalides: {str(e)}")
        except Exception as e:
            logger.error(f"Erreur technique preprocessing: {str(e)}", exc_info=True)
            return Result.err(f"Erreur preprocessing: {str(e)}")


    @staticmethod
    def _detect_data_format_static(X: np.ndarray) -> str:
        """
        Méthode statique pour détection de format (sans self).
        Utile dans _setup_preprocessing avant création du preprocessor.
        """
        if X.ndim != 4:
            return 'channels_last'  # Fallback
        
        n, dim1, dim2, dim3 = X.shape
        
        # Format channels_last: (N, H, W, C)
        if dim3 in [1, 3, 4] and dim3 < dim1 and dim3 < dim2:
            if abs(dim1 - dim2) / max(dim1, dim2) < 0.5:
                return 'channels_last'
        
        # Format channels_first: (N, C, H, W)
        if dim1 in [1, 3, 4] and dim1 < dim2 and dim1 < dim3:
            if abs(dim2 - dim3) / max(dim2, dim3) < 0.5:
                return 'channels_first'
        
        # Heuristique finale
        if dim1 < min(dim2, dim3) * 0.1:
            return 'channels_first'
        elif dim3 < min(dim1, dim2) * 0.1:
            return 'channels_last'
        else:
            # Ambiguïté: fallback channels_last
            return 'channels_last'
    
    def _build_model(self) -> Result:
        """Construit le modèle via ModelBuilder"""
        # Import local pour éviter la boucle circulaire
        from src.models.computer_vision.model_builder import ModelBuilder
        
        builder = ModelBuilder(self.device_manager)
        result = builder.build(self.model_config)
        
        if result.success:
            self.model = result.data
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            logger.info(
                f"Modèle construit avec succès - "
                f"model_type: {self.model_config.model_type.value}, "
                f"total_params: {total_params}, "
                f"trainable_params: {trainable_params}"
            )
        
        return result


    def _setup_training(self, y_train: Optional[np.ndarray]) -> Result:
        """
        Setup optimizer, scheduler, et criterion.

        GARANTIT:
        - Optimizer configuré
        - Scheduler configuré
        - Criterion configuré selon type de modèle
        - Cohérence CRITIQUE entre données, labels et num_classes
        """
        try:
            # ============================================================
            # 1. OPTIMIZER
            # ============================================================
            self.optimizer = OptimizerFactory.create(self.model, self.training_config)

            # ============================================================
            # 2. SCHEDULER
            # ============================================================
            self.scheduler = SchedulerFactory.create(self.optimizer, self.training_config)

            # ============================================================
            # 3. DÉTECTION TYPE MODÈLE
            # ============================================================
            is_autoencoder = self.model_config.model_type in [
                ModelType.CONV_AUTOENCODER,
                ModelType.VAE,
                ModelType.DENOISING_AE,
                ModelType.PATCH_CORE
            ]

            # ============================================================
            # 4. VALIDATION CRITIQUE DES LABELS
            # ============================================================
            # ✅ CORRECTION: Gérer y_train=None pour non supervisé
            if y_train is None:
                # Non supervisé: doit être autoencoder
                if not is_autoencoder:
                    raise ValueError(
                        "❌ y_train=None mais modèle n'est pas un autoencoder. "
                        "Non supervisé nécessite autoencoder/VAE."
                    )
                logger.info("✅ Mode non supervisé confirmé (y_train=None)")
            elif not is_autoencoder:
                model_num_classes = self.model_config.num_classes
                data_classes = np.unique(y_train)
                data_num_classes = len(data_classes)
                data_min_label = int(np.min(y_train))
                data_max_label = int(np.max(y_train))

                # ❌ Incohérence modèle / données
                if model_num_classes != data_num_classes:
                    raise ValueError(
                        f"❌ INCOHÉRENCE MODEL / DATA:\n"
                        f"   • model.num_classes = {model_num_classes}\n"
                        f"   • classes réelles dans y_train = {data_num_classes}\n"
                        f"   • labels présents = {data_classes}\n"
                        f"   • BUG orchestrateur (ne devrait jamais arriver)"
                    )

                # ❌ Labels mal encodés
                if data_min_label != 0:
                    raise ValueError(
                        f"❌ Labels invalides: min(y_train)={data_min_label}. "
                        f"Les labels doivent commencer à 0."
                    )

                if data_max_label >= model_num_classes:
                    raise ValueError(
                        f"❌ LABEL HORS LIMITES:\n"
                        f"   • max(y_train) = {data_max_label}\n"
                        f"   • model.num_classes = {model_num_classes}\n"
                        f"   • CrossEntropyLoss va crasher"
                    )

                logger.info(
                    f"✅ Validation labels OK - "
                    f"num_classes={model_num_classes}, "
                    f"labels=[{data_min_label}..{data_max_label}], "
                    f"distribution={dict(Counter(y_train))}"
                )

            # ============================================================
            # 5. SETUP CRITERION
            # ============================================================
            if is_autoencoder:
                # --- Autoencoders ---
                self.train_criterion = nn.MSELoss()
                self.val_criterion = nn.MSELoss()

                if self.training_config.use_class_weights:
                    logger.warning(
                        "⚠️ class_weights ignorés pour autoencoders "
                        "(MSELoss ≠ classification)"
                    )
                    self.training_config.use_class_weights = False

                logger.info("✅ Criterion: MSELoss (autoencoder)")

            else:
                # --- Classification ---
                # ✅ CORRECTION: y_train ne peut pas être None ici (déjà vérifié)
                if y_train is None:
                    raise ValueError("y_train ne peut pas être None pour classification")
                
                if self.training_config.use_class_weights:
                    classes = np.unique(y_train)

                    weights = compute_class_weight(
                        class_weight="balanced",
                        classes=classes,
                        y=y_train
                    )

                    weights_tensor = torch.tensor(
                        weights,
                        dtype=torch.float32,
                        device=self.device_manager.device
                    )

                    self.train_criterion = nn.CrossEntropyLoss(weight=weights_tensor)

                    logger.info(
                        f"✅ CrossEntropyLoss avec class weights (train): "
                        f"{dict(zip(classes, weights.round(3)))}"
                    )
                else:
                    self.train_criterion = nn.CrossEntropyLoss()
                    logger.info("✅ CrossEntropyLoss standard (train)")

                # Validation TOUJOURS sans class weights
                self.val_criterion = nn.CrossEntropyLoss()
                logger.info("✅ CrossEntropyLoss validation (sans class weights)")

            # ============================================================
            # 6. LOG FINAL
            # ============================================================
            scheduler_name = self.training_config.scheduler.value if self.scheduler else "none"

            logger.info(
                f"🎯 Training setup OK - "
                f"model_type={self.model_config.model_type}, "
                f"criterion={'MSELoss' if is_autoencoder else 'CrossEntropyLoss'}, "
                f"use_class_weights={self.training_config.use_class_weights}, "
                f"optimizer={self.training_config.optimizer.value}, "
                f"scheduler={scheduler_name}"
            )

            return Result.ok(None)

        except Exception as e:
            logger.error(f"❌ Setup training échoué: {e}", exc_info=True)
            return Result.err(str(e))
    
    def _setup_training_unsupervised(self) -> Result:
        """
        Setup training pour mode non supervisé (autoencoder uniquement).
        ✅ Gère le cas y_train=None
        """
        try:
            # 1. OPTIMIZER
            self.optimizer = OptimizerFactory.create(self.model, self.training_config)

            # 2. SCHEDULER
            self.scheduler = SchedulerFactory.create(self.optimizer, self.training_config)

            # 3. CRITERION (MSELoss pour autoencoder)
            self.train_criterion = nn.MSELoss()
            self.val_criterion = nn.MSELoss()

            logger.info(
                f"✅ Training setup non supervisé OK - "
                f"criterion=MSELoss, "
                f"optimizer={self.training_config.optimizer.value}"
            )

            return Result.ok(None)

        except Exception as e:
            logger.error(f"❌ Setup training non supervisé échoué: {e}", exc_info=True)
            return Result.err(str(e))

    
    def _training_loop(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        y_val: Optional[np.ndarray]
    ) -> Result:
        """Boucle d'entraînement avec early stopping et checkpointing"""
        try:
            is_autoencoder = self.model_config.model_type in [
                ModelType.CONV_AUTOENCODER, ModelType.VAE, ModelType.DENOISING_AE
            ]
            
            best_val_metric = float('inf') if is_autoencoder else 0.0
            best_model_state = None
            best_epoch = 0
            patience_counter = 0
            
            start_time = time.time()
            
            # CORRECTION: Callbacks avec try/catch
            for cb in self.callbacks:
                try:
                    cb.on_train_begin()
                except Exception as e:
                    logger.warning(f"Callback on_train_begin échoué: {e}")

            # SETUP FINE-TUNING SCHEDULER
            finetuning_scheduler = None
            
            if self.model_config.model_type == ModelType.TRANSFER_LEARNING:
                # Détermine le stratégie selon dataset size
                train_size = len(train_loader.dataset)
                
                if train_size < 500:
                    strategy = "gradual"
                    unfreeze_epochs = [3, 6, 9]
                    
                    from src.models.computer_vision.classification.transfer_learning import FineTuningScheduler
                    
                    finetuning_scheduler = FineTuningScheduler(
                        self.model,
                        strategy=strategy,
                        unfreeze_epochs=unfreeze_epochs
                    )
                    
                    logger.info(
                        f"✅ FineTuningScheduler activé:\n"
                        f"   • Strategy: {strategy}\n"
                        f"   • Unfreeze epochs: {unfreeze_epochs}\n"
                        f"   • Dataset size: {train_size}"
                    )
            
            # === BOUCLE EPOCHS ===
            for epoch in range(self.training_config.epochs):
                epoch_start = time.time()
                
                # === CALLBACKS on_epoch_begin ===
                for cb in self.callbacks:
                    try:
                        cb.on_epoch_begin(epoch)
                    except Exception as e:
                        logger.warning(f"Callback on_epoch_begin échoué: {e}")
                
                # DÉGEL PROGRESSIF
                if finetuning_scheduler:
                    finetuning_scheduler.step(epoch)
                
                # === TRAIN ===
                train_loss = self._train_epoch(train_loader, is_autoencoder)
                
                # === VALIDATION ===
                if is_autoencoder:
                    val_loss = self._validate_epoch_autoencoder(val_loader)
                    val_metrics = {'loss': val_loss}
                else:
                    val_loss, val_metrics = self._validate_epoch(val_loader, y_val)
                
                # Indentation fix - Mise à jour historique
                self.history['train_loss'].append(float(train_loss))
                self.history['val_loss'].append(float(val_loss))
                
                if not is_autoencoder:
                    self.history['val_accuracy'].append(float(val_metrics['accuracy']))
                    self.history['val_f1'].append(float(val_metrics['f1']))
                
                current_lr = float(self.optimizer.param_groups[0]['lr'])
                self.history['learning_rates'].append(current_lr)
                
                # Scheduler step
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                
                # CORRECTION: Callbacks avec try/catch et logging corrigé
                logs = {
                    'epoch': epoch,
                    'train_loss': float(train_loss),
                    'val_loss': float(val_loss),
                    'lr': current_lr,
                    'epoch_time': time.time() - epoch_start,
                    'model_state_dict': copy.deepcopy(self.model.state_dict()),
                    'optimizer_state_dict': copy.deepcopy(self.optimizer.state_dict())
                }
                
                if not is_autoencoder:
                    logs.update({
                        'val_accuracy': float(val_metrics['accuracy']),
                        'val_f1': float(val_metrics['f1'])
                    })
                
                for cb in self.callbacks:
                    try:
                        cb.on_epoch_end(epoch, logs)
                    except Exception as e:
                        logger.warning(f"Callback on_epoch_end échoué: {e}")
                
                # === EARLY STOPPING ===
                if is_autoencoder:
                    # Autoencoder: minimiser la loss
                    improved = val_loss < best_val_metric
                    best_val_metric = min(best_val_metric, val_loss)
                else:
                    # Classification: maximiser F1
                    improved = val_metrics['f1'] > best_val_metric
                    best_val_metric = max(best_val_metric, val_metrics['f1'])
                
                if improved:
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    best_epoch = epoch + 1
                    patience_counter = 0
                    
                    metric_name = "loss" if is_autoencoder else "F1"
                    logger.info(
                        f"✨ Nouveau meilleur modèle ({metric_name}={best_val_metric:.4f}) - "
                        f"epoch: {epoch+1}"
                    )
                    
                    # Sauvegarde checkpoint si configuré
                    if self.training_config.checkpoint_dir and self.training_config.save_best_only:
                        self._save_checkpoint(epoch, best_val_metric, best_model_state)
                else:
                    patience_counter += 1
                
                # Check early stopping
                if patience_counter >= self.training_config.early_stopping_patience:
                    logger.info(
                        f"🛑 Early stopping déclenché - "
                        f"epoch: {epoch+1}, "
                        f"patience: {patience_counter}"
                    )
                    break
            
            # Restauration du meilleur modèle
            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)
                logger.info(f"✅ Meilleur modèle restauré (epoch {best_epoch})")
            
            training_time = time.time() - start_time
            
            # CORRECTION: Callbacks avec try/catch
            for cb in self.callbacks:
                try:
                    cb.on_train_end({'training_time': training_time})
                except Exception as e:
                    logger.warning(f"Callback on_train_end échoué: {e}")
            
            logger.info(
                f"🎯 Entraînement terminé avec succès - "
                f"total_epochs: {epoch+1}, "
                f"best_epoch: {best_epoch}, "
                f"best_metric: {best_val_metric}, "
                f"training_time: {training_time:.1f}s, "
                f"avg_epoch_time: {training_time/(epoch+1):.2f}s"
            )
            
            result_metadata = {
                'best_epoch': best_epoch,
                'training_time': training_time,
                'total_epochs': epoch + 1
            }
            
            if is_autoencoder:
                result_metadata['best_loss'] = best_val_metric
            else:
                result_metadata['best_f1'] = best_val_metric
            
            return Result.ok(None, **result_metadata)
            
        except Exception as e:
            logger.error(f"Erreur boucle training: {e}", exc_info=True)
            return Result.err(f"Training loop échoué: {str(e)}")
        

    def _train_epoch(
        self,
        train_loader: DataLoader,
        is_autoencoder: bool = False
    ) -> float:
        """
        Entraîne une époque avec VALIDATION RUNTIME des labels.
        
        VALIDATIONS:
        - Labels min=0 à chaque batch (premier batch uniquement en production)
        - Labels max < num_classes
        - Shape output correcte
        - Pas de NaN/Inf
        
        Returns:
            Average training loss
        """
        # PatchCore: pas de training classique
        is_patchcore = self.model_config.model_type == ModelType.PATCH_CORE
        if is_patchcore:
            logger.info("⚠️ PatchCore détecté - Skip training epoch (fit() utilisé)")
            return 0.0

        self.model.train()
        running_loss = 0.0
        first_batch = True  # Pour validation runtime (debug mode)

        for batch_idx, (data, target_labels) in enumerate(train_loader):
            data = data.to(self.device_manager.device)

            if is_autoencoder:
                target = data
            else:
                target = target_labels.to(self.device_manager.device)

                # ================================================================
                # VALIDATION RUNTIME DES LABELS (PREMIER BATCH)
                # ================================================================
                if first_batch:
                    batch_min = int(target.min().item())
                    batch_max = int(target.max().item())
                    batch_unique = torch.unique(target).cpu().tolist()
                    model_output_size = self.model_config.num_classes

                    # ❌ ERREUR: Labels ne commencent pas à 0
                    if batch_min != 0:
                        raise RuntimeError(
                            f"❌ LABELS INVALIDES (batch {batch_idx}):\n"
                            f"   • min(labels) = {batch_min} (DOIT être 0)\n"
                            f"   • max(labels) = {batch_max}\n"
                            f"   • Labels présents: {batch_unique}\n"
                            f"\n"
                            f"   💡 BUG DANS DATALOADER/DATASET\n"
                            f"   Les labels ont été corrompus entre la création\n"
                            f"   du dataset et ce batch.\n"
                            f"\n"
                            f"   🛠️ SOLUTION:\n"
                            f"   1. Vérifier AugmentedImageDataset.__getitem__()\n"
                            f"   2. Ajouter logging pour tracer la corruption\n"
                            f"   3. Exécuter diagnostic_labels.py"
                        )

                    # ❌ ERREUR: Labels hors limites
                    if batch_max >= model_output_size:
                        raise RuntimeError(
                            f"❌ LABEL HORS LIMITES (batch {batch_idx}):\n"
                            f"   • max(labels) = {batch_max}\n"
                            f"   • model.num_classes = {model_output_size}\n"
                            f"   • CrossEntropyLoss attend [0, {model_output_size-1}]\n"
                            f"\n"
                            f"   💡 INCOHÉRENCE MODEL/DATA\n"
                            f"   Le modèle a été construit pour {model_output_size} classes\n"
                            f"   mais les données contiennent des labels jusqu'à {batch_max}"
                        )
                    
                    # ✅ Log validation OK
                    logger.info(
                        f"✅ Batch {batch_idx} validé:\n"
                        f"   • labels range: [{batch_min}, {batch_max}]\n"
                        f"   • labels unique: {batch_unique}\n"
                        f"   • num_classes: {model_output_size}"
                    )
                    
                    first_batch = False

            # ================================================================
            # FORWARD PASS
            # ================================================================
            self.optimizer.zero_grad()
            output = self.model(data)

            if output is None:
                logger.warning(f"⚠️ Modèle a retourné None - Skip batch {batch_idx}")
                continue

            # ================================================================
            # VALIDATION SHAPE OUTPUT
            # ================================================================
            if not is_autoencoder:
                expected_shape = (data.size(0), self.model_config.num_classes)
                if output.shape != expected_shape:
                    raise RuntimeError(
                        f"❌ MODEL OUTPUT SHAPE INCORRECTE (batch {batch_idx}):\n"
                        f"   • Attendue: {expected_shape}\n"
                        f"   • Reçue: {output.shape}\n"
                        f"\n"
                        f"   💡 CAUSE PROBABLE:\n"
                        f"   - Classifier mal configuré\n"
                        f"   - num_classes incorrect dans model_config"
                    )
            else:
                if output.shape != data.shape:
                    raise RuntimeError(
                        f"❌ SHAPE MISMATCH AUTOENCODER (batch {batch_idx}):\n"
                        f"   • Input shape: {data.shape}\n"
                        f"   • Output shape: {output.shape}"
                    )

            # ================================================================
            # LOSS COMPUTATION
            # ================================================================
            if (
                self.model_config.model_type == ModelType.VAE
                and hasattr(self.model, "compute_vae_loss")
            ):
                loss, recon_loss, kl_loss = self.model.compute_vae_loss(target, output)
            else:
                loss = self.train_criterion(output, target)

            # Vérification NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(
                    f"❌ Loss invalide détectée (batch {batch_idx}): {loss.item()}"
                )
                continue

            # ================================================================
            # BACKWARD PASS
            # ================================================================
            loss.backward()

            # Gradient clipping
            if self.training_config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.training_config.gradient_clip
                )

            self.optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        
        logger.info(
            f"✅ Epoch terminée:\n"
            f"   • Batches processed: {len(train_loader)}\n"
            f"   • Average loss: {avg_loss:.6f}"
        )
        
        return avg_loss

    
    def _validate_epoch_autoencoder(self, val_loader: DataLoader) -> float:
        """
        Validation pour autoencoder.
        """
        self.model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(self.device_manager.device)
                output = self.model(data)
                
                target = data
                
                # Vérifi la cohérence (debug mode)
                if output.shape != target.shape:
                    logger.error(
                        f"❌ INCOHÉRENCE VALIDATION: "
                        f"output={output.shape} vs target={target.shape}"
                    )
                    raise ValueError(
                        f"Shape mismatch validation: output={output.shape} vs target={target.shape}"
                    )
                
                # Loss computation
                if (self.model_config.model_type == ModelType.VAE and 
                    hasattr(self.model, 'compute_vae_loss')):
                    loss, _, _ = self.model.compute_vae_loss(target, output)
                else:
                    loss = self.val_criterion(output, target)
                
                running_loss += loss.item()
        
        return running_loss / len(val_loader)
    
    def _validate_epoch(
        self,
        val_loader: DataLoader,
        y_val: Optional[np.ndarray]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Validation pour classificateur.
        
        CRITIQUE: Utilise val_criterion SANS class weights pour évaluation honnête.
        ✅ Support y_val=None (non utilisé, mais cohérent avec signature)
        """
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(self.device_manager.device)
                target = target.to(self.device_manager.device)
                
                output = self.model(data)
                loss = self.val_criterion(output, target)
                running_loss += loss.item()
                
                # Prédictions
                preds = output.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(target.cpu().numpy())
        
        # Calcul métriques métier
        val_loss = running_loss / len(val_loader)
        
        metrics = {
            'accuracy': accuracy_score(all_targets, all_preds),
            'precision': precision_score(all_targets, all_preds, average='weighted', zero_division=0),
            'recall': recall_score(all_targets, all_preds, average='weighted', zero_division=0),
            'f1': f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        }
        
        return val_loss, metrics
    
    def _save_checkpoint(
        self, 
        epoch: int, 
        metric: float, 
        model_state: Dict[str, Any]
    ) -> None:
        """Sauvegarde un checkpoint"""
        try:
            if self.training_config.checkpoint_dir is None:
                return
            
            checkpoint_dir = self.training_config.checkpoint_dir
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = checkpoint_dir / f"best_model_epoch_{epoch}.pt"
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'metric': metric,
                'model_config': self.model_config,
                'training_config': self.training_config
            }, checkpoint_path)
            
            logger.info(f"💾 Checkpoint sauvegardé: {checkpoint_path}")
            
        except Exception as e:
            logger.warning(f"Impossible de sauvegarder le checkpoint: {e}")

    # ==================
    # MÉTHODES PUBLIQUES
    # ==================
    
    def predict(
        self,
        X: np.ndarray,
        return_reconstructed: bool = False,
        batch_size: Optional[int] = None,
        device_override: Optional[str] = None
    ) -> Result:
        """
        Prédictions robustes avec gestion automatique de la mémoire.
        
        Nouveautés:
        - Auto-détection batch_size optimal
        - Gestion mémoire GPU
        - Validation format d'entrée
        """
        try:
            # Validation
            if self.model is None:
                return Result.err("Modèle non entraîné")
            if self.preprocessor is None:
                return Result.err("Preprocessor non disponible")
            
            logger.info(f"🔮 Début prédictions sur {len(X)} échantillons")
            
            # ====================================================================
            # 1. VALIDATION FORMAT D'ENTRÉE
            # ====================================================================
            if X.ndim not in [3, 4]:
                return Result.err(
                    f"Format d'entrée invalide: {X.ndim}D. "
                    f"Attendu 3D (H,W,C) ou 4D (N,H,W,C)"
                )
            
            # Ajouter dimension batch si nécessaire
            if X.ndim == 3:
                X = np.expand_dims(X, axis=0)
                logger.info(f"Ajout dimension batch: {X.shape}")
            
            # ====================================================================
            # 2. PREPROCESSING
            # ====================================================================
            try:
                X_processed = self.preprocessor.transform(
                    X, 
                    output_format="channels_first"
                )
            except Exception as e:
                return Result.err(f"Échec preprocessing: {str(e)}")
            
            # ====================================================================
            # 3. DÉTERMINATION BATCH_SIZE OPTIMAL
            # ====================================================================
            if batch_size is None:
                batch_size = self._auto_detect_batch_size(
                    X_processed, 
                    self.training_config.batch_size
                )
                logger.info(f"⚙️ Batch_size auto-détecté: {batch_size}")
            
            # ====================================================================
            # 4. CRÉATION DATALOADER
            # ====================================================================
            dummy_labels = np.zeros(len(X_processed))
            
            test_loader = DataLoaderFactory.create(
                X_processed, 
                dummy_labels,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,  # Pas de multiprocessing pour prédiction
                pin_memory=False,
                augmentation_config=None,
                is_training=False,
                output_format="channels_first",
                device_manager=self.device_manager
            )
            
            # ====================================================================
            # 5. PRÉDICTION
            # ====================================================================
            is_autoencoder = self.model_config.model_type in [
                ModelType.CONV_AUTOENCODER, 
                ModelType.VAE, 
                ModelType.DENOISING_AE,
                ModelType.PATCH_CORE
            ]
            
            self.model.eval()
            
            if is_autoencoder:
                return self._predict_autoencoder(
                    test_loader, 
                    X_processed, 
                    return_reconstructed
                )
            else:
                return self._predict_classifier(test_loader)
            
        except torch.cuda.OutOfMemoryError:
            return Result.err(
                "❌ GPU OUT OF MEMORY\n"
                "Solution: Réduire batch_size ou utiliser CPU"
            )
        except Exception as e:
            logger.error(f"❌ Erreur prédiction: {e}", exc_info=True)
            return Result.err(f"Prédiction échouée: {str(e)}")

    def _auto_detect_batch_size(
        self,
        X: np.ndarray,
        initial_batch_size: int
    ) -> int:
        """
        Détecte automatiquement le batch_size optimal.
        
        Stratégie:
        1. Commencer avec batch_size initial
        2. Tester avec un batch factice
        3. Réduire si out of memory
        """
        if not torch.cuda.is_available():
            return min(initial_batch_size, 32)  # Limite CPU
        
        batch_size = initial_batch_size
        max_iterations = 5  # Éviter boucle infinie
        
        for i in range(max_iterations):
            try:
                # Tester avec un batch factice
                test_tensor = torch.randn(
                    batch_size, 
                    *X.shape[1:],
                    device=self.device_manager.device
                )
                
                # Tester forward pass
                with torch.no_grad():
                    _ = self.model(test_tensor)
                
                # Nettoyer
                del test_tensor
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                logger.info(f"✅ Batch_size {batch_size} validé")
                return batch_size
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    old_batch_size = batch_size
                    batch_size = max(1, batch_size // 2)
                    
                    logger.warning(
                        f"⚠️ Batch_size réduit: {old_batch_size} → {batch_size} "
                        f"(itération {i+1})"
                    )
                    
                    # Nettoyer
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                    if batch_size == 1:
                        logger.warning("⚠️ Batch_size minimum atteint (1)")
                        return 1
                else:
                    raise
        
        logger.warning(f"⚠️ Auto-détection échouée, fallback batch_size=8")
        return 8
    
    
    def _predict_autoencoder(
        self, 
        test_loader: DataLoader, 
        X_processed: np.ndarray,
        return_reconstructed: bool
    ) -> Result:
        """
        Prédictions pour autoencoder avec génération automatique des heatmaps.
        """
        reconstruction_errors = []
        reconstructed_images = [] if return_reconstructed else None
        error_maps_list = []
        
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(self.device_manager.device)
                reconstructed = self.model(data)
                
                # Erreur de reconstruction par échantillon (scalar)
                errors = torch.mean(
                    (data - reconstructed) ** 2,
                    dim=tuple(range(1, data.ndim))
                ).cpu().numpy()
                
                reconstruction_errors.extend(errors)
                
                # Error map spatiale: (B, C, H, W) → (B, H, W)
                batch_error_maps = torch.mean(
                    (data - reconstructed) ** 2, 
                    dim=1  # Moyenne sur les canaux
                ).cpu().numpy()
                
                error_maps_list.append(batch_error_maps)
                
                # Reconstructions si demandé
                if return_reconstructed:
                    reconstructed_images.append(reconstructed.cpu().numpy())
        
        reconstruction_errors = np.array(reconstruction_errors)
        
        # Seuil automatique (95ème percentile par défaut)
        # Pour MVTec AD, utiliser _compute_adaptive_threshold() pour meilleure précision
        threshold = np.percentile(reconstruction_errors, 95)
        predictions = (reconstruction_errors > threshold).astype(int)
        
        result_data = {
            'reconstruction_errors': reconstruction_errors,
            'predictions': predictions,
            'threshold': float(threshold)
        }
        
        if return_reconstructed and reconstructed_images:
            result_data['reconstructed'] = np.concatenate(reconstructed_images, axis=0)
        
        # Error maps et heatmaps
        if error_maps_list:
            error_maps = np.concatenate(error_maps_list, axis=0)
            result_data['error_maps'] = error_maps
            
            # Génération heatmaps normalisées [0, 1]
            heatmaps = []
            for error_map in error_maps:
                min_val = error_map.min()
                max_val = error_map.max()
                
                if max_val > min_val:
                    normalized = (error_map - min_val) / (max_val - min_val + 1e-8)
                else:
                    normalized = np.zeros_like(error_map)
                
                heatmaps.append(normalized)
            
            result_data['heatmaps'] = np.array(heatmaps)
            
            logger.info(
                f"✅ Error maps générées: shape={error_maps.shape}, "
                f"heatmaps shape={result_data['heatmaps'].shape}, "
                f"threshold={threshold:.6f}"
            )
        
        return Result.ok(result_data)
    
    def _predict_classifier(self, test_loader: DataLoader) -> Result:
        """Prédictions pour classificateur"""
        all_probs = []
        all_preds = []
        
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(self.device_manager.device)
                output = self.model(data)
                
                probs = torch.softmax(output, dim=1).cpu().numpy()
                preds = output.argmax(dim=1).cpu().numpy()
                
                all_probs.append(probs)
                all_preds.extend(preds)
        
        return Result.ok({
            'probabilities': np.concatenate(all_probs, axis=0),
            'predictions': np.array(all_preds)
        })
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Result:
        """
        Évaluation complète sur données de test.
        
        Gère automatiquement classification ET autoencoders.
        
        Args:
            X_test: Données de test (JAMAIS VU pendant training)
            y_test: Labels de test
            
        Returns:
            Result avec métriques complètes
        """
        try:
            # Validation du test set
            test_val = DataValidator.validate_input_data(X_test, y_test, "test")
            if not test_val.success:
                return test_val
            
            # Preprocessing (TRANSFORM uniquement, pas fit!)
            if self.preprocessor is None:
                return Result.err("Modèle non entraîné (pas de preprocessor)")
            
            X_test_norm = self.preprocessor.transform(X_test, output_format="channels_first")
            
            # Détection du type de modèle
            is_autoencoder = self.model_config.model_type in [
                ModelType.CONV_AUTOENCODER, ModelType.VAE, ModelType.DENOISING_AE
            ]
            
            if is_autoencoder:
                return self._evaluate_autoencoder(X_test_norm, y_test)
            else:
                return self._evaluate_classifier(X_test_norm, y_test)
            
        except Exception as e:
            logger.error(f"Erreur évaluation: {e}", exc_info=True)
            return Result.err(f"Évaluation échouée: {str(e)}")
    
    def _evaluate_classifier(
        self,
        X_test_norm: np.ndarray,
        y_test: np.ndarray
    ) -> Result:
        """Évaluation pour modèles de classification"""
        try:
            test_loader = DataLoaderFactory.create(
                X_test_norm, y_test,
                batch_size=self.training_config.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False
            )
            
            self.model.eval()
            all_preds = []
            all_probs = []
            all_targets = []
            
            with torch.no_grad():
                for data, target in test_loader:
                    data = data.to(self.device_manager.device)
                    output = self.model(data)
                    
                    probs = torch.softmax(output, dim=1).cpu().numpy()
                    preds = output.argmax(dim=1).cpu().numpy()
                    
                    all_probs.extend(probs)
                    all_preds.extend(preds)
                    all_targets.extend(target.numpy())
            
            all_preds = np.array(all_preds)
            all_probs = np.array(all_probs)
            all_targets = np.array(all_targets)
            
            # Métriques complètes
            metrics = {
                'accuracy': float(accuracy_score(all_targets, all_preds)),
                'precision': float(precision_score(all_targets, all_preds, average='weighted', zero_division=0)),
                'recall': float(recall_score(all_targets, all_preds, average='weighted', zero_division=0)),
                'f1': float(f1_score(all_targets, all_preds, average='weighted', zero_division=0)),
                'confusion_matrix': confusion_matrix(all_targets, all_preds).tolist(),
                'n_samples': len(X_test_norm),
                'n_classes': len(np.unique(y_test))
            }
            
            # AUC-ROC si binaire
            if self.model_config.num_classes == 2:
                try:
                    metrics['auc_roc'] = float(roc_auc_score(all_targets, all_probs[:, 1]))
                except:
                    metrics['auc_roc'] = None
            
            # Rapport de classification
            metrics['classification_report'] = classification_report(
                all_targets, all_preds,
                output_dict=True,
                zero_division=0
            )
            
            logger.info(
                f"✅ Évaluation classification complétée - "
                f"accuracy: {metrics['accuracy']}, "
                f"f1: {metrics['f1']}"
            )
            
            return Result.ok(metrics)
            
        except Exception as e:
            logger.error(f"Erreur évaluation classifier: {e}", exc_info=True)
            return Result.err(f"Évaluation classifier échouée: {str(e)}")
    
    def _evaluate_autoencoder(
        self,
        X_test_norm: np.ndarray,
        y_test: np.ndarray
    ) -> Result:
        """
        Évaluation pour autoencoders.
        """
        try:
            test_loader = DataLoaderFactory.create(
                X_test_norm, y_test,
                batch_size=self.training_config.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False
            )
            
            # Calcul des erreurs de reconstruction
            self.model.eval()
            reconstruction_errors = []
            all_targets = []
            
            with torch.no_grad():
                for data, target in test_loader:
                    data = data.to(self.device_manager.device)
                    reconstructed = self.model(data)
                    
                    # Erreur de reconstruction par échantillon
                    errors = torch.mean(
                        (data - reconstructed) ** 2,
                        dim=tuple(range(1, data.ndim))
                    ).cpu().numpy()
                    
                    reconstruction_errors.extend(errors)
                    all_targets.extend(target.numpy())
            
            reconstruction_errors = np.array(reconstruction_errors)
            all_targets = np.array(all_targets)
            
            # SEUIL ADAPTATIF SELON RATIO ANOMALIES
            threshold = self._compute_adaptive_threshold(reconstruction_errors, all_targets)
            
            y_pred = (reconstruction_errors > threshold).astype(int)
            
            # Métriques
            metrics = {
                'mean_reconstruction_error': float(np.mean(reconstruction_errors)),
                'std_reconstruction_error': float(np.std(reconstruction_errors)),
                'median_reconstruction_error': float(np.median(reconstruction_errors)),
                'threshold_adaptive': float(threshold),
                'accuracy': float(accuracy_score(all_targets, y_pred)),
                'precision': float(precision_score(all_targets, y_pred, zero_division=0)),
                'recall': float(recall_score(all_targets, y_pred, zero_division=0)),
                'f1': float(f1_score(all_targets, y_pred, zero_division=0)),
                'confusion_matrix': confusion_matrix(all_targets, y_pred).tolist(),
                'n_samples': len(X_test_norm)
            }
            
            # AUC-ROC
            try:
                metrics['auc_roc'] = float(roc_auc_score(all_targets, reconstruction_errors))
            except:
                metrics['auc_roc'] = None
            
            logger.info(
                f"✅ Évaluation autoencoder complétée - "
                f"mean_error: {metrics['mean_reconstruction_error']:.6f}, "
                f"threshold: {threshold:.6f}, "
                f"accuracy: {metrics['accuracy']:.4f}, "
                f"f1: {metrics['f1']:.4f}"
            )
            
            return Result.ok(metrics)
            
        except Exception as e:
            logger.error(f"❌ Erreur évaluation autoencoder: {e}", exc_info=True)
            return Result.err(f"Évaluation autoencoder échouée: {str(e)}")

    def _compute_adaptive_threshold(
        self, 
        errors: np.ndarray, 
        y_true: np.ndarray
    ) -> float:
        """
        🆕 CORRECTION MAJEURE #5: Calcul du seuil adaptatif basé sur le ratio anomalies.
        
        Args:
            errors: Erreurs de reconstruction
            y_true: Labels réels (0=normal, 1=anomalie)
            
        Returns:
            Seuil optimal adapté au dataset
        """
        # Calcul du ratio anomalies réelles
        anomaly_ratio = np.mean(y_true == 1)
        
        # Stratégie adaptative selon le ratio
        if anomaly_ratio < 0.01:  # < 1% anomalies (MVTec AD typique)
            percentile = 99.5
            strategy = "MVTec AD (< 1% anomalies)"
        elif anomaly_ratio < 0.05:  # < 5% anomalies
            percentile = 98.0
            strategy = "Faible ratio (< 5% anomalies)"
        elif anomaly_ratio < 0.10:  # < 10% anomalies
            percentile = 95.0
            strategy = "Ratio modéré (< 10% anomalies)"
        elif anomaly_ratio < 0.20:  # < 20% anomalies
            percentile = 90.0
            strategy = "Ratio élevé (< 20% anomalies)"
        else:  # >= 20% anomalies (équilibré)
            percentile = 85.0
            strategy = "Dataset équilibré (>= 20% anomalies)"
        
        threshold = np.percentile(errors, percentile)
        
        # Validation: seuil doit être dans l'intervalle raisonnable
        min_error = np.min(errors)
        max_error = np.max(errors)
        error_range = max_error - min_error
        
        if threshold <= min_error:
            threshold = min_error + 0.9 * error_range
            logger.warning(f"⚠️ Seuil trop bas, ajusté à {threshold:.6f}")
        elif threshold >= max_error:
            threshold = max_error - 0.05 * error_range
            logger.warning(f"⚠️ Seuil trop haut, ajusté à {threshold:.6f}")
        
        logger.info(
            f"✅ Seuil adaptatif calculé: {threshold:.6f} - "
            f"Stratégie: {strategy} ({percentile}ème percentile), "
            f"Ratio anomalies: {anomaly_ratio:.2%}"
        )
        
        return threshold
    
    def save_model(self, filepath: Union[str, Path]) -> Result:
        """
        Sauvegarde le modèle complet.
        
        Args:
            filepath: Chemin de sauvegarde
            
        Returns:
            Result indiquant le succès
        """
        try:
            if self.model is None:
                return Result.err("Aucun modèle à sauvegarder")
            
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_config': self.model_config,
                'training_config': self.training_config,
                'history': self.history,
                'preprocessor': self.preprocessor
            }, filepath)
            
            logger.info(f"💾 Modèle sauvegardé: {filepath}")
            return Result.ok(str(filepath))
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde modèle: {e}", exc_info=True)
            return Result.err(f"Sauvegarde échouée: {str(e)}")
    
    @classmethod
    def load_model(cls, filepath: Union[str, Path]) -> Result:
        """
        Charge un modèle sauvegardé.
        
        Args:
            filepath: Chemin du modèle
            
        Returns:
            Result avec instance de ComputerVisionTrainer
        """
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                return Result.err(f"Fichier introuvable: {filepath}")
            
            checkpoint = torch.load(filepath, map_location='cpu')
            
            trainer = cls(
                model_config=checkpoint['model_config'],
                training_config=checkpoint['training_config']
            )
            
            # Reconstruction du modèle
            model_result = trainer._build_model()
            if not model_result.success:
                return model_result
            
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            trainer.history = checkpoint.get('history', {})
            trainer.preprocessor = checkpoint.get('preprocessor')
            
            logger.info(f"✅ Modèle chargé: {filepath}")
            return Result.ok(trainer)
            
        except Exception as e:
            logger.error(f"Erreur chargement modèle: {e}", exc_info=True)
            return Result.err(f"Chargement échoué: {str(e)}")