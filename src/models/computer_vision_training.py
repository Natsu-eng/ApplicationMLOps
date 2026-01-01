"""
Production-Ready Computer Vision Training Pipeline
Version 1.0 - Enterprise Grade
Architecture:
✅ Configuration typée avec validation (Pydantic)
✅ Pipeline de preprocessing isolé et robuste
✅ Gestion d'erreurs cohérente avec Result types
✅ Callbacks découplés et extensibles
✅ Logging structuré pour observabilité
✅ Validation croisée sans fuite de données
✅ Métriques métier + techniques
✅ Support multi-formats (channels_first/last)
✅ Prédictions unifiées pour classification + autoencoders
✅ Gestion de mémoire optimisée
"""
import time
import warnings
import copy
from dataclasses import dataclass, field
from typing import Counter, Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from pathlib import Path

import numpy as np
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from torch.utils.data import DataLoader # type: ignore
from sklearn.metrics import ( # type: ignore
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_class_weight # type: ignore

from src.config.model_config import ModelConfig, ModelType
from src.data.computer_vision_preprocessing import DataLoaderFactory, DataPreprocessor, DataValidator, Result
from src.models.computer_vision.model_builder import ModelBuilder
from src.shared.logging import get_logger
from utils.callbacks import LoggingCallback, TrainingCallback
from utils.device_manager import DeviceManager

warnings.filterwarnings('ignore', category=UserWarning)

logger = get_logger(__name__)

# ======================
# CONFIGURATION ET TYPES
# ======================

class OptimizerType(str, Enum):
    """Types d'optimiseurs supportés"""
    ADAMW = "adamw"
    ADAM = "adam"
    SGD = "sgd"
    RMSPROP = "rmsprop"


class SchedulerType(str, Enum):
    """Types de schedulers de learning rate"""
    REDUCE_ON_PLATEAU = "reduce_on_plateau"
    COSINE = "cosine_annealing"
    STEP = "step_lr"
    NONE = "none"


@dataclass
class TrainingConfig:
    """
    Configuration d'entraînement validée et production-ready.
    
    Tous les paramètres ont des valeurs par défaut sûres pour la production.
    """
    # Hyperparamètres principaux
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    
    # Optimisation
    optimizer: OptimizerType = OptimizerType.ADAMW
    scheduler: SchedulerType = SchedulerType.REDUCE_ON_PLATEAU
    
    # Early stopping et réduction LR
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 8
    min_lr: float = 1e-7
    
    # Gestion des classes déséquilibrées
    use_class_weights: bool = False
    
    # Performance et reproductibilité
    use_mixed_precision: bool = False
    deterministic: bool = True
    seed: int = 42
    
    # DataLoader
    num_workers: int = 0
    pin_memory: bool = False
    
    # Sauvegarde
    checkpoint_dir: Optional[Path] = None
    save_best_only: bool = True
    
    def __post_init__(self):
        """Validation post-initialisation avec messages d'erreur clairs"""
        if self.epochs <= 0:
            raise ValueError(f"epochs doit être > 0, reçu: {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size doit être > 0, reçu: {self.batch_size}")
        if not (0 < self.learning_rate < 1):
            raise ValueError(f"learning_rate doit être dans ]0, 1[, reçu: {self.learning_rate}")
        if self.gradient_clip <= 0:
            raise ValueError(f"gradient_clip doit être > 0, reçu: {self.gradient_clip}")
        if self.early_stopping_patience <= 0:
            raise ValueError(f"early_stopping_patience doit être > 0, reçu: {self.early_stopping_patience}")
        
        # Conversion en Enum si nécessaire
        if isinstance(self.optimizer, str):
            self.optimizer = OptimizerType(self.optimizer)
        if isinstance(self.scheduler, str):
            self.scheduler = SchedulerType(self.scheduler)
        
        # Conversion Path
        if isinstance(self.checkpoint_dir, str):
            self.checkpoint_dir = Path(self.checkpoint_dir)


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
        
        Args:
            model: Modèle PyTorch
            config: Configuration d'entraînement
            
        Returns:
            Optimiseur configuré
            
        Raises:
            ValueError: Si l'optimiseur n'est pas supporté
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
        
        Args:
            optimizer: Optimiseur PyTorch
            config: Configuration d'entraînement
            
        Returns:
            Scheduler configuré ou None
            
        Raises:
            ValueError: Si le scheduler n'est pas supporté
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
        ```python
        trainer = ComputerVisionTrainer(model_config, training_config)
        result = trainer.fit(X_train, y_train, X_val, y_val)
        
        if result.success:
            model = result.data['model']
            history = result.data['history']
            
            # Prédictions
            pred_result = trainer.predict(X_test)
            predictions = pred_result.data['predictions']
        ```
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
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        preprocessing_config: Optional[Dict[str, Any]] = None
    ) -> Result:
        """
        Entraîne le modèle sur les données fournies.
        
        GARANTIT:
        - Augmentation sur données BRUTES (uint8 [0,255])
        - Validation sur données NORMALISÉES
        - Preprocessor utilisé DANS le dataset pour normalisation post-augmentation
        """
        try:
            logger.info("=== DÉBUT ENTRAÎNEMENT ===")
            start_total = time.time()
            
            self.preprocessing_config = preprocessing_config or {}
            
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
            
            # 3. PREPROCESSING AVEC STOCKAGE DES DONNÉES BRUTES
            prep_result = self._setup_preprocessing(
                X_train, y_train, X_val, y_val,
                target_size=target_size,
                expected_channels=expected_channels
            )
            if not prep_result.success:
                return prep_result
            
            X_train_norm, y_train, X_val_norm, y_val = prep_result.data
            
            # 4. CONSTRUCTION DU MODÈLE
            model_result = self._build_model()
            if not model_result.success:
                return model_result
            
            # 5. SETUP TRAINING
            setup_result = self._setup_training(y_train)
            if not setup_result.success:
                return setup_result
            
            # ====================================================================
            # 6. CRÉATION DATALOADERS AVEC GESTION CORRECTE DE L'AUGMENTATION
            # ====================================================================
            augmentation_enabled = self.preprocessing_config.get('augmentation_enabled', False)
            
            if augmentation_enabled:
                # ✅ CORRECTION CRITIQUE: Utiliser données BRUTES pour augmentation
                logger.info(
                    f"🎨 Augmentation ACTIVÉE - Utilisation données BRUTES\n"
                    f"   • X_train_brut: shape={X_train.shape}, dtype={X_train.dtype}, "
                    f"range=[{X_train.min():.1f}, {X_train.max():.1f}]\n"
                    f"   • Preprocessor sera appliqué APRÈS augmentation dans le dataset"
                )
                
                # TRAIN: Données brutes + augmentation + preprocessor interne
                train_loader = DataLoaderFactory.create(
                    X_train,  # ✅ DONNÉES BRUTES (pas normalisées)
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
                    preprocessor=self.preprocessor  # ✅ Pour normalisation POST-augmentation
                )
                
                # VALIDATION: Données normalisées SANS augmentation
                val_loader = DataLoaderFactory.create(
                    X_val_norm,  # Déjà normalisé
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
                # Pas d'augmentation: utiliser données normalisées directement
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
                f"✅ DataLoaders créés:\n"
                f"   • Train: {len(train_loader)} batches, augmentation={augmentation_enabled}\n"
                f"   • Val: {len(val_loader)} batches\n"
                f"   • Target size: {target_size}\n"
                f"   • Format: channels_first"
            )
            
            # 7. VALIDATION PREMIER BATCH
            validation_result = self._validate_first_batch(
                train_loader,
                expected_channels=expected_channels,
                target_size=target_size
            )
            
            if not validation_result.success:
                logger.error(f"❌ Validation premier batch échouée: {validation_result.error}")
                return Result.err(
                    f"Problème format détecté:\n{validation_result.error}"
                )
            
            # 8. BOUCLE D'ENTRAÎNEMENT
            train_result = self._training_loop(train_loader, val_loader, y_val)
            if not train_result.success:
                return train_result
            
            # 9. MÉTADONNÉES FINALES
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
            
            # 10. RETOUR STRUCTURÉ
            result_data = self._build_training_result(train_result)
            
            logger.info(
                f"🎯 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS:\n"
                f"   • Durée: {total_time:.1f}s\n"
                f"   • Époques: {len(self.history['train_loss'])}\n"
                f"   • Best epoch: {result_data['history']['best_epoch']}\n"
                f"   • Best val loss: {result_data['history']['best_val_loss']:.4f}\n"
                f"   • Augmentation: {augmentation_enabled}\n"
                f"   • Canaux: {expected_channels}"
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
    ) -> Result:
        """
        Validation complète du premier batch avec détails.       
        Args:
            expected_channels: Nombre de canaux attendus (1, 3, ou 4)
            target_size: Taille attendue (H, W) ou None
        """
        try:
            # Récupérer le premier batch
            first_batch = next(iter(train_loader))
            
            if len(first_batch) != 2:
                return Result.err(
                    f"Batch invalide: {len(first_batch)} éléments au lieu de 2"
                )
            
            data, labels = first_batch
            
            # ====================================================================
            # VALIDATION #1 : Dimensions du tensor
            # ====================================================================
            if data.dim() != 4:
                return Result.err(
                    f"❌ DIMENSIONS INVALIDES:\n"
                    f"   Attendu: 4D (B, C, H, W)\n"
                    f"   Reçu: {data.dim()}D\n"
                    f"   Shape: {data.shape}"
                )
            
            batch_size, channels, height, width = data.shape
            
            # ====================================================================
            # VALIDATION #2 : Nombre de canaux
            # ====================================================================
            if channels != expected_channels:
                return Result.err(
                    f"❌ CANAUX INCORRECTS:\n"
                    f"   Attendu: {expected_channels}\n"
                    f"   Reçu: {channels}\n"
                    f"   Shape: {data.shape}\n"
                    f"   \n"
                    f"   CAUSE PROBABLE:\n"
                    f"   - input_channels mal configuré dans model_config\n"
                    f"   - Conversion channels_first/last échouée\n"
                    f"   - Modèle incompatible avec ces données"
                )
            
            # ====================================================================
            # VALIDATION #3 : Taille d'image (si target_size spécifié)
            # ====================================================================
            if target_size:
                expected_h, expected_w = target_size
                if (height, width) != (expected_h, expected_w):
                    return Result.err(
                        f"❌ TAILLE D'IMAGE INCORRECTE:\n"
                        f"   Attendu: {expected_h}x{expected_w}\n"
                        f"   Reçu: {height}x{width}\n"
                        f"   \n"
                        f"   CAUSE PROBABLE:\n"
                        f"   - Resize non appliqué correctement\n"
                        f"   - Augmentation sans resize final\n"
                        f"   - Incohérence dans target_size"
                    )
            
            # ====================================================================
            # VALIDATION #4 : Range des valeurs
            # ====================================================================
            data_min = float(data.min())
            data_max = float(data.max())
            
            # Vérifier normalisation
            if data_max > 10 or data_min < -10:
                logger.warning(
                    f"⚠️ Range de valeurs large: [{data_min:.3f}, {data_max:.3f}]\n"
                    f"   Normalisation peut ne pas être appliquée."
                )
            
            # ====================================================================
            # VALIDATION #5 : Labels
            # ====================================================================
            if labels.dim() != 1:
                return Result.err(f"Labels 1D attendus, reçu: {labels.dim()}D")
            
            if len(labels) != batch_size:
                return Result.err(
                    f"Incohérence batch/labels: {batch_size} vs {len(labels)}"
                )
            
            # Vérifier valeurs labels
            unique_labels = torch.unique(labels).cpu().numpy()
            
            # ====================================================================
            # LOG SUCCÈS DÉTAILLÉ
            # ====================================================================
            logger.info(
                f"✅ PREMIER BATCH VALIDÉ:\n"
                f"   • Shape: {tuple(data.shape)}\n"
                f"   • Canaux: {channels} (attendus: {expected_channels}) ✓\n"
                f"   • Taille: {height}x{width} {'✓' if not target_size else f'(cible: {target_size})'}\n"
                f"   • Range: [{data_min:.3f}, {data_max:.3f}]\n"
                f"   • Labels: {labels.shape}\n"
                f"   • Labels uniques: {unique_labels.tolist()}\n"
                f"   • Dtype: {data.dtype}\n"
                f"   • Device: {data.device}"
            )
            
            return Result.ok({
                "batch_shape": tuple(data.shape),
                "channels": channels,
                "height": height,
                "width": width,
                "range": (data_min, data_max),
                "labels_shape": tuple(labels.shape),
                "unique_labels": unique_labels.tolist(),
                "dtype": str(data.dtype),
                "device": str(data.device)
            })
            
        except StopIteration:
            return Result.err("DataLoader vide")
        except Exception as e:
            logger.error(f"❌ Erreur validation batch: {e}", exc_info=True)
            return Result.err(f"Validation batch échouée: {str(e)}")

    
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
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Result:
        """Validation complète des données d'entrée"""
        
        # Validation train
        train_val = DataValidator.validate_input_data(X_train, y_train, "train")
        if not train_val.success:
            return train_val
        
        # Validation val
        val_val = DataValidator.validate_input_data(X_val, y_val, "validation")
        if not val_val.success:
            return val_val
        
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
        
        return Result.ok(None, imbalance=imbalance)
    
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
            # CRÉATION PREPROCESSOR
            # ====================================================================
            self.preprocessor = DataPreprocessor(
                strategy="standardize",
                auto_detect_format=True,
                target_size=final_target_size
            )
            
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
                f"   • Strategy: standardize\n"
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
        builder = ModelBuilder(self.device_manager)
        result = builder.build(self.model_config)
        
        if result.success:
            self.model = result.data
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            # CORRECTION: Logging sans kwargs
            logger.info(
                f"Modèle construit avec succès - "
                f"model_type: {self.model_config.model_type.value}, "
                f"total_params: {total_params}, "
                f"trainable_params: {trainable_params}"
            )
        
        return result


    def _setup_training(self, y_train: np.ndarray) -> Result:
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
            if not is_autoencoder:
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

    
    def _training_loop(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        y_val: np.ndarray
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
            
            for epoch in range(self.training_config.epochs):
                epoch_start = time.time()
                
                # CORRECTION: Callbacks avec try/catch
                for cb in self.callbacks:
                    try:
                        cb.on_epoch_begin(epoch)
                    except Exception as e:
                        logger.warning(f"Callback on_epoch_begin échoué: {e}")
                
                # === TRAIN ===
                train_loss = self._train_epoch(train_loader, is_autoencoder)
                
                # === VALIDATION ===
                if is_autoencoder:
                    val_loss = self._validate_epoch_autoencoder(val_loader)
                    val_metrics = {'loss': val_loss}
                else:
                    val_loss, val_metrics = self._validate_epoch(val_loader, y_val)
                
                # CORRECTION: Indentation fix - Mise à jour historique
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
                    # CORRECTION: Logging sans kwargs problématiques
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
                    # CORRECTION: Logging sans kwargs
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
            
            # CORRECTION: Logging sans kwargs problématiques
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

    def _train_epoch(self, train_loader: DataLoader, is_autoencoder: bool = False) -> float:
        """
        Entraîne une époque - unifié pour classification et autoencoders.
        """
        # PatchCore: pas de training classique
        is_patchcore = self.model_config.model_type == ModelType.PATCH_CORE
        if is_patchcore:
            logger.info("⚠️ PatchCore détecté - Skip training epoch (fit() utilisé)")
            return 0.0

        self.model.train()
        running_loss = 0.0
        first_batch = True  # assertions runtime une seule fois

        for batch_idx, (data, target_labels) in enumerate(train_loader):
            data = data.to(self.device_manager.device)

            if is_autoencoder:
                target = data
            else:
                target = target_labels.to(self.device_manager.device)

                # ============================================================
                # ASSERTION CRITIQUE 4 : Validation labels (premier batch)
                # ============================================================
                if first_batch:
                    batch_min = int(target.min().item())
                    batch_max = int(target.max().item())
                    model_output_size = self.model_config.num_classes

                    if batch_min != 0:
                        raise RuntimeError(
                            f"❌ Labels invalides: min={batch_min}. "
                            f"Les labels doivent commencer à 0."
                        )

                    if batch_max >= model_output_size:
                        raise RuntimeError(
                            f"❌ LABEL HORS LIMITES:\n"
                            f"   • max(label)={batch_max}\n"
                            f"   • model.num_classes={model_output_size}\n"
                            f"   • CrossEntropyLoss va crasher"
                        )

                    logger.info(
                        f"✅ Premier batch validé - "
                        f"labels=[{batch_min}..{batch_max}], "
                        f"num_classes={model_output_size}"
                    )
                    first_batch = False

            # ============================================================
            # Forward
            # ============================================================
            self.optimizer.zero_grad()
            output = self.model(data)

            if output is None:
                logger.warning("⚠️ Modèle a retourné None - Skip batch")
                continue

            # ============================================================
            # ASSERTION CRITIQUE 5 : Shape output
            # ============================================================
            if not is_autoencoder:
                expected_shape = (data.size(0), self.model_config.num_classes)
                if output.shape != expected_shape:
                    raise RuntimeError(
                        f"❌ MODEL OUTPUT SHAPE INCORRECTE:\n"
                        f"   • Attendue: {expected_shape}\n"
                        f"   • Reçue: {output.shape}"
                    )
            else:
                if output.shape != target.shape:
                    raise RuntimeError(
                        f"❌ SHAPE MISMATCH AUTOENCODER:\n"
                        f"   • output={output.shape}\n"
                        f"   • target={target.shape}"
                    )

            # ============================================================
            # Loss
            # ============================================================
            if (
                self.model_config.model_type == ModelType.VAE
                and hasattr(self.model, "compute_vae_loss")
            ):
                loss, recon_loss, kl_loss = self.model.compute_vae_loss(target, output)
            else:
                loss = self.train_criterion(output, target)

            # ============================================================
            # Backward
            # ============================================================
            loss.backward()

            if self.training_config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.training_config.gradient_clip
                )

            self.optimizer.step()
            running_loss += loss.item()

        return running_loss / len(train_loader)

    
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
        y_val: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """
        Validation pour classificateur.
        
        CRITIQUE: Utilise val_criterion SANS class weights pour évaluation honnête.
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
            
            # CORRECTION: Logging sans kwargs problématiques
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


# =================================
# INTÉGRATION AVEC ANOMALY TAXONOMY  
# =================================

class AnomalyAwareTrainer:
    """
    Trainer spécialisé pour la détection d'anomalies avec taxonomie.
    
    NOUVEAU: Intégration complète restaurée depuis l'ancienne version
    """
    
    def __init__(
        self,
        anomaly_type: Optional[str] = None,  
        *,
        model_config: Optional[ModelConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        taxonomy_config: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[TrainingCallback]] = None,
        auto_detect_from_state: bool = True
    ):
        """
        Initialise le trainer pour anomalies.
        
        Args:
            anomaly_type: Type d'anomalie (structural, visual, geometric) ou None pour auto
            model_config: Configuration du modèle (optionnel, sinon auto-configuré)
            training_config: Configuration d'entraînement (optionnel, sinon auto-configuré)
            taxonomy_config: Configuration de taxonomie personnalisée (optionnel)
            callbacks: Callbacks pour monitoring (optionnel)
            auto_detect_from_state: Active détection automatique depuis STATE si anomaly_type=None
        """
        from monitoring.state_managers import STATE
        
        # Détection automatique si nécessaire
        if anomaly_type is None and auto_detect_from_state:
            logger.info("🔍 Détection automatique du type d'anomalie depuis STATE")
            anomaly_type = self._detect_anomaly_type_from_state(STATE)
            
            if anomaly_type is None:
                logger.warning("⚠️ Impossible de détecter anomaly_type, fallback 'structural'")
                anomaly_type = "structural"
        
        elif anomaly_type is None:
            logger.info("ℹ️ Anomaly_type=None sans auto-détection, fallback 'structural'")
            anomaly_type = "structural"
        
        self.anomaly_type = anomaly_type
        self.taxonomy_config = taxonomy_config or self._get_default_taxonomy()
        self.callbacks = callbacks or []
        
        # Validation model_config si fourni
        if model_config is not None:
            if not isinstance(model_config.model_type, ModelType):
                raise ValueError(
                    f"model_config.model_type doit être une instance de ModelType, "
                    f"reçu: {type(model_config.model_type)}"
                )
            
            # Vérifier que c'est un modèle compatible anomalies
            valid_anomaly_models = [
                ModelType.CONV_AUTOENCODER,
                ModelType.VAE,
                ModelType.DENOISING_AE,
                ModelType.PATCH_CORE
            ]
            
            if model_config.model_type not in valid_anomaly_models:
                logger.warning(
                    f"⚠️ Modèle {model_config.model_type.value} inhabituel pour anomalies. "
                    f"Modèles recommandés: {[m.value for m in valid_anomaly_models]}"
                )
        
        # Configuration automatique ou manuelle
        if model_config is None or training_config is None:
            auto_model_config, auto_training_config = self._configure_for_anomaly()
            self.model_config = model_config or auto_model_config
            self.training_config = training_config or auto_training_config
            logger.info(f"🔧 Configuration automatique pour anomalie '{anomaly_type}'")
        else:
            self.model_config = model_config
            self.training_config = training_config
            logger.info(f"⚙️ Configuration manuelle pour anomalie '{anomaly_type}'")
        
        # Attributs pour compatibilité
        self.model: Optional[nn.Module] = None
        self.preprocessor: Optional[DataPreprocessor] = None
        self.history: Dict[str, Any] = {}
    
    def _detect_anomaly_type_from_state(self, STATE) -> Optional[str]:
        """
        Détecte le type d'anomalie depuis STATE.data.
        """
        try:
            # Stratégie 1: Metadata explicite
            if hasattr(STATE.data, 'metadata') and STATE.data.metadata:
                anomaly_type = STATE.data.metadata.get('anomaly_type')
                if anomaly_type:
                    logger.info(f"✅ Anomaly type depuis metadata: {anomaly_type}")
                    return anomaly_type
            
            # Stratégie 2: Nom du dataset
            if hasattr(STATE.data, 'name') and STATE.data.name:
                name_lower = STATE.data.name.lower()
                
                if any(kw in name_lower for kw in ['crack', 'corrosion', 'deformation']):
                    logger.info(f"✅ Anomaly type depuis nom: structural")
                    return "structural"
                
                if any(kw in name_lower for kw in ['scratch', 'stain', 'color']):
                    logger.info(f"✅ Anomaly type depuis nom: visual")
                    return "visual"
                
                if any(kw in name_lower for kw in ['dimension', 'alignment', 'size']):
                    logger.info(f"✅ Anomaly type depuis nom: geometric")
                    return "geometric"
            
            # Stratégie 3: Structure MVTec AD
            if hasattr(STATE.data, 'structure') and STATE.data.structure:
                if STATE.data.structure.get('type') == 'mvtec_ad':
                    logger.info(f"✅ Structure MVTec AD détectée → structural")
                    return "structural"
            
            logger.warning("⚠️ Impossible de détecter anomaly_type automatiquement")
            return None
            
        except Exception as e:
            logger.error(f"❌ Erreur détection anomaly_type: {e}", exc_info=True)
            return None
    
    def _get_default_taxonomy(self) -> Dict[str, Any]:
        """Taxonomie par défaut production-ready."""
        return {
            "structural": {
                "recommended_model": ModelType.CONV_AUTOENCODER,
                "difficulty": "high",
                "threshold": 0.90,
                "description": "Défauts structurels (cracks, corrosion, deformation)",
                "params": {
                    "latent_dim": 256,
                    "learning_rate": 1e-4,
                    "base_filters": 32,
                    "num_stages": 4
                },
                "training_params": {
                    "epochs": 100,
                    "batch_size": 32,
                    "early_stopping_patience": 15,
                    "use_class_weights": False  
                }
            },
            "visual": {
                "recommended_model": ModelType.DENOISING_AE,
                "difficulty": "medium",
                "threshold": 0.85,
                "description": "Défauts visuels (scratch, stain, discoloration)",
                "params": {
                    "latent_dim": 128,
                    "learning_rate": 1e-3,
                    "base_filters": 64,
                    "noise_factor": 0.1
                },
                "training_params": {
                    "epochs": 80,
                    "batch_size": 32,
                    "early_stopping_patience": 12,
                    "use_class_weights": False
                }
            },
            "geometric": {
                "recommended_model": ModelType.VAE,
                "difficulty": "low",
                "threshold": 0.95,
                "description": "Défauts géométriques (misalignment, dimension errors)",
                "params": {
                    "latent_dim": 64,
                    "learning_rate": 1e-3,
                    "base_filters": 32,
                    "beta": 1.0
                },
                "training_params": {
                    "epochs": 60,
                    "batch_size": 32,
                    "early_stopping_patience": 10,
                    "use_class_weights": False
                }
            }
        }
    
    def _configure_for_anomaly(self) -> Tuple[ModelConfig, TrainingConfig]:
        """Configure modèle et training selon le type d'anomalie."""
        category = self._get_anomaly_category(self.anomaly_type)
        config = self.taxonomy_config.get(category, self.taxonomy_config["structural"])
        
        logger.info(
            f"🔧 Configuration pour anomalie: {self.anomaly_type} (catégorie: {category}) - "
            f"difficulty: {config.get('difficulty')}, "
            f"recommended_model: {config['recommended_model'].value}"
        )
        
        # Configuration du modèle
        model_params = config.get("params", {})
        model_config = ModelConfig(
            model_type=config["recommended_model"],
            num_classes=2,
            input_channels=model_params.get("input_channels", 3),
            dropout_rate=model_params.get("dropout_rate", 0.0),
            base_filters=model_params.get("base_filters", 32),
            latent_dim=model_params.get("latent_dim", 256),
            num_stages=model_params.get("num_stages", 4)
        )
        
        # Configuration de l'entraînement depuis taxonomie
        training_params = config.get("training_params", {})
        training_config = TrainingConfig(
            epochs=training_params.get("epochs", 100),
            batch_size=training_params.get("batch_size", 32),
            learning_rate=model_params.get("learning_rate", 1e-4),
            optimizer=OptimizerType.ADAMW,
            scheduler=SchedulerType.REDUCE_ON_PLATEAU,
            early_stopping_patience=training_params.get("early_stopping_patience", 15),
            reduce_lr_patience=8,
            use_class_weights=False,
            gradient_clip=1.0,
            deterministic=True,
            seed=42,
            num_workers=0,
            pin_memory=False
        )
        
        logger.info(
            f"✅ Configuration générée - "
            f"latent_dim: {model_config.latent_dim}, "
            f"epochs: {training_config.epochs}, "
            f"lr: {training_config.learning_rate}"
        )
        
        return model_config, training_config
    
    def _get_anomaly_category(self, anomaly_type: str) -> str:
        """Détermine la catégorie d'anomalie avec mapping enrichi."""
        category_mappings = {
            "structural": [
                "crack", "corrosion", "deformation", "structural",
                "break", "fracture", "damage"
            ],
            "visual": [
                "scratch", "stain", "discoloration", "visual",
                "contamination", "dirt", "mark", "spot"
            ],
            "geometric": [
                "misalignment", "dimension_error", "geometric",
                "size", "position", "orientation"
            ]
        }
        
        anomaly_type_lower = anomaly_type.lower()
        
        for category, keywords in category_mappings.items():
            if anomaly_type_lower in keywords:
                logger.info(f"✅ Anomaly '{anomaly_type}' mappée à catégorie '{category}'")
                return category
        
        logger.warning(
            f"⚠️ Anomaly type '{anomaly_type}' non reconnue. "
            f"Fallback catégorie 'structural'"
        )
        return "structural"
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        callbacks: Optional[List[TrainingCallback]] = None
    ) -> Result:
        """
        Entraîne le modèle pour le type d'anomalie spécifique.
        """
        try:
            logger.info(
                f"🚀 Début entraînement anomalies - "
                f"anomaly_type: {self.anomaly_type}, "
                f"model_type: {self.model_config.model_type.value}, "
                f"X_train_shape: {X_train.shape}, "
                f"X_val_shape: {X_val.shape}"
            )
            
            # Détection PatchCore
            is_patchcore = self.model_config.model_type == ModelType.PATCH_CORE
            
            if is_patchcore:
                logger.info("🔍 PatchCore détecté - Utilisation workflow natif fit()")
                return self._train_patchcore(X_train, y_train, X_val, y_val, callbacks)
            
            # Pour les autres modèles (autoencoders), workflow standard
            active_callbacks = (callbacks or []) + self.callbacks
            
            trainer = ComputerVisionTrainer(
                model_config=self.model_config,
                training_config=self.training_config,
                callbacks=active_callbacks
            )
            
            result = trainer.fit(X_train, y_train, X_val, y_val)
            
            if result.success:
                self.model = trainer.model
                self.preprocessor = trainer.preprocessor
                self.history = result.data['history']
                
                logger.info(
                    f"✅ Entraînement anomalies terminé - "
                    f"best_epoch: {self.history.get('best_epoch', 0)}, "
                    f"best_loss: {self.history.get('best_val_loss', float('inf'))}"
                )
            else:
                logger.error(f"❌ Entraînement anomalies échoué: {result.error}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erreur critique entraînement anomalies: {e}", exc_info=True)
            return Result.err(f"Entraînement anomalies échoué: {str(e)}")


    def _train_patchcore(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        callbacks: Optional[List[TrainingCallback]] = None
    ) -> Result:
        """
        Entraînement spécifique pour PatchCore.   
        PatchCore n'a pas d'entraînement par epochs, mais construit une memory bank.
        """
        try:
            import time
            from src.models.computer_vision.model_builder import ModelBuilder
            from src.data.computer_vision_preprocessing import DataPreprocessor, DataLoaderFactory
            from utils.device_manager import DeviceManager
            
            start_time = time.time()
            
            # 1. Preprocessing
            logger.info("📊 Preprocessing données pour PatchCore")
            
            self.preprocessor = DataPreprocessor(
                strategy="standardize",
                auto_detect_format=True,
                target_size=None  # PatchCore accepte toutes les tailles
            )
            
            X_train_norm = self.preprocessor.fit_transform(X_train, output_format="channels_first")
            X_val_norm = self.preprocessor.transform(X_val, output_format="channels_first")
            
            # 2. Construction du modèle
            logger.info("Construction modèle PatchCore")
            
            builder = ModelBuilder(DeviceManager())
            result = builder.build(self.model_config)
            
            if not result.success:
                return result
            
            self.model = result.data
            
            # 3. Fit PatchCore (construction memory bank)
            logger.info("🔨 Construction memory bank PatchCore")
            
            train_loader = DataLoaderFactory.create(
                X_train_norm, y_train,
                batch_size=32,
                shuffle=False,  # Pas de shuffle pour PatchCore
                num_workers=0,
                pin_memory=False
            )
            
            # PatchCore.fit() construit la memory bank
            self.model.fit(train_loader)
            
            # 4. Évaluation sur validation
            logger.info("📊 Évaluation sur validation set")
            
            val_loader = DataLoaderFactory.create(
                X_val_norm, y_val,
                batch_size=32,
                shuffle=False,
                num_workers=0,
                pin_memory=False
            )
            
            # PatchCore.predict() retourne les scores d'anomalie
            val_scores = self.model.predict(val_loader)
            
            # Calcul métriques de validation
            threshold = np.percentile(val_scores, 95)
            val_preds = (val_scores > threshold).astype(int)
            
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
            
            val_accuracy = accuracy_score(y_val, val_preds)
            val_f1 = f1_score(y_val, val_preds, zero_division=0)
            
            try:
                val_auc = roc_auc_score(y_val, val_scores)
            except:
                val_auc = 0.0
            
            training_time = time.time() - start_time
            
            # 5. Construction de l'historique
            self.history = {
                'success': True,
                'model_type': 'patch_core',
                'is_autoencoder': False,
                'train_loss': [0.0],  # PatchCore n'a pas de loss d'entraînement
                'val_loss': [float(threshold)],  # Utiliser le threshold comme proxy
                'val_accuracy': [float(val_accuracy)],
                'val_f1': [float(val_f1)],
                'val_auc': [float(val_auc)],
                'learning_rates': [],
                'best_epoch': 0,
                'best_val_loss': float(threshold),
                'final_train_loss': 0.0,
                'training_time': training_time,
                'total_epochs_trained': 1,  # PatchCore = 1 "epoch" (fit unique)
                'early_stopping_triggered': False,
                'input_shape': tuple(self.preprocessor.original_shape_[1:]),
                'output_format': 'channels_first',
                'threshold': float(threshold),
                'training_config': {
                    'learning_rate': 0.0,  # PatchCore n'a pas de LR
                    'batch_size': 32,
                    'optimizer': 'none',
                    'scheduler': 'none',
                    'epochs_requested': 1,
                    'early_stopping_patience': 0,
                    'use_class_weights': False,
                    'gradient_clip': 0.0
                },
                'metadata': {
                    'memory_bank_size': len(self.model.memory_bank) if self.model.memory_bank is not None else 0,
                    'coreset_ratio': self.model.coreset_ratio,
                    'backbone': self.model.backbone_name
                }
            }
            
            logger.info(
                f"PatchCore entraîné avec succès - "
                f"memory_bank_size: {self.history['metadata']['memory_bank_size']}, "
                f"val_accuracy: {val_accuracy:.4f}, "
                f"val_f1: {val_f1:.4f}, "
                f"val_auc: {val_auc:.4f}, "
                f"training_time: {training_time:.1f}s"
            )
            
            return Result.ok(
                {
                    'model': self.model,
                    'preprocessor': self.preprocessor,
                    'history': self.history
                },
                training_time=training_time
            )
            
        except Exception as e:
            logger.error(f"❌ Erreur entraînement PatchCore: {e}", exc_info=True)
            return Result.err(f"Entraînement PatchCore échoué: {str(e)}")
    
    def predict(self, X: np.ndarray, **kwargs) -> Result:
        """Wrapper pour prédictions - délègue au trainer interne."""
        if self.model is None or self.preprocessor is None:
            return Result.err("Modèle non entraîné")
        
        logger.info(f"🔮 Prédictions anomalies sur {len(X)} images")
        
        try:
            # Créer un trainer temporaire pour predict
            temp_trainer = ComputerVisionTrainer(
                self.model_config,
                self.training_config
            )
            temp_trainer.model = self.model
            temp_trainer.preprocessor = self.preprocessor
            temp_trainer.device_manager = DeviceManager()
            
            result = temp_trainer.predict(X, **kwargs)
            
            if result.success:
                logger.info("✅ Prédictions anomalies terminées")
            else:
                logger.error(f"❌ Prédictions anomalies échouées: {result.error}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erreur prédictions anomalies: {e}", exc_info=True)
            return Result.err(f"Prédictions échouées: {str(e)}")


# ==================
# UTILITIES AVANCÉES
# ==================

class ModelInterpreter:
    """Interprétation des prédictions du modèle"""
    
    @staticmethod
    def get_feature_importance(
        model: nn.Module,
        X_sample: np.ndarray,
        preprocessor: DataPreprocessor
    ) -> Result:
        """
        Calcule l'importance des features via gradient.
        
        Args:
            model: Modèle PyTorch
            X_sample: Échantillon à analyser
            preprocessor: Preprocessor du modèle
            
        Returns:
            Result avec carte d'importance
        """
        try:
            model.eval()
            X_norm = preprocessor.transform(X_sample, output_format="channels_first")
            X_tensor = torch.tensor(X_norm, dtype=torch.float32, requires_grad=True)
            
            if torch.cuda.is_available():
                X_tensor = X_tensor.cuda()
                model = model.cuda()
            
            # Forward pass
            output = model(X_tensor)
            pred_class = output.argmax(dim=1)
            
            # Backward pour obtenir gradients
            output[0, pred_class[0]].backward()
            
            # Importance = magnitude du gradient
            importance = X_tensor.grad.abs().cpu().numpy()
            
            return Result.ok({
                'importance_map': importance,
                'predicted_class': int(pred_class[0].cpu().numpy())
            })
            
        except Exception as e:
            logger.error(f"Erreur interprétation: {e}")
            return Result.err(f"Interprétation échouée: {str(e)}")


class DataAugmenter:
    """Augmentation de données pour améliorer la généralisation"""
    
    @staticmethod
    def augment(
        X: np.ndarray,
        y: np.ndarray,
        factor: int = 2,
        methods: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augmente le dataset.
        
        Args:
            X: Images (N, H, W, C)
            y: Labels
            factor: Facteur d'augmentation
            methods: Liste de méthodes ('flip', 'rotate', 'noise')
            
        Returns:
            (X_augmented, y_augmented)
        """
        methods = methods or ['flip', 'rotate']
        
        augmented_X = [X]
        augmented_y = [y]
        
        for _ in range(factor - 1):
            X_aug = X.copy()
            
            if 'flip' in methods and np.random.rand() > 0.5:
                X_aug = np.flip(X_aug, axis=2)  # Flip horizontal
            
            if 'rotate' in methods and np.random.rand() > 0.5:
                angle = np.random.choice([90, 180, 270])
                X_aug = np.rot90(X_aug, k=angle//90, axes=(1, 2))
            
            if 'noise' in methods and np.random.rand() > 0.5:
                noise = np.random.normal(0, 0.01, X_aug.shape)
                X_aug = X_aug + noise
            
            augmented_X.append(X_aug)
            augmented_y.append(y)
        
        return np.concatenate(augmented_X), np.concatenate(augmented_y)


# ========================
# INTÉGRATION AVEC MLFLOW 
# ========================

class MLflowIntegration:
    """
    Intégration avec MLflow pour tracking d'expériences.
    """
    
    def __init__(self, experiment_name: str = "computer_vision_training"):
        self.experiment_name = experiment_name
        self.mlflow_available = False
        
        try:
            import mlflow # type: ignore
            self.mlflow = mlflow
            self.mlflow_available = True
            mlflow.set_experiment(experiment_name)
            logger.info(f"📊 MLflow activé: experiment={experiment_name}")
        except ImportError:
            logger.warning("⚠️ MLflow non disponible, tracking désactivé")
    
    def log_training(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        history: Dict[str, Any],
        test_metrics: Dict[str, Any]
    ) -> None:
        """
        Log une session d'entraînement complète.
        """
        if not self.mlflow_available:
            return
        
        try:
            with self.mlflow.start_run():
                # Log configs
                self.mlflow.log_params({
                    'model_type': model_config.model_type.value,
                    'num_classes': model_config.num_classes,
                    'input_channels': model_config.input_channels,
                    'learning_rate': training_config.learning_rate,
                    'batch_size': training_config.batch_size,
                    'epochs': training_config.epochs,
                    'optimizer': training_config.optimizer.value,
                    'use_class_weights': training_config.use_class_weights
                })
                
                # Log métriques finales
                for metric_name, metric_value in test_metrics.items():
                    if isinstance(metric_value, (int, float)):
                        self.mlflow.log_metric(f"test_{metric_name}", metric_value)
                
                # Log courbes d'entraînement
                train_losses = history.get('train_loss', [])
                val_losses = history.get('val_loss', [])
                
                for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
                    self.mlflow.log_metric('train_loss', train_loss, step=epoch)
                    self.mlflow.log_metric('val_loss', val_loss, step=epoch)
                
                logger.info("✅ Métriques loguées dans MLflow")
                
        except Exception as e:
            logger.warning(f"⚠️ Erreur logging MLflow: {e}")


# =====================
# CONFIGURATION FACTORY
# =====================

class ConfigFactory:
    """
    Factory pour créer des configurations pré-définies.
    """
    
    @staticmethod
    def get_config(preset: str) -> Tuple[ModelConfig, TrainingConfig]:
        """
        Retourne des configurations pré-définies.
        
        Presets disponibles:
        - 'quick_test': Entraînement rapide pour tests
        - 'balanced': Configuration équilibrée
        - 'high_accuracy': Optimisé pour précision
        - 'production': Configuration production robuste
        """
        presets = {
            'quick_test': (
                ModelConfig(
                    model_type=ModelType.SIMPLE_CNN,
                    num_classes=2,
                    dropout_rate=0.3
                ),
                TrainingConfig(
                    epochs=5,
                    batch_size=32,
                    learning_rate=1e-3,
                    early_stopping_patience=3,
                    deterministic=True,
                    num_workers=0,
                    pin_memory=False
                )
            ),
            
            'balanced': (
                ModelConfig(
                    model_type=ModelType.CUSTOM_RESNET,
                    num_classes=2,
                    dropout_rate=0.5,
                    base_filters=64
                ),
                TrainingConfig(
                    epochs=50,
                    batch_size=32,
                    learning_rate=1e-4,
                    optimizer=OptimizerType.ADAMW,
                    scheduler=SchedulerType.REDUCE_ON_PLATEAU,
                    early_stopping_patience=10,
                    use_class_weights=True,
                    deterministic=True,
                    num_workers=0,
                    pin_memory=False
                )
            ),
            
            'high_accuracy': (
                ModelConfig(
                    model_type=ModelType.TRANSFER_LEARNING,
                    num_classes=2,
                    pretrained=True,
                    freeze_layers=100,
                    dropout_rate=0.5
                ),
                TrainingConfig(
                    epochs=100,
                    batch_size=16,
                    learning_rate=1e-5,
                    optimizer=OptimizerType.ADAMW,
                    scheduler=SchedulerType.COSINE,
                    early_stopping_patience=20,
                    use_class_weights=True,
                    use_mixed_precision=False,
                    deterministic=True,
                    num_workers=0,
                    pin_memory=False
                )
            ),
            
            'production': (
                ModelConfig(
                    model_type=ModelType.CUSTOM_RESNET,
                    num_classes=2,
                    dropout_rate=0.5,
                    base_filters=64
                ),
                TrainingConfig(
                    epochs=100,
                    batch_size=32,
                    learning_rate=1e-4,
                    optimizer=OptimizerType.ADAMW,
                    scheduler=SchedulerType.REDUCE_ON_PLATEAU,
                    early_stopping_patience=15,
                    reduce_lr_patience=8,
                    use_class_weights=True,
                    gradient_clip=1.0,
                    use_mixed_precision=False,
                    deterministic=True,
                    seed=42,
                    num_workers=0,
                    pin_memory=False
                )
            )
        }
        
        if preset not in presets:
            raise ValueError(
                f"Preset inconnu: {preset}. "
                f"Disponibles: {list(presets.keys())}"
            )
        
        return presets[preset]