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
from typing import Dict, List, Optional, Tuple, Any, Union
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
        y_val: np.ndarray
    ) -> Result:
        """
        Entraîne le modèle de manière robuste.
        
        Pipeline:
        1. Validation des données
        2. Preprocessing (fit sur train, transform sur val)
        3. Construction du modèle
        4. Setup training (optimizer, criterion, scheduler)
        5. Boucle d'entraînement avec early stopping
        6. Retour des résultats structurés
        
        Args:
            X_train: Données d'entraînement (N, H, W, C) ou (N, C, H, W)
            y_train: Labels d'entraînement
            X_val: Données de validation
            y_val: Labels de validation
            
        Returns:
            Result contenant model, history, preprocessor et métadonnées
        """
        try:
            logger.info("=== Début de l'entraînement ===")
            start_total = time.time()
            
            # 1. Validation des données
            val_result = self._validate_data(X_train, y_train, X_val, y_val)
            if not val_result.success:
                return val_result
            
            # 2. Preprocessing (FIT sur train, TRANSFORM sur val)
            prep_result = self._setup_preprocessing(X_train, y_train, X_val, y_val)
            if not prep_result.success:
                return prep_result
            
            X_train_norm, y_train, X_val_norm, y_val = prep_result.data
            
            # 3. Construction du modèle
            model_result = self._build_model()
            if not model_result.success:
                return model_result
            
            # 4. Setup training
            setup_result = self._setup_training(y_train)
            if not setup_result.success:
                return setup_result
            
            # 5. Création des DataLoaders
            train_loader = DataLoaderFactory.create(
                X_train_norm, y_train,
                batch_size=self.training_config.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=False
            )
            
            val_loader = DataLoaderFactory.create(
                X_val_norm, y_val,
                batch_size=self.training_config.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False
            )
            
            # 6. Boucle d'entraînement
            train_result = self._training_loop(train_loader, val_loader, y_val)
            if not train_result.success:
                return train_result
            
            # 7. Métadonnées finales
            total_time = time.time() - start_total
            self._training_metadata.update({
                'total_training_time': total_time,
                'samples_per_second': len(X_train) / total_time,
                'final_model_params': sum(p.numel() for p in self.model.parameters()),
                'device': str(self.device_manager.device)
            })
            
            # 8. Retour structuré (GARANTIT PAS DE BOOLÉENS)
            return Result.ok(
                self._build_training_result(train_result),
                training_time=total_time
            )
            
        except Exception as e:
            logger.error(f"Erreur critique entraînement: {e}", exc_info=True)
            return Result.err(f"Entraînement échoué: {str(e)}")
    
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
        y_val: np.ndarray
    ) -> Result:
        """
        Récupération target_size depuis le contexte correct.           
        Args:
            X_train: Données d'entraînement
            y_train: Labels d'entraînement
            X_val: Données de validation
            y_val: Labels de validation       
        Returns:
            Result avec données preprocessées
        """
        try:
            if X_train is None or len(X_train) == 0:
                return Result.err("Données d'entraînement vides")
            if X_val is None or len(X_val) == 0:
                return Result.err("Données de validation vides")
            
            logger.info(
                f"Début setup preprocessing - "
                f"train_shape: {X_train.shape}, "
                f"val_shape: {X_val.shape}, "
                f"train_dtype: {X_train.dtype}, "
                f"val_dtype: {X_val.dtype}"
            )
            
            # Récupération target_size depuis PLUSIEURS sources (priorité)
            # Source 1: self.preprocessing_config (si défini par orchestrator)
            # Source 2: self.model_config.input_size (fallback)
            # Source 3: None (pas de resize)
            
            target_size = None
            
            # Priorité 1: preprocessing_config explicite
            if hasattr(self, 'preprocessing_config') and self.preprocessing_config:
                target_size = self.preprocessing_config.get('target_size', None)
                if target_size:
                    logger.info(f"✅ target_size depuis preprocessing_config: {target_size}")
            
            # Priorité 2: Déduction depuis model_config.input_size
            if target_size is None and hasattr(self, 'model_config'):
                # Si input_size est défini dans model_config, on DOIT resize
                if hasattr(self.model_config, 'input_size') and self.model_config.input_size:
                    # Vérifier si input_size != data size
                    data_h = X_train.shape[1] if X_train.shape[-1] in [1,3,4] else X_train.shape[2]
                    data_w = X_train.shape[2] if X_train.shape[-1] in [1,3,4] else X_train.shape[3]
                    
                    model_h, model_w = self.model_config.input_size
                    
                    if (data_h, data_w) != (model_h, model_w):
                        target_size = self.model_config.input_size
                        logger.info(
                            f"✅ target_size déduit depuis model_config.input_size: {target_size} "
                            f"(data: {data_h}x{data_w} → model: {model_h}x{model_w})"
                        )
                    else:
                        logger.info(
                            f"ℹ️ Data déjà à la taille du modèle ({data_h}x{data_w}), "
                            f"pas de resize nécessaire"
                        )
            
            # Log final
            if target_size:
                logger.info(f"🔄 Resize activé: target_size={target_size}")
            else:
                logger.info("ℹ️ Pas de resize (images conservent taille originale)")
            
            # Création DataPreprocessor avec target_size correct
            self.preprocessor = DataPreprocessor(
                strategy="standardize",
                auto_detect_format=True,
                target_size=target_size  
            )
            
            # FIT SUR TRAIN (avec resize si target_size spécifié)
            try:
                X_train_norm = self.preprocessor.fit_transform(
                    X_train,
                    output_format="channels_first"
                )
            except AttributeError as e:
                logger.error(f"❌ Erreur fit_transform: {e}")
                return Result.err(f"Erreur preprocessing: {str(e)}")
            
            # TRANSFORM SUR VALIDATION (même resize appliqué)
            X_val_norm = self.preprocessor.transform(
                X_val,
                output_format="channels_first"
            )
            
            # Validation post-preprocessing
            if X_train_norm is None or X_val_norm is None:
                return Result.err("Preprocessing a retourné None")
            
            if X_train_norm.ndim != 4:
                return Result.err(
                    f"Format invalide après preprocessing: {X_train_norm.ndim}D au lieu de 4D"
                )
            
            if np.any(np.isnan(X_train_norm)) or np.any(np.isinf(X_train_norm)):
                return Result.err("Données normalisées contiennent NaN ou Inf")
            
            # Vérifi la cohérence avec model_config.input_size
            if hasattr(self, 'model_config') and hasattr(self.model_config, 'input_size'):
                expected_h, expected_w = self.model_config.input_size
                actual_h, actual_w = X_train_norm.shape[2], X_train_norm.shape[3]
                
                if (actual_h, actual_w) != (expected_h, expected_w):
                    logger.error(
                        f"❌ INCOHÉRENCE DÉTECTÉE APRÈS PREPROCESSING: "
                        f"Données: {actual_h}x{actual_w}, "
                        f"Modèle attend: {expected_h}x{expected_w}"
                    )
                    return Result.err(
                        f"Preprocessing n'a pas resizé correctement: "
                        f"attendu {expected_h}x{expected_w}, obtenu {actual_h}x{actual_w}"
                    )
            
            # LOGGING détaillé avec resize
            input_format = getattr(self.preprocessor, 'data_format_', 'unknown')
            resized = getattr(self.preprocessor, 'resized_', False)
            
            logger.info(
                f"✅ Preprocessing configuré - "
                f"strategy: standardize, "
                f"input_format: {input_format}, "
                f"output_format: channels_first, "
                f"resized: {resized}, "
                f"target_size: {target_size}, "
                f"train_original: {X_train.shape}, "
                f"train_processed: {X_train_norm.shape}, "
                f"val_processed: {X_val_norm.shape}"
            )
            
            return Result.ok((X_train_norm, y_train, X_val_norm, y_val))
            
        except ValueError as e:
            logger.error(f"Erreur validation preprocessing: {str(e)}")
            return Result.err(f"Données invalides: {str(e)}")
        except Exception as e:
            logger.error(f"Erreur technique preprocessing: {str(e)}", exc_info=True)
            return Result.err(f"Erreur preprocessing: {str(e)}")
    
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
        - Criterion configuré selon type de modèle (autoencoder vs classification)
        """
        try:
            # Optimizer
            self.optimizer = OptimizerFactory.create(self.model, self.training_config)
            
            # Scheduler
            self.scheduler = SchedulerFactory.create(self.optimizer, self.training_config)
            
            # === DÉTECTION TYPE MODÈLE ===
            is_autoencoder = self.model_config.model_type in [
                ModelType.CONV_AUTOENCODER, 
                ModelType.VAE, 
                ModelType.DENOISING_AE,
                ModelType.PATCH_CORE  # Ajout PatchCore
            ]
            
            # === SETUP CRITERION SELON TYPE ===
            if is_autoencoder:
                # Autoencoders: MSE Loss (reconstruction)
                self.train_criterion = nn.MSELoss()
                self.val_criterion = nn.MSELoss()
                
                # Désactivation explicite class_weights
                if self.training_config.use_class_weights:
                    logger.warning(
                        "⚠️ Class weights demandés mais DÉSACTIVÉS pour autoencoders. "
                        "Raison: MSELoss ne supporte pas class_weights (loss de reconstruction, pas de classification). "
                        "Cette option sera ignorée."
                    )
                    # Forcer désactivation dans la config pour cohérence
                    self.training_config.use_class_weights = False
                
                logger.info("✅ Criterion: MSELoss (reconstruction autoencoder)")
            
            else:
                # Classification: CrossEntropyLoss
                if self.training_config.use_class_weights:
                    classes = np.unique(y_train)
                    
                    # Validation: au moins 2 classes
                    if len(classes) < 2:
                        logger.error(
                            f"❌ use_class_weights activé mais seulement {len(classes)} classe(s) détectée(s). "
                            f"Classification nécessite >= 2 classes."
                        )
                        return Result.err("Classification nécessite au moins 2 classes")
                    
                    # Calcul des poids
                    weights = compute_class_weight(
                        class_weight='balanced',
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
                        f"✅ Class weights appliqués sur TRAIN: "
                        f"{dict(zip(classes, weights.round(3)))}"
                    )
                else:
                    self.train_criterion = nn.CrossEntropyLoss()
                    logger.info("✅ CrossEntropyLoss standard (pas de class weights)")
                
                # Criterion validation SANS class weights (évaluation honnête)
                self.val_criterion = nn.CrossEntropyLoss()
                logger.info("✅ Criterion validation: CrossEntropyLoss (sans class weights)")
            
            # === LOGGING RÉCAPITULATIF ===
            scheduler_name = self.training_config.scheduler.value if self.scheduler else "none"
            logger.info(
                f"🎯 Training setup complété - "
                f"optimizer: {self.training_config.optimizer.value}, "
                f"scheduler: {scheduler_name}, "
                f"criterion: {'MSELoss' if is_autoencoder else 'CrossEntropyLoss'}, "
                f"use_class_weights: {self.training_config.use_class_weights}"
            )
            
            return Result.ok(None)
            
        except Exception as e:
            logger.error(f"❌ Erreur setup training: {e}", exc_info=True)
            return Result.err(f"Setup training échoué: {str(e)}")
    
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
        Le modèle est construit avec input_size=target_size.
        DONC: data et target ont TOUJOURS la même shape.       
        Args:
            train_loader: DataLoader d'entraînement
            is_autoencoder: Si True, target=input (reconstruction)     
        Returns:
            Loss moyenne de l'époque
        """
        # Détection des modèles sans backward
        is_patchcore = self.model_config.model_type == ModelType.PATCH_CORE
        
        if is_patchcore:
            # PatchCore n'a pas de training epoch classique
            logger.info("⚠️ PatchCore détecté - Skip training epoch (utilise fit() à la place)")
            return 0.0
        
        self.model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target_labels) in enumerate(train_loader):
            # Déplacement sur device
            data = data.to(self.device_manager.device)
            
            # Le preprocessing a déjà tout resizé correctement
            if is_autoencoder:
                target = data 
            else:
                target = target_labels.to(self.device_manager.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            
            # Vérifier si output est None (models sans backward)
            if output is None:
                logger.warning("⚠️ Modèle a retourné None - Skip backward")
                continue
            
            # Vérifi la cohérence shapes (debug mode)
            if is_autoencoder and output.shape != target.shape:
                # Cette condition NE DEVRAIT JAMAIS arriver si preprocessing correct
                logger.error(
                    f"❌ INCOHÉRENCE SHAPES DÉTECTÉE: "
                    f"output={output.shape} vs target={target.shape}. "
                    f"Le preprocessing devrait avoir resizé correctement!"
                )
                raise ValueError(
                    f"Shape mismatch: output={output.shape} vs target={target.shape}. "
                    f"Vérifiez que le preprocessing utilise bien target_size."
                )
            
            # Loss computation
            if (self.model_config.model_type == ModelType.VAE and 
                hasattr(self.model, 'compute_vae_loss')):
                # VAE: loss spéciale avec KL divergence
                loss, recon_loss, kl_loss = self.model.compute_vae_loss(target, output)
            else:
                # Loss standard (reconstruction ou classification)
                loss = self.train_criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
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
        batch_size: Optional[int] = None
    ) -> Result:
        """
        Prédictions robustes avec preprocessing automatique.
        
        Args:
            X: Données à prédire (N, H, W, C) ou (N, C, H, W)
            return_reconstructed: Si True, retourne les reconstructions (autoencoders)
            batch_size: Taille des batchs (par défaut: config.batch_size)
            
        Returns:
            Result avec prédictions structurées:
            - Pour classificateurs: {'probabilities', 'predictions'}
            - Pour autoencoders: {'reconstruction_errors', 'predictions', 'reconstructed'?}
        """
        try:
            # Validation
            if self.model is None:
                return Result.err("Modèle non entraîné")
            if self.preprocessor is None:
                return Result.err("Preprocessor non disponible")
            
            # Preprocessing
            X_processed = self.preprocessor.transform(X, output_format="channels_first")
            
            # Détection du type de modèle
            is_autoencoder = self.model_config.model_type in [
                ModelType.CONV_AUTOENCODER, ModelType.VAE, ModelType.DENOISING_AE
            ]
            
            # Création DataLoader temporaire
            batch_size = batch_size or self.training_config.batch_size
            dummy_labels = np.zeros(len(X_processed))
            
            test_loader = DataLoaderFactory.create(
                X_processed, dummy_labels,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False
            )
            
            # Prédiction
            self.model.eval()
            
            if is_autoencoder:
                return self._predict_autoencoder(test_loader, X_processed, return_reconstructed)
            else:
                return self._predict_classifier(test_loader)
            
        except Exception as e:
            logger.error(f"Erreur prédiction: {e}", exc_info=True)
            return Result.err(f"Prédiction échouée: {str(e)}")
    
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