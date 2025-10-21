"""
Wrapper pour compatibilité avec l'ancien code
Fichier: src/models/pipeline_visio/legacy_wrapper.py
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch.nn as nn

from src.data.computer_vision_preprocessing import DataPreprocessor
from src.models.computer_vision_training import (
    AnomalyAwareTrainer, 
    ComputerVisionTrainer, 
    ModelConfig, 
    ModelType, 
    OptimizerType, 
    SchedulerType, 
    TrainingConfig
)
from src.shared.logging import StructuredLogger
from utils.callbacks import LoggingCallback, StreamlitCallback

logger = StructuredLogger(__name__)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def training_config_to_dict(config: Union[TrainingConfig, Dict]) -> Dict:
    """
    Convertit un TrainingConfig en dictionnaire pour affichage.
    
    Args:
        config: TrainingConfig ou Dict
        
    Returns:
        Dict avec tous les paramètres
    """
    if isinstance(config, TrainingConfig):
        return {
            'epochs': config.epochs,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'weight_decay': config.weight_decay,
            'gradient_clip': config.gradient_clip,
            'optimizer': config.optimizer.value if hasattr(config.optimizer, 'value') else str(config.optimizer),
            'scheduler': config.scheduler.value if hasattr(config.scheduler, 'value') else str(config.scheduler),
            'early_stopping_patience': config.early_stopping_patience,
            'reduce_lr_patience': config.reduce_lr_patience,
            'use_class_weights': config.use_class_weights,
            'deterministic': config.deterministic,
            'seed': config.seed,
            'num_workers': config.num_workers,
            'pin_memory': config.pin_memory,
            'use_mixed_precision': config.use_mixed_precision
        }
    return config if isinstance(config, dict) else {}


# ============================================================================
# MAIN WRAPPER
# ============================================================================

def train_computer_vision_model_production(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_type: str = "simple_cnn",
    model_params: Dict[str, Any] = None,
    training_config: Union[Dict[str, Any], TrainingConfig] = None,
    streamlit_components: Dict = None,
    imbalance_config: Dict[str, Any] = None,
    anomaly_type: str = None
) -> Tuple[Optional[nn.Module], Dict]:
    """
    Fonction wrapper pour compatibilité avec l'ancien code.   
    Utilise le nouveau pipeline mais retourne le format attendu.
    
    Args:
        X_train: Training features (N, H, W, C)
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        model_type: Type de modèle (str ou ModelType)
        model_params: Paramètres du modèle
        training_config: Configuration d'entraînement (Dict ou TrainingConfig)
        streamlit_components: Composants Streamlit pour UI
        imbalance_config: Configuration pour gérer le déséquilibre
        anomaly_type: Type d'anomalie pour AnomalyAwareTrainer
        
    Returns:
        Tuple (model, history) où:
        - model: Modèle entraîné (nn.Module)
        - history: Dict avec historique d'entraînement
    """
    try:
        # Configs de base
        model_params = model_params or {}
        imbalance_config = imbalance_config or {}

        # ✅ CORRECTION: Gestion intelligente de training_config
        if isinstance(training_config, TrainingConfig):
            train_config = training_config
        elif isinstance(training_config, dict):
            train_config = TrainingConfig(
                epochs=training_config.get('epochs', 100),
                batch_size=training_config.get('batch_size', 32),
                learning_rate=training_config.get('learning_rate', 1e-4),
                weight_decay=training_config.get('weight_decay', 0.01),
                gradient_clip=training_config.get('gradient_clip', 1.0),
                optimizer=OptimizerType(training_config.get('optimizer', 'adamw')),
                scheduler=SchedulerType(training_config.get('scheduler', 'reduce_on_plateau')),
                early_stopping_patience=training_config.get('early_stopping_patience', 15),
                reduce_lr_patience=training_config.get('reduce_lr_patience', 8),
                use_class_weights=imbalance_config.get('use_class_weights', False),
                deterministic=training_config.get('deterministic', True),
                seed=training_config.get('seed', 42)
            )
        else:
            train_config = TrainingConfig(
                use_class_weights=imbalance_config.get('use_class_weights', False)
            )

        # Filtrer les paramètres valides pour ModelConfig
        allowed_params = [
            'num_classes', 'input_channels', 'dropout_rate', 
            'base_filters', 'latent_dim', 'num_stages', 
            'pretrained', 'freeze_layers'
        ]
        valid_model_params = {k: v for k, v in model_params.items() if k in allowed_params}

        # ModelConfig
        model_config = ModelConfig(
            model_type=ModelType(model_type) if isinstance(model_type, str) else model_type,
            num_classes=valid_model_params.get('num_classes', len(np.unique(y_train))),
            input_channels=valid_model_params.get('input_channels', X_train.shape[-1] if len(X_train.shape) > 3 else 3),
            dropout_rate=valid_model_params.get('dropout_rate', 0.5),
            base_filters=valid_model_params.get('base_filters', 32),
            latent_dim=valid_model_params.get('latent_dim', 256),
            num_stages=valid_model_params.get('num_stages', 4),
            anomaly_type=anomaly_type
        )

        # Callbacks
        callbacks = []
        if streamlit_components:
            callbacks.append(StreamlitCallback(
                progress_bar=streamlit_components.get('progress_bar'),
                status_text=streamlit_components.get('status_text'),
                total_epochs=train_config.epochs
            ))
        callbacks.append(LoggingCallback(log_every_n_epochs=5))

        # ✅ CORRECTION: Trainer avec signature correcte
        if anomaly_type:
            # AnomalyAwareTrainer accepte model_config et training_config en paramètres
            trainer = AnomalyAwareTrainer(
                anomaly_type=anomaly_type,
                taxonomy_config=None,
                model_config=model_config,
                training_config=train_config,
                callbacks=callbacks
            )
            result = trainer.train(X_train, y_train, X_val, y_val)
        else:
            trainer = ComputerVisionTrainer(
                model_config=model_config,
                training_config=train_config,
                callbacks=callbacks
            )
            result = trainer.fit(X_train, y_train, X_val, y_val)

        # Gestion des erreurs
        if not result.success:
            logger.error(f"Entraînement échoué: {result.error}")
            return None, {
                'success': False, 
                'error': result.error, 
                'train_loss': [], 
                'val_loss': []
            }

        # ✅ CORRECTION: Conversion pour historique
        training_config_dict = training_config_to_dict(train_config)

        # Historique compatible avec ancien format
        history = {
            'success': True,
            'train_loss': trainer.history['train_loss'],
            'val_loss': trainer.history['val_loss'],
            'val_accuracy': trainer.history.get('val_accuracy', []),
            'val_f1': trainer.history.get('val_f1', []),
            'learning_rates': trainer.history['learning_rates'],
            'best_epoch': result.metadata.get('best_epoch', 0),
            'best_val_loss': min(trainer.history['val_loss']) if trainer.history['val_loss'] else float('inf'),
            'training_time': result.metadata.get('training_time', 0),
            'total_epochs_trained': len(trainer.history['train_loss']),
            'early_stopping_triggered': len(trainer.history['train_loss']) < train_config.epochs,
            'model_type': model_type,
            'input_shape': X_train.shape,
            'training_config': training_config_dict,
            'anomaly_type': anomaly_type
        }

        return trainer.model, history

    except Exception as e:
        logger.error(f"Erreur wrapper production: {e}", exc_info=True)
        return None, {
            'success': False, 
            'error': str(e), 
            'train_loss': [], 
            'val_loss': []
        }


def evaluate_computer_vision_model_production(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str = "simple_cnn",
    threshold: float = 0.5,
    preprocessor: DataPreprocessor = None
) -> Dict[str, Any]:
    """
    Fonction wrapper pour évaluation compatible avec ancien code.
    
    Args:
        model: Modèle PyTorch entraîné
        X_test: Test features
        y_test: Test labels
        model_type: Type de modèle
        threshold: Seuil de classification
        preprocessor: Preprocessor utilisé pendant l'entraînement
        
    Returns:
        Dict avec métriques d'évaluation
    """
    try:
        # Si pas de preprocessor fourni, créer un basique
        if preprocessor is None:
            preprocessor = DataPreprocessor(strategy="none")
            preprocessor.fitted = True
        
        # Création config minimale
        model_config = ModelConfig(
            model_type=ModelType(model_type) if isinstance(model_type, str) else model_type,
            num_classes=2
        )
        
        training_config = TrainingConfig(batch_size=32)
        
        # Création trainer temporaire pour évaluation
        trainer = ComputerVisionTrainer(
            model_config=model_config,
            training_config=training_config
        )
        trainer.model = model
        trainer.preprocessor = preprocessor
        
        # Évaluation
        result = trainer.evaluate(X_test, y_test)
        
        if result.success:
            return result.data
        else:
            return {
                'success': False,
                'error': result.error
            }
            
    except Exception as e:
        logger.error(f"Erreur wrapper évaluation: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }