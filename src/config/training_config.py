"""
Configuration d'entraînement - Séparée pour éviter les imports circulaires
"""
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from enum import Enum

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