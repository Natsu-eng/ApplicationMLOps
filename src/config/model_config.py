"""
Définition UNIQUE et CENTRALISÉE de ModelType
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ModelType(str, Enum):
    """Types de modèles supportés - DÉFINITION CENTRALE"""
    SIMPLE_CNN = "simple_cnn"
    CUSTOM_RESNET = "custom_resnet"
    TRANSFER_LEARNING = "transfer_learning"
    CONV_AUTOENCODER = "conv_autoencoder"
    VAE = "variational_autoencoder"
    DENOISING_AE = "denoising_autoencoder"


@dataclass
class ModelConfig:
    """Configuration du modèle"""
    model_type: ModelType
    input_channels: int = 3
    num_classes: int = 2
    dropout_rate: float = 0.5
    base_filters: int = 32
    latent_dim: int = 128
    num_stages: int = 3
    pretrained: bool = False
    freeze_layers: int = 0  # Nombre de couches, pas boolean
    anomaly_type: Optional[str] = None
    
    def __post_init__(self):
        """Validation et conversion automatique"""
        # Conversion automatique string -> ModelType
        if isinstance(self.model_type, str):
            self.model_type = ModelType(self.model_type)
        
        # Validation
        assert isinstance(self.model_type, ModelType), "model_type doit être un ModelType"
        assert self.input_channels > 0, "input_channels doit être > 0"
        assert self.num_classes >= 2, "num_classes doit être >= 2"
