"""
Définition UNIQUE et CENTRALISÉE de ModelType
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Tuple


class ModelType(str, Enum):
    """Types de modèles supportés - DÉFINITION CENTRALE"""
    SIMPLE_CNN = "simple_cnn"
    CUSTOM_RESNET = "custom_resnet"
    TRANSFER_LEARNING = "transfer_learning"
    CONV_AUTOENCODER = "conv_autoencoder"
    VAE = "variational_autoencoder"
    DENOISING_AE = "denoising_autoencoder"
    PATCH_CORE = "patch_core"  
    SIAMESE_NETWORK = "siamese_network" 


@dataclass
class ModelConfig:
    """
    Configuration d'un modèle Computer Vision.
    Contient les hyperparamètres et options spécifiques au type de modèle.
    """
    model_type: ModelType
    num_classes: int = 2
    input_channels: int = 3
    dropout_rate: float = 0.5
    
    # CNN/ResNet
    base_filters: int = 32
    
    # AutoEncoders
    latent_dim: int = 128
    num_stages: int = 4
    input_size: Optional[Tuple[int, int]] = None  
    
    # Transfer Learning
    pretrained: bool = True
    freeze_layers: int = 0
    backbone_name: str = "resnet50"
    
    # PatchCore
    patchcore_layers: list = None
    coreset_ratio: float = 0.01
    
    # Siamese
    embedding_dim: int = 128
    margin: float = 1.0
    
    def __post_init__(self):
        """Validation et conversions post-initialisation"""
        # Conversion ModelType si string
        if isinstance(self.model_type, str):
            self.model_type = ModelType(self.model_type)
        
        # Validation input_size pour AutoEncoders
        if self.model_type in [
            ModelType.CONV_AUTOENCODER,
            ModelType.VAE,
            ModelType.DENOISING_AE
        ]:
            if self.input_size is None:
                raise ValueError(
                    f"input_size est OBLIGATOIRE pour {self.model_type.value}. "
                    f"Spécifiez input_size=(hauteur, largeur) dans ModelConfig."
                )
            
            # Validation dimensions
            if not isinstance(self.input_size, (tuple, list)) or len(self.input_size) != 2:
                raise ValueError(
                    f"input_size doit être un tuple (H, W), reçu: {self.input_size}"
                )
            
            h, w = self.input_size
            if h < 32 or w < 32:
                raise ValueError(
                    f"input_size trop petit: {self.input_size}. Minimum: (32, 32)"
                )