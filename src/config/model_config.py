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
    PATCH_CORE = "patch_core"  # NOUVEAU
    SIAMESE_NETWORK = "siamese_network"  # NOUVEAU


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
    
    # NOUVEAUX paramètres pour PatchCore et Siamese
    backbone_name: str = "resnet18"
    patchcore_layers: List[str] = None
    faiss_index_type: str = "Flat"
    coreset_ratio: float = 0.01
    num_neighbors: int = 1
    embedding_dim: int = 128
    margin: float = 1.0
    input_size: Tuple[int, int] = (224, 224)
    
    def __post_init__(self):
        """Validation et conversion automatique"""
        # Conversion automatique string -> ModelType
        if isinstance(self.model_type, str):
            self.model_type = ModelType(self.model_type)
        
        # Initialisation des valeurs par défaut pour les nouvelles listes
        if self.patchcore_layers is None:
            self.patchcore_layers = ['layer2', 'layer3']
        
        # Validation
        assert isinstance(self.model_type, ModelType), "model_type doit être un ModelType"
        assert self.input_channels > 0, "input_channels doit être > 0"
        assert self.num_classes >= 2, "num_classes doit être >= 2"
        assert 0 <= self.dropout_rate <= 1, "dropout_rate doit être entre 0 et 1"
        assert self.base_filters > 0, "base_filters doit être > 0"
        assert self.latent_dim >= 16, "latent_dim doit être >= 16"
        assert self.num_stages >= 2, "num_stages doit être >= 2"
        
        # Validation des nouveaux paramètres
        assert self.backbone_name in ["resnet18", "wide_resnet50_2"], "backbone_name non supporté"
        assert 0 < self.coreset_ratio <= 1, "coreset_ratio doit être entre 0 et 1"
        assert self.num_neighbors > 0, "num_neighbors doit être > 0"
        assert self.embedding_dim > 0, "embedding_dim doit être > 0"
        assert self.margin > 0, "margin doit être > 0"
        assert len(self.input_size) == 2, "input_size doit être un tuple (H, W)"
        assert self.input_size[0] >= 32 and self.input_size[1] >= 32, "input_size trop petit (min 32x32)"