import numpy as np
from src.config.model_config import ModelConfig, ModelType
from src.data.computer_vision_preprocessing import Result
from utils.device_manager import DeviceManager
import torch.nn as nn # type: ignore
import torch # type: ignore

from src.shared.logging import get_logger

logger = get_logger(__name__)

# ==========================
# IMPORTS DES MODÈLES RÉELS
# ==========================
# Import des vrais modèles avec gestion d'erreurs
MODELS_AVAILABLE = {
    'cnn': False,
    'transfer': False,
    'autoencoder': False,
    'patchcore': False,  # NOUVEAU
    'siamese': False     # NOUVEAU
}

try:
    from src.models.computer_vision.classification.cnn_models import (
        SimpleCNN, CustomResNet
    )
    MODELS_AVAILABLE['cnn'] = True
    logger.info("✅ Modèles CNN importés")
except ImportError as e:
    logger.warning(f"⚠️ Modèles CNN non disponibles: {e}")
    SimpleCNN = None
    CustomResNet = None

try:
    from src.models.computer_vision.classification.transfer_learning import (
        TransferLearningModel, FineTuningScheduler
    )
    MODELS_AVAILABLE['transfer'] = True
    logger.info("✅ Modèles Transfer Learning importés")
except ImportError as e:
    logger.warning(f"⚠️ Modèles Transfer Learning non disponibles: {e}")
    TransferLearningModel = None
    FineTuningScheduler = None

try:
    from src.models.computer_vision.anomaly_detection.autoencoders import (
        ConvAutoEncoder, VariationalAutoEncoder, DenoisingAutoEncoder
    )
    MODELS_AVAILABLE['autoencoder'] = True
    logger.info("✅ Modèles Autoencoder importés")
except ImportError as e:
    logger.warning(f"⚠️ Modèles Autoencoder non disponibles: {e}")
    ConvAutoEncoder = None
    VariationalAutoEncoder = None
    DenoisingAutoEncoder = None

# NOUVEAUX IMPORTS - JUSTE LES 2 MODÈLES AJOUTÉS
try:
    from src.models.computer_vision.anomaly_detection.patch_core import ProfessionalPatchCore
    MODELS_AVAILABLE['patchcore'] = True
    logger.info("✅ Modèle PatchCore professionnel importé")
except ImportError as e:
    logger.warning(f"⚠️ Modèle PatchCore non disponible: {e}")
    ProfessionalPatchCore = None

try:
    from src.models.computer_vision.anomaly_detection.siamese_networks import ProfessionalSiameseNetwork
    MODELS_AVAILABLE['siamese'] = True
    logger.info("✅ Modèle Siamese Network professionnel importé")
except ImportError as e:
    logger.warning(f"⚠️ Modèle Siamese Network non disponible: {e}")
    ProfessionalSiameseNetwork = None


# =======================
# CONSTRUCTION DE MODÈLES
# =======================
class ModelBuilder:
    """
    Factory pour construire des modèles.   
    Utilise les VRAIS modèles de votre codebase si disponibles,
    sinon utilise des placeholders pour tests.
    """
    
    def __init__(self, device_manager: DeviceManager):
        self.device_manager = device_manager
        self.registry = {}
        self._register_builders()
    
    def _register_builders(self):
        """Enregistre les builders de modèles réels"""
        self.registry = {
            ModelType.SIMPLE_CNN: self._build_simple_cnn,
            ModelType.CUSTOM_RESNET: self._build_custom_resnet,
            ModelType.TRANSFER_LEARNING: self._build_transfer_learning,
            ModelType.CONV_AUTOENCODER: self._build_conv_autoencoder,
            ModelType.VAE: self._build_variational_autoencoder,
            ModelType.DENOISING_AE: self._build_denoising_autoencoder,
            # JUSTE LES 2 NOUVEAUX MODÈLES AJOUTÉS
            ModelType.PATCH_CORE: self._build_patchcore,
            ModelType.SIAMESE_NETWORK: self._build_siamese_network,
        }
    
    def build(self, config: ModelConfig) -> Result:
        """
        Construit un modèle selon la config.        
        Returns:
            Result avec model si succès
        """
        try:
            if config.model_type not in self.registry:
                return Result.err(
                    f"Type de modèle non supporté: {config.model_type}"
                )
            
            builder = self.registry[config.model_type]
            model = builder(config)
            
            if model is None:
                return Result.err(f"Échec construction modèle: {config.model_type}")
            
            # Déplacement sur device
            model = model.to(self.device_manager.device)
            
            # Comptage paramètres
            n_params = sum(p.numel() for p in model.parameters())
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            logger.info(
                f"Modèle construit: {config.model_type.value} - "
                f"total_params: {n_params}, "
                f"trainable_params: {n_trainable}, "
                f"device: {str(self.device_manager.device)}"
            )
            
            return Result.ok(
                model,
                n_params=n_params,
                n_trainable=n_trainable
            )
            
        except Exception as e:
            logger.error(f"Erreur construction modèle: {e}", exc_info=True)
            return Result.err(f"Construction modèle échouée: {str(e)}")
    
    # VOTRE CODE EXISTANT POUR LES 6 PREMIERS MODÈLES (JE NE TOUCHE PAS)
    def _build_simple_cnn(self, config: ModelConfig) -> nn.Module:
        """Builder pour SimpleCNN - UTILISE LE VRAI MODÈLE"""
        if SimpleCNN is not None:
            # Utilisation du VRAI modèle
            return SimpleCNN(
                input_channels=config.input_channels,
                num_classes=config.num_classes,
                dropout_rate=config.dropout_rate,
                filters=[config.base_filters, config.base_filters * 2, config.base_filters * 4]
            )
        else:
            # Fallback : placeholder simple pour tests
            logger.warning("SimpleCNN non disponible, utilisation placeholder")
            return self._build_placeholder_cnn(config)
    
    def _build_custom_resnet(self, config: ModelConfig) -> nn.Module:
        """Builder pour CustomResNet - UTILISE LE VRAI MODÈLE"""
        if CustomResNet is not None:
            # Utilisation du VRAI modèle
            return CustomResNet(
                input_channels=config.input_channels,
                num_classes=config.num_classes,
                num_blocks=[2, 2, 2, 2],
                base_filters=config.base_filters
            )
        else:
            logger.warning("CustomResNet non disponible, utilisation placeholder")
            return self._build_placeholder_cnn(config)
    
    def _build_transfer_learning(self, config: ModelConfig) -> nn.Module:
        """Builder pour TransferLearning - UTILISE LE VRAI MODÈLE"""
        if TransferLearningModel is not None:
            # Utilisation du VRAI modèle
            return TransferLearningModel(
                model_name='resnet50',
                num_classes=config.num_classes,
                pretrained=config.pretrained,
                freeze_layers=config.freeze_layers,
                dropout_rate=config.dropout_rate,
                use_custom_classifier=True
            )
        else:
            logger.warning("TransferLearningModel non disponible, utilisation placeholder")
            return self._build_placeholder_cnn(config)
    
    def _build_conv_autoencoder(self, config: ModelConfig) -> nn.Module:
        """Builder pour ConvAutoEncoder - UTILISE LE VRAI MODÈLE"""
        if ConvAutoEncoder is not None:
            # Utilisation du VRAI modèle
            return ConvAutoEncoder(
                input_channels=config.input_channels,
                latent_dim=config.latent_dim,
                base_filters=config.base_filters,
                num_stages=config.num_stages,
                dropout_rate=config.dropout_rate,
                use_skip_connections=False,
                use_vae=False
            )
        else:
            logger.warning("ConvAutoEncoder non disponible, utilisation placeholder")
            return self._build_placeholder_autoencoder(config)
    
    def _build_variational_autoencoder(self, config: ModelConfig) -> nn.Module:
        """Builder pour VariationalAutoEncoder - UTILISE LE VRAI MODÈLE"""
        if VariationalAutoEncoder is not None:
            # Utilisation du VRAI modèle
            return VariationalAutoEncoder(
                input_channels=config.input_channels,
                latent_dim=config.latent_dim,
                base_filters=config.base_filters,
                num_stages=config.num_stages,
                dropout_rate=config.dropout_rate,
                use_skip_connections=False
            )
        else:
            logger.warning("VariationalAutoEncoder non disponible, utilisation placeholder")
            return self._build_placeholder_autoencoder(config)
    
    def _build_denoising_autoencoder(self, config: ModelConfig) -> nn.Module:
        """Builder pour DenoisingAutoEncoder - UTILISE LE VRAI MODÈLE"""
        if DenoisingAutoEncoder is not None:
            # Utilisation du VRAI modèle
            return DenoisingAutoEncoder(
                input_channels=config.input_channels,
                latent_dim=config.latent_dim,
                base_filters=config.base_filters,
                num_stages=config.num_stages,
                dropout_rate=config.dropout_rate,
                use_skip_connections=False,
                noise_factor=0.2
            )
        else:
            logger.warning("DenoisingAutoEncoder non disponible, utilisation placeholder")
            return self._build_placeholder_autoencoder(config)
    
    # JUSTE LES 2 NOUVELLES FONCTIONS AJOUTÉES
    def _build_patchcore(self, config: ModelConfig) -> nn.Module:
        """Builder pour ProfessionalPatchCore - UTILISE LE VRAI MODÈLE"""
        if ProfessionalPatchCore is not None:
            # Utilisation du VRAI modèle PatchCore professionnel
            return ProfessionalPatchCore(
                backbone_name=getattr(config, 'backbone_name', 'wide_resnet50_2'),
                layers=getattr(config, 'patchcore_layers', ['layer2', 'layer3']),
                faiss_index_type=getattr(config, 'faiss_index_type', 'Flat'),
                coreset_ratio=getattr(config, 'coreset_ratio', 0.01),
                num_neighbors=getattr(config, 'num_neighbors', 1)
            )
        else:
            logger.warning("ProfessionalPatchCore non disponible, utilisation placeholder")
            return self._build_placeholder_patchcore(config)
    
    def _build_siamese_network(self, config: ModelConfig) -> nn.Module:
        """Builder pour ProfessionalSiameseNetwork - UTILISE LE VRAI MODÈLE"""
        if ProfessionalSiameseNetwork is not None:
            # Utilisation du VRAI modèle Siamese professionnel
            return ProfessionalSiameseNetwork(
                backbone_name=getattr(config, 'backbone_name', 'resnet18'),
                embedding_dim=getattr(config, 'embedding_dim', 128),
                margin=getattr(config, 'margin', 1.0)
            )
        else:
            logger.warning("ProfessionalSiameseNetwork non disponible, utilisation placeholder")
            return self._build_placeholder_siamese(config)

    # VOS PLACEHOLDERS EXISTANTS (JE NE TOUCHE PAS)
    def _build_placeholder_cnn(self, config: ModelConfig) -> nn.Module:
        """Placeholder simple pour tests uniquement"""
        return nn.Sequential(
            nn.Conv2d(config.input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(64, config.num_classes)
        )
    
    def _build_placeholder_autoencoder(self, config: ModelConfig) -> nn.Module:
        """Placeholder autoencoder pour tests uniquement"""
        class SimpleAutoencoder(nn.Module):
            def __init__(self, input_channels, latent_dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(input_channels, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(64, latent_dim)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, 64),
                    nn.ReLU(),
                    nn.Unflatten(1, (64, 1, 1)),
                    nn.Upsample(scale_factor=32, mode='nearest'),
                    nn.Conv2d(64, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, input_channels, 3, padding=1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                z = self.encoder(x)
                return self.decoder(z)
        
        return SimpleAutoencoder(config.input_channels, config.latent_dim)
    
    # JUSTE LES 2 NOUVEAUX PLACEHOLDERS AJOUTÉS
    def _build_placeholder_patchcore(self, config: ModelConfig) -> nn.Module:
        """Placeholder PatchCore pour tests uniquement"""
        class SimplePatchCore(nn.Module):
            def __init__(self, input_channels):
                super().__init__()
                self.feature_extractor = nn.Sequential(
                    nn.Conv2d(input_channels, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU()
                )
                self.memory_bank = None
            
            def forward(self, x):
                return self.feature_extractor(x)
            
            def fit(self, dataloader):
                # Implémentation simplifiée pour tests
                self.memory_bank = torch.randn(100, 64)  # Placeholder
            
            def predict(self, dataloader):
                if self.memory_bank is None:
                    raise ValueError("Modèle non entraîné")
                return np.random.rand(32)  # Scores aléatoires pour tests
        
        return SimplePatchCore(config.input_channels)
    
    def _build_placeholder_siamese(self, config: ModelConfig) -> nn.Module:
        """Placeholder Siamese Network pour tests uniquement"""
        class SimpleSiamese(nn.Module):
            def __init__(self, input_channels, embedding_dim=128):
                super().__init__()
                self.embedding_dim = embedding_dim
                self.feature_extractor = nn.Sequential(
                    nn.Conv2d(input_channels, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Flatten(),
                    nn.Linear(64 * (config.input_size[0]//4) * (config.input_size[1]//4), embedding_dim),
                    nn.ReLU()
                )
            
            def forward(self, x1, x2=None):
                emb1 = self.feature_extractor(x1)
                if x2 is not None:
                    emb2 = self.feature_extractor(x2)
                    return torch.norm(emb1 - emb2, dim=1) # 
                return emb1
            
            def predict_anomaly_score(self, query_images, reference_embeddings):
                # Implémentation simplifiée
                query_emb = self.forward(query_images)
                distances = torch.cdist(query_emb, reference_embeddings) 
                return distances.min(dim=1)[0]
        
        return SimpleSiamese(config.input_channels)