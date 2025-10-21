from src.config.model_config import ModelConfig, ModelType
from src.data.computer_vision_preprocessing import Result
from utils.device_manager import DeviceManager
import torch.nn as nn

from src.shared.logging import StructuredLogger

logger = StructuredLogger(__name__)

# ==========================
# IMPORTS DES MODÈLES RÉELS
# ==========================
# Import des vrais modèles avec gestion d'erreurs
MODELS_AVAILABLE = {
    'cnn': False,
    'transfer': False,
    'autoencoder': False
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
                f"Modèle construit: {config.model_type.value}",
                total_params=n_params,
                trainable_params=n_trainable,
                device=str(self.device_manager.device)
            )
            
            return Result.ok(
                model,
                n_params=n_params,
                n_trainable=n_trainable
            )
            
        except Exception as e:
            logger.error(f"Erreur construction modèle: {e}", exc_info=True)
            return Result.err(f"Construction modèle échouée: {str(e)}")
    
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
               #input_channels=config.input_channels,
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
        

    # ========================================================================
    # PLACEHOLDERS (pour tests si vrais modèles non disponibles)
    # ========================================================================
    
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
