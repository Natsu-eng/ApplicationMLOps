"""
Modèles CNN professionnels pour classification d'images.
Version production-ready avec validation, logging et flexibilité.

À placer dans: src/models/computer_vision/classification/cnn_models.py
"""
 
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
from typing import Tuple, Optional, List
from src.shared.logging import get_logger

logger = get_logger(__name__)


class SimpleCNN(nn.Module):
    """
    CNN simple et robuste pour classification d'images.
    
    Architecture:
        - 3 blocs convolutionnels (Conv2D + BatchNorm + ReLU + MaxPool + Dropout)
        - Global Average Pooling
        - 2 couches fully connected avec Dropout
    
    Args:
        input_channels: Nombre de canaux d'entrée (3 pour RGB, 1 pour grayscale)
        num_classes: Nombre de classes à prédire
        dropout_rate: Taux de dropout (régularisation)
        filters: Liste des filtres pour chaque couche conv
        
    Example:
        >>> model = SimpleCNN(input_channels=3, num_classes=10)
        >>> x = torch.randn(32, 3, 224, 224)
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([32, 10])
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 2,
        dropout_rate: float = 0.5,
        filters: Optional[List[int]] = None
    ):
        super(SimpleCNN, self).__init__()
        
        # Validation des entrées
        if input_channels not in [1, 3, 4]:
            raise ValueError(f"input_channels doit être 1, 3 ou 4, reçu: {input_channels}")
        
        if num_classes < 2:
            raise ValueError(f"num_classes doit être >= 2, reçu: {num_classes}")
        
        if not 0 <= dropout_rate <= 1:
            raise ValueError(f"dropout_rate doit être entre 0 et 1, reçu: {dropout_rate}")
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Filtres par défaut si non spécifiés
        if filters is None:
            filters = [32, 64, 128]
        
        self.filters = filters
        
        # === FEATURE EXTRACTOR ===
        layers = []
        in_channels = input_channels
        
        for i, out_channels in enumerate(filters):
            # Bloc convolutionnel
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(p=dropout_rate * 0.5)  # Dropout spatial léger
            ])
            in_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        
        # === CLASSIFIER ===
        # Global Average Pooling pour réduire les dimensions
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters[-1], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Initialisation des poids
        self._initialize_weights()
        
        logger.info(
            f"SimpleCNN initialisé: "
            f"input_channels={input_channels}, "
            f"num_classes={num_classes}, "
            f"filters={filters}, "
            f"dropout={dropout_rate}"
        )
    
    def _initialize_weights(self):
        """Initialisation Xavier/Kaiming des poids."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass avec validation des entrées.
        
        Args:
            x: Tensor d'images (batch_size, channels, height, width)
            
        Returns:
            Logits (batch_size, num_classes)
            
        Raises:
            ValueError: Si la forme d'entrée est invalide
        """
        # Validation
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor (B, C, H, W), got {x.dim()}D tensor")
        
        if x.size(1) != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels, "
                f"got {x.size(1)} channels"
            )
        
        try:
            # Feature extraction
            features = self.features(x)
            
            # Global pooling
            pooled = self.global_pool(features)
            
            # Classification
            output = self.classifier(pooled)
            
            return output
        
        except Exception as e:
            logger.error(f"Erreur dans forward SimpleCNN: {e}")
            logger.error(f"Input shape: {x.shape}")
            raise
    
    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extrait les feature maps avant le classifier.
        Utile pour visualisation et debugging.
        
        Args:
            x: Images d'entrée
            
        Returns:
            Feature maps (batch_size, channels, height, width)
        """
        return self.features(x)
    
    def count_parameters(self) -> int:
        """Compte le nombre de paramètres entraînables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def summary(self) -> dict:
        """Retourne un résumé du modèle."""
        return {
            "model_type": "SimpleCNN",
            "input_channels": self.input_channels,
            "num_classes": self.num_classes,
            "filters": self.filters,
            "dropout_rate": self.dropout_rate,
            "total_parameters": self.count_parameters(),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "non_trainable": sum(p.numel() for p in self.parameters() if not p.requires_grad)
        }


class ResidualBlock(nn.Module):
    """
    Bloc résiduel avec skip connection.
    Permet un entraînement plus profond sans dégradation.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class CustomResNet(nn.Module):
    """
    ResNet personnalisé avec blocs résiduels.
    Plus performant que SimpleCNN pour datasets complexes.
    
    Args:
        input_channels: Nombre de canaux d'entrée
        num_classes: Nombre de classes
        num_blocks: Nombre de blocs résiduels par stage
        filters: Filtres de base (doublés à chaque stage)
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 2,
        num_blocks: List[int] = [2, 2, 2, 2],
        base_filters: int = 64
    ):
        super(CustomResNet, self).__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.in_channels = base_filters
        
        # Stem (initial conv)
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, base_filters, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual stages
        self.stage1 = self._make_stage(base_filters, num_blocks[0], stride=1)
        self.stage2 = self._make_stage(base_filters * 2, num_blocks[1], stride=2)
        self.stage3 = self._make_stage(base_filters * 4, num_blocks[2], stride=2)
        self.stage4 = self._make_stage(base_filters * 8, num_blocks[3], stride=2)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_filters * 8, num_classes)
        
        self._initialize_weights()
        
        logger.info(f"CustomResNet initialisé: {self.count_parameters():,} paramètres")
    
    def _make_stage(self, out_channels: int, num_blocks: int, stride: int):
        """Crée un stage de blocs résiduels."""
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialisation des poids."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor, got {x.dim()}D")
        
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def count_parameters(self) -> int:
        """Compte les paramètres."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# === FONCTION FACTORY ===

def get_cnn_model(
    model_type: str = "simple",
    input_channels: int = 3,
    num_classes: int = 2,
    **kwargs
) -> nn.Module:
    """
    Factory pour créer des modèles CNN.
    
    Args:
        model_type: "simple" ou "resnet"
        input_channels: Canaux d'entrée
        num_classes: Nombre de classes
        **kwargs: Arguments additionnels pour le modèle
        
    Returns:
        Modèle PyTorch
        
    Example:
        >>> model = get_cnn_model("simple", input_channels=3, num_classes=10)
        >>> model = get_cnn_model("resnet", num_classes=5, num_blocks=[2,2,2,2])
    """
    models_registry = {
        "simple": SimpleCNN,
        "resnet": CustomResNet
    }
    
    if model_type not in models_registry:
        raise ValueError(
            f"model_type '{model_type}' non supporté. "
            f"Options: {list(models_registry.keys())}"
        )
    
    model_class = models_registry[model_type]
    
    try:
        model = model_class(
            input_channels=input_channels,
            num_classes=num_classes,
            **kwargs
        )
        
        logger.info(f"Modèle {model_type} créé avec succès")
        return model
    
    except Exception as e:
        logger.error(f"Erreur création modèle {model_type}: {e}")
        raise


