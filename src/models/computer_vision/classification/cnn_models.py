"""
Modèles CNN professionnels pour classification d'images.
Avec Global Average Pooling - accepte toutes les tailles d'images.

Dans: src/models/computer_vision/classification/cnn_models.py
"""
 
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
from typing import Tuple, Optional, List
from src.shared.logging import get_logger

logger = get_logger(__name__)

class SimpleCNN(nn.Module):
    """
    CNN simple avec Global Average Pooling. 
    Accepte toutes les tailles d'images grâce à AdaptiveAvgPool2d
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 2,
        dropout_rate: float = 0.5,
        filters: Optional[List[int]] = None,
        input_size: Optional[Tuple[int, int]] = None  
    ):
        super(SimpleCNN, self).__init__()
        
        if input_channels not in [1, 3, 4]:
            raise ValueError(f"input_channels doit être 1, 3 ou 4")
        
        if num_classes < 2:
            raise ValueError(f"num_classes doit être >= 2")
        
        if not 0 <= dropout_rate <= 1:
            raise ValueError(f"dropout_rate doit être entre 0 et 1")
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.input_size = input_size  
        
        if filters is None:
            filters = [32, 64, 128]
        
        self.filters = filters
        
        # Feature extractor
        layers = []
        in_channels = input_channels
        
        for i, out_channels in enumerate(filters):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(p=dropout_rate * 0.5)
            ])
            in_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        
        # Global Average Pooling - accepte toutes les tailles
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters[-1], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()
        
        logger.info(
            f"SimpleCNN initialisé: "
            f"input_channels={input_channels}, "
            f"num_classes={num_classes}, "
            f"filters={filters}"
        )
    
    def _initialize_weights(self):
        """Initialisation Xavier/Kaiming des poids."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.       
        Accepte toutes les tailles grâce à Global Average Pooling
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor (B, C, H, W), got {x.dim()}D tensor")
        
        if x.size(1) != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels, "
                f"got {x.size(1)} channels"
            )
        
        try:
            features = self.features(x)
            pooled = self.global_pool(features)  
            output = self.classifier(pooled)
            return output
        
        except Exception as e:
            logger.error(f"Erreur dans forward SimpleCNN: {e}")
            logger.error(f"Input shape: {x.shape}")
            raise
    
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
            "total_parameters": self.count_parameters()
        }


class CustomResNet(nn.Module):
    """
    ResNet personnalisé avec blocs résiduels.
    Accepte toutes les tailles d'images.
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 2,
        num_blocks: List[int] = None,
        base_filters: int = 64,
        input_size: Optional[Tuple[int, int]] = None  
    ):
        super(CustomResNet, self).__init__()
        
        if num_blocks is None:
            num_blocks = [2, 2, 2, 2]
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.in_channels = base_filters
        self.input_size = input_size  
        
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, base_filters, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.stage1 = self._make_stage(base_filters, num_blocks[0], stride=1)
        self.stage2 = self._make_stage(base_filters * 2, num_blocks[1], stride=2)
        self.stage3 = self._make_stage(base_filters * 4, num_blocks[2], stride=2)
        self.stage4 = self._make_stage(base_filters * 8, num_blocks[3], stride=2)
        
        # Adaptive pooling - accepte toutes les tailles
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_filters * 8, num_classes)
        
        self._initialize_weights()
        
        logger.info(f"CustomResNet initialisé: {self.count_parameters():,} paramètres")
    
    def _make_stage(self, out_channels: int, num_blocks: int, stride: int):
        """Crée un stage de blocs résiduels."""
        from torch import nn as nn_alias  
        
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
        """Forward pass - accepte toutes les tailles."""
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor, got {x.dim()}D")
        
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.avgpool(x)  # (B, C, 1, 1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def count_parameters(self) -> int:
        """Compte les paramètres."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResidualBlock(nn.Module):
    """Bloc résiduel avec skip connection."""
    
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


def get_cnn_model(
    model_type: str = "simple",
    input_channels: int = 3,
    num_classes: int = 2,
    **kwargs
) -> nn.Module:
    """Factory pour créer des modèles CNN."""
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