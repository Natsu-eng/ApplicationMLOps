"""
Modèles de Transfer Learning .
Avec gestion correcte des types et compatibilité complète.

Les modèles pré-entraînés (ResNet, VGG, EfficientNet) utilisent déjà
AdaptiveAvgPool2d en interne, donc ils acceptent toutes les tailles d'images.

Pas besoin de resize dynamique supplémentaire.

Dans: src/models/computer_vision/classification/transfer_learning.py
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, List, Dict, Any, Tuple
from src.shared.logging import get_logger

logger = get_logger(__name__)


class TransferLearningModel(nn.Module):
    """
    Modèle de Transfer Learning robuste et flexible.   
    Accepte toutes les tailles d'images (grâce aux backbones pré-entraînés)
    Gestion correcte des types et paramètres
    
    Supporte:
        - ResNet (18, 34, 50, 101, 152)
        - VGG (11, 13, 16, 19)
        - EfficientNet (b0-b7)
        - MobileNet (v2, v3)
        - DenseNet (121, 161, 169, 201)
    
    Features:
        - Fine-tuning progressif (geler certaines couches)
        - Classifier personnalisable
        - Feature extraction mode
        - Compatible avec tous pipelines
    
    Args:
        model_name: Nom du modèle pré-entraîné
        num_classes: Nombre de classes
        pretrained: Charger les poids ImageNet
        freeze_layers: Nombre de couches à geler (0 = tout entraînable, -1 = tout sauf classifier)
        dropout_rate: Dropout avant la couche finale
        use_custom_classifier: Utiliser un classifier multi-couches
        input_size: Taille d'entrée (stockée pour compatibilité mais non utilisée)
        
    Example:
        >>> model = TransferLearningModel("resnet50", num_classes=10, freeze_layers=100)
        >>> x = torch.randn(8, 3, 224, 224)
        >>> output = model(x)
    """
    
    # Modèles supportés avec leurs caractéristiques
    SUPPORTED_MODELS = {
        # ResNet family
        "resnet18": {"type": "resnet", "layers": 18, "features": 512},
        "resnet34": {"type": "resnet", "layers": 34, "features": 512},
        "resnet50": {"type": "resnet", "layers": 50, "features": 2048},
        "resnet101": {"type": "resnet", "layers": 101, "features": 2048},
        "resnet152": {"type": "resnet", "layers": 152, "features": 2048},
        
        # VGG family
        "vgg11": {"type": "vgg", "features": 4096},
        "vgg13": {"type": "vgg", "features": 4096},
        "vgg16": {"type": "vgg", "features": 4096},
        "vgg19": {"type": "vgg", "features": 4096},
        
        # EfficientNet
        "efficientnet_b0": {"type": "efficientnet", "features": 1280},
        "efficientnet_b1": {"type": "efficientnet", "features": 1280},
        "efficientnet_b2": {"type": "efficientnet", "features": 1408},
        "efficientnet_b3": {"type": "efficientnet", "features": 1536},
        "efficientnet_b4": {"type": "efficientnet", "features": 1792},
        
        # MobileNet
        "mobilenet_v2": {"type": "mobilenet", "features": 1280},
        "mobilenet_v3_small": {"type": "mobilenet", "features": 576},
        "mobilenet_v3_large": {"type": "mobilenet", "features": 960},
        
        # DenseNet
        "densenet121": {"type": "densenet", "features": 1024},
        "densenet161": {"type": "densenet", "features": 2208},
        "densenet169": {"type": "densenet", "features": 1664},
        "densenet201": {"type": "densenet", "features": 1920}
    }
    
    def __init__(
        self,
        model_name: str = "resnet50",
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_layers: int = 0,
        dropout_rate: float = 0.5,
        use_custom_classifier: bool = True,
        input_size: Optional[Tuple[int, int]] = None,  
        input_channels: int = 3  
    ):
        super(TransferLearningModel, self).__init__()
        
        # === VALIDATION ===
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"model_name '{model_name}' non supporté. "
                f"Options: {list(self.SUPPORTED_MODELS.keys())}"
            )
        
        if num_classes < 2:
            raise ValueError(f"num_classes doit être >= 2, reçu: {num_classes}")
        
        if not 0 <= dropout_rate <= 1:
            raise ValueError(f"dropout_rate doit être entre 0 et 1")
        
        if input_channels != 3:
            logger.warning(
                f"⚠️ Transfer learning nécessite input_channels=3 (RGB). "
                f"Valeur {input_channels} ignorée."
            )
            input_channels = 3
        
        # === STOCKAGE DES PARAMÈTRES ===
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freeze_layers = freeze_layers
        self.dropout_rate = dropout_rate
        self.input_size = input_size if input_size else (224, 224) 
        self.input_channels = input_channels
        
        # Récupére les infos du modèle
        model_info = self.SUPPORTED_MODELS[model_name]
        self.num_features = model_info["features"]
        
        # === CHARGEMENT DU BACKBONE ===
        try:
            if pretrained:
                # PyTorch 0.13+ syntax
                weights = "IMAGENET1K_V1"
                self.backbone = models.__dict__[model_name](weights=weights)
                logger.info(f"✅ Modèle {model_name} chargé avec poids ImageNet")
            else:
                self.backbone = models.__dict__[model_name](weights=None)
                logger.info(f"✅ Modèle {model_name} chargé sans poids")
        
        except Exception as e:
            logger.warning(f"⚠️ Erreur chargement avec 'weights': {e}")
            # Fallback pour anciennes versions PyTorch
            try:
                self.backbone = models.__dict__[model_name](pretrained=pretrained)
                logger.info(f"✅ Modèle {model_name} chargé (fallback pretrained={pretrained})")
            except Exception as e2:
                logger.error(f"❌ Erreur chargement {model_name}: {e2}")
                raise
        
        # === REMPLACEMENT DU CLASSIFIER ===
        self._replace_classifier(use_custom_classifier)
        
        # === GEL DES COUCHES ===
        if freeze_layers > 0 or freeze_layers == -1:
            self._freeze_layers(freeze_layers)
        
        logger.info(
            f"TransferLearningModel initialisé: "
            f"model={model_name}, "
            f"classes={num_classes}, "
            f"frozen_layers={freeze_layers}, "
            f"params={self.count_parameters():,}"
        )
    
    def _replace_classifier(self, use_custom: bool):
        """
        Remplace le classifier du modèle backbone.       
        Gère automatiquement les différentes architectures (ResNet, VGG, etc)
        """
        if use_custom:
            # Classifier multi-couches avec BatchNorm et Dropout
            classifier = nn.Sequential(
                nn.Linear(self.num_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=self.dropout_rate),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=self.dropout_rate * 0.5),
                nn.Linear(256, self.num_classes)
            )
        else:
            # Classifier simple
            classifier = nn.Linear(self.num_features, self.num_classes)
        
        # Identifie et remplace le classifier selon le type de modèle
        if hasattr(self.backbone, 'fc'):  # ResNet, DenseNet
            self.backbone.fc = classifier
            logger.debug(f"Classifier remplacé via 'fc' pour {self.model_name}")
        
        elif hasattr(self.backbone, 'classifier'):  # VGG, MobileNet, EfficientNet
            if isinstance(self.backbone.classifier, nn.Sequential):
                # Remplace la dernière couche du Sequential
                last_layer_idx = len(self.backbone.classifier) - 1
                self.backbone.classifier[last_layer_idx] = classifier
                logger.debug(f"Classifier remplacé via 'classifier[{last_layer_idx}]'")
            else:
                self.backbone.classifier = classifier
                logger.debug(f"Classifier remplacé via 'classifier'")
        
        else:
            raise ValueError(
                f"Impossible de remplacer le classifier pour {self.model_name}. "
                f"Attributs disponibles: {dir(self.backbone)}"
            )
    
    def _freeze_layers(self, num_layers: int):
        """
        Gèle les N premières couches du modèle.
        Utile pour fine-tuning progressif.
        
        Args:
            num_layers: Nombre de couches à geler
                       -1 = Geler tout sauf le classifier
                        0 = Rien de gelé
                       >0 = Geler les N premières couches
        """
        if num_layers == -1:
            # Geler tout sauf le classifier
            for name, param in self.backbone.named_parameters():
                # Ne pas geler les paramètres du classifier
                if 'fc' not in name and 'classifier' not in name:
                    param.requires_grad = False
            
            logger.info("✅ Toutes les couches gelées sauf le classifier")
            return
        
        # Geler les N premières couches
        frozen_count = 0
        for param in self.backbone.parameters():
            if frozen_count < num_layers:
                param.requires_grad = False
                frozen_count += 1
            else:
                break
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        
        logger.info(
            f"✅ Gelé {frozen_count} couches. "
            f"Paramètres entraînables: {trainable:,}/{total:,} "
            f"({trainable/total*100:.1f}%)"
        )
    
    def unfreeze_layers(self, num_layers: int = -1):
        """
        Dégèle des couches pour fine-tuning progressif.       
        Args:
            num_layers: Nombre de couches à dégeler
                       -1 = Tout dégeler
                       >0 = Dégeler les N dernières couches
        """
        if num_layers == -1:
            # Dégeler tout
            for param in self.backbone.parameters():
                param.requires_grad = True
            logger.info("✅ Toutes les couches dégelées")
        else:
            # Dégeler les N dernières couches
            params_list = list(self.backbone.parameters())
            for param in params_list[-num_layers:]:
                param.requires_grad = True
            logger.info(f"✅ Dégelé les {num_layers} dernières couches")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass avec validation.    
        Accepte toutes les tailles d'images (grâce aux backbones pré-entraînés)      
        Args:
            x: Images (batch_size, 3, height, width)
            
        Returns:
            Logits (batch_size, num_classes)
            
        Raises:
            ValueError: Si dimensions invalides
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor (B,C,H,W), got {x.dim()}D")
        
        if x.size(1) != 3:
            raise ValueError(
                f"Transfer learning nécessite 3 canaux (RGB), "
                f"reçu: {x.size(1)} canaux"
            )
        
        try:
            return self.backbone(x)
        
        except Exception as e:
            logger.error(f"Erreur dans forward {self.model_name}: {e}")
            logger.error(f"Input shape: {x.shape}")
            raise
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extrait les features avant le classifier.
        Utile pour feature extraction ou visualisation.
        
        Args:
            x: Images d'entrée
            
        Returns:
            Features (batch_size, num_features)
        """
        # Retirer temporairement le classifier
        if hasattr(self.backbone, 'fc'):
            original_fc = self.backbone.fc
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, 'classifier'):
            original_classifier = self.backbone.classifier
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError("Cannot extract features from this model")
        
        # Forward sans classifier
        features = self.backbone(x)
        
        # Restaurer le classifier
        if hasattr(self.backbone, 'fc'):
            self.backbone.fc = original_fc
        else:
            self.backbone.classifier = original_classifier
        
        return features
    
    def count_parameters(self, trainable_only: bool = False) -> int:
        """
        Compte les paramètres du modèle.        
        Args:
            trainable_only: Compter uniquement les paramètres entraînables
            
        Returns:
            Nombre de paramètres
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
    
    def summary(self) -> Dict[str, Any]:
        """Retourne un résumé complet du modèle."""
        trainable = self.count_parameters(trainable_only=True)
        total = self.count_parameters(trainable_only=False)
        
        return {
            "model_type": "TransferLearning",
            "backbone": self.model_name,
            "num_classes": self.num_classes,
            "pretrained": self.pretrained,
            "frozen_layers": self.freeze_layers,
            "dropout_rate": self.dropout_rate,
            "input_size": self.input_size,
            "input_channels": self.input_channels,
            "num_features": self.num_features,
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": total - trainable,
            "trainable_percentage": (trainable / total * 100) if total > 0 else 0
        }
    
    def enable_gradient_checkpointing(self):
        """
        Active le gradient checkpointing pour économiser la mémoire.
        Utile pour les gros modèles ou petits GPU.
        """
        if hasattr(self.backbone, 'gradient_checkpointing_enable'):
            self.backbone.gradient_checkpointing_enable()
            logger.info("✅ Gradient checkpointing activé")
        else:
            logger.warning(
                f"⚠️ Gradient checkpointing non supporté pour {self.model_name}"
            )


# === FONCTION FACTORY ===

def get_transfer_learning_model(
    model_name: str = "resnet50",
    num_classes: int = 2,
    pretrained: bool = True,
    **kwargs
) -> TransferLearningModel:
    """
    Factory pour créer des modèles de transfer learning.
    
    Args:
        model_name: Nom du modèle backbone
        num_classes: Nombre de classes
        pretrained: Charger les poids pré-entraînés
        **kwargs: Arguments additionnels (freeze_layers, dropout_rate, input_size, etc.)
        
    Returns:
        TransferLearningModel configuré
        
    Example:
        >>> model = get_transfer_learning_model(
        ...     "resnet50",
        ...     num_classes=10,
        ...     freeze_layers=100,
        ...     dropout_rate=0.3
        ... )
    """
    try:
        model = TransferLearningModel(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )
        
        logger.info(f"✅ Modèle {model_name} créé avec succès")
        return model
    
    except Exception as e:
        logger.error(f"❌ Erreur création modèle {model_name}: {e}")
        raise

# === STRATÉGIES DE FINE-TUNING ===

class FineTuningScheduler:
    """
    Gestionnaire de stratégies de fine-tuning progressif.
    
    Permet de dégeler progressivement les couches selon une stratégie:
        - "immediate": Tout entraînable dès le début
        - "gradual": Dégel progressif par étapes
        - "top_down": Dégel de haut en bas (classifier → features)
    
    Example:
        >>> model = get_transfer_learning_model("resnet50", freeze_layers=-1)
        >>> scheduler = FineTuningScheduler(model, strategy="gradual")
        >>> scheduler.step(epoch=5)  # Dégèle des couches à l'époque 5
    """
    
    def __init__(
        self,
        model: TransferLearningModel,
        strategy: str = "gradual",
        unfreeze_epochs: Optional[List[int]] = None
    ):
        """
        Args:
            model: Modèle à fine-tuner
            strategy: Stratégie de fine-tuning ("immediate", "gradual", "top_down")
            unfreeze_epochs: Époques auxquelles dégeler des couches
        """
        self.model = model
        self.strategy = strategy
        
        if unfreeze_epochs is None:
            # Dégel par défaut aux époques 3, 6, 9
            self.unfreeze_epochs = [3, 6, 9]
        else:
            self.unfreeze_epochs = sorted(unfreeze_epochs)
        
        self.current_step = 0
        
        logger.info(
            f"FineTuningScheduler initialisé: "
            f"strategy={strategy}, "
            f"unfreeze_epochs={self.unfreeze_epochs}"
        )
    
    def step(self, epoch: int):
        """
        Exécute l'étape de fine-tuning pour l'époque donnée.       
        Args:
            epoch: Numéro de l'époque actuelle
        """
        if self.strategy == "immediate":
            # Tout dégelé dès le début
            if epoch == 0:
                self.model.unfreeze_layers(-1)
                logger.info("Strategy 'immediate': Toutes les couches dégelées")
        
        elif self.strategy == "gradual":
            # Dégel progressif
            if epoch in self.unfreeze_epochs:
                step_idx = self.unfreeze_epochs.index(epoch)
                
                # Calculer combien de couches dégeler
                total_layers = len(list(self.model.backbone.parameters()))
                layers_to_unfreeze = total_layers // len(self.unfreeze_epochs)
                
                self.model.unfreeze_layers(layers_to_unfreeze)
                
                logger.info(
                    f"Strategy 'gradual': Dégelé {layers_to_unfreeze} couches "
                    f"à l'époque {epoch}"
                )
        
        elif self.strategy == "top_down":
            # Dégel de haut en bas (classifier → features)
            if epoch in self.unfreeze_epochs:
                # Dégeler progressivement depuis le classifier
                step_idx = self.unfreeze_epochs.index(epoch)
                total_steps = len(self.unfreeze_epochs)
                
                # Calculer quelle proportion dégeler
                unfreeze_ratio = (step_idx + 1) / total_steps
                
                params_list = list(self.model.backbone.parameters())
                layers_to_unfreeze = int(len(params_list) * unfreeze_ratio)
                
                # Dégeler depuis la fin
                for param in params_list[-layers_to_unfreeze:]:
                    param.requires_grad = True
                
                logger.info(
                    f"Strategy 'top_down': Dégelé {unfreeze_ratio*100:.0f}% "
                    f"des couches à l'époque {epoch}"
                )
        
        else:
            logger.warning(f"⚠️ Stratégie '{self.strategy}' non reconnue")


# === UTILITAIRES ===

def list_available_models() -> List[str]:
    """Liste tous les modèles disponibles."""
    return list(TransferLearningModel.SUPPORTED_MODELS.keys())


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Retourne les informations détaillées d'un modèle.
    
    Args:
        model_name: Nom du modèle
        
    Returns:
        Dictionnaire avec infos (type, features, etc.)
    """
    if model_name not in TransferLearningModel.SUPPORTED_MODELS:
        raise ValueError(f"Modèle '{model_name}' non supporté")
    
    return TransferLearningModel.SUPPORTED_MODELS[model_name]


def compare_models(
    model_names: List[str],
    num_classes: int = 2,
    input_shape: tuple = (1, 3, 224, 224)
) -> Dict[str, Dict[str, Any]]:
    """
    Compare plusieurs modèles en termes de paramètres et vitesse.
    
    Args:
        model_names: Liste des modèles à comparer
        num_classes: Nombre de classes
        input_shape: Shape d'entrée pour benchmark
        
    Returns:
        Dictionnaire avec statistiques pour chaque modèle
    """
    import time
    
    results = {}
    
    for model_name in model_names:
        try:
            # Créer le modèle
            model = get_transfer_learning_model(
                model_name,
                num_classes=num_classes,
                pretrained=False  # Plus rapide sans poids
            )
            
            model.eval()
            
            # Comptage des paramètres
            summary = model.summary()
            
            # Benchmark vitesse (inference)
            dummy_input = torch.randn(input_shape)
            
            with torch.no_grad():
                # Warmup
                for _ in range(5):
                    _ = model(dummy_input)
                
                # Mesure
                start = time.time()
                for _ in range(50):
                    _ = model(dummy_input)
                end = time.time()
            
            avg_time = (end - start) / 50 * 1000  # en ms
            
            results[model_name] = {
                "total_params": summary["total_parameters"],
                "trainable_params": summary["trainable_parameters"],
                "inference_time_ms": avg_time,
                "throughput_imgs_per_sec": 1000 / avg_time * input_shape[0]
            }
            
            logger.info(f"✅ Benchmark {model_name}: {avg_time:.2f}ms par forward")
        
        except Exception as e:
            logger.error(f"❌ Erreur benchmark {model_name}: {e}")
            results[model_name] = {"error": str(e)}
    
    return results
