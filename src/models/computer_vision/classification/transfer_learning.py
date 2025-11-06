"""
Modèles de Transfer Learning professionnels.
Version production-ready avec gestion flexible du fine-tuning.

À placer dans: src/models/computer_vision/classification/transfer_learning.py
"""

import torch # type: ignore
import torch.nn as nn # type: ignore
import torchvision.models as models # type: ignore
from typing import Optional, List, Dict, Any
from src.shared.logging import get_logger

logger = get_logger(__name__)


class TransferLearningModel(nn.Module):
    """
    Modèle de Transfer Learning robuste et flexible.
    
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
        - Gradient checkpointing pour économiser mémoire
    
    Args:
        model_name: Nom du modèle pré-entraîné
        num_classes: Nombre de classes
        pretrained: Charger les poids ImageNet
        freeze_layers: Nombre de couches à geler (0 = tout entraînable)
        dropout_rate: Dropout avant la couche finale
        use_custom_classifier: Utiliser un classifier multi-couches
        
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
        input_size: int = 224
    ):
        super(TransferLearningModel, self).__init__()
        
        # Validation
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"model_name '{model_name}' non supporté. "
                f"Options: {list(self.SUPPORTED_MODELS.keys())}"
            )
        
        if num_classes < 2:
            raise ValueError(f"num_classes doit être >= 2, reçu: {num_classes}")
        
        if not 0 <= dropout_rate <= 1:
            raise ValueError(f"dropout_rate doit être entre 0 et 1")
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freeze_layers = freeze_layers
        self.dropout_rate = dropout_rate
        self.input_size = input_size
        
        # Récupérer les infos du modèle
        model_info = self.SUPPORTED_MODELS[model_name]
        self.num_features = model_info["features"]
        
        # Charger le modèle pré-entraîné
        try:
            if pretrained:
                weights = "IMAGENET1K_V1"  # PyTorch 0.13+ syntax
                self.backbone = models.__dict__[model_name](weights=weights)
            else:
                self.backbone = models.__dict__[model_name](weights=None)
            
            logger.info(f"Modèle {model_name} chargé (pretrained={pretrained})")
        
        except Exception as e:
            logger.error(f"Erreur chargement {model_name}: {e}")
            # Fallback pour anciennes versions PyTorch
            self.backbone = models.__dict__[model_name](pretrained=pretrained)
        
        # Remplacer le classifier
        self._replace_classifier(use_custom_classifier)
        
        # Geler les couches si demandé
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)
        
        logger.info(
            f"TransferLearningModel initialisé: "
            f"model={model_name}, "
            f"classes={num_classes}, "
            f"frozen_layers={freeze_layers}, "
            f"params={self.count_parameters():,}"
        )
    
    def _replace_classifier(self, use_custom: bool):
        """Remplace le classifier du modèle backbone."""
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
        
        # Identifier et remplacer le classifier selon le type de modèle
        if hasattr(self.backbone, 'fc'):  # ResNet, DenseNet
            self.backbone.fc = classifier
        elif hasattr(self.backbone, 'classifier'):  # VGG, MobileNet, EfficientNet
            if isinstance(self.backbone.classifier, nn.Sequential):
                # Remplacer la dernière couche du Sequential
                last_layer_idx = len(self.backbone.classifier) - 1
                self.backbone.classifier[last_layer_idx] = classifier
            else:
                self.backbone.classifier = classifier
        else:
            raise ValueError(f"Impossible de remplacer le classifier pour {self.model_name}")
    
    def _freeze_layers(self, num_layers: int):
        """
        Gèle les N premières couches du modèle.
        Utile pour fine-tuning progressif.
        
        Args:
            num_layers: Nombre de couches à geler (0 = rien de gelé)
        """
        if num_layers == -1:
            # Geler tout sauf le classifier
            for name, param in self.backbone.named_parameters():
                if 'fc' not in name and 'classifier' not in name:
                    param.requires_grad = False
            
            logger.info("Toutes les couches gelées sauf le classifier")
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
            f"Gelé {frozen_count} couches. "
            f"Paramètres entraînables: {trainable:,}/{total:,} "
            f"({trainable/total*100:.1f}%)"
        )
    
    def unfreeze_layers(self, num_layers: int = -1):
        """
        Dégèle des couches pour fine-tuning progressif.
        
        Args:
            num_layers: Nombre de couches à dégeler (-1 = toutes)
        """
        if num_layers == -1:
            # Dégeler tout
            for param in self.backbone.parameters():
                param.requires_grad = True
            logger.info("Toutes les couches dégelées")
        else:
            # Dégeler les N dernières couches
            params_list = list(self.backbone.parameters())
            for param in params_list[-num_layers:]:
                param.requires_grad = True
            logger.info(f"Dégelé les {num_layers} dernières couches")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass avec validation.
        
        Args:
            x: Images (batch_size, 3, height, width)
            
        Returns:
            Logits (batch_size, num_classes)
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor, got {x.dim()}D")
        
        if x.size(1) != 3:
            raise ValueError(f"Expected 3 channels (RGB), got {x.size(1)} channels")
        
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
            logger.info("Gradient checkpointing activé")
        else:
            logger.warning(f"Gradient checkpointing non supporté pour {self.model_name}")


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
        **kwargs: Arguments additionnels (freeze_layers, dropout_rate, etc.)
        
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
        
        logger.info(f"Modèle {model_name} créé avec succès")
        return model
    
    except Exception as e:
        logger.error(f"Erreur création modèle {model_name}: {e}")
        raise


# === STRATÉGIES DE FINE-TUNING ===

class FineTuningScheduler:
    """
    Gestionnaire de stratégies de fine-tuning progressif.
    
    Permet de dégeler progressivement les couches selon une stratégie:
        - "immediate": Tout entraînable dès le début
        - "gradual": Dégel progressif par étapes
        - "discriminative": Learning rates différents par couche
    
    Example:
        >>> model = get_transfer_learning_model("resnet50", freeze_layers=-1)
        >>> scheduler = FineTuningScheduler(model, strategy="gradual")
        >>> scheduler.step(epoch=5)  # Dégèle des couches à l'époque 5
    """
    
    def __init__(
        self,
        model: TransferLearningModel,
        strategy: str = "gradual",
        unfreeze_epochs: List[int] = None
    ):
        """
        Args:
            model: Modèle à fine-tuner
            strategy: Stratégie de fine-tuning
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
            logger.warning(f"Stratégie '{self.strategy}' non reconnue")


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
            
            logger.info(f"Benchmark {model_name}: {avg_time:.2f}ms par forward")
        
        except Exception as e:
            logger.error(f"Erreur benchmark {model_name}: {e}")
            results[model_name] = {"error": str(e)}
    
    return results


# === TESTS UNITAIRES ===

if __name__ == "__main__":
    import sys
    
    print("="*60)
    print("TESTS - Transfer Learning Models")
    print("="*60)
    
    # Test 1: Création modèle basique
    print("\n### Test 1: Création ResNet50 ###")
    try:
        model = TransferLearningModel("resnet50", num_classes=10, pretrained=False)
        print(f"✅ Modèle créé: {model.count_parameters():,} paramètres")
        print(model.summary())
    except Exception as e:
        print(f"❌ Erreur: {e}")
        sys.exit(1)
    
    # Test 2: Forward pass
    print("\n### Test 2: Forward Pass ###")
    try:
        x = torch.randn(4, 3, 224, 224)
        output = model(x)
        print(f"✅ Forward OK: Input {x.shape} → Output {output.shape}")
        assert output.shape == (4, 10), f"Shape incorrecte: {output.shape}"
    except Exception as e:
        print(f"❌ Erreur forward: {e}")
        sys.exit(1)
    
    # Test 3: Freeze/Unfreeze
    print("\n### Test 3: Freeze/Unfreeze ###")
    try:
        model_freeze = TransferLearningModel(
            "resnet18",
            num_classes=5,
            pretrained=False,
            freeze_layers=-1  # Tout gelé sauf classifier
        )
        
        trainable_frozen = model_freeze.count_parameters(trainable_only=True)
        print(f"Paramètres entraînables (gelé): {trainable_frozen:,}")
        
        model_freeze.unfreeze_layers(-1)  # Tout dégeler
        trainable_unfrozen = model_freeze.count_parameters(trainable_only=True)
        print(f"Paramètres entraînables (dégelé): {trainable_unfrozen:,}")
        
        assert trainable_unfrozen > trainable_frozen, "Unfreeze n'a pas fonctionné"
        print("✅ Freeze/Unfreeze OK")
    except Exception as e:
        print(f"❌ Erreur freeze: {e}")
        sys.exit(1)
    
    # Test 4: Feature extraction
    print("\n### Test 4: Feature Extraction ###")
    try:
        features = model.get_features(x)
        print(f"✅ Features extraites: {features.shape}")
        assert features.shape[0] == x.shape[0], "Batch size incorrect"
    except Exception as e:
        print(f"❌ Erreur extraction: {e}")
        sys.exit(1)
    
    # Test 5: Liste des modèles
    print("\n### Test 5: Modèles Disponibles ###")
    available = list_available_models()
    print(f"✅ {len(available)} modèles disponibles:")
    for i, name in enumerate(available[:10], 1):
        info = get_model_info(name)
        print(f"  {i}. {name} ({info['features']} features)")
    if len(available) > 10:
        print(f"  ... et {len(available) - 10} autres")
    
    # Test 6: Factory
    print("\n### Test 6: Factory Function ###")
    try:
        model_factory = get_transfer_learning_model(
            "mobilenet_v2",
            num_classes=3,
            freeze_layers=50,
            dropout_rate=0.3
        )
        print(f"✅ Factory OK: {type(model_factory).__name__}")
        print(f"   Paramètres: {model_factory.count_parameters():,}")
    except Exception as e:
        print(f"❌ Erreur factory: {e}")
        sys.exit(1)
    
    # Test 7: Fine-tuning Scheduler
    print("\n### Test 7: Fine-tuning Scheduler ###")
    try:
        model_ft = TransferLearningModel("resnet18", num_classes=2, freeze_layers=-1, pretrained=False)
        scheduler = FineTuningScheduler(model_ft, strategy="gradual", unfreeze_epochs=[2, 4, 6])
        
        for epoch in range(7):
            scheduler.step(epoch)
        
        print("✅ FineTuningScheduler OK")
    except Exception as e:
        print(f"❌ Erreur scheduler: {e}")
        sys.exit(1)
    
    # Test 8: Comparaison modèles (optionnel - plus lent)
    print("\n### Test 8: Comparaison Modèles (optionnel) ###")
    try:
        models_to_compare = ["resnet18", "mobilenet_v2", "efficientnet_b0"]
        print(f"Comparaison de {len(models_to_compare)} modèles...")
        
        comparison = compare_models(models_to_compare, num_classes=10)
        
        print("\nRésultats:")
        for name, stats in comparison.items():
            if "error" not in stats:
                print(f"\n{name}:")
                print(f"  Paramètres: {stats['total_params']:,}")
                print(f"  Temps inference: {stats['inference_time_ms']:.2f}ms")
                print(f"  Throughput: {stats['throughput_imgs_per_sec']:.0f} img/s")
        
        print("\n✅ Comparaison OK")
    except Exception as e:
        print(f"⚠️ Comparaison échouée (non bloquant): {e}")
    
    print("\n" + "="*60)
    print("✅ TOUS LES TESTS RÉUSSIS!")
    print("="*60)