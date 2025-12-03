"""
Module de classification des types d'anomalies.
Permet d'identifier crack, scratch, hole, contamination, etc.
"""
import numpy as np
import torch # type: ignore
import torch.nn as nn # type: ignore
from typing import Dict, Any, Optional, Tuple, List
from src.shared.logging import get_logger

logger = get_logger(__name__)


class AnomalyTypeClassifier(nn.Module):
    """
    Classificateur des types d'anomalies.
    Entraîné en plus de l'autoencoder pour identifier crack/scratch/hole/contamination/etc.
    
    ✅ CORRECTION #14: Nouveau module pour classification multi-classes des types d'erreurs
    """
    
    # Mapping standard des types d'anomalies
    ANOMALY_TYPE_MAPPING = {
        0: "normal",
        1: "crack",
        2: "scratch",
        3: "hole",
        4: "contamination",
        5: "deformation",
        6: "stain",
        7: "misalignment",
        8: "unknown"
    }
    
    def __init__(
        self,
        backbone: nn.Module,  # Autoencoder ou extracteur de features pré-entraîné
        num_anomaly_types: int = 9,  # normal + 8 types d'anomalies
        embedding_dim: Optional[int] = None,
        freeze_backbone: bool = True
    ):
        """
        Args:
            backbone: Modèle pré-entraîné (autoencoder, CNN, etc.)
            num_anomaly_types: Nombre de types d'anomalies à classifier
            embedding_dim: Dimension de l'espace latent (si None, détecté automatiquement)
            freeze_backbone: Si True, gèle les poids du backbone
        """
        super(AnomalyTypeClassifier, self).__init__()
        
        self.backbone = backbone
        self.num_anomaly_types = num_anomaly_types
        self.freeze_backbone = freeze_backbone
        
        # Geler le backbone si demandé
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("✅ Backbone gelé pour classification types")
        
        # Détection automatique de la dimension de l'espace latent
        if embedding_dim is None:
            embedding_dim = self._detect_embedding_dim(backbone)
        
        self.embedding_dim = embedding_dim
        
        # Classificateur sur l'espace latent
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_anomaly_types)
        )
        
        logger.info(
            f"✅ AnomalyTypeClassifier initialisé - "
            f"embedding_dim: {embedding_dim}, "
            f"num_types: {num_anomaly_types}, "
            f"freeze_backbone: {freeze_backbone}"
        )
    
    def _detect_embedding_dim(self, backbone: nn.Module) -> int:
        """Détecte automatiquement la dimension de l'espace latent."""
        try:
            # Stratégie 1: Vérifier si le modèle a un attribut latent_dim
            if hasattr(backbone, 'latent_dim'):
                return backbone.latent_dim
            
            # Stratégie 2: Vérifier si c'est un autoencoder avec encoder
            if hasattr(backbone, 'encoder'):
                # Forward pass test pour déterminer la dimension
                test_input = torch.randn(1, 3, 64, 64)
                with torch.no_grad():
                    if hasattr(backbone, 'forward') and not hasattr(backbone, 'encode'):
                        # Autoencoder standard: passer par encoder puis flatten
                        encoded = backbone.encoder(test_input)
                        if hasattr(encoded, 'flatten'):
                            encoded_flat = encoded.flatten(start_dim=1)
                        else:
                            encoded_flat = encoded.view(encoded.size(0), -1)
                        return encoded_flat.shape[1]
            
            # Stratégie 3: Vérifier si c'est un modèle avec fc_encode
            if hasattr(backbone, 'fc_encode'):
                for layer in backbone.fc_encode:
                    if isinstance(layer, nn.Linear):
                        return layer.out_features
            
            # Fallback: dimension par défaut
            logger.warning(
                "⚠️ Impossible de détecter embedding_dim automatiquement, utilisation valeur par défaut 256"
            )
            return 256
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur détection embedding_dim: {e}, utilisation valeur par défaut 256")
            return 256
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extrait les features depuis le backbone.
        
        Args:
            x: Images (B, C, H, W)
        
        Returns:
            Features (B, embedding_dim)
        """
        with torch.set_grad_enabled(not self.freeze_backbone):
            if hasattr(self.backbone, 'encoder'):
                # Autoencoder: utiliser encoder uniquement
                encoded = self.backbone.encoder(x)
                # Flatten si nécessaire
                if encoded.dim() > 2:
                    encoded = encoded.view(encoded.size(0), -1)
                return encoded
            elif hasattr(self.backbone, 'fc_encode'):
                # Autoencoder avec fc_encode
                encoded = self.backbone.fc_encode(x.view(x.size(0), -1))
                return encoded
            else:
                # CNN ou autre: utiliser forward puis adapter
                features = self.backbone(x)
                if features.dim() > 2:
                    # Global average pooling
                    features = features.mean(dim=(2, 3))
                return features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass complet.
        
        Args:
            x: Images (B, C, H, W)
        
        Returns:
            Logits de classification (B, num_anomaly_types)
        """
        # Extraction features
        features = self.extract_features(x)
        
        # Classification
        logits = self.classifier(features)
        
        return logits
    
    def predict_anomaly_types(
        self,
        x: torch.Tensor,
        return_proba: bool = True
    ) -> Dict[str, Any]:
        """
        Prédit les types d'anomalies.
        
        Args:
            x: Images (B, C, H, W)
            return_proba: Si True, retourne aussi les probabilités
        
        Returns:
            Dict avec:
                - 'types': (B,) Types d'anomalies prédits (indices)
                - 'type_names': (B,) Noms des types
                - 'probabilities': (B, num_types) Probabilités si return_proba=True
        """
        self.eval()
        
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            pred_types = torch.argmax(probs, dim=1)
        
        # Conversion en numpy
        pred_types_np = pred_types.cpu().numpy()
        type_names = [self.ANOMALY_TYPE_MAPPING.get(int(t), "unknown") for t in pred_types_np]
        
        result = {
            'types': pred_types_np,
            'type_names': type_names
        }
        
        if return_proba:
            result['probabilities'] = probs.cpu().numpy()
        
        return result
    
    @classmethod
    def get_type_name(cls, type_id: int) -> str:
        """Retourne le nom d'un type d'anomalie."""
        return cls.ANOMALY_TYPE_MAPPING.get(type_id, "unknown")
    
    @classmethod
    def get_type_id(cls, type_name: str) -> int:
        """Retourne l'ID d'un type d'anomalie."""
        reverse_mapping = {v: k for k, v in cls.ANOMALY_TYPE_MAPPING.items()}
        return reverse_mapping.get(type_name.lower(), 8)  # 8 = unknown


def load_anomaly_type_labels_from_mvtec(data_dir: str) -> Optional[np.ndarray]:
    """
    Charge les labels de types d'anomalies depuis la structure MVTec AD.
    
    Les dossiers test/crack/, test/scratch/, etc. contiennent les types.
    
    Args:
        data_dir: Chemin vers le dataset MVTec AD
    
    Returns:
        Array de labels de types (None si impossible de charger)
    """
    from pathlib import Path
    
    try:
        test_path = Path(data_dir) / "test"
        if not test_path.exists():
            return None
        
        # Mapping nom dossier → type_id
        folder_to_type = {
            "good": 0,  # normal
            "crack": 1,
            "scratch": 2,
            "hole": 3,
            "contamination": 4,
            "deformation": 5,
            "stain": 6,
            "misalignment": 7
        }
        
        type_labels = []
        
        for category_folder in test_path.iterdir():
            if category_folder.is_dir():
                category_name = category_folder.name.lower()
                
                if category_name in folder_to_type:
                    type_id = folder_to_type[category_name]
                    # Compter les images dans ce dossier
                    image_files = [
                        f for f in category_folder.iterdir()
                        if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
                    ]
                    # Ajouter les labels pour toutes les images de ce type
                    type_labels.extend([type_id] * len(image_files))
        
        if len(type_labels) == 0:
            logger.warning("⚠️ Aucun label de type d'anomalie trouvé dans MVTec AD")
            return None
        
        logger.info(
            f"✅ Labels de types chargés: {len(type_labels)} images, "
            f"types: {set(type_labels)}"
        )
        
        return np.array(type_labels)
        
    except Exception as e:
        logger.error(f"Erreur chargement labels types MVTec: {e}", exc_info=True)
        return None

