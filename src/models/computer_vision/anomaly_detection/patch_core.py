"""
PatchCore professionnel avec coreset subsampling et backbone pré-entraîné.
"""
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
import torchvision.models as models # type: ignore
import faiss # type: ignore
import numpy as np
from typing import List, Tuple, Optional
from src.shared.logging import get_logger

logger = get_logger(__name__)

class ProfessionalPatchCore(nn.Module):
    """
    Implémentation professionnelle de PatchCore.
    """
    
    def __init__(
        self,
        backbone_name: str = "wide_resnet50_2",
        layers: List[str] = ["layer2", "layer3"],
        faiss_index_type: str = "Flat",
        coreset_ratio: float = 0.01,
        num_neighbors: int = 1
    ):
        super().__init__()
        
        # Backbone pré-entraîné
        self.backbone = self._get_backbone(backbone_name)
        self.layers = layers
        self.coreset_ratio = coreset_ratio
        self.num_neighbors = num_neighbors
        
        # Memory bank et index FAISS
        self.memory_bank = None
        self.faiss_index = None
        self.feature_dim = None
        
        # Extraction des features
        self.feature_extractor = FeatureExtractor(self.backbone, layers)
        
    def _get_backbone(self, name: str) -> nn.Module:
        """Charge un backbone pré-entraîné."""
        if name == "wide_resnet50_2":
            model = models.wide_resnet50_2(pretrained=True)
        elif name == "resnet18":
            model = models.resnet18(pretrained=True)
        else:
            raise ValueError(f"Backbone {name} non supporté")
        
        # Geler les paramètres
        for param in model.parameters():
            param.requires_grad = False
            
        return model
    
    def _coreset_subsampling(self, features: np.ndarray) -> np.ndarray:
        """
        Coreset subsampling avec algorithme greedy k-center.
        Réduit la taille de la mémoire bank tout en préservant la couverture.
        """
        n_samples = features.shape[0]
        n_coreset = max(1, int(n_samples * self.coreset_ratio))
        
        if n_coreset >= n_samples:
            return features
        
        # Algorithme greedy k-center
        indices = [np.random.randint(n_samples)]
        distances = np.full(n_samples, np.inf)
        
        for _ in range(1, n_coreset):
            # Met à jour les distances
            new_distances = np.linalg.norm(
                features - features[indices[-1]], axis=1
            )
            distances = np.minimum(distances, new_distances)
            
            # Sélectionne le point le plus éloigné
            indices.append(np.argmax(distances))
        
        return features[indices]
    
    def fit(self, dataloader):
        """Construit la mémoire bank avec coreset subsampling."""
        all_features = []
        
        self.eval()
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch
                
                # Extraction des features
                features = self.feature_extractor(images)
                features = features.cpu().numpy()
                all_features.append(features)
        
        # Concatenation et reshaping
        all_features = np.concatenate(all_features, axis=0)
        all_features = all_features.reshape(all_features.shape[0], -1)
        
        # Normalisation L2
        all_features = all_features / np.linalg.norm(
            all_features, axis=1, keepdims=True
        )
        
        # Coreset subsampling
        self.memory_bank = self._coreset_subsampling(all_features)
        self.feature_dim = self.memory_bank.shape[1]
        
        # Construction de l'index FAISS
        self.faiss_index = faiss.IndexFlatL2(self.feature_dim)
        self.faiss_index.add(self.memory_bank)
        
        logger.info(
            f"PatchCore entraîné: {len(self.memory_bank)} patchs "
            f"({self.coreset_ratio*100:.1f}% du dataset original)"
        )
    
    def predict(self, dataloader) -> np.ndarray:
        """Calcule les scores d'anomalie."""
        if self.faiss_index is None:
            raise ValueError("Modèle non entraîné. Appelez fit() d'abord.")
        
        all_scores = []
        
        self.eval()
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch
                
                # Extraction et reshaping
                features = self.feature_extractor(images)
                batch_size, num_features, h, w = features.shape
                features = features.permute(0, 2, 3, 1).reshape(-1, num_features)
                features = features.cpu().numpy()
                
                # Normalisation
                features = features / np.linalg.norm(
                    features, axis=1, keepdims=True
                )
                
                # Recherche des plus proches voisins
                distances, _ = self.faiss_index.search(features, self.num_neighbors)
                patch_scores = distances[:, 0]
                
                # Reshape vers (batch_size, h, w)
                patch_scores = patch_scores.reshape(batch_size, h, w)
                
                # Score d'anomalie par image (max pooling)
                image_scores = patch_scores.max(axis=(1, 2))
                all_scores.append(image_scores)
        
        return np.concatenate(all_scores)


class FeatureExtractor(nn.Module):
    """Extracteur de features depuis plusieurs couches."""
    
    def __init__(self, backbone: nn.Module, layers: List[str]):
        super().__init__()
        self.backbone = backbone
        self.layers = layers
        self.features = {}
        
        # Enregistrement des hooks
        for layer_name in layers:
            layer = dict([*self.backbone.named_modules()])[layer_name]
            layer.register_forward_hook(self._get_hook(layer_name))
    
    def _get_hook(self, layer_name: str):
        def hook(module, input, output):
            self.features[layer_name] = output
        return hook
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.features.clear()
        _ = self.backbone(x)
        
        # Aggrégation des features multi-échelle
        feature_maps = []
        for layer_name in self.layers:
            feat = self.features[layer_name]
            # Adaptive pooling pour uniformiser la taille
            feat = F.adaptive_avg_pool2d(feat, (14, 14))
            feature_maps.append(feat)
        
        return torch.cat(feature_maps, dim=1)