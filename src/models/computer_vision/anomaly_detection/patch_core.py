"""
PatchCore avec coreset subsampling et backbone pré-entraîné.
Version avec gestion correcte des types et compatibility training pipeline.

Dans: src/models/computer_vision/anomaly_detection/patch_core.py

Pas besoin de resize dynamique car PatchCore accepte toutes les tailles via
adaptive pooling des feature maps.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import faiss
import numpy as np
from typing import List, Tuple, Optional, Union, Any
from src.shared.logging import get_logger

logger = get_logger(__name__)

class ProfessionalPatchCore(nn.Module):
    """
    PatchCore pour détection d'anomalies.  
    Accepte toutes les tailles d'images (adaptive pooling)
    Compatible avec training pipeline standard
    """
    
    def __init__(
        self,
        backbone_name: str = "wide_resnet50_2",
        layers: List[str] = None,
        faiss_index_type: str = "Flat",
        coreset_ratio: float = 0.01,
        num_neighbors: int = 1,
        input_channels: int = 3,
        input_size: Optional[Tuple[int, int]] = None,  
        **kwargs
    ):
        super().__init__()
        
        if layers is None:
            layers = ["layer2", "layer3"]
        
        self.backbone_name = backbone_name
        self.layers = layers
        self.coreset_ratio = coreset_ratio
        self.num_neighbors = num_neighbors
        self.input_channels = input_channels
        self.input_size = input_size  
        
        self.memory_bank = None
        self.faiss_index = None
        self.feature_dim = None
        self._is_fitted = False
        
        self.backbone = self._get_backbone(backbone_name)
        self.feature_extractor = FeatureExtractor(self.backbone, layers)
        
        if input_channels != 3:
            self.channel_adapter = nn.Conv2d(input_channels, 3, kernel_size=1, bias=False)
            nn.init.xavier_uniform_(self.channel_adapter.weight)
        else:
            self.channel_adapter = None
        
        logger.info(
            f"PatchCore initialisé - "
            f"backbone: {backbone_name}, "
            f"layers: {layers}, "
            f"coreset_ratio: {coreset_ratio}"
        )
    
    def _get_backbone(self, name: str) -> nn.Module:
        """Charge un backbone pré-entraîné et gèle ses paramètres."""
        if name == "wide_resnet50_2":
            model = models.wide_resnet50_2(pretrained=True)
        elif name == "resnet18":
            model = models.resnet18(pretrained=True)
        elif name == "resnet50":
            model = models.resnet50(pretrained=True)
        else:
            raise ValueError(
                f"Backbone '{name}' non supporté. "
                f"Disponibles: wide_resnet50_2, resnet18, resnet50"
            )
        
        for param in model.parameters():
            param.requires_grad = False
        
        model.eval()
        logger.info(f"Backbone '{name}' chargé et gelé")
        return model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass pour compatibilité avec le training pipeline standard.       
        Retourne None au lieu d'un dummy tensor avec requires_grad.
        Le training loop détectera None et skip le backward.
        
        PatchCore n'utilise PAS de backpropagation !
        Le vrai entraînement se fait via fit().     
        Args:
            x: Tensor d'entrée (B, C, H, W)         
        Returns:
            None (signale au training loop de skip le backward)
        """
        # PatchCore ne fait pas de forward pass classique
        # Retourner None pour signaler qu'il n'y a pas de backward à faire
        return None

    
    def _adapt_channels(self, x: torch.Tensor) -> torch.Tensor:
        """Adapte le nombre de canaux si nécessaire."""
        if self.channel_adapter is not None:
            return self.channel_adapter(x)
        return x
    
    def _coreset_subsampling(self, features: np.ndarray) -> np.ndarray:
        """Coreset subsampling avec algorithme greedy k-center."""
        n_samples = features.shape[0]
        n_coreset = max(1, int(n_samples * self.coreset_ratio))
        
        if n_coreset >= n_samples:
            logger.info("Coreset ratio >= 1.0, utilisation de tous les features")
            return features
        
        logger.info(
            f"Début coreset subsampling: {n_samples} → {n_coreset} samples "
            f"({self.coreset_ratio*100:.1f}%)"
        )
        
        indices = [np.random.randint(n_samples)]
        distances = np.full(n_samples, np.inf)
        
        for i in range(1, n_coreset):
            new_distances = np.linalg.norm(
                features - features[indices[-1]], axis=1
            )
            distances = np.minimum(distances, new_distances)
            indices.append(np.argmax(distances))
            
            if (i + 1) % max(1, n_coreset // 10) == 0:
                logger.debug(f"Coreset progress: {i+1}/{n_coreset}")
        
        coreset_features = features[indices]
        
        logger.info(
            f"Coreset subsampling terminé - "
            f"coverage: {len(indices)} points"
        )
        
        return coreset_features
    
    def fit(self, dataloader) -> None:
        """Construit la memory bank avec coreset subsampling."""
        logger.info("🔨 Début construction memory bank PatchCore")
        
        all_features = []
        
        self.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch
                
                images = self._adapt_channels(images)
                features = self.feature_extractor(images)
                features_np = features.cpu().numpy()
                all_features.append(features_np)
                
                if (batch_idx + 1) % 10 == 0:
                    logger.debug(f"Features extraction: batch {batch_idx+1}")
        
        all_features = np.concatenate(all_features, axis=0)
        logger.info(f"Features extraites: shape={all_features.shape}")
        
        # Reshape (N, C, H, W) → (N*H*W, C)
        n_samples, n_channels, h, w = all_features.shape
        all_features = all_features.transpose(0, 2, 3, 1)
        all_features = all_features.reshape(-1, n_channels)
        
        logger.info(f"Features reshaped: {all_features.shape}")
        
        # Normalisation L2
        norms = np.linalg.norm(all_features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        all_features = all_features / norms
        
        logger.info(f"Features normalisées (L2)")
        
        # Coreset subsampling
        self.memory_bank = self._coreset_subsampling(all_features)
        self.feature_dim = self.memory_bank.shape[1]
        
        # Construction index FAISS
        logger.info(f"Construction index FAISS (dim={self.feature_dim})")
        self.faiss_index = faiss.IndexFlatL2(self.feature_dim)
        self.faiss_index.add(self.memory_bank.astype(np.float32))
        
        self._is_fitted = True
        
        logger.info(
            f"✅ PatchCore entraîné - "
            f"memory_bank: {len(self.memory_bank)} patches"
        )
    
    def predict(self, dataloader) -> np.ndarray:
        """Calcule les scores d'anomalie sur un dataset."""
        if not self._is_fitted or self.faiss_index is None:
            raise ValueError(
                "Modèle non entraîné. Appelez fit(train_loader) avant predict()."
            )
        
        logger.info("🔮 Début prédictions PatchCore")
        
        all_scores = []
        
        self.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch
                
                images = self._adapt_channels(images)
                features = self.feature_extractor(images)
                batch_size, num_features, h, w = features.shape
                
                features = features.permute(0, 2, 3, 1).reshape(-1, num_features)
                features_np = features.cpu().numpy()
                
                # Normalisation L2
                norms = np.linalg.norm(features_np, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-8)
                features_np = features_np / norms
                
                distances, _ = self.faiss_index.search(
                    features_np.astype(np.float32),
                    self.num_neighbors
                )
                
                patch_scores = distances[:, 0]
                patch_scores = patch_scores.reshape(batch_size, h, w)
                image_scores = patch_scores.max(axis=(1, 2))
                all_scores.append(image_scores)
                
                if (batch_idx + 1) % 10 == 0:
                    logger.debug(f"Predictions: batch {batch_idx+1}")
        
        scores = np.concatenate(all_scores)
        
        logger.info(
            f"✅ Prédictions terminées - "
            f"n_samples: {len(scores)}"
        )
        
        return scores
    
    def get_anomaly_map(self, image: torch.Tensor) -> np.ndarray:
        """Génère une heatmap d'anomalie spatiale."""
        if not self._is_fitted:
            raise ValueError("Modèle non entraîné")
        
        if image.ndim == 3:
            image = image.unsqueeze(0)
        
        self.eval()
        with torch.no_grad():
            image = self._adapt_channels(image)
            features = self.feature_extractor(image)
            _, num_features, h, w = features.shape
            
            features = features.permute(0, 2, 3, 1).reshape(-1, num_features)
            features_np = features.cpu().numpy()
            
            norms = np.linalg.norm(features_np, axis=1, keepdims=True)
            features_np = features_np / np.maximum(norms, 1e-8)
            
            distances, _ = self.faiss_index.search(
                features_np.astype(np.float32),
                self.num_neighbors
            )
            
            anomaly_map = distances[:, 0].reshape(h, w)
            
            if anomaly_map.max() > anomaly_map.min():
                anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
            
            return anomaly_map


class FeatureExtractor(nn.Module):
    """Extracteur de features multi-échelles depuis un backbone CNN."""
    
    def __init__(self, backbone: nn.Module, layers: List[str]):
        super().__init__()
        self.backbone = backbone
        self.layers = layers
        self.features = {}
        
        layer_dict = dict([*self.backbone.named_modules()])
        
        for layer_name in layers:
            if layer_name not in layer_dict:
                raise ValueError(
                    f"Couche '{layer_name}' introuvable dans le backbone"
                )
            
            layer = layer_dict[layer_name]
            layer.register_forward_hook(self._get_hook(layer_name))
        
        logger.info(f"FeatureExtractor initialisé avec couches: {layers}")
    
    def _get_hook(self, layer_name: str):
        """Crée un hook pour capturer les activations."""
        def hook(module, input, output):
            self.features[layer_name] = output
        return hook
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extrait et agrège les features multi-échelles."""
        self.features.clear()
        _ = self.backbone(x)
        
        feature_maps = []
        
        for layer_name in self.layers:
            feat = self.features[layer_name]
            # Adaptive pooling pour uniformiser → (14, 14)
            feat_pooled = F.adaptive_avg_pool2d(feat, (14, 14))
            feature_maps.append(feat_pooled)
        
        aggregated = torch.cat(feature_maps, dim=1)
        return aggregated