"""
PatchCore professionnel avec coreset subsampling et backbone pré-entraîné.
Version production-ready avec intégration complète au training pipeline.
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
    Implémentation professionnelle de PatchCore pour détection d'anomalies.
    
    Architecture:
    - Extraction de features multi-échelles via backbone pré-entraîné
    - Coreset subsampling (greedy k-center) pour efficacité mémoire
    - Index FAISS pour recherche rapide des plus proches voisins
    - Pas de backpropagation (modèle non supervisé)
    
    Usage:
        model = ProfessionalPatchCore(backbone_name="wide_resnet50_2")
        model.fit(train_loader)  # Construction memory bank
        scores = model.predict(test_loader)  # Détection anomalies
    """
    
    def __init__(
        self,
        backbone_name: str = "wide_resnet50_2",
        layers: List[str] = ["layer2", "layer3"],
        faiss_index_type: str = "Flat",
        coreset_ratio: float = 0.01,
        num_neighbors: int = 1,
        input_channels: int = 3,
        **kwargs  # Absorbe les paramètres non utilisés pour compatibilité
    ):
        """
        Initialise PatchCore.
        
        Args:
            backbone_name: Nom du backbone ('wide_resnet50_2', 'resnet18')
            layers: Couches d'extraction des features
            faiss_index_type: Type d'index FAISS (actuellement 'Flat' uniquement)
            coreset_ratio: Ratio de subsampling (0.01 = 1% du dataset)
            num_neighbors: Nombre de voisins pour le scoring
            input_channels: Nombre de canaux d'entrée (1=grayscale, 3=RGB)
            **kwargs: Paramètres additionnels ignorés (pour compatibilité)
        """
        super().__init__()
        
        # Configuration
        self.backbone_name = backbone_name
        self.layers = layers
        self.coreset_ratio = coreset_ratio
        self.num_neighbors = num_neighbors
        self.input_channels = input_channels
        
        # Memory bank et index FAISS (initialisés à None)
        self.memory_bank = None
        self.faiss_index = None
        self.feature_dim = None
        self._is_fitted = False
        
        # Backbone pré-entraîné (frozen)
        self.backbone = self._get_backbone(backbone_name)
        
        # Extraction des features
        self.feature_extractor = FeatureExtractor(self.backbone, layers)
        
        # Adaptateur de canaux si nécessaire
        if input_channels != 3:
            self.channel_adapter = nn.Conv2d(input_channels, 3, kernel_size=1, bias=False)
            nn.init.xavier_uniform_(self.channel_adapter.weight)
        else:
            self.channel_adapter = None
        
        logger.info(
            f"PatchCore initialisé - "
            f"backbone: {backbone_name}, "
            f"layers: {layers}, "
            f"coreset_ratio: {coreset_ratio}, "
            f"input_channels: {input_channels}"
        )
    
    def _get_backbone(self, name: str) -> nn.Module:
        """
        Charge un backbone pré-entraîné et gèle ses paramètres.
        
        Args:
            name: Nom du backbone
            
        Returns:
            Backbone PyTorch avec paramètres gelés
            
        Raises:
            ValueError: Si le backbone n'est pas supporté
        """
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
        
        # Geler tous les paramètres (pas de fine-tuning)
        for param in model.parameters():
            param.requires_grad = False
        
        model.eval()
        
        logger.info(f"Backbone '{name}' chargé et gelé (requires_grad=False)")
        return model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass pour compatibilité avec le training pipeline standard.
        
        IMPORTANT: PatchCore n'utilise PAS de backpropagation !
        Cette méthode retourne un tenseur dummy avec gradient pour éviter
        les erreurs dans les boucles d'entraînement supervisées.
        
        Le vrai entraînement se fait via fit().
        
        Args:
            x: Tensor d'entrée (B, C, H, W)
            
        Returns:
            Tensor dummy (B, 1) avec requires_grad=True
        """
        batch_size = x.size(0)
        
        # Retourne un tenseur dummy qui simule une loss nulle
        # Cela permet au training loop standard de fonctionner sans erreur
        dummy = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
        
        # CRITIQUE: Ajouter requires_grad=True pour compatibilité backward()
        dummy.requires_grad = True
        
        return dummy
    
    def _adapt_channels(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adapte le nombre de canaux si nécessaire (grayscale → RGB).
        
        Args:
            x: Tensor d'entrée (B, C, H, W)
            
        Returns:
            Tensor avec 3 canaux (B, 3, H, W)
        """
        if self.channel_adapter is not None:
            return self.channel_adapter(x)
        return x
    
    def _coreset_subsampling(self, features: np.ndarray) -> np.ndarray:
        """
        Coreset subsampling avec algorithme greedy k-center.
        
        Réduit la taille de la memory bank tout en préservant la couverture
        spatiale maximale du dataset d'entraînement.
        
        Algorithme:
        1. Sélectionne un point initial aléatoire
        2. Itérativement sélectionne le point le plus éloigné de tous les points déjà sélectionnés
        3. Continue jusqu'à atteindre le ratio souhaité
        
        Args:
            features: Features normalisées (N, D)
            
        Returns:
            Subset de features (N_coreset, D) avec N_coreset = N * coreset_ratio
        """
        n_samples = features.shape[0]
        n_coreset = max(1, int(n_samples * self.coreset_ratio))
        
        if n_coreset >= n_samples:
            logger.info("Coreset ratio >= 1.0, utilisation de tous les features")
            return features
        
        logger.info(
            f"Début coreset subsampling: {n_samples} → {n_coreset} samples "
            f"({self.coreset_ratio*100:.1f}%)"
        )
        
        # Initialisation: premier point aléatoire
        indices = [np.random.randint(n_samples)]
        distances = np.full(n_samples, np.inf)
        
        # Algorithme greedy k-center
        for i in range(1, n_coreset):
            # Calcul des distances au dernier point ajouté
            new_distances = np.linalg.norm(
                features - features[indices[-1]], axis=1
            )
            
            # Mise à jour: distance minimale à l'ensemble des points sélectionnés
            distances = np.minimum(distances, new_distances)
            
            # Sélection: point le plus éloigné
            indices.append(np.argmax(distances))
            
            # Logging progressif
            if (i + 1) % max(1, n_coreset // 10) == 0:
                logger.debug(f"Coreset progress: {i+1}/{n_coreset}")
        
        coreset_features = features[indices]
        
        logger.info(
            f"Coreset subsampling terminé - "
            f"coverage: {len(indices)} points, "
            f"compression: {100*(1-n_coreset/n_samples):.1f}%"
        )
        
        return coreset_features
    
    def fit(self, dataloader) -> None:
        """
        Construit la memory bank avec coreset subsampling.
        
        Étapes:
        1. Extraction des features multi-échelles sur le dataset d'entraînement
        2. Normalisation L2 des features
        3. Coreset subsampling pour réduction mémoire
        4. Construction de l'index FAISS pour recherche efficace
        
        Args:
            dataloader: DataLoader PyTorch avec données d'entraînement (normales uniquement)
            
        Raises:
            RuntimeError: Si l'extraction de features échoue
        """
        logger.info("🔨 Début construction memory bank PatchCore")
        
        all_features = []
        
        self.eval()  # Mode évaluation
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Gestion format batch (data, labels) ou data seul
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch
                
                # Adaptation canaux si nécessaire
                images = self._adapt_channels(images)
                
                # Extraction des features via backbone
                features = self.feature_extractor(images)
                features_np = features.cpu().numpy()
                all_features.append(features_np)
                
                if (batch_idx + 1) % 10 == 0:
                    logger.debug(f"Features extraction: batch {batch_idx+1}")
        
        # Concatenation et reshape
        all_features = np.concatenate(all_features, axis=0)
        
        logger.info(f"Features extraites: shape={all_features.shape}")
        
        # Reshape (N, C, H, W) → (N*H*W, C)
        n_samples, n_channels, h, w = all_features.shape
        all_features = all_features.transpose(0, 2, 3, 1)  # (N, H, W, C)
        all_features = all_features.reshape(-1, n_channels)  # (N*H*W, C)
        
        logger.info(f"Features reshaped: {all_features.shape}")
        
        # Normalisation L2 (crucial pour distance cosine)
        norms = np.linalg.norm(all_features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # Éviter division par zéro
        all_features = all_features / norms
        
        logger.info(f"Features normalisées (L2)")
        
        # Coreset subsampling
        self.memory_bank = self._coreset_subsampling(all_features)
        self.feature_dim = self.memory_bank.shape[1]
        
        # Construction de l'index FAISS
        logger.info(f"Construction index FAISS (dim={self.feature_dim})")
        
        self.faiss_index = faiss.IndexFlatL2(self.feature_dim)
        self.faiss_index.add(self.memory_bank.astype(np.float32))
        
        self._is_fitted = True
        
        logger.info(
            f"✅ PatchCore entraîné avec succès - "
            f"memory_bank: {len(self.memory_bank)} patches, "
            f"feature_dim: {self.feature_dim}, "
            f"compression: {self.coreset_ratio*100:.1f}%"
        )
    
    def predict(self, dataloader) -> np.ndarray:
        """
        Calcule les scores d'anomalie sur un dataset.
        
        Pour chaque image:
        1. Extraction features multi-échelles
        2. Pour chaque patch: recherche du plus proche voisin dans memory bank
        3. Score image = distance maximale (anomalie la plus forte)
        
        Args:
            dataloader: DataLoader PyTorch avec données à évaluer
            
        Returns:
            Scores d'anomalie (N,) - Plus le score est élevé, plus l'image est anormale
            
        Raises:
            ValueError: Si le modèle n'est pas entraîné (fit() non appelé)
        """
        if not self._is_fitted or self.faiss_index is None:
            raise ValueError(
                "Modèle non entraîné. Appelez fit(train_loader) avant predict()."
            )
        
        logger.info("🔮 Début prédictions PatchCore")
        
        all_scores = []
        
        self.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Gestion format batch
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch
                
                # Adaptation canaux
                images = self._adapt_channels(images)
                
                # Extraction features
                features = self.feature_extractor(images)
                batch_size, num_features, h, w = features.shape
                
                # Reshape (B, C, H, W) → (B*H*W, C)
                features = features.permute(0, 2, 3, 1).reshape(-1, num_features)
                features_np = features.cpu().numpy()
                
                # Normalisation L2
                norms = np.linalg.norm(features_np, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-8)
                features_np = features_np / norms
                
                # Recherche des plus proches voisins dans memory bank
                distances, _ = self.faiss_index.search(
                    features_np.astype(np.float32),
                    self.num_neighbors
                )
                
                # Scores par patch (distance au plus proche voisin)
                patch_scores = distances[:, 0]
                
                # Reshape vers (batch_size, h, w)
                patch_scores = patch_scores.reshape(batch_size, h, w)
                
                # Score image = max pooling (anomalie la plus forte)
                image_scores = patch_scores.max(axis=(1, 2))
                all_scores.append(image_scores)
                
                if (batch_idx + 1) % 10 == 0:
                    logger.debug(f"Predictions: batch {batch_idx+1}")
        
        scores = np.concatenate(all_scores)
        
        logger.info(
            f"✅ Prédictions terminées - "
            f"n_samples: {len(scores)}, "
            f"score_mean: {scores.mean():.4f}, "
            f"score_std: {scores.std():.4f}"
        )
        
        return scores
    
    def get_anomaly_map(self, image: torch.Tensor) -> np.ndarray:
        """
        Génère une heatmap d'anomalie spatiale pour une image.
        
        Args:
            image: Tensor (1, C, H, W) ou (C, H, W)
            
        Returns:
            Heatmap normalisée (H, W) avec valeurs dans [0, 1]
        """
        if not self._is_fitted:
            raise ValueError("Modèle non entraîné")
        
        # Gestion dimension batch
        if image.ndim == 3:
            image = image.unsqueeze(0)
        
        self.eval()
        with torch.no_grad():
            # Adaptation canaux
            image = self._adapt_channels(image)
            
            # Extraction features
            features = self.feature_extractor(image)
            _, num_features, h, w = features.shape
            
            # Reshape
            features = features.permute(0, 2, 3, 1).reshape(-1, num_features)
            features_np = features.cpu().numpy()
            
            # Normalisation
            norms = np.linalg.norm(features_np, axis=1, keepdims=True)
            features_np = features_np / np.maximum(norms, 1e-8)
            
            # Distances
            distances, _ = self.faiss_index.search(
                features_np.astype(np.float32),
                self.num_neighbors
            )
            
            # Heatmap
            anomaly_map = distances[:, 0].reshape(h, w)
            
            # Normalisation [0, 1]
            if anomaly_map.max() > anomaly_map.min():
                anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
            
            return anomaly_map


class FeatureExtractor(nn.Module):
    """
    Extracteur de features multi-échelles depuis un backbone CNN.
    
    Utilise des hooks PyTorch pour capturer les activations de plusieurs couches
    et les agréger en un vecteur de features unifié.
    """
    
    def __init__(self, backbone: nn.Module, layers: List[str]):
        """
        Initialise l'extracteur.
        
        Args:
            backbone: Modèle CNN pré-entraîné
            layers: Noms des couches à extraire (ex: ['layer2', 'layer3'])
        """
        super().__init__()
        self.backbone = backbone
        self.layers = layers
        self.features = {}
        
        # Enregistrement des hooks sur les couches cibles
        layer_dict = dict([*self.backbone.named_modules()])
        
        for layer_name in layers:
            if layer_name not in layer_dict:
                raise ValueError(
                    f"Couche '{layer_name}' introuvable dans le backbone. "
                    f"Couches disponibles: {list(layer_dict.keys())}"
                )
            
            layer = layer_dict[layer_name]
            layer.register_forward_hook(self._get_hook(layer_name))
        
        logger.info(f"FeatureExtractor initialisé avec couches: {layers}")
    
    def _get_hook(self, layer_name: str):
        """
        Crée un hook pour capturer les activations d'une couche.
        
        Args:
            layer_name: Nom de la couche
            
        Returns:
            Fonction hook
        """
        def hook(module, input, output):
            self.features[layer_name] = output
        return hook
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extrait et agrège les features multi-échelles.
        
        Args:
            x: Tensor d'entrée (B, C, H, W)
            
        Returns:
            Features agrégées (B, C_total, H_uniform, W_uniform)
        """
        self.features.clear()
        
        # Forward pass pour déclencher les hooks
        _ = self.backbone(x)
        
        # Agrégation des features multi-échelles
        feature_maps = []
        
        for layer_name in self.layers:
            feat = self.features[layer_name]
            
            # Adaptive pooling pour uniformiser les tailles spatiales
            # Toutes les feature maps → (14, 14)
            feat_pooled = F.adaptive_avg_pool2d(feat, (14, 14))
            feature_maps.append(feat_pooled)
        
        # Concatenation sur la dimension des canaux
        aggregated = torch.cat(feature_maps, dim=1)
        
        return aggregated
    
    def get_output_channels(self) -> int:
        """
        Retourne le nombre total de canaux en sortie.
        
        Returns:
            Nombre de canaux agrégés
        """
        # Nécessite un forward pass pour connaître les dimensions
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = self.forward(dummy_input)
        return output.shape[1]