"""
Réseau siamois professionnel avec contrastive loss et weight tying.
"""
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
import torchvision.models as models # type: ignore
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ProfessionalSiameseNetwork(nn.Module):
    """
    Réseau siamois complet avec contrastive learning.
    """
    
    def __init__(
        self,
        backbone_name: str = "resnet18",
        embedding_dim: int = 128,
        margin: float = 1.0
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.margin = margin
        
        # Backbone partagé avec weight tying
        self.backbone = self._get_backbone(backbone_name)
        self.embedder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True)
        )
        
    def _get_backbone(self, name: str) -> nn.Module:
        """Backbone pré-entraîné avec feature extraction."""
        if name == "resnet18":
            model = models.resnet18(pretrained=True)
            # Supprimer la dernière couche FC
            model = nn.Sequential(*list(model.children())[:-1])
        else:
            raise ValueError(f"Backbone {name} non supporté")
        
        # Geler les premières couches
        for param in list(model.parameters())[:-4]:  # Dégeler dernières couches
            param.requires_grad = False
            
        return model
    
    def forward(self, x1: torch.Tensor, x2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass pour une ou deux images."""
        embedding1 = self.embedder(self.backbone(x1))
        embedding1 = F.normalize(embedding1, p=2, dim=1)
        
        if x2 is not None:
            embedding2 = self.embedder(self.backbone(x2))
            embedding2 = F.normalize(embedding2, p=2, dim=1)
            return embedding1, embedding2
        
        return embedding1
    
    def contrastive_loss(
        self, 
        embedding1: torch.Tensor, 
        embedding2: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Contrastive loss pour l'apprentissage siamois.
        
        Args:
            embedding1, embedding2: Embeddings des paires d'images
            labels: 1 pour similaire, 0 pour différent
        """
        euclidean_distance = F.pairwise_distance(embedding1, embedding2)
        
        # Contrastive loss
        loss_similar = labels * torch.pow(euclidean_distance, 2)
        loss_dissimilar = (1 - labels) * torch.pow(
            torch.clamp(self.margin - euclidean_distance, min=0.0), 2
        )
        
        return torch.mean(loss_similar + loss_dissimilar)
    
    def predict_anomaly_score(
        self, 
        query_images: torch.Tensor, 
        reference_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcule les scores d'anomalie par rapport à des embeddings de référence.
        
        Args:
            query_images: Images à évaluer
            reference_embeddings: Embeddings des images normales de référence
        """
        self.eval()
        with torch.no_grad():
            query_embeddings = self.forward(query_images)
            distances = torch.cdist(query_embeddings, reference_embeddings)
            anomaly_scores = distances.min(dim=1)[0]  # Distance au plus proche voisin
            
        return anomaly_scores


# Factory function pour une utilisation facile
def get_siamese_network(
    backbone_name: str = "resnet18",
    embedding_dim: int = 128,
    margin: float = 1.0
) -> ProfessionalSiameseNetwork:
    """Factory pour créer un réseau siamois."""
    return ProfessionalSiameseNetwork(
        backbone_name=backbone_name,
        embedding_dim=embedding_dim,
        margin=margin
    )