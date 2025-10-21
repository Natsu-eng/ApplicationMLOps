"""
Implémentation d'un réseau siamois pour la détection d'anomalies.
"""
import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class SiameseNetwork(nn.Module):
    """
    Réseau siamois pour comparer paires d'images et détecter anomalies.
    """
    def __init__(self, input_shape: tuple = (3, 224, 224), embedding_dim: int = 128, margin: float = 1.0):
        super(SiameseNetwork, self).__init__()
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        self.margin = margin
        
        # Réseau de feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * (input_shape[1] // 4) * (input_shape[2] // 4), embedding_dim),
            nn.ReLU()
        )
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor = None) -> torch.Tensor:
        try:
            emb1 = self.feature_extractor(x1)
            if x2 is not None:
                emb2 = self.feature_extractor(x2)
                return torch.norm(emb1 - emb2, dim=1)
            return emb1
        except Exception as e:
            logger.error(f"Erreur dans forward SiameseNetwork: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit les scores d'anomalie (distance à une référence normale).
        
        Args:
            X (np.ndarray): Images d'entrée (N, C, H, W).
        
        Returns:
            np.ndarray: Scores d'anomalie (distance moyenne).
        """
        try:
            self.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(device)
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            
            # Utiliser une image normale moyenne comme référence (simplifié)
            ref_tensor = X_tensor.mean(dim=0, keepdim=True)
            
            with torch.no_grad():
                distances = self(X_tensor, ref_tensor.repeat(len(X), 1, 1, 1)).cpu().numpy()
            return distances
        except Exception as e:
            logger.error(f"Erreur dans predict SiameseNetwork: {e}")
            return np.zeros(len(X))