"""
Implémentation de PatchCore pour la détection d'anomalies.
Basé sur l'extraction de patchs et la détection des anomalies locales.
"""
import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class PatchCore(nn.Module):
    """
    Modèle PatchCore pour détection d'anomalies basée sur patchs.
    """
    def __init__(self, input_shape: tuple = (3, 224, 224), patch_size: int = 32, feature_dim: int = 256):
        super(PatchCore, self).__init__()
        self.input_shape = input_shape
        self.patch_size = patch_size
        
        # Feature extractor simplifié (à remplacer par ResNet pré-entraîné si besoin)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, feature_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        # Pooling pour patchs
        self.pool = nn.AdaptiveAvgPool2d((patch_size, patch_size))
        self.memory_bank = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            features = self.feature_extractor(x)
            patches = self.pool(features)
            return patches
        except Exception as e:
            logger.error(f"Erreur dans forward PatchCore: {e}")
            raise
    
    def fit(self, X: np.ndarray):
        """
        Construit la banque de mémoire des patchs normaux.
        
        Args:
            X (np.ndarray): Images normales pour entraînement (N, C, H, W).
        """
        try:
            self.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(device)
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            with torch.no_grad():
                features = self.feature_extractor(X_tensor)
                patches = self.pool(features).view(len(X), -1, self.patch_size * self.patch_size)
                self.memory_bank = patches.mean(dim=0).cpu().numpy()
            logger.info("Banque de mémoire PatchCore construite")
        except Exception as e:
            logger.error(f"Erreur dans fit PatchCore: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit les scores d'anomalie basés sur la distance aux patchs normaux.
        
        Args:
            X (np.ndarray): Images d'entrée (N, C, H, W).
        
        Returns:
            np.ndarray: Scores d'anomalie (distance max par image).
        """
        try:
            if self.memory_bank is None:
                raise ValueError("Banque de mémoire non initialisée. Appelez fit() d'abord.")
            
            self.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(device)
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            memory_bank = torch.tensor(self.memory_bank, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                features = self.feature_extractor(X_tensor)
                patches = self.pool(features).view(len(X), -1, self.patch_size * self.patch_size)
                distances = torch.cdist(patches, memory_bank.unsqueeze(0)).min(dim=2)[0]
                scores = distances.max(dim=1)[0].cpu().numpy()
            return scores
        except Exception as e:
            logger.error(f"Erreur dans predict PatchCore: {e}")
            return np.zeros(len(X))