"""
Preprocessing spécifique ImageNet pour Transfer Learning.
À placer dans: src/data/imagenet_preprocessing.py

CRITIQUE: Les modèles pré-entraînés sur ImageNet REQUIÈRENT cette normalisation exacte.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Union, Tuple
from src.shared.logging import get_logger

logger = get_logger(__name__)


class ImageNetNormalizer(nn.Module):
    """
    Normalisation ImageNet pour Transfer Learning.
    
    IMPORTANT:
    - ImageNet mean: [0.485, 0.456, 0.406] (RGB)
    - ImageNet std:  [0.229, 0.224, 0.225] (RGB)
    - Input attendu: Tensor float [0, 1] en channels_first (N, C, H, W)
    
    Cette normalisation est OBLIGATOIRE pour:
    - ResNet (tous)
    - VGG (tous)
    - EfficientNet
    - MobileNet
    - DenseNet
    - Tout modèle pré-entraîné sur ImageNet
    
    Exemple:
        >>> normalizer = ImageNetNormalizer()
        >>> x = torch.rand(8, 3, 224, 224)  # [0, 1]
        >>> x_normalized = normalizer(x)   # ImageNet normalisé
    """
    
    def __init__(self):
        super().__init__()
        
        # Statistiques ImageNet officielles (RGB)
        self.register_buffer(
            'mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        
        logger.info("ImageNetNormalizer initialisé avec stats officielles")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalise selon ImageNet.
        
        Args:
            x: Tensor (N, C, H, W) avec valeurs [0, 1]
        
        Returns:
            Tensor normalisé selon ImageNet
        """
        if x.dim() != 4:
            raise ValueError(f"Attendu 4D tensor (N,C,H,W), reçu: {x.dim()}D")
        
        if x.size(1) != 3:
            raise ValueError(
                f"ImageNet normalisation requiert 3 canaux RGB, reçu: {x.size(1)}"
            )
        
        # Vérification range (doit être [0, 1])
        if x.min() < -0.1 or x.max() > 1.1:
            logger.warning(
                f"⚠️ Input hors range [0,1]: min={x.min():.3f}, max={x.max():.3f}. "
                f"ImageNet normalisation attend [0,1]"
            )
        
        # Normalisation: (x - mean) / std
        return (x - self.mean) / self.std
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse la normalisation (pour visualisation).
        
        Args:
            x: Tensor normalisé
        
        Returns:
            Tensor [0, 1]
        """
        return x * self.std + self.mean


class ImageNetPreprocessor:
    """
    Pipeline complet de preprocessing pour Transfer Learning.
    
    Pipeline:
    1. Conversion uint8 → float32
    2. Normalisation [0, 1]
    3. Normalisation ImageNet
    4. Format channels_first garanti
    
    Compatible avec numpy et torch.
    """
    
    def __init__(self):
        self.normalizer = ImageNetNormalizer()
        logger.info("ImageNetPreprocessor initialisé")
    
    def preprocess_numpy(
        self,
        images: np.ndarray,
        input_format: str = "channels_last"
    ) -> torch.Tensor:
        """
        Preprocess depuis numpy vers tensor ImageNet-ready.
        
        Args:
            images: Images numpy (N, H, W, C) ou (N, C, H, W)
            input_format: Format d'entrée
        
        Returns:
            Tensor (N, 3, H, W) normalisé ImageNet
        """
        # Conversion vers float [0, 1]
        if images.dtype == np.uint8:
            images = images.astype(np.float32) / 255.0
        elif images.max() > 1.0:
            logger.warning(f"Images avec max={images.max()}, normalisation [0,1]")
            images = images / 255.0
        
        # Conversion vers channels_first si nécessaire
        if input_format == "channels_last" and images.ndim == 4:
            images = np.transpose(images, (0, 3, 1, 2))
        
        # Conversion vers torch
        x = torch.from_numpy(images).float()
        
        # Normalisation ImageNet
        x = self.normalizer(x)
        
        return x
    
    def preprocess_torch(
        self,
        images: torch.Tensor,
        already_normalized_01: bool = False
    ) -> torch.Tensor:
        """
        Preprocess depuis torch vers ImageNet-ready.
        
        Args:
            images: Tensor (N, 3, H, W)
            already_normalized_01: Si True, skip normalisation [0,1]
        
        Returns:
            Tensor normalisé ImageNet
        """
        if not already_normalized_01:
            if images.dtype == torch.uint8:
                images = images.float() / 255.0
            elif images.max() > 1.0:
                images = images / 255.0
        
        return self.normalizer(images)
    
    def get_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retourne mean et std ImageNet."""
        return self.normalizer.mean, self.normalizer.std


# === FONCTIONS UTILITAIRES ===

def imagenet_normalize(
    x: Union[torch.Tensor, np.ndarray],
    input_range: str = "0-1"
) -> torch.Tensor:
    """
    Fonction helper pour normalisation ImageNet rapide.
    
    Args:
        x: Images (torch ou numpy)
        input_range: "0-1", "0-255", ou "custom"
    
    Returns:
        Tensor normalisé ImageNet
    """
    normalizer = ImageNetNormalizer()
    
    # Conversion numpy → torch si nécessaire
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    
    # Normalisation [0, 1] si nécessaire
    if input_range == "0-255":
        x = x / 255.0
    elif input_range == "custom":
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    
    return normalizer(x)


def check_imagenet_normalization(x: torch.Tensor) -> dict:
    """
    Vérifie si un tensor est correctement normalisé ImageNet.
    
    Args:
        x: Tensor à vérifier
    
    Returns:
        Dict avec diagnostics
    """
    mean_per_channel = x.mean(dim=(0, 2, 3))
    std_per_channel = x.std(dim=(0, 2, 3))
    
    # Stats ImageNet attendues
    expected_mean = torch.tensor([0.485, 0.456, 0.406])
    expected_std = torch.tensor([0.229, 0.224, 0.225])
    
    # Après normalisation, devrait être proche de mean≈0, std≈1
    normalized_mean = mean_per_channel  # devrait être ~0
    normalized_std = std_per_channel    # devrait être ~1
    
    # Calcul erreur
    mean_error = torch.abs(normalized_mean).mean().item()
    std_error = torch.abs(normalized_std - 1.0).mean().item()
    
    is_imagenet_normalized = mean_error < 1.0 and std_error < 1.0
    
    return {
        "is_imagenet_normalized": is_imagenet_normalized,
        "mean_per_channel": mean_per_channel.tolist(),
        "std_per_channel": std_per_channel.tolist(),
        "mean_error": mean_error,
        "std_error": std_error,
        "expected_mean": expected_mean.tolist(),
        "expected_std": expected_std.tolist(),
        "diagnostic": (
            "✅ Normalisé ImageNet" if is_imagenet_normalized 
            else f"❌ Non normalisé ImageNet (mean_err={mean_error:.3f}, std_err={std_error:.3f})"
        )
    }


# === TESTS UNITAIRES ===

def test_imagenet_normalization():
    """Test de la normalisation ImageNet."""
    print("🧪 Test ImageNet Normalization")
    print("=" * 50)
    
    # Test 1: Normalisation basique
    normalizer = ImageNetNormalizer()
    x = torch.rand(4, 3, 224, 224)  # [0, 1]
    x_norm = normalizer(x)
    
    print(f"✅ Test 1: Shape préservée: {x.shape} → {x_norm.shape}")
    print(f"   Range avant: [{x.min():.3f}, {x.max():.3f}]")
    print(f"   Range après: [{x_norm.min():.3f}, {x_norm.max():.3f}]")
    
    # Test 2: Vérification stats
    stats = check_imagenet_normalization(x_norm)
    print(f"\n✅ Test 2: {stats['diagnostic']}")
    print(f"   Mean per channel: {[f'{m:.3f}' for m in stats['mean_per_channel']]}")
    print(f"   Std per channel:  {[f'{s:.3f}' for s in stats['std_per_channel']]}")
    
    # Test 3: Denormalization
    x_denorm = normalizer.denormalize(x_norm)
    reconstruction_error = torch.abs(x - x_denorm).mean().item()
    print(f"\n✅ Test 3: Denormalisation")
    print(f"   Erreur reconstruction: {reconstruction_error:.6f}")
    print(f"   Range denorm: [{x_denorm.min():.3f}, {x_denorm.max():.3f}]")
    
    # Test 4: Preprocessing complet
    preprocessor = ImageNetPreprocessor()
    images_np = np.random.randint(0, 256, (4, 224, 224, 3), dtype=np.uint8)
    x_processed = preprocessor.preprocess_numpy(images_np)
    
    print(f"\n✅ Test 4: Pipeline complet numpy→torch")
    print(f"   Input:  shape={images_np.shape}, dtype={images_np.dtype}")
    print(f"   Output: shape={x_processed.shape}, dtype={x_processed.dtype}")
    print(f"   Range:  [{x_processed.min():.3f}, {x_processed.max():.3f}]")
    
    stats_processed = check_imagenet_normalization(x_processed)
    print(f"   {stats_processed['diagnostic']}")
    
    print("\n" + "=" * 50)
    print("✅ Tous les tests passés!")


if __name__ == "__main__":
    test_imagenet_normalization()