"""
AutoEncoders pour la détection d'anomalies.
Modifications majeures:
- Resize automatique des images (auto_resize=True par défaut)
- Gestion robuste des dimensions
- Logging détaillé
- Compatible avec tous pipelines d'entraînement

À placer dans: src/models/computer_vision/anomaly_detection/autoencoders.py
"""

import numpy as np
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
from typing import Tuple, Optional, Dict, Any, List
from src.shared.logging import get_logger

logger = get_logger(__name__)


class ConvAutoEncoder(nn.Module):
    """
    AutoEncoder Convolutionnel avec RESIZE AUTOMATIQUE.  
    Resize dynamique des images si taille incorrecte  
    Architecture:
        Encoder: Conv2D → BatchNorm → ReLU → MaxPool (xN stages)
        Latent: Espace latent compressé (calcul dynamique)
        Decoder: ConvTranspose2D → BatchNorm → ReLU (xN stages) 
    Features:
        - Resize automatique si taille d'entrée incorrecte
        - Calcul dynamique des dimensions basé sur input_size
        - Skip connections optionnelles (U-Net style)
        - Variational mode (VAE) optionnel
        - Validation robuste des shapes
        - Logging complet pour debugging
    
    Args:
        input_channels: Nombre de canaux d'entrée (1=grayscale, 3=RGB, 4=RGBA)
        latent_dim: Dimension de l'espace latent (bottleneck)
        base_filters: Nombre de filtres de base (doublés à chaque stage)
        num_stages: Nombre de stages de convolution (2-5 recommandé)
        dropout_rate: Taux de dropout pour les couches fully connected
        use_skip_connections: Activer les connexions résiduelles (style U-Net)
        use_vae: Mode Variational AutoEncoder
        input_size: Taille d'entrée des images (hauteur, largeur)
        auto_resize: ✅ NOUVEAU - Resize automatique si taille incorrecte (défaut: True)
    
    Example:
        >>> # Images 256x256 redimensionnées automatiquement en 128x128
        >>> model = ConvAutoEncoder(input_size=(128, 128), latent_dim=128, auto_resize=True)
        >>> x = torch.randn(8, 3, 256, 256)  # Taille différente
        >>> reconstructed = model(x)  # Resize automatique → (8, 3, 128, 128)
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        latent_dim: int = 256,
        base_filters: int = 32,
        num_stages: int = 4,
        dropout_rate: float = 0.2,
        use_skip_connections: bool = False,
        use_vae: bool = False,
        input_size: Tuple[int, int] = (256, 256),
        auto_resize: bool = True 
    ):
        super(ConvAutoEncoder, self).__init__()
        
        # === VALIDATION DES PARAMÈTRES ===
        if input_channels not in [1, 3, 4]:
            raise ValueError(f"input_channels doit être 1, 3 ou 4, reçu: {input_channels}")
        
        if latent_dim < 16:
            raise ValueError(f"latent_dim trop petit: {latent_dim} (min: 16)")
        
        if num_stages < 2 or num_stages > 5:
            raise ValueError(f"num_stages doit être entre 2 et 5, reçu: {num_stages}")
        
        if input_size[0] < 32 or input_size[1] < 32:
            raise ValueError(f"Taille d'entrée trop petite: {input_size} (min: 32x32)")
        
        # Stockage des paramètres
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.base_filters = base_filters
        self.num_stages = num_stages
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.use_vae = use_vae
        self.input_size = input_size
        self.auto_resize = auto_resize 
        
        # === CONSTRUCTION DE L'ENCODEUR ===
        self.encoder_blocks = nn.ModuleList()
        in_channels = input_channels
        
        current_size = input_size
        self.encoder_sizes = [current_size]
        
        for stage in range(num_stages):
            out_channels = base_filters * (2 ** stage)
            
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            
            self.encoder_blocks.append(block)
            in_channels = out_channels
            
            current_size = (current_size[0] // 2, current_size[1] // 2)
            self.encoder_sizes.append(current_size)
            
            if current_size[0] < 4 or current_size[1] < 4:
                raise ValueError(
                    f"Taille trop petite après {stage+1} stages: {current_size}. "
                    f"Réduisez num_stages ou augmentez input_size."
                )
        
        # === CALCUL DES DIMENSIONS ENCODÉES ===
        self.encoded_channels = base_filters * (2 ** (num_stages - 1))
        self.encoded_size = self.encoder_sizes[-1]
        self.flat_features = self.encoded_channels * self.encoded_size[0] * self.encoded_size[1]
        
        logger.info(
            f"Calcul dynamique - Taille encodée: {self.encoded_channels} canaux "
            f"x {self.encoded_size[0]}x{self.encoded_size[1]} = {self.flat_features} features"
        )
        
        # === ESPACE LATENT ===
        if use_vae:
            self.fc_mu = nn.Linear(self.flat_features, latent_dim)
            self.fc_logvar = nn.Linear(self.flat_features, latent_dim)
            self.fc_decode = nn.Linear(latent_dim, self.flat_features)
        else:
            self.fc_encode = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.flat_features, latent_dim),
                nn.Dropout(dropout_rate),
                nn.ReLU(inplace=True)
            )
            
            self.fc_decode = nn.Sequential(
                nn.Linear(latent_dim, self.flat_features),
                nn.ReLU(inplace=True)
            )
        
        # === CONSTRUCTION DU DÉCODEUR ===
        self.decoder_blocks = nn.ModuleList()
        
        for stage in reversed(range(num_stages)):
            in_channels = base_filters * (2 ** stage)
            out_channels = base_filters * (2 ** (stage - 1)) if stage > 0 else input_channels
            
            if use_skip_connections and stage < num_stages - 1:
                decoder_in_channels = in_channels * 2
            else:
                decoder_in_channels = in_channels
            
            block = nn.Sequential(
                nn.ConvTranspose2d(
                    decoder_in_channels if stage < num_stages - 1 else in_channels,
                    in_channels,
                    kernel_size=2,
                    stride=2,
                    bias=False
                ),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels) if stage > 0 else nn.Identity(),
                nn.ReLU(inplace=True) if stage > 0 else nn.Sigmoid()
            )
            
            self.decoder_blocks.append(block)
        
        # === INITIALISATION DES POIDS ===
        self._initialize_weights()
        
        logger.info(
            f"ConvAutoEncoder initialisé: "
            f"input_size={input_size}, "
            f"latent_dim={latent_dim}, "
            f"auto_resize={auto_resize}, "
            f"params={self.count_parameters():,}"
        )
    
    def _initialize_weights(self):
        """Initialisation des poids pour convergence optimale."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _resize_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Resize automatique des images si nécessaire.      
        Args:
            x: Tensor (B, C, H, W)           
        Returns:
            Tensor redimensionné à input_size (B, C, target_H, target_W)            
        Raises:
            ValueError: Si auto_resize=False et taille incorrecte
        """
        _, _, h, w = x.shape
        target_h, target_w = self.input_size
        
        # Pas besoin de resize si taille correcte
        if h == target_h and w == target_w:
            return x
        
        # Resize automatique si activé
        if self.auto_resize:
            logger.debug(
                f"🔧 Resize automatique: ({h}, {w}) → ({target_h}, {target_w})"
            )
            x_resized = F.interpolate(
                x,
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            )
            return x_resized
        else:
            # Mode strict: lever une erreur
            raise ValueError(
                f"Taille d'entrée incorrecte: ({h}, {w}) vs attendu ({target_h}, {target_w}). "
                f"Activez auto_resize=True pour resize automatique."
            )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Passe forward à travers l'encodeur.      
        Args:
            x: Tensor d'entrée (B, C, H, W) - doit avoir la bonne taille           
        Returns:
            z: Représentation latente (B, latent_dim)
            skip_features: Liste des features intermédiaires
        """
        skip_features = []
        
        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            if self.use_skip_connections and i < len(self.encoder_blocks) - 1:
                skip_features.append(x)
        
        # Flatten
        batch_size = x.size(0)
        x_flat = x.reshape(batch_size, -1)
        
        # Validation (ne devrait jamais échouer avec auto_resize)
        expected_features = self.flat_features
        actual_features = x_flat.size(1)
        
        if actual_features != expected_features:
            raise ValueError(
                f"Erreur interne: features={actual_features} vs attendu={expected_features}. "
                f"Cela ne devrait jamais arriver avec auto_resize=True. "
                f"Shape x avant flatten: {x.shape}"
            )
        
        # Passage vers l'espace latent
        if self.use_vae:
            mu = self.fc_mu(x_flat)
            logvar = self.fc_logvar(x_flat)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            self.mu = mu
            self.logvar = logvar
            return z, skip_features
        else:
            z = self.fc_encode(x_flat)
            return z, skip_features
    
    def decode(self, z: torch.Tensor, skip_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Passe forward à travers le décodeur.       
        Args:
            z: Représentation latente (B, latent_dim)
            skip_features: Features intermédiaires de l'encodeur           
        Returns:
            Image reconstruite (B, C, H, W)
        """
        if self.use_vae:
            x = self.fc_decode(z)
        else:
            x = self.fc_decode(z)
        
        x = x.reshape(-1, self.encoded_channels, self.encoded_size[0], self.encoded_size[1])
        
        for i, block in enumerate(self.decoder_blocks):
            if self.use_skip_connections and i < len(skip_features):
                skip_idx = len(skip_features) - 1 - i
                x = torch.cat([x, skip_features[skip_idx]], dim=1)
            
            x = block(x)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass complet avec RESIZE AUTOMATIQUE.       
        Les images sont redimensionnées si nécessaire       
        Args:
            x: Images d'entrée (B, C, H, W) - toute taille acceptée si auto_resize=True           
        Returns:
            Images reconstruites (B, C, input_size[0], input_size[1])            
        Raises:
            ValueError: Si dimensions invalides
        """
        # Validation basique
        if x.dim() != 4:
            raise ValueError(f"Attendu tensor 4D (B,C,H,W), reçu: {x.dim()}D")
        
        if x.size(1) != self.input_channels:
            raise ValueError(
                f"Attendu {self.input_channels} canaux, reçu: {x.size(1)} canaux"
            )
        
        # RESIZE AUTOMATIQUE si nécessaire
        x = self._resize_input(x)
        
        # Encodage et décodage
        try:
            z, skip_features = self.encode(x)
            reconstructed = self.decode(z, skip_features)
            return reconstructed
        
        except Exception as e:
            logger.error(f"Échec forward pass: {e}")
            logger.error(f"Shape après resize: {x.shape}")
            logger.error(f"Config modèle: {self.summary()}")
            raise
    
    def compute_anomaly_scores(
        self,
        x: torch.Tensor,
        method: str = "mse"
    ) -> torch.Tensor:
        """
        Calcule les scores d'anomalie basés sur l'erreur de reconstruction.        
        Args:
            x: Images originales (toute taille si auto_resize=True)
            method: Méthode de calcul ("mse", "mae", "ssim", "combined")            
        Returns:
            Scores d'anomalie (B,) - Plus élevé = plus anormal
        """
        self.eval()
        
        with torch.no_grad():
            reconstructed = self.forward(x)
            
            # Important: resize l'original pour match la reconstruction
            if x.shape != reconstructed.shape:
                x = F.interpolate(
                    x,
                    size=(reconstructed.size(2), reconstructed.size(3)),
                    mode='bilinear',
                    align_corners=False
                )
            
            if method == "mse":
                scores = torch.mean((x - reconstructed) ** 2, dim=(1, 2, 3))
            
            elif method == "mae":
                scores = torch.mean(torch.abs(x - reconstructed), dim=(1, 2, 3))
            
            elif method == "ssim":
                try:
                    from pytorch_msssim import ssim # type: ignore
                    ssim_val = ssim(x, reconstructed, data_range=1.0, size_average=False)
                    scores = 1 - ssim_val
                except ImportError:
                    logger.warning("pytorch_msssim non installé, utilisation de MSE")
                    scores = torch.mean((x - reconstructed) ** 2, dim=(1, 2, 3))
            
            elif method == "combined":
                mse_score = torch.mean((x - reconstructed) ** 2, dim=(1, 2, 3))
                grad_x = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
                grad_recon = torch.abs(reconstructed[:, :, :-1, :] - reconstructed[:, :, 1:, :])
                perceptual_score = torch.mean((grad_x - grad_recon) ** 2, dim=(1, 2, 3))
                scores = 0.7 * mse_score + 0.3 * perceptual_score
            
            else:
                raise ValueError(f"Méthode '{method}' non supportée")
            
            return scores
    
    def get_reconstruction_error_map(
        self,
        x: torch.Tensor,
        method: str = "mse"
    ) -> torch.Tensor:
        """
        Génère une carte spatiale des erreurs de reconstruction.        
        Args:
            x: Images originales (B, C, H, W)
            method: Méthode de calcul ("mse", "mae", "l1_l2")           
        Returns:
            Carte d'erreur (B, 1, H', W') où H', W' = input_size
        """
        self.eval()
        
        with torch.no_grad():
            reconstructed = self.forward(x)
            
            # Resize original pour match
            if x.shape != reconstructed.shape:
                x = F.interpolate(
                    x,
                    size=(reconstructed.size(2), reconstructed.size(3)),
                    mode='bilinear',
                    align_corners=False
                )
            
            if method == "mse":
                error_map = torch.mean((x - reconstructed) ** 2, dim=1, keepdim=True)
            elif method == "mae":
                error_map = torch.mean(torch.abs(x - reconstructed), dim=1, keepdim=True)
            elif method == "l1_l2":
                l2 = torch.mean((x - reconstructed) ** 2, dim=1, keepdim=True)
                l1 = torch.mean(torch.abs(x - reconstructed), dim=1, keepdim=True)
                error_map = 0.7 * l2 + 0.3 * l1
            else:
                logger.warning(f"⚠️ Méthode '{method}' non reconnue, utilisation MSE")
                error_map = torch.mean((x - reconstructed) ** 2, dim=1, keepdim=True)
            
            if torch.isnan(error_map).any() or torch.isinf(error_map).any():
                logger.warning("⚠️ NaN/Inf détectés dans error_map")
                error_map = torch.nan_to_num(error_map, nan=0.0, posinf=0.0, neginf=0.0)
            
            return error_map
    
    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Prédit les scores d'anomalie pour un batch numpy.
        Compatible avec l'API scikit-learn.        
        Args:
            X: Images (N, C, H, W) en numpy
            batch_size: Taille des batchs            
        Returns:
            Scores d'anomalie (N,)
        """
        self.eval()
        device = next(self.parameters()).device
        
        if X is None or len(X) == 0:
            logger.error("Données d'entrée vides")
            return np.array([])
        
        all_scores = []
        
        try:
            for i in range(0, len(X), batch_size):
                batch = X[i:i+batch_size]
                batch_tensor = torch.tensor(batch, dtype=torch.float32).to(device)
                scores = self.compute_anomaly_scores(batch_tensor, method="mse")
                all_scores.append(scores.cpu().numpy())
            
            return np.concatenate(all_scores) if all_scores else np.array([])
        
        except Exception as e:
            logger.error(f"Échec prédiction: {e}")
            return np.zeros(len(X)) if X is not None else np.array([])
    
    def count_parameters(self, trainable_only: bool = False) -> int:
        """Compte les paramètres du modèle."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
    
    def summary(self) -> Dict[str, Any]:
        """Résumé complet du modèle."""
        return {
            "model_type": "ConvAutoEncoder",
            "input_channels": self.input_channels,
            "input_size": self.input_size,
            "latent_dim": self.latent_dim,
            "encoded_size": self.encoded_size,
            "flat_features": self.flat_features,
            "num_stages": self.num_stages,
            "base_filters": self.base_filters,
            "skip_connections": self.use_skip_connections,
            "variational": self.use_vae,
            "auto_resize": self.auto_resize, 
            "total_parameters": self.count_parameters(),
            "trainable_parameters": self.count_parameters(trainable_only=True)
        }


# === VARIATIONAL AUTOENCODER ===
class VariationalAutoEncoder(ConvAutoEncoder):
    """
    Variational AutoEncoder (VAE) avec resize automatique.
    """
    
    def __init__(self, **kwargs):
        kwargs['use_vae'] = True
        super().__init__(**kwargs)
        
        self.kl_weight = 0.001
        logger.info("VariationalAutoEncoder initialisé avec auto_resize")
    
    def compute_vae_loss(
        self,
        x: torch.Tensor,
        reconstructed: torch.Tensor,
        kl_weight: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calcule la loss VAE = Reconstruction Loss + KL Divergence.
        
        Args:
            x: Images originales
            reconstructed: Images reconstruites
            kl_weight: Poids du terme KL
            
        Returns:
            (total_loss, reconstruction_loss, kl_loss)
        """
        if kl_weight is None:
            kl_weight = self.kl_weight
        
        # Resize x si nécessaire
        if x.shape != reconstructed.shape:
            x = F.interpolate(
                x,
                size=(reconstructed.size(2), reconstructed.size(3)),
                mode='bilinear',
                align_corners=False
            )
        
        recon_loss = F.mse_loss(reconstructed, x, reduction='mean')
        
        kl_loss = -0.5 * torch.sum(
            1 + self.logvar - self.mu.pow(2) - self.logvar.exp()
        )
        kl_loss /= x.size(0) * x.size(1) * x.size(2) * x.size(3)
        
        total_loss = recon_loss + kl_weight * kl_loss
        
        return total_loss, recon_loss, kl_loss


# === DENOISING AUTOENCODER ===

class DenoisingAutoEncoder(ConvAutoEncoder):
    """
    Denoising AutoEncoder avec resize automatique.
    """
    
    def __init__(self, noise_factor: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        
        if not 0 <= noise_factor <= 1:
            raise ValueError(f"noise_factor doit être entre 0 et 1")
        
        self.noise_factor = noise_factor
        logger.info(f"DenoisingAutoEncoder initialisé: noise_factor={noise_factor}")
    
    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Ajoute du bruit gaussien."""
        noise = torch.randn_like(x) * self.noise_factor
        noisy_x = torch.clamp(x + noise, 0.0, 1.0)
        return noisy_x
    
    def forward(self, x: torch.Tensor, add_noise_training: bool = True) -> torch.Tensor:
        """Forward avec ajout optionnel de bruit."""
        if self.training and add_noise_training:
            x_input = self.add_noise(x)
        else:
            x_input = x
        
        return super().forward(x_input)


# === FACTORY FUNCTION ===

def get_autoencoder(
    model_type: str = "conv",
    input_channels: int = 3,
    latent_dim: int = 256,
    input_size: Tuple[int, int] = (256, 256),
    auto_resize: bool = True, 
    **kwargs
) -> nn.Module:
    """
    Factory pour créer des AutoEncoders avec resize automatique.   
    Args:
        model_type: Type ("conv", "vae", "denoising")
        input_channels: Nombre de canaux
        latent_dim: Dimension espace latent
        input_size: Taille cible des images
        auto_resize: Resize automatique si taille incorrecte
        **kwargs: Arguments additionnels       
    Returns:
        Instance d'AutoEncoder configurée
    """
    models_registry = {
        "conv": ConvAutoEncoder,
        "vae": VariationalAutoEncoder,
        "denoising": DenoisingAutoEncoder
    }
    
    if model_type not in models_registry:
        raise ValueError(
            f"model_type '{model_type}' non supporté. "
            f"Options: {list(models_registry.keys())}"
        )
    
    model_class = models_registry[model_type]
    
    try:
        model = model_class(
            input_channels=input_channels,
            latent_dim=latent_dim,
            input_size=input_size,
            auto_resize=auto_resize,  # ✅ NOUVEAU
            **kwargs
        )
        
        logger.info(
            f"AutoEncoder {model_type} créé - "
            f"Taille: {input_size}, "
            f"Auto-resize: {auto_resize}"
        )
        return model
    
    except Exception as e:
        logger.error(f"Erreur création {model_type}: {e}", exc_info=True)
        raise
