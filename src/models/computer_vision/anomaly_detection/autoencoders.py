"""
AutoEncoders professionnels pour détection d'anomalies.
Version production-ready avec architecture dynamique et monitoring.

À placer dans: src/models/computer_vision/anomaly_detection/autoencoders.py
"""

import numpy as np
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
from typing import Tuple, Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class ConvAutoEncoder(nn.Module):
    """
    AutoEncoder Convolutionnel avec architecture dynamique.
    S'adapte automatiquement à la taille d'entrée des images.
    
    Architecture:
        Encoder: Conv2D → BatchNorm → ReLU → MaxPool (xN stages)
        Latent: Espace latent compressé (calcul dynamique)
        Decoder: ConvTranspose2D → BatchNorm → ReLU (xN stages)
    
    Features:
        - Calcul dynamique des dimensions basé sur la taille d'entrée
        - Skip connections optionnelles (U-Net style)
        - Variational mode (VAE) optionnel
        - Validation robuste des shapes
        - Logging complet pour le debugging
        - Initialisation optimisée des poids
    
    Args:
        input_channels: Nombre de canaux d'entrée (1=grayscale, 3=RGB, 4=RGBA)
        latent_dim: Dimension de l'espace latent (bottleneck)
        base_filters: Nombre de filtres de base (doublés à chaque stage)
        num_stages: Nombre de stages de convolution (2-5 recommandé)
        dropout_rate: Taux de dropout pour les couches fully connected
        use_skip_connections: Activer les connexions résiduelles (style U-Net)
        use_vae: Mode Variational AutoEncoder
        input_size: Taille d'entrée des images (hauteur, largeur) - CRITIQUE
    
    Example:
        >>> # Pour images 256x256 RGB
        >>> model = ConvAutoEncoder(input_size=(256, 256), latent_dim=128)
        >>> x = torch.randn(8, 3, 256, 256)
        >>> reconstructed = model(x)
        >>> scores = model.compute_anomaly_scores(x)
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
        input_size: Tuple[int, int] = (256, 256)  # NOUVEAU: taille dynamique
    ):
        super(ConvAutoEncoder, self).__init__()
        
        # === VALIDATION DES PARAMÈTRES ===
        if input_channels not in [1, 3, 4]:
            raise ValueError(f"input_channels doit être 1, 3 ou 4, reçu: {input_channels}")
        
        if latent_dim < 16:
            raise ValueError(f"latent_dim trop petit: {latent_dim} (min: 16)")
        
        if num_stages < 2 or num_stages > 5:
            raise ValueError(f"num_stages doit être entre 2 et 5, reçu: {num_stages}")
        
        # Vérification de la taille d'entrée
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
        self.input_size = input_size  # Stockage de la taille d'entrée
        
        # === CONSTRUCTION DE L'ENCODEUR ===
        self.encoder_blocks = nn.ModuleList()
        in_channels = input_channels
        
        # Calcul dynamique de la taille à travers les stages
        current_size = input_size  # Taille courante (H, W)
        self.encoder_sizes = [current_size]  # Historique des tailles pour debug
        
        # Construction des blocs encodeur
        for stage in range(num_stages):
            # Double le nombre de filtres à chaque stage
            out_channels = base_filters * (2 ** stage)
            
            # Bloc encodeur: Conv → BatchNorm → ReLU → Conv → BatchNorm → ReLU → MaxPool
            block = nn.Sequential(
                # Première convolution
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                # Seconde convolution
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                # Réduction de dimension
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            
            self.encoder_blocks.append(block)
            in_channels = out_channels  # Mise à jour pour le prochain bloc
            
            # Mise à jour de la taille après maxpool (division par 2)
            current_size = (current_size[0] // 2, current_size[1] // 2)
            self.encoder_sizes.append(current_size)
            
            # Vérification que la taille reste valide
            if current_size[0] < 4 or current_size[1] < 4:
                raise ValueError(
                    f"Taille trop petite après {stage+1} stages: {current_size}. "
                    f"Réduisez num_stages ou augmentez input_size."
                )
        
        # === CALCUL DES DIMENSIONS ENCODÉES (DYNAMIQUE) ===
        # Nombre de canaux à la sortie de l'encodeur
        self.encoded_channels = base_filters * (2 ** (num_stages - 1))
        # Taille spatiale finale (H, W)
        self.encoded_size = self.encoder_sizes[-1]
        # Nombre total de features après flatten
        self.flat_features = self.encoded_channels * self.encoded_size[0] * self.encoded_size[1]
        
        logger.info(
            f"Calcul dynamique - Taille encodée: {self.encoded_channels} canaux "
            f"x {self.encoded_size[0]}x{self.encoded_size[1]} = {self.flat_features} features"
        )
        
        # === ESPACE LATENT (BOTTLENECK) ===
        if use_vae:
            # Variational AutoEncoder: génère mu et logvar
            self.fc_mu = nn.Linear(self.flat_features, latent_dim)        # Moyenne
            self.fc_logvar = nn.Linear(self.flat_features, latent_dim)    # Variance (log)
            self.fc_decode = nn.Linear(latent_dim, self.flat_features)    # Décodeur latent
        else:
            # AutoEncoder standard
            self.fc_encode = nn.Sequential(
                nn.Flatten(),  # (B, C, H, W) → (B, C*H*W)
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
        
        # Construction en ordre inverse (du plus profond au plus superficiel)
        for stage in reversed(range(num_stages)):
            in_channels = base_filters * (2 ** stage)
            # Détermine les canaux de sortie (retour vers les canaux d'origine)
            out_channels = base_filters * (2 ** (stage - 1)) if stage > 0 else input_channels
            
            # Gestion des skip connections (concaténation des features)
            if use_skip_connections and stage < num_stages - 1:
                decoder_in_channels = in_channels * 2  # Double les canaux d'entrée
            else:
                decoder_in_channels = in_channels
            
            # Bloc décodeur: ConvTranspose → BatchNorm → ReLU → Conv → BatchNorm → Activation
            block = nn.Sequential(
                # Upsampling
                nn.ConvTranspose2d(
                    decoder_in_channels if stage < num_stages - 1 else in_channels,
                    in_channels,
                    kernel_size=2,
                    stride=2,
                    bias=False
                ),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                # Convolution de reconstruction
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                # Pas de BatchNorm sur la dernière couche
                nn.BatchNorm2d(out_channels) if stage > 0 else nn.Identity(),
                # Sigmoid sur la dernière couche pour normaliser les pixels [0,1]
                nn.ReLU(inplace=True) if stage > 0 else nn.Sigmoid()
            )
            
            self.decoder_blocks.append(block)
        
        # === INITIALISATION DES POIDS ===
        self._initialize_weights()
        
        logger.info(
            f"ConvAutoEncoder dynamique initialisé: "
            f"input_size={input_size}, "
            f"latent_dim={latent_dim}, "
            f"encoded_size={self.encoded_size}, "
            f"flat_features={self.flat_features}, "
            f"params={self.count_parameters():,}"
        )
    
    def _initialize_weights(self):
        """
        Initialisation des poids pour une convergence optimale.
        Utilise Kaiming pour les convs et Xavier pour les linear.
        """
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
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Passe forward à travers l'encodeur.     
        Args:
            x: Tensor d'entrée (B, C, H, W)          
        Returns:
            z: Représentation latente (B, latent_dim)
            skip_features: Liste des features intermédiaires pour skip connections          
        Raises:
            ValueError: Si la taille des features ne correspond pas aux attentes
        """
        skip_features = []
        
        # Passage à travers chaque bloc encodeur
        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            # Stocke les features pour les skip connections (sauf le dernier bloc)
            if self.use_skip_connections and i < len(self.encoder_blocks) - 1:
                skip_features.append(x)
        
        # Aplatissement pour les couches fully connected
        batch_size = x.size(0)
        x_flat = x.reshape(batch_size, -1)
        
        # === VALIDATION CRITIQUE DE LA TAILLE ===
        expected_features = self.flat_features
        actual_features = x_flat.size(1)
        
        if actual_features != expected_features:
            raise ValueError(
                f"Taille features inattendue: {actual_features} vs {expected_features}. "
                f"Vérifiez que l'entrée a la taille {self.input_size}. "
                f"Calculé pour {self.num_stages} stages avec input {self.input_size}"
            )
        
        # === PASSAGE VERS L'ESPACE LATENT ===
        if self.use_vae:
            # VAE: génère distribution gaussienne
            mu = self.fc_mu(x_flat)           # Moyenne
            logvar = self.fc_logvar(x_flat)   # Log variance
            
            # Reparametrization trick pour backprop
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            
            # Stockage pour calcul de la loss KL
            self.mu = mu
            self.logvar = logvar
            
            return z, skip_features
        else:
            # AE standard: passage direct
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
        # Reconstruction depuis l'espace latent
        if self.use_vae:
            x = self.fc_decode(z)
        else:
            x = self.fc_decode(z)
        
        # Reshape vers la forme spatiale encodée
        x = x.reshape(-1, self.encoded_channels, self.encoded_size[0], self.encoded_size[1])
        
        # Passage à travers chaque bloc décodeur
        for i, block in enumerate(self.decoder_blocks):
            # Skip connections: concaténation avec les features correspondantes
            if self.use_skip_connections and i < len(skip_features):
                skip_idx = len(skip_features) - 1 - i  # Index inverse
                x = torch.cat([x, skip_features[skip_idx]], dim=1)
            
            x = block(x)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass complet: encode → latent → decode.
        
        Args:
            x: Images d'entrée (B, C, H, W)
            
        Returns:
            Images reconstruites (B, C, H, W)
            
        Raises:
            ValueError: Si les dimensions d'entrée sont incorrectes
        """
        # === VALIDATION DE LA TAILLE D'ENTRÉE ===
        if x.dim() != 4:
            raise ValueError(f"Attendu tensor 4D (B,C,H,W), reçu: {x.dim()}D")
        
        if x.size(1) != self.input_channels:
            raise ValueError(
                f"Attendu {self.input_channels} canaux, "
                f"reçu: {x.size(1)} canaux"
            )
        
        # Extraction de la taille
        _, _, h, w = x.shape
        expected_h, expected_w = self.input_size
        
        # Warning si taille différente de celle configurée
        if h != expected_h or w != expected_w:
            logger.warning(
                f"Taille d'entrée différente de celle configurée: "
                f"({h}, {w}) vs ({expected_h}, {expected_w}). "
                f"Le modèle peut ne pas fonctionner correctement."
            )
        
        try:
            # Encodeur → Espace latent → Décodeur
            z, skip_features = self.encode(x)
            reconstructed = self.decode(z, skip_features)
            return reconstructed
        
        except Exception as e:
            # Log détaillé en cas d'erreur
            logger.error(f"Échec forward pass: {e}")
            logger.error(f"Shape entrée: {x.shape}")
            logger.error(f"Taille attendue: {self.input_size}")
            logger.error(f"Configuration modèle: {self.summary()}")
            raise
    
    def compute_anomaly_scores(
        self,
        x: torch.Tensor,
        method: str = "mse"
    ) -> torch.Tensor:
        """
        Calcule les scores d'anomalie basés sur l'erreur de reconstruction.
        
        Args:
            x: Images originales
            method: Méthode de calcul ("mse", "mae", "ssim", "combined")
            
        Returns:
            Scores d'anomalie (B,) - Plus élevé = plus anormal
        """
        self.eval()
        
        with torch.no_grad():
            reconstructed = self.forward(x)
            
            if method == "mse":
                # Mean Squared Error par image
                scores = torch.mean((x - reconstructed) ** 2, dim=(1, 2, 3))
            
            elif method == "mae":
                # Mean Absolute Error
                scores = torch.mean(torch.abs(x - reconstructed), dim=(1, 2, 3))
            
            elif method == "ssim":
                # SSIM-based (inversé car SSIM élevé = similaire)
                try:
                    from pytorch_msssim import ssim # type: ignore
                    ssim_val = ssim(x, reconstructed, data_range=1.0, size_average=False)
                    scores = 1 - ssim_val  # Inversion: anomalie = faible SSIM
                except ImportError:
                    logger.warning("pytorch_msssim non installé, utilisation de MSE")
                    scores = torch.mean((x - reconstructed) ** 2, dim=(1, 2, 3))
            
            elif method == "combined":
                # Combinaison MSE + perceptual
                mse_score = torch.mean((x - reconstructed) ** 2, dim=(1, 2, 3))
                
                # Perceptual loss simple (gradient-based)
                grad_x = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
                grad_recon = torch.abs(reconstructed[:, :, :-1, :] - reconstructed[:, :, 1:, :])
                perceptual_score = torch.mean((grad_x - grad_recon) ** 2, dim=(1, 2, 3))
                
                scores = 0.7 * mse_score + 0.3 * perceptual_score
            
            else:
                raise ValueError(f"Méthode '{method}' non supportée")
            
            return scores
    
    def get_reconstruction_error_map(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Génère une carte spatiale des erreurs de reconstruction.
        Utile pour visualiser les zones anormales.
        
        Args:
            x: Images originales (B, C, H, W)
            
        Returns:
            Carte d'erreur (B, 1, H, W)
        """
        self.eval()
        
        with torch.no_grad():
            reconstructed = self.forward(x)
            
            # Erreur par pixel, moyennée sur les canaux
            error_map = torch.mean((x - reconstructed) ** 2, dim=1, keepdim=True)
            
            return error_map
    
    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Prédit les scores d'anomalie pour un batch numpy.
        Compatible avec l'API scikit-learn.
        
        Args:
            X: Images (N, C, H, W) en numpy
            batch_size: Taille des batchs pour traitement
            
        Returns:
            Scores d'anomalie (N,)
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Validation des données d'entrée
        if X is None or len(X) == 0:
            logger.error("Données d'entrée vides")
            return np.array([])
        
        all_scores = []
        
        try:
            # Traitement par batch pour économiser la mémoire
            for i in range(0, len(X), batch_size):
                batch = X[i:i+batch_size]
                
                # Conversion en tensor
                batch_tensor = torch.tensor(batch, dtype=torch.float32).to(device)
                
                # Calcul des scores
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
        """Résumé complet du modèle pour le debug et le logging."""
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
            "total_parameters": self.count_parameters(),
            "trainable_parameters": self.count_parameters(trainable_only=True)
        }


# === VARIATIONAL AUTOENCODER ===

class VariationalAutoEncoder(ConvAutoEncoder):
    """
    Variational AutoEncoder (VAE) pour détection d'anomalies.
    Hérite de ConvAutoEncoder avec loss KL-divergence.
    
    VAE apprend une distribution latente (μ, σ) plutôt qu'un point unique.
    Meilleur pour généralisation et détection d'anomalies hors distribution.
    
    Example:
        >>> model = VariationalAutoEncoder(input_size=(256, 256), latent_dim=128)
        >>> x = torch.randn(8, 3, 256, 256)
        >>> reconstructed = model(x)
        >>> loss, recon_loss, kl_loss = model.compute_vae_loss(x, reconstructed)
    """
    
    def __init__(self, **kwargs):
        kwargs['use_vae'] = True  # Force VAE mode
        super().__init__(**kwargs)
        
        self.kl_weight = 0.001  # Poids du terme KL
        
        logger.info("VariationalAutoEncoder initialisé")
    
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
            kl_weight: Poids du terme KL (optionnel)
            
        Returns:
            (total_loss, reconstruction_loss, kl_loss)
        """
        if kl_weight is None:
            kl_weight = self.kl_weight
        
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstructed, x, reduction='mean')
        
        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(
            1 + self.logvar - self.mu.pow(2) - self.logvar.exp()
        )
        kl_loss /= x.size(0) * x.size(1) * x.size(2) * x.size(3)  # Normalisation
        
        # Loss totale
        total_loss = recon_loss + kl_weight * kl_loss
        
        return total_loss, recon_loss, kl_loss


# === DENOISING AUTOENCODER ===

class DenoisingAutoEncoder(ConvAutoEncoder):
    """
    Denoising AutoEncoder pour détection robuste d'anomalies.
    Ajoute du bruit pendant l'entraînement pour meilleure robustesse.
    
    Example:
        >>> model = DenoisingAutoEncoder(input_size=(256, 256), noise_factor=0.2)
        >>> noisy_x = model.add_noise(x)
        >>> reconstructed = model(noisy_x)
    """
    
    def __init__(self, noise_factor: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        
        if not 0 <= noise_factor <= 1:
            raise ValueError(f"noise_factor doit être entre 0 et 1, reçu: {noise_factor}")
        
        self.noise_factor = noise_factor
        
        logger.info(f"DenoisingAutoEncoder initialisé: noise_factor={noise_factor}")
    
    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ajoute du bruit gaussien aux images.
        
        Args:
            x: Images propres
            
        Returns:
            Images bruitées
        """
        noise = torch.randn_like(x) * self.noise_factor
        noisy_x = torch.clamp(x + noise, 0.0, 1.0)
        return noisy_x
    
    def forward(self, x: torch.Tensor, add_noise_training: bool = True) -> torch.Tensor:
        """
        Forward avec ajout optionnel de bruit.
        
        Args:
            x: Images d'entrée
            add_noise_training: Ajouter du bruit si en mode training
            
        Returns:
            Images reconstruites
        """
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
    input_size: Tuple[int, int] = (256, 256),  # NOUVEAU: paramètre requis
    **kwargs
) -> nn.Module:
    """
    Factory pour créer des AutoEncoders avec taille dynamique.
    
    Args:
        model_type: Type d'autoencoder ("conv", "vae", "denoising")
        input_channels: Nombre de canaux d'entrée
        latent_dim: Dimension de l'espace latent
        input_size: Taille des images d'entrée (H, W) - OBLIGATOIRE
        **kwargs: Arguments additionnels pour le modèle
        
    Returns:
        Instance d'AutoEncoder configurée
        
    Raises:
        ValueError: Si le type de modèle n'est pas supporté
        Exception: Si la création du modèle échoue
        
    Example:
        >>> # Pour images 256x256 RGB
        >>> model = get_autoencoder("conv", input_size=(256, 256), latent_dim=128)
        >>> # Pour images 128x128 grayscale  
        >>> model = get_autoencoder("vae", input_channels=1, input_size=(128, 128))
    """
    # Registre des modèles disponibles
    models_registry = {
        "conv": ConvAutoEncoder,
        "vae": VariationalAutoEncoder,
        "denoising": DenoisingAutoEncoder
    }
    
    # Validation du type de modèle
    if model_type not in models_registry:
        raise ValueError(
            f"model_type '{model_type}' non supporté. "
            f"Options: {list(models_registry.keys())}"
        )
    
    model_class = models_registry[model_type]
    
    try:
        # Création du modèle avec les paramètres dynamiques
        model = model_class(
            input_channels=input_channels,
            latent_dim=latent_dim,
            input_size=input_size,  # Passage crucial de la taille
            **kwargs
        )
        
        logger.info(
            f"AutoEncoder {model_type} créé avec succès - "
            f"Taille: {input_size}, "
            f"Canaux: {input_channels}, "
            f"Latent: {latent_dim}"
        )
        return model
    
    except Exception as e:
        logger.error(
            f"Erreur création AutoEncoder {model_type}: {e}",
            exc_info=True,
            input_size=input_size,
            input_channels=input_channels,
            latent_dim=latent_dim
        )
        raise


# === TESTS UNITAIRES ===

if __name__ == "__main__":
    print("="*60)
    print("TESTS - AutoEncoders Dynamiques")
    print("="*60)
    
    # Test 1: ConvAutoEncoder avec différentes tailles
    print("\n### Test 1: ConvAutoEncoder Dynamique ###")
    model_256 = ConvAutoEncoder(input_size=(256, 256), latent_dim=128)
    print(f"✅ Modèle 256x256: {model_256.summary()}")
    
    model_128 = ConvAutoEncoder(input_size=(128, 128), latent_dim=64)
    print(f"✅ Modèle 128x128: {model_128.summary()}")
    
    # Test 2: Forward pass
    print("\n### Test 2: Forward Pass ###")
    x = torch.randn(4, 3, 256, 256)
    recon = model_256(x)
    print(f"✅ Forward OK: {x.shape} → {recon.shape}")
    
    # Test 3: Anomaly scores
    print("\n### Test 3: Anomaly Scores ###")
    scores = model_256.compute_anomaly_scores(x, method="mse")
    print(f"✅ Scores calculés: {scores.shape}, range: [{scores.min():.4f}, {scores.max():.4f}]")
    
    # Test 4: VAE
    print("\n### Test 4: Variational AutoEncoder ###")
    vae = VariationalAutoEncoder(input_size=(256, 256), latent_dim=128)
    recon_vae = vae(x)
    total_loss, recon_loss, kl_loss = vae.compute_vae_loss(x, recon_vae)
    print(f"✅ VAE Loss: total={total_loss:.4f}, recon={recon_loss:.4f}, kl={kl_loss:.4f}")
    
    # Test 5: Denoising
    print("\n### Test 5: Denoising AutoEncoder ###")
    dae = DenoisingAutoEncoder(input_size=(256, 256), noise_factor=0.3, latent_dim=128)
    noisy = dae.add_noise(x)
    print(f"✅ Bruit ajouté: original range=[{x.min():.2f}, {x.max():.2f}], noisy range=[{noisy.min():.2f}, {noisy.max():.2f}]")
    
    # Test 6: Factory
    print("\n### Test 6: Factory ###")
    model_factory = get_autoencoder("conv", input_size=(256, 256), latent_dim=256, use_skip_connections=True)
    print(f"✅ Factory OK: {type(model_factory).__name__}")
    
    print("\n" + "="*60)
    print("✅ TOUS LES TESTS RÉUSSIS!")
    print("="*60)