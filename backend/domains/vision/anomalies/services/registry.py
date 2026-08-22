"""Registre des architectures de détection d'anomalies visuelles — pilier
Vision, Lot 15 sous-lot C (structure normal/défaut).

Même esprit que `services/vision_classification_registry.py` : ajouter une
architecture = ajouter une entrée, jamais toucher le moteur
(`vision_anomaly_training.py`) au-delà d'un branchement explicite sur
`AnomalyModelSpec.loss_kind`. Le legacy propose 4+ architectures (VAE,
denoising, PatchCore, Siamese, `src/models/computer_vision/anomaly_detection/`)
— 3 reprises ici (`conv_autoencoder`/`denoising_autoencoder`/`conv_vae`),
toutes des variantes du MÊME autoencodeur convolutif propre (pas la version
legacy à bottleneck dense/`auto_resize` dynamique, source du bug #5).
PatchCore et Siamese restent volontairement absents : PatchCore ajoute une
dépendance lourde (`faiss`) et un pipeline mémoire de patches + recherche de
plus proche voisin à l'inférence, Siamese exige un pipeline d'entraînement
par PAIRES (contrastive loss) totalement différent de l'entraînement par
reconstruction ici — les deux sont un changement de pipeline, pas une
entrée de registre, voir DECISIONS.md."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Fixe, pas un paramètre modèle : la responsabilité du redimensionnement est
# entièrement dans le pipeline de données (vision_anomaly_training.py),
# jamais dans le modèle lui-même — contrairement au legacy
# (`ConvAutoEncoder.auto_resize`), source du bug #5 (incohérence de format)
# documenté dans docs/legacy/ANALYSE_COMPLETE_COMPUTER_VISION.md.
# Vit dans `domains/vision/shared.py` (Lot 8, §Phase 0) — `vision/datasets`
# en a aussi besoin, jamais un import direct d'un sous-domaine vision vers
# un autre.
from domains.vision.shared import ANOMALY_IMAGE_SIZE as IMAGE_SIZE


class ConvAutoEncoder(nn.Module):
    """Autoencodeur convolutif simple, entièrement convolutif (pas de
    bottleneck dense) — évite par construction toute la classe de bugs de
    calcul dynamique de `flat_features` du legacy. Reçoit TOUJOURS des
    images `IMAGE_SIZE`×`IMAGE_SIZE` (garanti par le pipeline de données),
    jamais de resize interne."""

    def __init__(self, base_filters: int = 32, num_stages: int = 3):
        super().__init__()
        encoder_layers: list[nn.Module] = []
        in_channels = 3
        for stage in range(num_stages):
            out_channels = base_filters * (2**stage)
            encoder_layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]
            in_channels = out_channels
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        for stage in reversed(range(num_stages)):
            out_channels = base_filters * (2 ** (stage - 1)) if stage > 0 else 3
            decoder_layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2))
            if stage > 0:
                decoder_layers += [nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
            in_channels = out_channels
        decoder_layers.append(nn.Sigmoid())  # sortie dans [0, 1], même espace que l'entrée (ToTensor, pas de normalisation ImageNet)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class DenoisingConvAutoEncoder(ConvAutoEncoder):
    """Variante « denoising » du legacy (`DenoisingAutoEncoder`), portée sur
    l'architecture propre ci-dessus plutôt que sur le bottleneck dense
    legacy — bruit gaussien ajouté à l'ENTRÉE pendant l'entraînement
    uniquement (`self.training`), la cible de reconstruction reste l'image
    PROPRE. Force le modèle à apprendre les traits structurels de "good"
    plutôt qu'à mémoriser le bruit de capture pixel à pixel — aucun autre
    changement de pipeline (même `MSELoss`, même boucle d'entraînement)."""

    def __init__(self, base_filters: int = 32, num_stages: int = 3, noise_factor: float = 0.1):
        super().__init__(base_filters=base_filters, num_stages=num_stages)
        self.noise_factor = noise_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.noise_factor > 0:
            x = torch.clamp(x + torch.randn_like(x) * self.noise_factor, 0.0, 1.0)
        return super().forward(x)


class ConvVAE(nn.Module):
    """Variational autoencoder SPATIAL — porte l'esprit du VAE legacy sans
    reprendre son bottleneck dense (`fc_mu`/`fc_logvar` sur des features
    aplaties, dont la taille dépend de `input_size`, source du bug #5) :
    `mu`/`logvar` sont ici des cartes de features (têtes `Conv2d` 1×1 sur la
    sortie spatiale de l'encodeur), jamais un vecteur — l'architecture reste
    entièrement convolutive, indépendante de la taille d'entrée. Nécessite
    un terme KL en plus de l'erreur de reconstruction (voir `compute_loss`,
    branché depuis `vision_anomaly_training.py` via `AnomalyModelSpec.loss_kind
    == "vae"`) : la reconstruction seule ne suffit pas à entraîner un VAE
    (`mu`/`logvar` ne reçoivent aucun gradient utile sans la divergence KL)."""

    kl_weight = 1e-4  # poids faible — la reconstruction doit rester le signal dominant

    def __init__(self, base_filters: int = 32, num_stages: int = 3):
        super().__init__()
        self._base = ConvAutoEncoder(base_filters=base_filters, num_stages=num_stages)
        bottleneck_channels = base_filters * (2 ** (num_stages - 1))
        self.mu_head = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=1)
        self.logvar_head = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=1)
        self._last_mu: Optional[torch.Tensor] = None
        self._last_logvar: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._base.encoder(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
        else:
            z = mu  # inférence déterministe — pas d'échantillonnage
        self._last_mu, self._last_logvar = mu, logvar
        return self._base.decoder(z)

    def compute_loss(self, x: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        recon_loss = F.mse_loss(reconstructed, x)
        if self._last_mu is None or self._last_logvar is None:
            return recon_loss  # éval (pas de forward training juste avant) — reconstruction seule
        kl = -0.5 * torch.mean(1 + self._last_logvar - self._last_mu.pow(2) - self._last_logvar.exp())
        return recon_loss + self.kl_weight * kl


@dataclass(frozen=True)
class AnomalyModelSpec:
    id: str
    label: str
    build_model: Callable[[], nn.Module]
    # "mse" (défaut) — `nn.MSELoss()` standard, appliqué par le moteur
    # d'entraînement. "vae" — le moteur appelle `model.compute_loss(x,
    # reconstructed)` à la place (reconstruction + KL, voir ConvVAE).
    loss_kind: str = "mse"


ANOMALY_MODEL_REGISTRY: list[AnomalyModelSpec] = [
    AnomalyModelSpec(id="conv_autoencoder", label="Autoencodeur convolutif", build_model=ConvAutoEncoder),
    AnomalyModelSpec(
        id="denoising_autoencoder",
        label="Autoencodeur débruiteur",
        build_model=DenoisingConvAutoEncoder,
    ),
    AnomalyModelSpec(
        id="conv_vae",
        label="Autoencodeur variationnel (VAE)",
        build_model=ConvVAE,
        loss_kind="vae",
    ),
]

DEFAULT_ANOMALY_MODEL_ID = "conv_autoencoder"

_REGISTRY_BY_ID = {s.id: s for s in ANOMALY_MODEL_REGISTRY}


def get_anomaly_model_spec(model_id: str) -> AnomalyModelSpec:
    if model_id not in _REGISTRY_BY_ID:
        raise ValueError(f"Modèle d'anomalie '{model_id}' inconnu. Options : {', '.join(_REGISTRY_BY_ID)}")
    return _REGISTRY_BY_ID[model_id]
