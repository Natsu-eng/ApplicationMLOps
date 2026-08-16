"""Registre des architectures de détection d'anomalies visuelles — pilier
Vision, Lot 15 sous-lot C (MVTec AD).

Même esprit que `services/vision_classification_registry.py` : ajouter une
architecture = ajouter une entrée, jamais toucher le moteur
(`vision_anomaly_training.py`). Une seule entrée pour l'instant
(autoencodeur convolutif simple) — le legacy en propose 4+ (VAE, denoising,
PatchCore, Siamese, `src/models/computer_vision/anomaly_detection/`),
volontairement pas repris ici : PatchCore notamment est nettement plus
lourd en calcul CPU (mémoire de patches de référence, recherche de plus
proche voisin sur l'ensemble du jeu d'entraînement à chaque inférence).
Extensible plus tard via ce registre si un besoin réel apparaît."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

# Fixe, pas un paramètre modèle : la responsabilité du redimensionnement est
# entièrement dans le pipeline de données (vision_anomaly_training.py),
# jamais dans le modèle lui-même — contrairement au legacy
# (`ConvAutoEncoder.auto_resize`), source du bug #5 (incohérence de format)
# documenté dans docs/legacy/ANALYSE_COMPLETE_COMPUTER_VISION.md.
IMAGE_SIZE = 128


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


@dataclass(frozen=True)
class AnomalyModelSpec:
    id: str
    label: str
    build_model: Callable[[], nn.Module]


ANOMALY_MODEL_REGISTRY: list[AnomalyModelSpec] = [
    AnomalyModelSpec(id="conv_autoencoder", label="Autoencodeur convolutif", build_model=ConvAutoEncoder),
]

DEFAULT_ANOMALY_MODEL_ID = "conv_autoencoder"

_REGISTRY_BY_ID = {s.id: s for s in ANOMALY_MODEL_REGISTRY}


def get_anomaly_model_spec(model_id: str) -> AnomalyModelSpec:
    if model_id not in _REGISTRY_BY_ID:
        raise ValueError(f"Modèle d'anomalie '{model_id}' inconnu. Options : {', '.join(_REGISTRY_BY_ID)}")
    return _REGISTRY_BY_ID[model_id]
