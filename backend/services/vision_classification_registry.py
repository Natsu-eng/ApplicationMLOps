"""Registre des backbones de classification d'images — pilier Vision,
Lot 15 sous-lot B.

Même esprit que `services/anomaly_registry.py` : chaque entrée déclare tout
ce qui distingue un backbone du reste, ajouter un backbone = ajouter une
entrée, jamais toucher le moteur (`vision_classification_training.py`).

Contrainte déterminante (voir le plan du chantier Vision, section
"Contrainte d'infrastructure déterminante") : aucun GPU dans
`docker-compose.yml`, un seul worker RQ physique CPU partagé avec tous les
autres types de job. Seuls des backbones légers sont proposés ici — pas de
ResNet50+/EfficientNet/VGG (legacy `transfer_learning.py` en propose 17),
qui rendraient un entraînement CPU impraticable dans le temps d'un job.
Extensible plus tard via ce registre si un besoin réel apparaît."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch.nn as nn
from torchvision import models


@dataclass(frozen=True)
class ClassificationBackboneSpec:
    id: str
    label: str
    # Construit le backbone pré-entraîné ET remplace sa couche finale par un
    # nn.Linear(in_features_réelles, num_classes) — in_features lu
    # dynamiquement sur la couche d'origine (jamais codé en dur : diffère
    # entre resnet.fc et mobilenet.classifier[-1]).
    build_model: Callable[[int, float], nn.Module]
    # Liste des sous-modules considérés comme le backbone (tout sauf la tête
    # de classification) — utilisée pour geler/dégeler explicitement,
    # jamais un simple compteur de couches arbitraire comme le legacy.
    backbone_children: Callable[[nn.Module], list[nn.Module]]
    # Dernière couche convolutive (avant pooling/classifier) — cible des
    # hooks Grad-CAM (sous-lot D, services/vision_gradcam.py). Diffère par
    # architecture, jamais devinée dynamiquement (fragile) : déclarée
    # explicitement ici, même esprit que `build_model`/`backbone_children`.
    gradcam_target_layer: Callable[[nn.Module], nn.Module]


def _build_resnet18(num_classes: int, dropout_rate: float) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(in_features, num_classes))
    return model


def _resnet18_backbone_children(model: nn.Module) -> list[nn.Module]:
    return [m for name, m in model.named_children() if name != "fc"]


def _build_mobilenet_v3_small(num_classes: int, dropout_rate: float) -> nn.Module:
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(in_features, num_classes))
    return model


def _mobilenet_backbone_children(model: nn.Module) -> list[nn.Module]:
    # `features` = tronc convolutif ; `classifier[:-1]` = couches Linear/
    # Hardswish/Dropout d'origine AVANT la tête qu'on vient de remplacer.
    return [model.features] + [m for m in model.classifier[:-1]]


CLASSIFICATION_BACKBONE_REGISTRY: list[ClassificationBackboneSpec] = [
    ClassificationBackboneSpec(
        id="mobilenet_v3_small",
        label="MobileNetV3-Small",
        build_model=_build_mobilenet_v3_small,
        backbone_children=_mobilenet_backbone_children,
        gradcam_target_layer=lambda model: model.features,
    ),
    ClassificationBackboneSpec(
        id="resnet18",
        label="ResNet18",
        build_model=_build_resnet18,
        backbone_children=_resnet18_backbone_children,
        gradcam_target_layer=lambda model: model.layer4,
    ),
]

# Le plus léger des deux (2.5M paramètres vs 11M) — meilleur défaut pour un
# entraînement CPU sans configuration explicite de l'utilisateur.
DEFAULT_BACKBONE_ID = "mobilenet_v3_small"

_REGISTRY_BY_ID = {s.id: s for s in CLASSIFICATION_BACKBONE_REGISTRY}


def get_backbone_spec(backbone_id: str) -> ClassificationBackboneSpec:
    if backbone_id not in _REGISTRY_BY_ID:
        raise ValueError(
            f"Backbone '{backbone_id}' inconnu. Options : {', '.join(_REGISTRY_BY_ID)}"
        )
    return _REGISTRY_BY_ID[backbone_id]


def freeze_backbone(model: nn.Module, spec: ClassificationBackboneSpec) -> None:
    """Gèle tous les paramètres du backbone (tronc pré-entraîné), laisse la
    tête de classification (déjà remplacée par `build_model`) entraînable —
    transfer learning au sens strict, comportement par défaut du sous-lot B."""
    for child in spec.backbone_children(model):
        for param in child.parameters():
            param.requires_grad = False


def unfreeze_backbone(model: nn.Module) -> None:
    """Dégèle tous les paramètres — utilisé pour le fine-tuning progressif
    optionnel (`unfreeze_after_epoch`, voir vision_classification_training.py)."""
    for param in model.parameters():
        param.requires_grad = True
