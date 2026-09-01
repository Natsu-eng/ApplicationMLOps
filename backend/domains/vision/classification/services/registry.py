"""Registre des backbones de classification d'images — pilier Vision,
Lot 15 sous-lot B, étendu Lot 6A.

Même esprit que `services/anomaly_registry.py` : chaque entrée déclare tout
ce qui distingue un backbone du reste, ajouter un backbone = ajouter une
entrée, jamais toucher le moteur (`vision_classification_training.py`).

Contrainte déterminante (voir le plan du chantier Vision, section
"Contrainte d'infrastructure déterminante") : aucun GPU dans
`docker-compose.yml`, un seul worker RQ physique CPU partagé avec tous les
autres types de job. Seuls des backbones raisonnablement légers sont
proposés ici — jamais les architectures les plus lourdes du registre
legacy (VGG16/19, ResNet101/152 — `transfer_learning.py` en proposait 17
au total), qui rendraient un entraînement CPU impraticable dans le temps
d'un job (peu d'époques complétées avant `max_training_seconds`, voir
`vision_classification_training.py`). 7 backbones ici (Lot 6A) contre 2
initialement (Lot 15) : assez de variété réelle (familles resnet/
mobilenet/efficientnet/shufflenet/densenet) sans reproduire l'impraticable
parité totale avec le legacy — décision explicite, voir DECISIONS.md."""
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
    # Nombre de paramètres (millions, tronc pré-entraîné torchvision, valeur
    # publiée officielle — https://pytorch.org/vision/stable/models.html —
    # PAS recalculée dynamiquement : instancier les 7 architectures au seul
    # import de ce module forcerait le téléchargement des 7 jeux de poids
    # pré-entraînés (des centaines de Mo) à chaque process qui importe ce
    # fichier (tests compris), pour une valeur qui ne change jamais tant que
    # la version de torchvision reste figée (voir requirements.txt). Lot 16F
    # (retour utilisateur : catalogue jamais assorti d'une indication de
    # vitesse) — sert à dériver `speed_tier` ci-dessous, jamais affiché seul.
    params_millions: float


# Seuils Lot 16F — jamais un jugement qualitatif arbitraire : bornés sur les
# 7 entrées réelles du registre ci-dessous (2,3 à 21,8 M de paramètres).
# "rapide" = le tiers le plus léger du registre (mobile-first), "lent" =
# au-delà de ResNet34, les 2 architectures les plus profondes.
_SPEED_TIER_FAST_MAX_PARAMS = 6.0
_SPEED_TIER_MODERATE_MAX_PARAMS = 12.0


def speed_tier(params_millions: float) -> str:
    if params_millions <= _SPEED_TIER_FAST_MAX_PARAMS:
        return "rapide"
    if params_millions <= _SPEED_TIER_MODERATE_MAX_PARAMS:
        return "modere"
    return "lent"


def _resnet_backbone_children(model: nn.Module) -> list[nn.Module]:
    # Générique à tout modèle dont la tête de classification s'appelle
    # "fc" (toute la famille resnet, mais aussi shufflenet_v2 ci-dessous —
    # filtrage par NOM, jamais par position, donc réutilisable tel quel).
    return [m for name, m in model.named_children() if name != "fc"]


def _build_resnet18(num_classes: int, dropout_rate: float) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(in_features, num_classes))
    return model


def _build_resnet34(num_classes: int, dropout_rate: float) -> nn.Module:
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(in_features, num_classes))
    return model


def _mobilenet_backbone_children(model: nn.Module) -> list[nn.Module]:
    # `features` = tronc convolutif ; `classifier[:-1]` = couches Linear/
    # Hardswish/Dropout d'origine AVANT la tête qu'on vient de remplacer.
    # Générique small/large (même structure `.features`/`.classifier`).
    return [model.features] + [m for m in model.classifier[:-1]]


def _build_mobilenet_v3_small(num_classes: int, dropout_rate: float) -> nn.Module:
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(in_features, num_classes))
    return model


def _build_mobilenet_v3_large(num_classes: int, dropout_rate: float) -> nn.Module:
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(in_features, num_classes))
    return model


def _build_efficientnet_b0(num_classes: int, dropout_rate: float) -> nn.Module:
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(in_features, num_classes))
    return model


def _efficientnet_backbone_children(model: nn.Module) -> list[nn.Module]:
    # Même structure `.features`/`.classifier` que mobilenet — EfficientNet
    # a en plus un `.avgpool` (pooling pur, sans paramètre entraînable, pas
    # besoin de l'inclure ici).
    return [model.features] + [m for m in model.classifier[:-1]]


def _build_shufflenet_v2(num_classes: int, dropout_rate: float) -> nn.Module:
    model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(in_features, num_classes))
    return model


def _build_densenet121(num_classes: int, dropout_rate: float) -> nn.Module:
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    # DenseNet : `classifier` est directement un nn.Linear (jamais un
    # Sequential comme mobilenet/efficientnet) — remplacé dans son
    # intégralité, pas seulement son dernier élément.
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(in_features, num_classes))
    return model


def _densenet_backbone_children(model: nn.Module) -> list[nn.Module]:
    # `classifier` déjà entièrement remplacé par build_model ci-dessus —
    # seul `.features` (tronc convolutif) doit pouvoir être gelé/dégelé.
    return [model.features]


CLASSIFICATION_BACKBONE_REGISTRY: list[ClassificationBackboneSpec] = [
    ClassificationBackboneSpec(
        id="mobilenet_v3_small",
        label="MobileNetV3-Small",
        build_model=_build_mobilenet_v3_small,
        backbone_children=_mobilenet_backbone_children,
        gradcam_target_layer=lambda model: model.features,
        params_millions=2.5,
    ),
    ClassificationBackboneSpec(
        id="resnet18",
        label="ResNet18",
        build_model=_build_resnet18,
        backbone_children=_resnet_backbone_children,
        gradcam_target_layer=lambda model: model.layer4,
        params_millions=11.7,
    ),
    ClassificationBackboneSpec(
        id="resnet34",
        label="ResNet34",
        build_model=_build_resnet34,
        backbone_children=_resnet_backbone_children,
        gradcam_target_layer=lambda model: model.layer4,
        params_millions=21.8,
    ),
    ClassificationBackboneSpec(
        id="mobilenet_v3_large",
        label="MobileNetV3-Large",
        build_model=_build_mobilenet_v3_large,
        backbone_children=_mobilenet_backbone_children,
        gradcam_target_layer=lambda model: model.features,
        params_millions=5.4,
    ),
    ClassificationBackboneSpec(
        id="efficientnet_b0",
        label="EfficientNet-B0",
        build_model=_build_efficientnet_b0,
        backbone_children=_efficientnet_backbone_children,
        gradcam_target_layer=lambda model: model.features,
        params_millions=5.3,
    ),
    ClassificationBackboneSpec(
        id="shufflenet_v2",
        label="ShuffleNetV2",
        build_model=_build_shufflenet_v2,
        backbone_children=_resnet_backbone_children,
        gradcam_target_layer=lambda model: model.conv5,
        params_millions=2.3,
    ),
    ClassificationBackboneSpec(
        id="densenet121",
        label="DenseNet121",
        build_model=_build_densenet121,
        backbone_children=_densenet_backbone_children,
        gradcam_target_layer=lambda model: model.features,
        params_millions=8.0,
    ),
]

# Le plus léger du registre (2.5M paramètres) — meilleur défaut pour un
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
