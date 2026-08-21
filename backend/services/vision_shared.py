"""Constantes et utilitaires d'augmentation partagés entre les deux
sous-domaines Vision — extrait de `services/vision_classification_training.py`
(Lot 8, correctif de frontières §Phase 0) : `vision_anomalies` (routeur +
moteur d'entraînement) et `vision_datasets` (ingestion/validation) en
avaient besoin et importaient jusqu'ici un module du sous-domaine
classification pour des symboles qui n'ont rien de spécifique à la
classification (taille d'image d'entrée du backbone, presets
d'augmentation applicables à toute image quel que soit le pilier vision
qui l'exploite ensuite).

Ne PAS confondre `IMAGE_SIZE` ci-dessous (224, taille d'entrée du backbone
de classification) avec `services/vision_anomaly_registry.py::IMAGE_SIZE`
(128, taille d'entrée de l'autoencodeur) — deux constantes légitimement
différentes, chacune propre à son propre modèle, pas un doublon à fusionner
(vérifié explicitement lors de l'extraction)."""
from __future__ import annotations

from torchvision import transforms

IMAGE_SIZE = 224

# Lot 6A (correctif I9, AUDIT_DATALAB_2026-08-16.md §I9) — 4 presets, du plus
# faible au plus fort — jamais de valeurs choisies au hasard : chaque niveau
# ajoute une transformation à celles du niveau précédent, jamais une
# combinaison disjointe (progression cohérente, prévisible pour
# l'utilisateur).
AUGMENTATION_PRESET_IDS = ("aucune", "legere", "standard", "forte")


def augmentation_transforms(preset: str) -> list:
    if preset == "aucune":
        return []
    if preset == "legere":
        return [transforms.RandomHorizontalFlip()]
    if preset == "standard":
        return [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    if preset == "forte":
        return [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ]
    raise ValueError(f"Preset d'augmentation inconnu : {preset!r} (attendu parmi {AUGMENTATION_PRESET_IDS})")


# Seuils empiriques (pas une science exacte) fondés sur la taille de la
# classe la plus PETITE — le goulot d'étranglement réel pour le risque de
# sur-apprentissage, jamais le total d'images (masquerait un déséquilibre
# sévère : 1000 images dont une classe à 5 reste une classe à 5). Peu
# d'images par classe → sur-apprentissage plus probable → augmentation
# plus forte pour diversifier artificiellement le peu de données
# disponibles ; beaucoup d'images → la variété réelle suffit déjà, une
# augmentation trop forte distordrait inutilement la distribution et
# ralentirait la convergence sans bénéfice.
_RECOMMENDATION_THRESHOLDS = ((20, "forte"), (50, "standard"), (150, "legere"))


def recommend_augmentation_preset(min_class_size: int) -> str:
    """Recommandation fondée sur la taille du dataset (I9) — indicative,
    jamais appliquée automatiquement : l'utilisateur choisit toujours
    explicitement le preset final (voir ClassificationConfig.augmentation_preset)."""
    for threshold, preset in _RECOMMENDATION_THRESHOLDS:
        if min_class_size < threshold:
            return preset
    return "aucune"
