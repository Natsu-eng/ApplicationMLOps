"""Inférence détection d'anomalies visuelles — note une NOUVELLE image à
partir du pipeline persisté par `workers/vision_anomaly_worker.py` (Lot 6B,
§F.2 — jusqu'ici, ce pilier n'avait AUCUNE capacité de notation d'une
nouvelle image, contrairement à la classification (`/explain`) : seules les
images du jeu de test déjà entraîné étaient consultables via `/examples`).

Contrairement au tabulaire/clustering (LOF nécessitait une instance dédiée
`novelty=True`), un autoencodeur est nativement inductif : la reconstruction
et l'erreur de reconstruction se calculent identiquement sur une image
jamais vue, aucun artifice nécessaire. Le seuil de décision (`threshold`,
calibré par le point de Youden sur la courbe ROC à l'entraînement, voir
`services/engine.py::train_and_evaluate_anomaly_vision`) est persisté dans
le bundle et réutilisé TEL QUEL, jamais recalculé — un seuil n'a de sens que
calibré sur un jeu de données de référence, pas sur une seule image."""
from __future__ import annotations

from typing import Any

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from domains.vision.anomalies.services.registry import IMAGE_SIZE, get_anomaly_model_spec
from domains.vision.localization import overlay_heatmap_on_image, resize_map_to_original


class VisionAnomalyInferenceError(ValueError):
    """L'image fournie ne peut pas être notée (fichier illisible...)."""


def _rebuild_model(artifact: dict[str, Any]):
    spec = get_anomaly_model_spec(artifact["model_id"])
    model = spec.build_model()
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    return model


def score_vision_anomaly(artifact: dict[str, Any], image: Image.Image) -> dict[str, Any]:
    """Note UNE image — reconstruit le modèle (voir `_rebuild_model`), calcule
    l'erreur de reconstruction (même formule EXACTE que l'entraînement,
    `services/engine.py` : MSE par pixel moyennée sur canaux/spatial) et la
    compare au seuil déjà calibré. Retourne aussi la carte de localisation
    (mêmes fonctions que les exemples d'entraînement, `services/localization.py`)
    — savoir OÙ se situe le défaut est le cœur de la valeur de ce pilier,
    pas seulement un score."""
    try:
        rgb_image = image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise VisionAnomalyInferenceError("Impossible de lire cette image") from exc

    model = _rebuild_model(artifact)
    image_size = artifact.get("image_size", IMAGE_SIZE)
    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    original_size = rgb_image.size  # (largeur, hauteur)
    input_tensor = transform(rgb_image).unsqueeze(0)

    with torch.no_grad():
        reconstructed = model(input_tensor)
        error_map = torch.mean((input_tensor - reconstructed) ** 2, dim=1)[0]
        score = float(error_map.mean().item())

    threshold = float(artifact["threshold"])
    error_map_original_size = resize_map_to_original(error_map.numpy(), original_size)

    return {
        "anomaly_score": score,
        "threshold": threshold,
        "is_anomaly": score > threshold,
        "heatmap_png": overlay_heatmap_on_image(rgb_image, error_map_original_size),
    }
