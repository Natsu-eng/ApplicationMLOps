"""Grad-CAM — explicabilité visuelle pour la classification d'images
(pilier Vision, Lot 15 sous-lot D).

Équivalent visuel de SHAP côté ML tabulaire (`services/ml_explainability.py`)
: pourquoi CE modèle a prédit CETTE classe pour CETTE image précise. Le
legacy prévoyait Grad-CAM (mentionné dans la documentation historique) mais
ne l'a jamais branché nulle part — aucune trace dans les fichiers audités
(`docs/legacy/`).

Réutilise intégralement `services/vision_localization.py` (sous-lot C) pour
le réalignement à la taille d'image originale et l'encodage PNG — un seul
mécanisme de superposition heatmap/image dans tout le module vision, jamais
deux implémentations séparées."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from PIL import Image

from domains.vision.classification.services.registry import get_backbone_spec
from domains.vision.classification.services.engine import build_eval_transform
from domains.vision.localization import overlay_heatmap_on_image, resize_map_to_original


class GradCamError(ValueError):
    """Requête d'explication invalide (classe cible inconnue, etc.)."""


@dataclass
class GradCamResult:
    predicted_label: str
    probabilities: dict[str, float]
    target_label: str  # classe réellement expliquée (= predicted_label si non précisée)
    heatmap_png: str


class _ActivationGradientCapture:
    """Capture l'activation et le gradient d'UNE couche cible via hooks —
    classe dédiée plutôt que des variables `nonlocal` éparpillées, pour
    rester lisible et testable isolément."""

    def __init__(self, target_layer: nn.Module):
        self.activation: torch.Tensor | None = None
        self.gradient: torch.Tensor | None = None
        target_layer.register_forward_hook(self._on_forward)
        target_layer.register_full_backward_hook(self._on_backward)

    def _on_forward(self, module: nn.Module, inputs: Any, output: torch.Tensor) -> None:
        self.activation = output

    def _on_backward(self, module: nn.Module, grad_input: Any, grad_output: Any) -> None:
        self.gradient = grad_output[0]


def _rebuild_model(artifact: dict[str, Any]) -> nn.Module:
    spec = get_backbone_spec(artifact["backbone_id"])
    model = spec.build_model(len(artifact["class_names"]), artifact["dropout_rate"])
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    return model


def explain_classification_prediction(
    artifact: dict[str, Any],
    image: Image.Image,
    target_label: str | None = None,
) -> GradCamResult:
    """Point d'entrée principal — `artifact` est le contenu de
    `VisionClassificationModel.file_path` (`torch.load`, voir
    `workers/vision_classification_worker.py`). `target_label` optionnel :
    explique la classe prédite par défaut, ou une classe précise choisie par
    l'utilisateur (ex. "pourquoi PAS la classe X ?")."""
    class_names: list[str] = artifact["class_names"]
    spec = get_backbone_spec(artifact["backbone_id"])
    model = _rebuild_model(artifact)
    capture = _ActivationGradientCapture(spec.gradcam_target_layer(model))

    transform = build_eval_transform()
    original_size = image.size  # (largeur, hauteur)
    input_tensor = transform(image.convert("RGB")).unsqueeze(0)
    # Le backbone est gelé par défaut (sous-lot B) : sans ceci, AUCUN tenseur
    # en amont de la couche cible n'exige de gradient, le graphe autograd
    # n'est jamais construit jusqu'à elle et le hook backward capte un
    # gradient vide/incorrect. Ne sert qu'à connecter le graphe — les poids
    # gelés ne sont jamais mis à jour (pas d'optimizer.step() ici).
    input_tensor.requires_grad_(True)

    model.zero_grad()
    logits = model(input_tensor)
    probabilities = torch.softmax(logits, dim=1)[0].detach()
    predicted_idx = int(probabilities.argmax().item())

    if target_label is None:
        target_idx = predicted_idx
    else:
        if target_label not in class_names:
            raise GradCamError(f"Classe inconnue : {target_label}. Options : {', '.join(class_names)}")
        target_idx = class_names.index(target_label)

    logits[0, target_idx].backward()

    # Poids par canal = moyenne spatiale du gradient (Grad-CAM original,
    # Selvaraju et al. 2017) — combinaison pondérée des activations, ReLU
    # pour ne garder que l'influence POSITIVE sur la classe cible.
    weights = capture.gradient.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * capture.activation).sum(dim=1, keepdim=True))
    cam_map = cam[0, 0].detach().numpy()

    # Même réalignement testé au sous-lot C (bugs #10/#17) : la carte Grad-CAM
    # est calculée à la résolution du dernier bloc convolutif (ex. 7x7 pour
    # un backbone à stride 32 sur une entrée 224x224), jamais superposée
    # telle quelle à l'image d'origine.
    cam_original_size = resize_map_to_original(cam_map, original_size)

    return GradCamResult(
        predicted_label=class_names[predicted_idx],
        probabilities={name: float(probabilities[i]) for i, name in enumerate(class_names)},
        target_label=class_names[target_idx],
        # Superposition (Lot 16A) — remplace la heatmap seule : les zones
        # rouges de l'image superposée sont celles qui ont le plus influencé
        # la prédiction, directement lisibles sur la photo elle-même.
        heatmap_png=overlay_heatmap_on_image(image, cam_original_size),
    )
