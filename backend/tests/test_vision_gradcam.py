"""Tests de `services/vision_gradcam.py` (pilier Vision, Lot 15 sous-lot D)
— Grad-CAM réel (pas mocké) sur un modèle réellement entraîné, même
approche que les autres modules du pilier Vision."""
from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image

from services.vision_classification_registry import CLASSIFICATION_BACKBONE_REGISTRY
from services.vision_classification_training import ClassificationConfig, train_and_evaluate_classification
from services.vision_gradcam import GradCamError, explain_classification_prediction


def _write_classification_dataset(root, n_per_class=10, size=(80, 80)):
    rng = np.random.default_rng(0)
    for class_name, color in [("rouge", (220, 20, 20)), ("bleu", (20, 20, 220))]:
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            noise = rng.integers(-15, 15, (size[1], size[0], 3))
            arr = np.clip(np.array(color) + noise, 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(class_dir / f"{i}.png")
    return root


@pytest.fixture(scope="module")
def trained_artifact(tmp_path_factory):
    """Un seul entraînement réel réutilisé par tous les tests de ce fichier
    — Grad-CAM n'entraîne rien, inutile de réentraîner à chaque test
    (économise plusieurs dizaines de secondes de calcul CPU réel)."""
    root = tmp_path_factory.mktemp("gradcam_dataset")
    _write_classification_dataset(root)
    config = ClassificationConfig(backbone_id="mobilenet_v3_small", num_epochs=2, batch_size=4, freeze_backbone=True)
    result = train_and_evaluate_classification(root, config, lambda step, pct: None)
    sample_image_path = root / "rouge" / "0.png"
    return result.model_artifact, sample_image_path


def test_explain_returns_valid_heatmap(trained_artifact):
    artifact, sample_image_path = trained_artifact
    with Image.open(sample_image_path) as image:
        explanation = explain_classification_prediction(artifact, image)

    assert explanation.predicted_label in {"rouge", "bleu"}
    assert explanation.target_label == explanation.predicted_label  # pas de cible précisée
    assert set(explanation.probabilities) == {"rouge", "bleu"}
    assert abs(sum(explanation.probabilities.values()) - 1.0) < 1e-4
    assert explanation.heatmap_png.startswith("data:image/png;base64,")


def test_explain_heatmap_is_not_degenerate(trained_artifact):
    """Le gradient doit réellement varier spatialement — un backbone gelé
    mal câblé (input sans requires_grad) produirait un gradient nul/plat,
    donc une heatmap uniforme."""
    artifact, sample_image_path = trained_artifact
    with Image.open(sample_image_path) as image:
        explanation = explain_classification_prediction(artifact, image)

    raw = base64.b64decode(explanation.heatmap_png.split(",", 1)[1])
    arr = np.array(Image.open(io.BytesIO(raw)))
    assert arr.std() > 0


def test_explain_heatmap_matches_original_image_size(trained_artifact):
    artifact, sample_image_path = trained_artifact
    with Image.open(sample_image_path) as image:
        original_size = image.size
        explanation = explain_classification_prediction(artifact, image)

    raw = base64.b64decode(explanation.heatmap_png.split(",", 1)[1])
    arr = np.array(Image.open(io.BytesIO(raw)))
    assert (arr.shape[1], arr.shape[0]) == original_size  # (largeur, hauteur) vs numpy (H, W)


def test_explain_with_explicit_target_label(trained_artifact):
    artifact, sample_image_path = trained_artifact
    with Image.open(sample_image_path) as image:
        explanation = explain_classification_prediction(artifact, image, target_label="bleu")

    assert explanation.target_label == "bleu"


def test_explain_rejects_unknown_target_label(trained_artifact):
    artifact, sample_image_path = trained_artifact
    with Image.open(sample_image_path) as image:
        with pytest.raises(GradCamError):
            explain_classification_prediction(artifact, image, target_label="vert")


def test_all_registered_backbones_support_gradcam():
    """Chaque backbone du registre déclare bien un `gradcam_target_layer` —
    garde-fou pour ne jamais ajouter un futur backbone sans support Grad-CAM
    (contrairement à l'entraînement, Grad-CAM n'a pas de mode dégradé)."""
    for spec in CLASSIFICATION_BACKBONE_REGISTRY:
        model = spec.build_model(2, 0.3)
        target = spec.gradcam_target_layer(model)
        assert target is not None
