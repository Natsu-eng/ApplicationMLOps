"""Tests de `services/vision_gradcam.py` (pilier Vision, Lot 15 sous-lot D)
— Grad-CAM réel (pas mocké) sur un modèle réellement entraîné, même
approche que les autres modules du pilier Vision."""
from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image

from domains.vision.classification.services.registry import CLASSIFICATION_BACKBONE_REGISTRY
from domains.vision.classification.services.engine import ClassificationConfig, train_and_evaluate_classification
from domains.vision.classification.services.gradcam import (
    GradCamError,
    explain_classification_prediction,
    explain_classification_predictions_batch,
)


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


# ── Lot 6A — nouveaux backbones : Grad-CAM bout-en-bout sur les motifs de
# gradcam_target_layer génuinement nouveaux (pas seulement "n'est pas None"
# comme ci-dessus) — `.layer4` (famille resnet) et `.conv5` (shufflenet).
# Le motif `.features` (mobilenet/efficientnet/densenet) est déjà couvert
# bout-en-bout par `trained_artifact` (mobilenet_v3_small) ci-dessus.


@pytest.mark.parametrize("backbone_id", ["resnet18", "shufflenet_v2"])
def test_explain_produces_a_non_degenerate_heatmap_for_new_backbones(tmp_path, backbone_id):
    _write_classification_dataset(tmp_path)
    config = ClassificationConfig(backbone_id=backbone_id, num_epochs=1, batch_size=4, freeze_backbone=True)
    result = train_and_evaluate_classification(tmp_path, config, lambda step, pct: None)

    sample_image_path = tmp_path / "rouge" / "0.png"
    with Image.open(sample_image_path) as image:
        explanation = explain_classification_prediction(result.model_artifact, image)

    raw = base64.b64decode(explanation.heatmap_png.split(",", 1)[1])
    arr = np.array(Image.open(io.BytesIO(raw)))
    assert arr.std() > 0


# ── Batch (retour utilisateur direct : "Grad-CAM devrait supporter le
# batch, pas une image à la fois") ──────────────────────────────────────


def test_batch_explains_every_image_and_preserves_order(trained_artifact):
    artifact, sample_image_path = trained_artifact
    root = sample_image_path.parent.parent
    paths = [("rouge/0.png", root / "rouge" / "0.png"), ("bleu/0.png", root / "bleu" / "0.png")]
    images = [(key, Image.open(path)) for key, path in paths]
    try:
        results = explain_classification_predictions_batch(artifact, images)
    finally:
        for _, image in images:
            image.close()

    assert [r.key for r in results] == ["rouge/0.png", "bleu/0.png"]
    for r in results:
        assert r.error is None
        assert r.result is not None
        assert r.result.predicted_label in {"rouge", "bleu"}
        # Toujours la classe PRÉDITE en mode batch (jamais de target_label
        # par image — voir docstring de explain_classification_predictions_batch).
        assert r.result.target_label == r.result.predicted_label
        assert r.result.heatmap_png.startswith("data:image/png;base64,")


def test_batch_matches_single_image_explain_bit_for_bit(trained_artifact):
    """Le mode batch doit produire EXACTEMENT le même résultat que
    l'explication image par image — jamais une approximation différente
    entre les deux chemins, seule la performance change."""
    artifact, sample_image_path = trained_artifact
    with Image.open(sample_image_path) as single_image:
        single = explain_classification_prediction(artifact, single_image)

    with Image.open(sample_image_path) as batch_image:
        batch_results = explain_classification_predictions_batch(artifact, [("img", batch_image)])

    assert batch_results[0].result.predicted_label == single.predicted_label
    assert batch_results[0].result.probabilities == pytest.approx(single.probabilities)
    assert batch_results[0].result.heatmap_png == single.heatmap_png


def test_batch_isolates_a_failing_image_from_the_rest(trained_artifact, monkeypatch):
    """Une image individuellement en échec ne doit jamais faire échouer les
    autres — dégradation par image, jamais par lot entier."""
    import domains.vision.classification.services.gradcam as gradcam_module

    artifact, sample_image_path = trained_artifact
    good_image = Image.open(sample_image_path)
    poison_image = Image.open(sample_image_path)

    real_run_gradcam = gradcam_module._run_gradcam
    call_count = {"n": 0}

    def _flaky_run_gradcam(model, capture, class_names, image, target_label):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("panne simulée sur la première image")
        return real_run_gradcam(model, capture, class_names, image, target_label)

    monkeypatch.setattr(gradcam_module, "_run_gradcam", _flaky_run_gradcam)
    try:
        results = explain_classification_predictions_batch(artifact, [("poison", poison_image), ("bon", good_image)])
    finally:
        good_image.close()
        poison_image.close()

    assert results[0].key == "poison"
    assert results[0].result is None
    assert results[0].error is not None
    assert results[1].key == "bon"
    assert results[1].result is not None
    assert results[1].error is None


def test_batch_on_empty_list_returns_empty_without_crashing(trained_artifact):
    artifact, _ = trained_artifact
    assert explain_classification_predictions_batch(artifact, []) == []
