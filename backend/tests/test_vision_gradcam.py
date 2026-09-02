"""Tests de `services/vision_gradcam.py` (pilier Vision, Lot 15 sous-lot D)
— Grad-CAM réel (pas mocké) sur un modèle réellement entraîné, même
approche que les autres modules du pilier Vision."""
from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image

from domains.vision.classification.services.engine import ClassificationConfig, train_and_evaluate_classification
from domains.vision.classification.services.gradcam import (
    MIN_IMAGES_FOR_SYNTHESIS,
    GradCamBatchItemResult,
    GradCamError,
    GradCamResult,
    compute_border_attention_fraction,
    explain_classification_prediction,
    explain_classification_predictions_batch,
    synthesize_attention_pattern,
)
from domains.vision.classification.services.registry import CLASSIFICATION_BACKBONE_REGISTRY


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
    with Image.open(sample_image_path) as image, pytest.raises(GradCamError):
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

    def _flaky_run_gradcam(model, capture, class_names, image, target_label, image_size):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("panne simulée sur la première image")
        return real_run_gradcam(model, capture, class_names, image, target_label, image_size)

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


# ── Mode expert : résolution d'entrée (retour utilisateur direct — "vision
# n'offre pas de réduire/augmenter la taille des images") ──────────────────


def test_explain_reconstructs_the_resolution_the_model_was_actually_trained_at(tmp_path):
    """Régression ciblée : un modèle entraîné à une résolution NON standard
    (64, pas le défaut 224) doit rester explicable — `_run_gradcam` doit lire
    `artifact["image_size"]`, jamais un 224 en dur, sous peine d'un mismatch
    de shape (backbone gelé attend l'entrée qu'il a vue à l'entraînement)."""
    root = tmp_path / "dataset"
    _write_classification_dataset(root)
    config = ClassificationConfig(
        backbone_id="mobilenet_v3_small", num_epochs=1, batch_size=4, freeze_backbone=True, image_size=64
    )
    result = train_and_evaluate_classification(root, config, lambda step, pct: None)

    with Image.open(root / "rouge" / "0.png") as image:
        explanation = explain_classification_prediction(result.model_artifact, image)

    assert explanation.predicted_label in {"rouge", "bleu"}


def test_explain_falls_back_to_224_for_artifacts_predating_the_image_size_field(trained_artifact):
    """Rétrocompatibilité par absence (même motif que les autres champs
    ajoutés à `model_card`/`model_artifact` dans ce projet) — un artefact
    d'un modèle entraîné avant ce correctif n'a pas la clé `image_size`."""
    artifact, sample_image_path = trained_artifact
    artifact_without_image_size = {k: v for k, v in artifact.items() if k != "image_size"}
    with Image.open(sample_image_path) as image:
        explanation = explain_classification_prediction(artifact_without_image_size, image)

    assert explanation.predicted_label in {"rouge", "bleu"}


# ── Synthèse agrégée (retour d'évaluation d'une maquette externe : "pas
# juste une heatmap par image, une observation transversale sur ce qui
# cloche") ─────────────────────────────────────────────────────────────


def test_border_attention_fraction_is_high_for_a_border_concentrated_map():
    cam = np.zeros((7, 7))
    cam[0, :] = 1.0  # toute l'attention sur la première rangée (bordure)
    fraction = compute_border_attention_fraction(cam)
    assert fraction == pytest.approx(1.0)


def test_border_attention_fraction_is_zero_for_a_center_concentrated_map():
    cam = np.zeros((7, 7))
    cam[3, 3] = 1.0  # toute l'attention sur la cellule centrale exacte
    fraction = compute_border_attention_fraction(cam)
    assert fraction == pytest.approx(0.0)


def test_border_attention_fraction_handles_an_all_zero_map_without_crashing():
    cam = np.zeros((7, 7))
    assert compute_border_attention_fraction(cam) == 0.0


def _fake_item(key: str, border_fraction: float | None, shape=(7, 7)) -> GradCamBatchItemResult:
    if border_fraction is None:
        return GradCamBatchItemResult(key=key, result=None, error="échec simulé")
    result = GradCamResult(
        predicted_label="rouge",
        probabilities={"rouge": 0.9, "bleu": 0.1},
        target_label="rouge",
        heatmap_png="data:image/png;base64,",
        border_attention_fraction=border_fraction,
        cam_map_shape=shape,
    )
    return GradCamBatchItemResult(key=key, result=result, error=None)


def test_synthesis_returns_none_below_the_minimum_sample_size():
    items = [_fake_item(str(i), 1.0) for i in range(MIN_IMAGES_FOR_SYNTHESIS - 1)]
    assert synthesize_attention_pattern(items) is None


def test_synthesis_detects_a_majority_border_bias():
    # Bordure maximale (1.0) pour tous — largement au-dessus de la
    # référence "au hasard" (~0,82 sur une grille 7x7 à 20 % de bordure).
    items = [_fake_item(str(i), 1.0) for i in range(MIN_IMAGES_FOR_SYNTHESIS)]
    synthesis = synthesize_attention_pattern(items)
    assert synthesis is not None
    assert synthesis.n_images == MIN_IMAGES_FOR_SYNTHESIS
    assert synthesis.n_border_biased == MIN_IMAGES_FOR_SYNTHESIS
    assert synthesis.border_biased_fraction == pytest.approx(1.0)
    assert synthesis.has_notable_pattern is True
    assert str(MIN_IMAGES_FOR_SYNTHESIS) in synthesis.observation


def test_synthesis_reports_no_notable_pattern_when_attention_stays_central():
    # Bordure nulle (0.0) pour tous — bien en dessous de la référence "au
    # hasard", jamais classée en biais de bordure.
    items = [_fake_item(str(i), 0.0) for i in range(MIN_IMAGES_FOR_SYNTHESIS)]
    synthesis = synthesize_attention_pattern(items)
    assert synthesis is not None
    assert synthesis.n_border_biased == 0
    assert synthesis.has_notable_pattern is False


def test_synthesis_ignores_failed_items_when_counting_the_sample():
    successful = [_fake_item(str(i), 1.0) for i in range(MIN_IMAGES_FOR_SYNTHESIS)]
    failed = [_fake_item("echec", None)]
    synthesis = synthesize_attention_pattern(successful + failed)
    assert synthesis is not None
    assert synthesis.n_images == MIN_IMAGES_FOR_SYNTHESIS  # l'échec n'est pas compté


def test_synthesis_over_a_real_batch_of_explained_images(trained_artifact):
    """Bout-en-bout — la synthèse doit fonctionner sur de VRAIES cartes
    Grad-CAM produites par un modèle réellement entraîné, pas seulement sur
    des données synthétiques."""
    artifact, sample_image_path = trained_artifact
    root = sample_image_path.parent.parent
    paths = [
        (f"rouge/{i}.png", root / "rouge" / f"{i}.png") for i in range(3)
    ] + [
        (f"bleu/{i}.png", root / "bleu" / f"{i}.png") for i in range(3)
    ]
    images = [(key, Image.open(path)) for key, path in paths]
    try:
        results = explain_classification_predictions_batch(artifact, images)
    finally:
        for _, image in images:
            image.close()

    synthesis = synthesize_attention_pattern(results)
    assert synthesis is not None
    assert synthesis.n_images == 6
    assert 0.0 <= synthesis.border_biased_fraction <= 1.0
    assert 0.0 <= synthesis.area_fraction_border <= 1.0
