"""Tests de `services/vision_anomaly_training.py` (pilier Vision, Lot 15
sous-lot C) — entraînement réel (pas mocké) sur un mini dataset MVTec AD
synthétique, même approche que `test_vision_classification_training.py`."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from domains.shared.ml_preprocessing import TrainingAbortedError
from domains.vision.anomalies.services.engine import (
    MAX_MODELS_PER_COMPARISON,
    MIN_IMAGES_PER_CATEGORY_FOR_CALIBRATION_SPLIT,
    MIN_TRAIN_GOOD_FOR_TRAINING,
    AnomalyVisionConfig,
    _split_calibration_evaluation,
    train_and_compare_anomaly_models,
    train_and_evaluate_anomaly_vision,
)


def _make_normal(rng, size=(64, 64)):
    arr = np.full((size[1], size[0], 3), (120, 120, 120), dtype=np.uint8)
    noise = rng.integers(-10, 10, (size[1], size[0], 3))
    return np.clip(arr.astype(int) + noise, 0, 255).astype(np.uint8)


def _make_defect(rng, size=(64, 64)):
    arr = _make_normal(rng, size)
    arr[20:40, 20:40] = [220, 20, 20]
    return arr


def _write_mvtec_dataset(root, n_train_good=14, n_test_good=4, n_test_defect=4, seed=0):
    rng = np.random.default_rng(seed)
    (root / "train" / "good").mkdir(parents=True, exist_ok=True)
    for i in range(n_train_good):
        Image.fromarray(_make_normal(rng)).save(root / "train" / "good" / f"{i}.png")

    (root / "test" / "good").mkdir(parents=True, exist_ok=True)
    for i in range(n_test_good):
        Image.fromarray(_make_normal(rng)).save(root / "test" / "good" / f"{i}.png")

    (root / "test" / "scratch").mkdir(parents=True, exist_ok=True)
    for i in range(n_test_defect):
        Image.fromarray(_make_defect(rng)).save(root / "test" / "scratch" / f"{i}.png")
    return root


def _noop_progress(step: str, percent: int) -> None:
    pass


def test_training_produces_valid_result(tmp_path):
    _write_mvtec_dataset(tmp_path)
    config = AnomalyVisionConfig(num_epochs=3, batch_size=4)

    result = train_and_evaluate_anomaly_vision(tmp_path, config, _noop_progress)

    assert result.n_train + result.n_val == 14
    assert result.n_test == 8
    assert len(result.history) == 3
    assert 0.0 <= result.roc_auc <= 1.0
    assert 0.0 <= result.test_accuracy <= 1.0
    assert len(result.confusion_matrix) == 2 and len(result.confusion_matrix[0]) == 2
    assert sum(sum(row) for row in result.confusion_matrix) == result.n_test
    assert len(result.examples) > 0
    assert "state_dict" in result.model_artifact
    assert result.model_card["defect_categories"] == ["scratch"]


def test_examples_have_aligned_heatmaps_and_masks(tmp_path):
    _write_mvtec_dataset(tmp_path)
    config = AnomalyVisionConfig(num_epochs=2, batch_size=4)

    result = train_and_evaluate_anomaly_vision(tmp_path, config, _noop_progress)

    for example in result.examples:
        assert example.heatmap_png.startswith("data:image/png;base64,")
        assert example.mask_png.startswith("data:image/png;base64,")
        assert example.defect_category in {"good", "scratch"}
        assert example.true_label == (0 if example.defect_category == "good" else 1)
        assert "\\" not in example.relative_path  # portable JSON/URL — voir test_vision_classification_training.py


def test_examples_sorted_by_descending_anomaly_score(tmp_path):
    _write_mvtec_dataset(tmp_path)
    config = AnomalyVisionConfig(num_epochs=2, batch_size=4)

    result = train_and_evaluate_anomaly_vision(tmp_path, config, _noop_progress)
    scores = [e.anomaly_score for e in result.examples]
    assert scores == sorted(scores, reverse=True)


def test_raises_when_train_good_below_minimum(tmp_path):
    _write_mvtec_dataset(tmp_path, n_train_good=MIN_TRAIN_GOOD_FOR_TRAINING - 1)
    config = AnomalyVisionConfig(num_epochs=1)
    with pytest.raises(TrainingAbortedError):
        train_and_evaluate_anomaly_vision(tmp_path, config, _noop_progress)


def test_time_budget_stops_training_early(tmp_path):
    _write_mvtec_dataset(tmp_path)
    config = AnomalyVisionConfig(num_epochs=10, batch_size=4, max_training_seconds=0)

    result = train_and_evaluate_anomaly_vision(tmp_path, config, _noop_progress)

    assert result.model_card["time_capped"] is True
    assert result.model_card["num_epochs_run"] < 10


def test_split_calibration_evaluation_disjoint_when_enough_images():
    """Correctif C2 (AUDIT_DATALAB_2026-08-16.md) — preuve directe que le
    seuil est calibré sur des images qui ne servent pas au calcul des
    métriques rapportées : les deux sous-ensembles sont disjoints et
    couvrent exactement les indices d'origine."""
    categories = ["good"] * 8 + ["scratch"] * 8
    calibration_idx, evaluation_idx, biased = _split_calibration_evaluation(categories, seed=0)
    assert biased is False
    assert set(calibration_idx).isdisjoint(evaluation_idx)
    assert sorted(calibration_idx + evaluation_idx) == list(range(16))
    # 50/50 stratifié sur deux catégories de taille paire égale : exactement 8/8.
    assert len(calibration_idx) == 8
    assert len(evaluation_idx) == 8


def test_split_calibration_evaluation_falls_back_when_category_too_small():
    """Le cas "trop petit" doit être un nombre, pas une intuition — en
    dessous de MIN_IMAGES_PER_CATEGORY_FOR_CALIBRATION_SPLIT pour UNE SEULE
    catégorie, repli explicite sur les mêmes indices des deux côtés
    (biaisé, mais honnêtement signalé), jamais un split déséquilibré
    silencieux."""
    categories = ["good"] * 8 + ["scratch"] * (MIN_IMAGES_PER_CATEGORY_FOR_CALIBRATION_SPLIT - 1)
    calibration_idx, evaluation_idx, biased = _split_calibration_evaluation(categories, seed=0)
    assert biased is True
    expected = list(range(len(categories)))
    assert calibration_idx == expected
    assert evaluation_idx == expected


def test_calibration_split_used_end_to_end_when_dataset_large_enough(tmp_path):
    """Bout en bout : avec assez d'images de test par catégorie, le seuil
    est calibré sur une partie de test/, les métriques rapportées viennent
    de l'autre partie — jamais du même sous-ensemble."""
    _write_mvtec_dataset(
        tmp_path, n_test_good=MIN_IMAGES_PER_CATEGORY_FOR_CALIBRATION_SPLIT, n_test_defect=MIN_IMAGES_PER_CATEGORY_FOR_CALIBRATION_SPLIT
    )
    config = AnomalyVisionConfig(num_epochs=2, batch_size=4)

    result = train_and_evaluate_anomaly_vision(tmp_path, config, _noop_progress)

    assert result.model_card["threshold_calibration_status"] == "ok"
    assert result.model_card["threshold_calibration_message"] is None
    assert result.n_calibration is not None and result.n_evaluation is not None
    assert result.n_calibration + result.n_evaluation == result.n_test
    assert result.n_calibration > 0 and result.n_evaluation > 0
    # Chaque exemple présenté vient de l'évaluation — jamais une image ayant
    # servi à calibrer le seuil présentée comme une prédiction non vue.
    assert len(result.examples) > 0


def test_calibration_falls_back_and_flags_bias_on_small_dataset(tmp_path):
    """Le jeu de test par défaut des fixtures (4 good + 4 scratch) est sous
    le plancher MIN_IMAGES_PER_CATEGORY_FOR_CALIBRATION_SPLIT=6 — repli
    attendu, explicitement signalé, jamais silencieux. C'est le cas
    FRÉQUENT sur MVTec AD réel (plusieurs sous-types de défaut ont moins de
    10 images de test), pas un cas rare à traiter à la légère."""
    _write_mvtec_dataset(tmp_path, n_test_good=4, n_test_defect=4)
    config = AnomalyVisionConfig(num_epochs=2, batch_size=4)

    result = train_and_evaluate_anomaly_vision(tmp_path, config, _noop_progress)

    assert result.model_card["threshold_calibration_status"] == "degraded"
    assert isinstance(result.model_card["threshold_calibration_message"], str)
    assert result.n_calibration == result.n_evaluation == result.n_test


def test_threshold_is_calibrated_not_a_fixed_percentile(tmp_path):
    """Correctif du bug #7/#12 (seuil fixe non calibré) : le seuil doit
    varier avec les données réelles (J de Youden sur test/), pas être une
    constante indépendante du dataset."""
    tmp_a = tmp_path / "a"
    tmp_b = tmp_path / "b"
    _write_mvtec_dataset(tmp_a, seed=1)
    _write_mvtec_dataset(tmp_b, seed=99)
    config = AnomalyVisionConfig(num_epochs=2, batch_size=4)

    result_a = train_and_evaluate_anomaly_vision(tmp_a, config, _noop_progress)
    result_b = train_and_evaluate_anomaly_vision(tmp_b, config, _noop_progress)

    assert result_a.threshold != result_b.threshold


# ── Diagnostics supplémentaires (retour utilisateur : "rendre l'onglet
# anomalies aussi riche/transparent que la classification" + "d'autres
# fonctionnalités modernes que les autres plateformes n'offrent pas") ────


def test_roc_curve_is_computed_on_evaluation_only(tmp_path):
    _write_mvtec_dataset(
        tmp_path, n_test_good=MIN_IMAGES_PER_CATEGORY_FOR_CALIBRATION_SPLIT, n_test_defect=MIN_IMAGES_PER_CATEGORY_FOR_CALIBRATION_SPLIT
    )
    config = AnomalyVisionConfig(num_epochs=2, batch_size=4)

    result = train_and_evaluate_anomaly_vision(tmp_path, config, _noop_progress)

    assert set(result.roc_curves.keys()) == {"Défaut"}
    curve = result.roc_curves["Défaut"]
    assert len(curve["fpr"]) == len(curve["tpr"])
    assert all(0.0 <= v <= 1.0 for v in curve["fpr"])
    assert all(0.0 <= v <= 1.0 for v in curve["tpr"])
    assert set(result.pr_curves.keys()) == {"Défaut"}
    pr = result.pr_curves["Défaut"]
    assert len(pr["precision"]) == len(pr["recall"])


def test_score_histogram_separates_normal_and_defect_counts(tmp_path):
    _write_mvtec_dataset(tmp_path)
    config = AnomalyVisionConfig(num_epochs=2, batch_size=4)

    result = train_and_evaluate_anomaly_vision(tmp_path, config, _noop_progress)

    hist = result.score_histogram
    assert set(hist.keys()) == {"bin_edges", "normal_counts", "defect_counts"}
    assert len(hist["bin_edges"]) == len(hist["normal_counts"]) + 1 == len(hist["defect_counts"]) + 1
    # Chaque image de l'évaluation est comptée exactement une fois, répartie
    # entre les deux histogrammes selon son label réel.
    assert sum(hist["normal_counts"]) + sum(hist["defect_counts"]) == result.n_evaluation


def test_category_breakdown_covers_every_category_with_full_evaluation_count(tmp_path):
    """Le taux de détection par catégorie doit porter sur la TOTALITÉ de
    l'évaluation, pas seulement les `MAX_EXAMPLES` exemples affichés."""
    _write_mvtec_dataset(tmp_path, n_test_good=10, n_test_defect=10)
    config = AnomalyVisionConfig(num_epochs=2, batch_size=4)

    result = train_and_evaluate_anomaly_vision(tmp_path, config, _noop_progress)

    categories = {row["category"] for row in result.category_breakdown}
    assert categories == {"good", "scratch"}
    total_n = sum(row["n"] for row in result.category_breakdown)
    assert total_n == result.n_evaluation
    for row in result.category_breakdown:
        assert 0.0 <= row["detection_rate"] <= 1.0


def test_diagnostics_persisted_even_in_biased_fallback_mode(tmp_path):
    """Repli biaisé (calibration == évaluation, jeu de test minuscule) : les
    3 diagnostics doivent quand même être calculés — jamais None/vides sans
    raison alors que le calcul reste possible sur les mêmes données."""
    _write_mvtec_dataset(tmp_path, n_test_good=4, n_test_defect=4)
    config = AnomalyVisionConfig(num_epochs=2, batch_size=4)

    result = train_and_evaluate_anomaly_vision(tmp_path, config, _noop_progress)

    assert result.roc_curves
    assert result.pr_curves
    assert result.score_histogram
    assert result.category_breakdown


# ── Mode expert : comparatif d'architectures (retour utilisateur direct,
# même parité que le comparatif de backbones côté classification) ──────────


def test_compare_models_returns_the_candidate_with_lowest_best_val_loss(tmp_path):
    _write_mvtec_dataset(tmp_path)
    base_config = AnomalyVisionConfig(num_epochs=1, batch_size=4)
    model_ids = ["conv_autoencoder", "denoising_autoencoder"]

    result = train_and_compare_anomaly_models(tmp_path, base_config, model_ids, _noop_progress)

    candidates = result.model_card["candidates"]
    assert result.model_card["comparison_mode"] is True
    assert {c["model_id"] for c in candidates} == set(model_ids)
    selected = [c for c in candidates if c["selected"]]
    assert len(selected) == 1
    assert selected[0]["model_id"] == min(candidates, key=lambda c: c["best_val_loss"])["model_id"]
    assert result.model_id == selected[0]["model_id"]


def test_compare_models_requires_at_least_two_candidates(tmp_path):
    _write_mvtec_dataset(tmp_path)
    config = AnomalyVisionConfig(num_epochs=1, batch_size=4)
    with pytest.raises(TrainingAbortedError):
        train_and_compare_anomaly_models(tmp_path, config, ["conv_autoencoder"], _noop_progress)


def test_compare_models_rejects_more_than_the_maximum(tmp_path):
    _write_mvtec_dataset(tmp_path)
    config = AnomalyVisionConfig(num_epochs=1, batch_size=4)
    too_many = ["conv_autoencoder", "denoising_autoencoder", "conv_vae", "conv_autoencoder"][
        : MAX_MODELS_PER_COMPARISON + 1
    ]
    with pytest.raises(TrainingAbortedError):
        train_and_compare_anomaly_models(tmp_path, config, too_many, _noop_progress)


def test_compare_models_rejects_duplicates(tmp_path):
    _write_mvtec_dataset(tmp_path)
    config = AnomalyVisionConfig(num_epochs=1, batch_size=4)
    with pytest.raises(TrainingAbortedError):
        train_and_compare_anomaly_models(
            tmp_path, config, ["conv_autoencoder", "conv_autoencoder"], _noop_progress
        )


def test_weight_decay_is_applied_without_error_and_reported_honestly(tmp_path):
    _write_mvtec_dataset(tmp_path)
    config = AnomalyVisionConfig(num_epochs=1, batch_size=4, weight_decay=0.01)

    result = train_and_evaluate_anomaly_vision(tmp_path, config, _noop_progress)

    assert result.model_card["weight_decay"] == 0.01
