"""Tests de `services/vision_anomaly_training.py` (pilier Vision, Lot 15
sous-lot C) — entraînement réel (pas mocké) sur un mini dataset MVTec AD
synthétique, même approche que `test_vision_classification_training.py`."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from services.ml_preprocessing import TrainingAbortedError
from services.vision_anomaly_training import (
    MIN_TRAIN_GOOD_FOR_TRAINING,
    AnomalyVisionConfig,
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
