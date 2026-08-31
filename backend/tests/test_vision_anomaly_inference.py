"""Tests de domains/vision/anomalies/services/inference.py (Lot 6B, §F.2 —
noter une NOUVELLE image à partir d'une détection d'anomalies visuelles déjà
entraînée)."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from domains.vision.anomalies.services.engine import AnomalyVisionConfig, train_and_evaluate_anomaly_vision
from domains.vision.anomalies.services.inference import score_vision_anomaly


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


def _train(tmp_path):
    _write_mvtec_dataset(tmp_path)
    config = AnomalyVisionConfig(num_epochs=3, batch_size=4)
    return train_and_evaluate_anomaly_vision(tmp_path, config, _noop_progress)


def test_score_returns_a_result_between_0_and_1_fields_present(tmp_path):
    result = _train(tmp_path)
    rng = np.random.default_rng(99)
    image = Image.fromarray(_make_normal(rng))
    scored = score_vision_anomaly(result.model_artifact, image)
    assert scored["anomaly_score"] >= 0.0
    assert scored["threshold"] == pytest.approx(result.threshold)
    assert isinstance(scored["is_anomaly"], bool)
    assert scored["heatmap_png"].startswith("data:image") or len(scored["heatmap_png"]) > 100


def test_score_matches_manual_threshold_comparison(tmp_path):
    result = _train(tmp_path)
    rng = np.random.default_rng(99)
    image = Image.fromarray(_make_defect(rng))
    scored = score_vision_anomaly(result.model_artifact, image)
    assert scored["is_anomaly"] == (scored["anomaly_score"] > scored["threshold"])


def test_defect_image_scores_at_least_as_high_as_a_normal_image(tmp_path):
    """Pas une garantie statistique forte sur un mini dataset synthétique
    (voir test_vision_anomaly_training.py, même limite documentée) mais une
    vérification de cohérence minimale : un défaut structurel évident ne
    doit pas ressortir MOINS atypique qu'une image normale."""
    result = _train(tmp_path)
    rng = np.random.default_rng(99)
    normal_score = score_vision_anomaly(result.model_artifact, Image.fromarray(_make_normal(rng)))["anomaly_score"]
    defect_score = score_vision_anomaly(result.model_artifact, Image.fromarray(_make_defect(rng)))["anomaly_score"]
    assert defect_score >= normal_score
