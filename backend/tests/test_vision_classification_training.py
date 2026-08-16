"""Tests de `services/vision_classification_training.py` (pilier Vision,
Lot 15 sous-lot B) — entraînement réel (pas mocké) sur un mini dataset
synthétique, même approche que `test_anomaly_training.py`. Epochs volontairement
réduits au minimum (temps réel CPU) — la correction du pipeline est vérifiée,
pas la qualité du modèle obtenu sur un jeu de données minuscule."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from services.ml_preprocessing import TrainingAbortedError
from services.vision_classification_training import (
    MIN_IMAGES_PER_CLASS_FOR_TRAINING,
    ClassificationConfig,
    train_and_evaluate_classification,
)


def _write_classification_dataset(root, class_counts: dict[str, int], size=(48, 48)):
    rng = np.random.default_rng(0)
    colors = {"classe_a": (220, 20, 20), "classe_b": (20, 20, 220), "classe_c": (20, 220, 20)}
    for class_name, count in class_counts.items():
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        base_color = colors.get(class_name, (128, 128, 128))
        for i in range(count):
            noise = rng.integers(-20, 20, (size[1], size[0], 3))
            arr = np.clip(np.array(base_color) + noise, 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(class_dir / f"{i}.png")
    return root


def _noop_progress(step: str, percent: int) -> None:
    pass


def test_training_produces_valid_result(tmp_path):
    _write_classification_dataset(tmp_path, {"classe_a": 8, "classe_b": 8})
    config = ClassificationConfig(backbone_id="mobilenet_v3_small", num_epochs=1, batch_size=4)

    result = train_and_evaluate_classification(tmp_path, config, _noop_progress)

    assert result.class_names == ["classe_a", "classe_b"]
    assert result.n_train + result.n_val + result.n_test == 16
    assert len(result.history) == 1
    assert 0.0 <= result.test_accuracy <= 1.0
    assert len(result.confusion_matrix) == 2 and len(result.confusion_matrix[0]) == 2
    assert sum(sum(row) for row in result.confusion_matrix) == result.n_test
    assert "state_dict" in result.model_artifact
    assert result.model_artifact["class_names"] == ["classe_a", "classe_b"]
    assert result.model_card["time_capped"] is False


def test_raises_with_fewer_than_two_classes(tmp_path):
    _write_classification_dataset(tmp_path, {"classe_a": 10})
    config = ClassificationConfig(num_epochs=1)
    with pytest.raises(TrainingAbortedError):
        train_and_evaluate_classification(tmp_path, config, _noop_progress)


def test_raises_when_class_below_minimum(tmp_path):
    _write_classification_dataset(
        tmp_path, {"classe_a": MIN_IMAGES_PER_CLASS_FOR_TRAINING - 1, "classe_b": MIN_IMAGES_PER_CLASS_FOR_TRAINING}
    )
    config = ClassificationConfig(num_epochs=1)
    with pytest.raises(TrainingAbortedError, match="classe_a"):
        train_and_evaluate_classification(tmp_path, config, _noop_progress)


def test_time_budget_stops_training_early(tmp_path):
    _write_classification_dataset(tmp_path, {"classe_a": 8, "classe_b": 8})
    config = ClassificationConfig(num_epochs=5, batch_size=4, max_training_seconds=0)

    result = train_and_evaluate_classification(tmp_path, config, _noop_progress)

    assert result.model_card["time_capped"] is True
    assert result.model_card["num_epochs_run"] < 5


def test_unfreeze_after_epoch_runs_without_error(tmp_path):
    _write_classification_dataset(tmp_path, {"classe_a": 8, "classe_b": 8})
    config = ClassificationConfig(num_epochs=2, batch_size=4, freeze_backbone=True, unfreeze_after_epoch=1)

    result = train_and_evaluate_classification(tmp_path, config, _noop_progress)

    assert len(result.history) == 2
    assert result.model_card["unfreeze_after_epoch"] == 1


def test_examples_include_errors_when_present(tmp_path):
    _write_classification_dataset(tmp_path, {"classe_a": 8, "classe_b": 8})
    config = ClassificationConfig(num_epochs=1, batch_size=4)

    result = train_and_evaluate_classification(tmp_path, config, _noop_progress)

    assert len(result.examples) > 0
    for example in result.examples:
        assert example.true_label in {"classe_a", "classe_b"}
        assert example.predicted_label in {"classe_a", "classe_b"}
        assert example.correct == (example.true_label == example.predicted_label)


def test_example_relative_path_uses_forward_slashes(tmp_path):
    """`relative_path` doit être portable (URL/JSON) — jamais un antislash
    Windows (`str()` nu sur un `Path` produit des antislashs sous Windows ;
    bug réel trouvé en testant l'app réelle en local sous Windows)."""
    _write_classification_dataset(tmp_path, {"classe_a": 8, "classe_b": 8})
    config = ClassificationConfig(num_epochs=1, batch_size=4)

    result = train_and_evaluate_classification(tmp_path, config, _noop_progress)

    assert len(result.examples) > 0
    for example in result.examples:
        assert "\\" not in example.relative_path
