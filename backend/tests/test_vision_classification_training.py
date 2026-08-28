"""Tests de `services/vision_classification_training.py` (pilier Vision,
Lot 15 sous-lot B) — entraînement réel (pas mocké) sur un mini dataset
synthétique, même approche que `test_anomaly_training.py`. Epochs volontairement
réduits au minimum (temps réel CPU) — la correction du pipeline est vérifiée,
pas la qualité du modèle obtenu sur un jeu de données minuscule."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from domains.shared.ml_preprocessing import TrainingAbortedError
from domains.vision.classification.services.engine import (
    AUGMENTATION_PRESET_IDS,
    MIN_IMAGES_PER_CLASS_FOR_TRAINING,
    ClassificationConfig,
    augmentation_transforms,
    _class_weights,
    _should_stop_early,
    recommend_augmentation_preset,
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


def test_training_handles_three_classes(tmp_path):
    """Multiclasse (pas seulement binaire) — ImageFolder/CrossEntropyLoss/
    métriques macro généralisent nativement à N classes, vérifié ici
    explicitement plutôt que supposé."""
    _write_classification_dataset(tmp_path, {"classe_a": 8, "classe_b": 8, "classe_c": 8})
    config = ClassificationConfig(backbone_id="mobilenet_v3_small", num_epochs=1, batch_size=4)

    result = train_and_evaluate_classification(tmp_path, config, _noop_progress)

    assert result.class_names == ["classe_a", "classe_b", "classe_c"]
    assert len(result.confusion_matrix) == 3 and all(len(row) == 3 for row in result.confusion_matrix)
    assert result.model_artifact["class_names"] == ["classe_a", "classe_b", "classe_c"]


# ── Lot 6A, correctif I8 — pondération de classes, arrêt anticipé, scheduler ─


def test_class_weights_are_inversely_proportional_to_frequency():
    # 3 classes, comptes déséquilibrés parmi les indices d'entraînement
    # (classe 0 : 3 fois, classe 1 : 1 fois, classe 2 : 2 fois).
    targets = [0, 0, 0, 1, 2, 2]
    train_idx = [0, 1, 2, 3, 4, 5]
    weights = _class_weights(targets, train_idx, n_classes=3)

    assert weights.shape == (3,)
    # total=6, n_classes=3 : poids = 6/(3*compte) -> classe la moins
    # fréquente (1 exemple) doit avoir le poids le plus élevé.
    assert weights[1] > weights[2] > weights[0]
    assert weights[1].item() == pytest.approx(6 / (3 * 1))
    assert weights[0].item() == pytest.approx(6 / (3 * 3))


def test_class_weights_only_uses_the_training_split():
    """Les comptes de validation/test ne doivent JAMAIS influencer les
    poids — seul train_idx compte, même si targets couvre tout le dataset."""
    targets = [0, 0, 0, 0, 0, 1]  # classe 1 très minoritaire dans TOUT le dataset...
    train_idx = [0, 1, 5]  # ...mais équilibrée dans le split d'entraînement (2 vs 1)
    weights = _class_weights(targets, train_idx, n_classes=2)
    assert weights[0].item() == pytest.approx(3 / (2 * 2))
    assert weights[1].item() == pytest.approx(3 / (2 * 1))


def test_should_stop_early_triggers_at_patience_threshold():
    assert _should_stop_early(epochs_without_improvement=2, patience=2) is True
    assert _should_stop_early(epochs_without_improvement=1, patience=2) is False


def test_should_stop_early_disabled_when_patience_is_none():
    assert _should_stop_early(epochs_without_improvement=1000, patience=None) is False


def test_class_weighting_can_be_disabled(tmp_path):
    _write_classification_dataset(tmp_path, {"classe_a": 8, "classe_b": 8})
    config = ClassificationConfig(num_epochs=1, batch_size=4, class_weighting=False)

    result = train_and_evaluate_classification(tmp_path, config, _noop_progress)
    assert result.model_card["class_weighting_applied"] is False


def test_class_weighting_enabled_by_default(tmp_path):
    _write_classification_dataset(tmp_path, {"classe_a": 8, "classe_b": 8})
    config = ClassificationConfig(num_epochs=1, batch_size=4)

    result = train_and_evaluate_classification(tmp_path, config, _noop_progress)
    assert result.model_card["class_weighting_applied"] is True


def test_lr_scheduler_can_be_disabled(tmp_path):
    _write_classification_dataset(tmp_path, {"classe_a": 8, "classe_b": 8})
    config = ClassificationConfig(num_epochs=2, batch_size=4, use_lr_scheduler=False)

    result = train_and_evaluate_classification(tmp_path, config, _noop_progress)
    assert result.model_card["lr_scheduler_used"] is False


def test_early_stopping_disabled_runs_full_budget(tmp_path):
    _write_classification_dataset(tmp_path, {"classe_a": 8, "classe_b": 8})
    config = ClassificationConfig(num_epochs=2, batch_size=4, early_stopping_patience=None)

    result = train_and_evaluate_classification(tmp_path, config, _noop_progress)
    assert result.model_card["early_stopped"] is False
    assert result.model_card["num_epochs_run"] == 2


def test_model_card_reports_are_internally_consistent(tmp_path):
    """early_stopped=True implique num_epochs_run < num_epochs_requested —
    jamais l'inverse, quelle que soit la dynamique réelle de convergence
    (non déterministe sur un dataset synthétique minuscule)."""
    _write_classification_dataset(tmp_path, {"classe_a": 8, "classe_b": 8})
    config = ClassificationConfig(num_epochs=6, batch_size=4, early_stopping_patience=1)

    result = train_and_evaluate_classification(tmp_path, config, _noop_progress)
    if result.model_card["early_stopped"]:
        assert result.model_card["num_epochs_run"] < 6


# ── Lot 6A, correctif I9 — presets d'augmentation configurables ─────────────


@pytest.mark.parametrize("preset", AUGMENTATION_PRESET_IDS)
def test_every_preset_id_is_a_valid_transform_list(preset):
    transforms_list = augmentation_transforms(preset)
    assert isinstance(transforms_list, list)


def test_augmentation_presets_form_a_strict_progression():
    """Chaque niveau ajoute une transformation à celles du niveau
    précédent — jamais une combinaison disjointe (progression cohérente,
    prévisible pour l'utilisateur)."""
    counts = [len(augmentation_transforms(p)) for p in AUGMENTATION_PRESET_IDS]
    assert counts == sorted(counts)
    assert counts[0] == 0  # "aucune" : pas d'augmentation du tout


def test_unknown_augmentation_preset_raises():
    with pytest.raises(ValueError):
        augmentation_transforms("extreme")


def test_training_rejects_unknown_augmentation_preset(tmp_path):
    _write_classification_dataset(tmp_path, {"classe_a": 8, "classe_b": 8})
    config = ClassificationConfig(num_epochs=1, batch_size=4, augmentation_preset="extreme")
    with pytest.raises(TrainingAbortedError):
        train_and_evaluate_classification(tmp_path, config, _noop_progress)


@pytest.mark.parametrize("preset", AUGMENTATION_PRESET_IDS)
def test_training_runs_with_every_augmentation_preset(tmp_path, preset):
    _write_classification_dataset(tmp_path, {"classe_a": 8, "classe_b": 8})
    config = ClassificationConfig(num_epochs=1, batch_size=4, augmentation_preset=preset)

    result = train_and_evaluate_classification(tmp_path, config, _noop_progress)
    assert result.model_card["augmentation_preset"] == preset


def test_recommend_augmentation_preset_thresholds():
    assert recommend_augmentation_preset(5) == "forte"
    assert recommend_augmentation_preset(19) == "forte"
    assert recommend_augmentation_preset(20) == "standard"
    assert recommend_augmentation_preset(49) == "standard"
    assert recommend_augmentation_preset(50) == "legere"
    assert recommend_augmentation_preset(149) == "legere"
    assert recommend_augmentation_preset(150) == "aucune"
    assert recommend_augmentation_preset(10_000) == "aucune"


# ── Répartition personnalisée (Lot 6A) ──────────────────────────────────────


def test_custom_split_ratios_are_respected(tmp_path):
    # 40 images/classe (80 total), val_ratio=0.2/test_ratio=0.1 — comptes
    # vérifiés directement contre le comportement réel de train_test_split
    # en cascade (arrondi sklearn au ceil à chaque étage, jamais supposé).
    _write_classification_dataset(tmp_path, {"classe_a": 40, "classe_b": 40})
    config = ClassificationConfig(num_epochs=1, batch_size=4, val_ratio=0.2, test_ratio=0.1)

    result = train_and_evaluate_classification(tmp_path, config, _noop_progress)

    assert result.n_train + result.n_val + result.n_test == 80
    assert result.n_val == 16
    assert result.n_test == 9
    assert result.n_train == 55
    assert result.model_card["val_ratio"] == 0.2
    assert result.model_card["test_ratio"] == 0.1


def test_default_split_ratios_reproduce_historical_70_15_15(tmp_path):
    _write_classification_dataset(tmp_path, {"classe_a": 40, "classe_b": 40})
    config = ClassificationConfig(num_epochs=1, batch_size=4)  # val_ratio/test_ratio par défaut

    result = train_and_evaluate_classification(tmp_path, config, _noop_progress)

    assert result.n_val == 12
    assert result.n_test == 12
    assert result.n_train == 56


def test_training_rejects_split_leaving_less_than_10_percent_for_train(tmp_path):
    _write_classification_dataset(tmp_path, {"classe_a": 40, "classe_b": 40})
    config = ClassificationConfig(num_epochs=1, batch_size=4, val_ratio=0.5, test_ratio=0.45)
    with pytest.raises(TrainingAbortedError):
        train_and_evaluate_classification(tmp_path, config, _noop_progress)


# ── ROC/AUC binaire et multiclasse (Lot 6A, correctif 16G) ──────────────────


def test_binary_classification_produces_a_single_roc_curve(tmp_path):
    _write_classification_dataset(tmp_path, {"classe_a": 20, "classe_b": 20})
    config = ClassificationConfig(backbone_id="mobilenet_v3_small", num_epochs=1, batch_size=4)

    result = train_and_evaluate_classification(tmp_path, config, _noop_progress)

    assert set(result.roc_curves.keys()) == {"classe_b"}  # classe positive = index 1
    assert set(result.pr_curves.keys()) == {"classe_b"}
    curve = result.roc_curves["classe_b"]
    assert len(curve["fpr"]) == len(curve["tpr"])
    assert result.test_roc_auc is not None
    assert 0.0 <= result.test_roc_auc <= 1.0
    assert result.model_card["test_roc_auc"] == result.test_roc_auc


def test_multiclass_classification_produces_one_roc_curve_per_class(tmp_path):
    _write_classification_dataset(tmp_path, {"classe_a": 15, "classe_b": 15, "classe_c": 15})
    config = ClassificationConfig(backbone_id="mobilenet_v3_small", num_epochs=1, batch_size=4)

    result = train_and_evaluate_classification(tmp_path, config, _noop_progress)

    assert set(result.roc_curves.keys()) == {"classe_a", "classe_b", "classe_c"}
    assert set(result.pr_curves.keys()) == {"classe_a", "classe_b", "classe_c"}
    assert result.test_roc_auc is not None
    assert 0.0 <= result.test_roc_auc <= 1.0


# ── Calibration (onglet "Fiabilité", retour utilisateur : "d'autres
# fonctionnalités modernes que les autres plateformes n'offrent pas") ────


def test_binary_classification_produces_a_calibration_curve(tmp_path):
    _write_classification_dataset(tmp_path, {"classe_a": 20, "classe_b": 20})
    config = ClassificationConfig(backbone_id="mobilenet_v3_small", num_epochs=1, batch_size=4)

    result = train_and_evaluate_classification(tmp_path, config, _noop_progress)

    assert set(result.calibration.keys()) <= {"classe_b"}
    if result.calibration:
        curve = result.calibration["classe_b"]
        assert len(curve["mean_predicted"]) == len(curve["fraction_positive"])
        assert all(0.0 <= v <= 1.0 for v in curve["mean_predicted"])
        assert all(0.0 <= v <= 1.0 for v in curve["fraction_positive"])
    assert result.model_card["calibration_status"]["status"] == "ok"


def test_multiclass_classification_produces_calibration_per_class(tmp_path):
    _write_classification_dataset(tmp_path, {"classe_a": 15, "classe_b": 15, "classe_c": 15})
    config = ClassificationConfig(backbone_id="mobilenet_v3_small", num_epochs=1, batch_size=4)

    result = train_and_evaluate_classification(tmp_path, config, _noop_progress)

    assert set(result.calibration.keys()) <= {"classe_a", "classe_b", "classe_c"}
    assert result.model_card["calibration_status"]["status"] == "ok"


# ── Sélection représentative des exemples (Lot 6A, correctif §G.4) ─────────


def test_representative_sample_round_robins_across_groups():
    from domains.vision.classification.services.engine import _representative_sample

    items = ["a1", "a2", "a3", "a4", "b1", "c1", "c2"]
    groups = {"a1": "a", "a2": "a", "a3": "a", "a4": "a", "b1": "b", "c1": "c", "c2": "c"}

    sample = _representative_sample(items, lambda x: groups[x], limit=3)

    # Round-robin : un de chaque groupe d'abord (a1, b1, c1), jamais les 3
    # premiers de la liste (qui seraient tous du groupe "a").
    assert sample == ["a1", "b1", "c1"]


def test_representative_sample_exhausts_small_groups_gracefully():
    from domains.vision.classification.services.engine import _representative_sample

    items = ["a1", "a2", "a3", "a4", "a5", "b1"]
    groups = {"a1": "a", "a2": "a", "a3": "a", "a4": "a", "a5": "a", "b1": "b"}

    sample = _representative_sample(items, lambda x: groups[x], limit=4)

    assert len(sample) == 4
    assert "b1" in sample  # le seul élément du petit groupe doit apparaître
