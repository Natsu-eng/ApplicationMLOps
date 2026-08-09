"""Tests de services/ml_training.py (Lot 3) — cœur du pipeline d'entraînement.

Essais Optuna volontairement réduits (3) pour un temps d'exécution
raisonnable en test — la logique exercée est identique à la production
(voir TrainingConfig).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from services.ml_preprocessing import split_dataset
from services.ml_training import TrainingConfig, train_and_evaluate

_FAST_CONFIG = TrainingConfig(optuna_trials=3, cv_folds=3)


def _make_regression_df(n=200, seed=42):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(50, 10, n)
    x2 = rng.normal(20, 5, n)
    y = 2.5 * x1 - 1.2 * x2 + rng.normal(0, 3, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "cible": y})


def _make_multiclass_df(n_per_class=40, seed=0):
    """Mêmes proportions qu'un jeu type Iris (3 classes équilibrées, features
    numériques) — reproduit le bug réel trouvé en usage réel : SHAP renvoie
    un tableau 3D en multiclasse selon la version/le backend, ce qui faisait
    planter `_compute_shap_summary` avec `IndexError: only integer scalar
    arrays can be converted to a scalar index` avant correction."""
    rng = np.random.default_rng(seed)
    rows = []
    for cls, center in enumerate([(5.0, 3.4), (6.0, 2.8), (6.6, 3.0)]):
        for _ in range(n_per_class):
            rows.append([rng.normal(center[0], 0.3), rng.normal(center[1], 0.3), cls])
    return pd.DataFrame(rows, columns=["f1", "f2", "cible"])


def test_regression_pipeline_end_to_end():
    df = _make_regression_df()
    split = split_dataset(df, "cible", ["x1", "x2"], "regression", None, 0.2, 42)
    result = train_and_evaluate(split, "regression", _FAST_CONFIG, lambda s, p: None)

    assert result.algorithm in ("LightGBM", "XGBoost", "CatBoost")
    assert result.metrics["r2_test"] > 0.8  # signal fort et peu bruité, doit être bien capté
    assert result.cqr is not None
    assert 0 <= result.cqr["empirical_coverage"] <= 1
    assert len(result.shap_summary) == 2


def test_multiclass_classification_shap_does_not_crash():
    """Test de non-régression pour le bug trouvé en usage réel (dataset Iris)."""
    df = _make_multiclass_df()
    split = split_dataset(df, "cible", ["f1", "f2"], "classification", None, 0.2, 42)
    result = train_and_evaluate(split, "classification", _FAST_CONFIG, lambda s, p: None)

    assert result.metrics["accuracy"] > 0.7
    assert len(result.shap_summary) == 2
    for entry in result.shap_summary:
        assert isinstance(entry["importance"], float)
    assert result.cqr is None  # pas de CQR en classification


def test_group_anti_leak_split_reflected_in_model_card():
    rng = np.random.default_rng(3)
    n = 150
    df = pd.DataFrame(
        {"x": rng.normal(size=n), "groupe": rng.integers(0, 30, n), "cible": rng.normal(size=n)}
    )
    split = split_dataset(df, "cible", ["x"], "regression", "groupe", 0.2, 42)
    result = train_and_evaluate(split, "regression", _FAST_CONFIG, lambda s, p: None)
    assert result.model_card["anti_leak_grouping"] is True
