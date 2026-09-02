"""Tests de domains/shared/drift.py — logique pure (PSI), données
synthétiques uniquement, même convention que test_model_verdict.py /
test_data_quality.py (ce module ne touche ni base ni entraînement réel)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from domains.shared.drift import (
    MIN_CURRENT_ROWS_FOR_DRIFT,
    compute_drift_report,
    compute_psi,
    psi_severity,
)

_RNG = np.random.default_rng(42)


# ── compute_psi — numérique ─────────────────────────────────────────────


def test_identical_numeric_distributions_have_near_zero_psi():
    ref = pd.Series(_RNG.normal(0, 1, 2000))
    cur = pd.Series(_RNG.normal(0, 1, 500))
    assert compute_psi(ref, cur) < 0.05


def test_fully_shifted_numeric_distribution_has_high_psi():
    ref = pd.Series(_RNG.normal(0, 1, 2000))
    cur = pd.Series(_RNG.normal(5, 1, 500))  # décalée de 5 écarts-types
    assert compute_psi(ref, cur) > 0.25


def test_constant_reference_returns_zero_without_crashing():
    ref = pd.Series([1.0] * 100)
    cur = pd.Series(_RNG.normal(0, 1, 100))
    assert compute_psi(ref, cur) == 0.0


def test_empty_current_returns_zero():
    ref = pd.Series(_RNG.normal(0, 1, 100))
    cur = pd.Series([], dtype=float)
    assert compute_psi(ref, cur) == 0.0


def test_current_values_outside_reference_range_are_captured():
    """Les bornes extrêmes de la référence sont remplacées par ±inf — une
    valeur jamais vue à l'entraînement doit compter dans un bucket, pas
    être silencieusement perdue par `np.histogram`."""
    ref = pd.Series(np.arange(100, dtype=float))  # 0..99
    cur = pd.Series([500.0] * 50)  # bien au-delà du max de la référence
    psi = compute_psi(ref, cur)
    assert psi > 0.25


# ── compute_psi — catégoriel ─────────────────────────────────────────────


def test_identical_categorical_distributions_have_near_zero_psi():
    ref = pd.Series(["a", "b", "c"] * 200)
    cur = pd.Series(["a", "b", "c"] * 50)
    assert compute_psi(ref, cur) < 0.05


def test_categorical_shift_towards_a_rare_reference_category_has_high_psi():
    ref = pd.Series(["a"] * 950 + ["b"] * 50)  # b rare à l'entraînement
    cur = pd.Series(["b"] * 100)  # devenue majoritaire en production
    assert compute_psi(ref, cur) > 0.25


def test_brand_new_category_unseen_in_reference_is_detected():
    ref = pd.Series(["a", "b"] * 200)
    cur = pd.Series(["c"] * 100)  # jamais vue à l'entraînement
    assert compute_psi(ref, cur) > 0.25


# ── psi_severity ──────────────────────────────────────────────────────────


def test_psi_severity_thresholds():
    assert psi_severity(0.0) == "stable"
    assert psi_severity(0.09) == "stable"
    assert psi_severity(0.1) == "modere"
    assert psi_severity(0.24) == "modere"
    assert psi_severity(0.25) == "significatif"
    assert psi_severity(1.0) == "significatif"


# ── compute_drift_report ──────────────────────────────────────────────────


def test_drift_report_flags_insufficient_data_below_threshold():
    reference_df = pd.DataFrame({"x": _RNG.normal(0, 1, 500)})
    current_df = pd.DataFrame({"x": _RNG.normal(0, 1, MIN_CURRENT_ROWS_FOR_DRIFT - 1)})
    report = compute_drift_report(reference_df, current_df, ["x"])
    assert report["insufficient_data"] is True
    assert report["features"] == []


def test_drift_report_computes_once_threshold_reached():
    reference_df = pd.DataFrame({"x": _RNG.normal(0, 1, 500)})
    current_df = pd.DataFrame({"x": _RNG.normal(0, 1, MIN_CURRENT_ROWS_FOR_DRIFT)})
    report = compute_drift_report(reference_df, current_df, ["x"])
    assert report["insufficient_data"] is False
    assert len(report["features"]) == 1
    assert report["features"][0]["feature"] == "x"


def test_drift_report_sorts_most_severe_features_first():
    reference_df = pd.DataFrame(
        {
            "stable_col": _RNG.normal(0, 1, 500),
            "shifted_col": _RNG.normal(0, 1, 500),
        }
    )
    current_df = pd.DataFrame(
        {
            "stable_col": _RNG.normal(0, 1, 200),
            "shifted_col": _RNG.normal(6, 1, 200),
        }
    )
    report = compute_drift_report(reference_df, current_df, ["stable_col", "shifted_col"])
    assert report["features"][0]["feature"] == "shifted_col"
    assert report["features"][0]["severity"] == "significatif"
    assert report["n_significant"] == 1
    assert report["n_moderate"] == 0


def test_drift_report_silently_skips_a_feature_column_absent_from_current():
    reference_df = pd.DataFrame({"x": _RNG.normal(0, 1, 500), "y": _RNG.normal(0, 1, 500)})
    current_df = pd.DataFrame({"x": _RNG.normal(0, 1, MIN_CURRENT_ROWS_FOR_DRIFT)})  # pas de "y"
    report = compute_drift_report(reference_df, current_df, ["x", "y"])
    assert [f["feature"] for f in report["features"]] == ["x"]
