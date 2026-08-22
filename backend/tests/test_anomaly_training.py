"""Tests de domains/anomalies/services/engine.py (Lot 14 — ML non supervisé)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import domains.anomalies.services.engine as anomaly_training
from domains.anomalies.services.engine import MAX_TOP_N, AnomalyConfig, train_and_evaluate_anomalies
from domains.shared.ml_preprocessing import TrainingAbortedError

_NOOP = lambda step, pct: None  # noqa: E731


def _make_dataset_with_injected_outliers(n_normal: int = 95, seed: int = 42) -> pd.DataFrame:
    """Un groupe normal serré + 5 points injectés loin dans l'espace —
    structure connue à l'avance : les 5 outliers injectés doivent apparaître
    en tête du classement par consensus, pas seulement "ça ne plante pas"."""
    rng = np.random.default_rng(seed)
    normal = rng.normal(0, 1, (n_normal, 3))
    outliers = np.array(
        [[15.0, 15.0, 15.0], [14.0, -14.0, 0.0], [-13.0, 13.0, -13.0], [16.0, 0.0, -16.0], [0.0, 16.0, 16.0]]
    )
    df = pd.DataFrame(np.vstack([normal, outliers]), columns=["a", "b", "c"])
    df["outlier_index"] = list(range(n_normal)) + list(range(n_normal, n_normal + 5))
    return df


def test_injected_outliers_rank_at_the_top_by_consensus():
    df = _make_dataset_with_injected_outliers()
    outlier_row_indices = set(range(95, 100))
    result = train_and_evaluate_anomalies(df[["a", "b", "c"]], AnomalyConfig(top_n=10, seed=42), _NOOP)
    top_5_row_indices = {o.row_index for o in result.top_observations[:5]}
    assert top_5_row_indices == outlier_row_indices


def test_most_extreme_outliers_have_both_agreement():
    df = _make_dataset_with_injected_outliers()
    result = train_and_evaluate_anomalies(df[["a", "b", "c"]], AnomalyConfig(top_n=10, seed=42), _NOOP)
    assert result.top_observations[0].agreement == "both"


def test_consensus_rate_never_exceeds_individual_rates():
    """Le consensus est une intersection (les deux algorithmes d'accord) —
    mathématiquement toujours <= au taux de chacun pris isolément."""
    df = _make_dataset_with_injected_outliers()
    result = train_and_evaluate_anomalies(df[["a", "b", "c"]], AnomalyConfig(seed=42), _NOOP)
    assert result.anomaly_rate_consensus <= result.anomaly_rate_isolation_forest + 1e-9
    assert result.anomaly_rate_consensus <= result.anomaly_rate_lof + 1e-9
    assert result.n_anomalies_consensus <= min(result.n_anomalies_isolation_forest, result.n_anomalies_lof)


def test_numeric_deviations_identify_the_perturbed_variable():
    """Un des points injectés est extrême uniquement sur `a`/`b` (pas `c`,
    proche de 0 pour celui-ci) — la déviation la plus forte doit pointer
    vers la bonne variable, pas une variable arbitraire."""
    df = _make_dataset_with_injected_outliers()
    result = train_and_evaluate_anomalies(df[["a", "b", "c"]], AnomalyConfig(top_n=10, seed=42), _NOOP)
    target = next(o for o in result.top_observations if o.row_index == 96)  # [14, -14, 0]
    assert set(target.numeric_deviations.keys()) >= {"a", "b"}
    assert "c" not in target.numeric_deviations or abs(target.numeric_deviations["c"]["z_score"]) < abs(
        target.numeric_deviations["a"]["z_score"]
    )


def test_top_n_clipped_to_max_top_n():
    df = _make_dataset_with_injected_outliers()
    result = train_and_evaluate_anomalies(df[["a", "b", "c"]], AnomalyConfig(top_n=99999, seed=42), _NOOP)
    assert len(result.top_observations) <= MAX_TOP_N
    assert len(result.top_observations) <= result.n_samples_used


def test_seed_is_wired_to_isolation_forest_estimator():
    df = _make_dataset_with_injected_outliers()
    result = train_and_evaluate_anomalies(df[["a", "b", "c"]], AnomalyConfig(seed=777), _NOOP)
    assert result.pipeline_bundle["isolation_forest"].random_state == 777


def test_progress_callback_increasing_and_reaches_100():
    df = _make_dataset_with_injected_outliers(n_normal=20)
    calls: list[int] = []
    train_and_evaluate_anomalies(df[["a", "b", "c"]], AnomalyConfig(seed=42), lambda step, pct: calls.append(pct))
    assert calls[0] < calls[-1]
    assert calls[-1] == 100
    assert all(0 <= p <= 100 for p in calls)


def test_too_few_rows_raises_actionable_error():
    df = pd.DataFrame({"a": list(range(5)), "b": list(range(5))}, dtype=float)
    with pytest.raises(TrainingAbortedError):
        train_and_evaluate_anomalies(df, AnomalyConfig(seed=42), _NOOP)


def test_all_identical_rows_degrade_gracefully_no_crash():
    """Aucune vraie structure d'anomalie (toutes les lignes identiques) —
    ne doit jamais lever une exception technique brute ; les z-scores
    doivent rester à 0 (division protégée), pas NaN."""
    df = pd.DataFrame({"a": [5.0] * 30, "b": [3.0] * 30})
    result = train_and_evaluate_anomalies(df, AnomalyConfig(top_n=5, seed=42), _NOOP)
    for obs in result.top_observations:
        for deviation in obs.numeric_deviations.values():
            assert deviation["z_score"] == 0.0


def test_score_histogram_counts_sum_to_n_samples_used():
    df = _make_dataset_with_injected_outliers()
    result = train_and_evaluate_anomalies(df[["a", "b", "c"]], AnomalyConfig(seed=42), _NOOP)
    assert sum(result.score_histogram["counts"]) == result.n_samples_used


def test_sampling_preserves_original_row_index_and_caps_at_limit(monkeypatch):
    monkeypatch.setattr(anomaly_training, "MAX_ROWS_FOR_ANOMALY", 50)
    df = _make_dataset_with_injected_outliers(n_normal=195)  # 200 lignes, > cap de test (50)
    result = train_and_evaluate_anomalies(df[["a", "b", "c"]], AnomalyConfig(seed=42), _NOOP)
    assert result.sampled is True
    assert result.n_samples_total == 200
    assert result.n_samples_used == 50
    row_indices = [o.row_index for o in result.top_observations]
    assert all(0 <= idx < 200 for idx in row_indices)


def test_categorical_flags_identify_rare_category():
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({"x": rng.normal(0, 1, n), "category": ["common"] * 97 + ["rare", "rare", "rare"]})
    result = train_and_evaluate_anomalies(df, AnomalyConfig(top_n=10, seed=42), _NOOP)
    rare_flagged = [o for o in result.top_observations if "category" in o.categorical_flags]
    assert any(o.categorical_flags["category"]["value"] == "rare" for o in rare_flagged)


# ── Lot 6B, §F.2 : exposition du taux de contamination ──────────────────────


def test_contamination_defaults_to_auto_when_not_configured():
    df = _make_dataset_with_injected_outliers()
    result = train_and_evaluate_anomalies(df[["a", "b", "c"]], AnomalyConfig(seed=42), _NOOP)
    assert result.model_card["contamination"] == "auto"


def test_explicit_contamination_is_reported_and_matches_configured_rate():
    """5 outliers injectés sur 100 lignes = 5 % — régler la contamination sur
    la vraie proportion doit rapprocher le taux d'anomalies détecté de cette
    valeur pour les DEUX méthodes (jusqu'ici codé en dur sur "auto", jamais
    réglable — Lot 6B, §F.2)."""
    df = _make_dataset_with_injected_outliers()
    result = train_and_evaluate_anomalies(df[["a", "b", "c"]], AnomalyConfig(seed=42, contamination=0.05), _NOOP)
    assert result.model_card["contamination"] == 0.05
    assert result.anomaly_rate_isolation_forest == pytest.approx(0.05, abs=0.02)
    assert result.anomaly_rate_lof == pytest.approx(0.05, abs=0.02)


def test_lower_contamination_flags_no_more_anomalies_than_default_auto():
    """Preuve que le paramètre est réellement câblé aux estimateurs (pas
    seulement stocké dans le model_card) : une contamination stricte (1 %)
    ne peut jamais flagger PLUS d'observations que le réglage "auto"."""
    df = _make_dataset_with_injected_outliers()
    result_auto = train_and_evaluate_anomalies(df[["a", "b", "c"]], AnomalyConfig(seed=42), _NOOP)
    result_strict = train_and_evaluate_anomalies(
        df[["a", "b", "c"]], AnomalyConfig(seed=42, contamination=0.01), _NOOP
    )
    assert result_strict.n_anomalies_isolation_forest <= result_auto.n_anomalies_isolation_forest
    assert result_strict.n_anomalies_lof <= result_auto.n_anomalies_lof
