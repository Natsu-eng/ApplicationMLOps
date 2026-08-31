"""Tests de domains/anomalies/services/inference.py (Lot 6B, §F.2 — noter une
nouvelle observation à partir d'une détection d'anomalies déjà entraînée)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from domains.anomalies.services.engine import AnomalyConfig, train_and_evaluate_anomalies
from domains.anomalies.services.inference import AnomalyInferenceError, score_anomaly, score_anomalies_batch

_NOOP = lambda step, pct: None  # noqa: E731


def _make_dataset_with_injected_outliers(n_normal: int = 95, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    normal = rng.normal(0, 1, (n_normal, 3))
    outliers = np.array(
        [[15.0, 15.0, 15.0], [14.0, -14.0, 0.0], [-13.0, 13.0, -13.0], [16.0, 0.0, -16.0], [0.0, 16.0, 16.0]]
    )
    return pd.DataFrame(np.vstack([normal, outliers]), columns=["a", "b", "c"])


def _train():
    df = _make_dataset_with_injected_outliers()
    return train_and_evaluate_anomalies(df[["a", "b", "c"]], AnomalyConfig(top_n=10, seed=42), _NOOP)


def test_new_point_far_from_the_normal_group_is_flagged_anomalous_by_both():
    result = _train()
    scored = score_anomaly(result.pipeline_bundle, result.feature_columns, {"a": 20.0, "b": 20.0, "c": 20.0})
    assert scored["is_anomaly_isolation_forest"] is True
    assert scored["is_anomaly_lof"] is True
    assert scored["is_anomaly_consensus"] is True
    assert scored["agreement"] == "both"
    assert scored["consensus_score"] > 0.9  # doit ressortir tout en haut du classement


def test_new_point_near_the_normal_group_center_is_not_anomalous():
    result = _train()
    scored = score_anomaly(result.pipeline_bundle, result.feature_columns, {"a": 0.0, "b": 0.0, "c": 0.0})
    assert scored["is_anomaly_isolation_forest"] is False
    assert scored["is_anomaly_lof"] is False
    assert scored["is_anomaly_consensus"] is False
    assert scored["agreement"] == "none"
    assert scored["consensus_score"] < 0.5


def test_consensus_score_is_consistent_with_training_time_ranking():
    """Une nouvelle observation IDENTIQUE à un outlier injecté doit obtenir un
    consensus proche de celui du même point vu à l'entraînement (même
    distribution de référence, même formule de rang percentile)."""
    result = _train()
    training_top1 = result.top_observations[0]
    training_row = {"a": 15.0, "b": 15.0, "c": 15.0}  # le premier outlier injecté
    scored = score_anomaly(result.pipeline_bundle, result.feature_columns, training_row)
    assert scored["consensus_score"] == pytest.approx(training_top1.consensus_score, abs=0.05)


def test_missing_feature_raises_actionable_error():
    result = _train()
    with pytest.raises(AnomalyInferenceError):
        score_anomaly(result.pipeline_bundle, result.feature_columns, {"a": 0.0, "b": 0.0})  # c manquant


def test_batch_scores_every_row_and_preserves_other_columns():
    result = _train()
    input_df = pd.DataFrame(
        {"a": [0.0, 20.0], "b": [0.0, 20.0], "c": [0.0, 20.0], "id_ligne": ["normal", "extreme"]}
    )
    scored = score_anomalies_batch(result.pipeline_bundle, result.feature_columns, input_df)
    assert list(scored["score_status"]) == ["scored", "scored"]
    assert scored.loc[0, "is_anomaly_consensus"] == False  # noqa: E712
    assert scored.loc[1, "is_anomaly_consensus"] == True  # noqa: E712
    assert "id_ligne" in scored.columns


def test_batch_reports_missing_column_without_crashing():
    result = _train()
    input_df = pd.DataFrame({"a": [0.0], "b": [0.0]})  # c manquant partout
    scored = score_anomalies_batch(result.pipeline_bundle, result.feature_columns, input_df)
    assert list(scored["score_status"]) == ["missing_column"]
