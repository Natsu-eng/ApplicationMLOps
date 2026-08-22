"""Tests de domains/clustering/services/inference.py (Lot 6B, §F.2 — assignation
d'une nouvelle observation à un clustering déjà entraîné)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from domains.clustering.services.inference import ClusterInferenceError, assign_cluster
from domains.clustering.services.engine import ClusteringConfig, train_and_evaluate_clustering

_NOOP = lambda step, pct: None  # noqa: E731


def _make_three_blobs_df(n_per_group: int = 50, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    group = np.repeat([0, 1, 2], n_per_group)
    centers = {0: (0, 0), 1: (12, 12), 2: (0, 12)}
    x1 = np.array([centers[g][0] for g in group]) + rng.normal(0, 0.5, len(group))
    x2 = np.array([centers[g][1] for g in group]) + rng.normal(0, 0.5, len(group))
    return pd.DataFrame({"x1": x1, "x2": x2})


def test_kmeans_assignment_is_exact_and_matches_nearest_center():
    df = _make_three_blobs_df()
    result = train_and_evaluate_clustering(df, ClusteringConfig(seed=42, algorithm_ids=["kmeans"]), _NOOP)
    assignment = assign_cluster(result.pipeline_bundle, result.feature_columns, {"x1": 0.0, "x2": 0.0})
    assert assignment["assignment_method"] == "exact"
    assert assignment["is_noise"] is False
    assert assignment["cluster_id"] is not None


def test_hierarchical_assignment_is_approximate_centroid():
    df = _make_three_blobs_df()
    result = train_and_evaluate_clustering(df, ClusteringConfig(seed=42, algorithm_ids=["hierarchical"]), _NOOP)
    assert "centroids" in result.pipeline_bundle
    assignment = assign_cluster(result.pipeline_bundle, result.feature_columns, {"x1": 12.0, "x2": 12.0})
    assert assignment["assignment_method"] == "approximate_centroid"
    assert assignment["cluster_id"] is not None


def test_dbscan_assignment_flags_far_point_as_noise():
    df = _make_three_blobs_df()
    result = train_and_evaluate_clustering(df, ClusteringConfig(seed=42, algorithm_ids=["dbscan"]), _NOOP)
    assert "core_points" in result.pipeline_bundle
    # Point très loin de tous les groupes construits (0,0)/(12,12)/(0,12).
    assignment = assign_cluster(result.pipeline_bundle, result.feature_columns, {"x1": 500.0, "x2": 500.0})
    assert assignment["assignment_method"] == "approximate_nearest_core"
    assert assignment["is_noise"] is True
    assert assignment["cluster_id"] is None


def test_dbscan_assignment_finds_cluster_for_point_near_a_group():
    df = _make_three_blobs_df()
    result = train_and_evaluate_clustering(df, ClusteringConfig(seed=42, algorithm_ids=["dbscan"]), _NOOP)
    assignment = assign_cluster(result.pipeline_bundle, result.feature_columns, {"x1": 0.1, "x2": 0.1})
    assert assignment["assignment_method"] == "approximate_nearest_core"
    assert assignment["is_noise"] is False
    assert assignment["cluster_id"] is not None


def test_missing_feature_raises_actionable_error():
    df = _make_three_blobs_df()
    result = train_and_evaluate_clustering(df, ClusteringConfig(seed=42, algorithm_ids=["kmeans"]), _NOOP)
    with pytest.raises(ClusterInferenceError):
        assign_cluster(result.pipeline_bundle, result.feature_columns, {"x1": 0.0})  # x2 manquant


def test_bundle_without_assignment_data_degrades_to_unsupported():
    """Rétrocompatibilité par absence — un clustering hiérarchique entraîné
    AVANT ce lot n'a pas de clé `centroids` dans son pipeline_bundle."""
    df = _make_three_blobs_df()
    result = train_and_evaluate_clustering(df, ClusteringConfig(seed=42, algorithm_ids=["hierarchical"]), _NOOP)
    legacy_bundle = {k: v for k, v in result.pipeline_bundle.items() if k != "centroids"}
    assignment = assign_cluster(legacy_bundle, result.feature_columns, {"x1": 0.0, "x2": 0.0})
    assert assignment["assignment_method"] == "unsupported"
    assert assignment["cluster_id"] is None
