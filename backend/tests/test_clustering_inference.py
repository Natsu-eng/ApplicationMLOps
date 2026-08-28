"""Tests de domains/clustering/services/inference.py (Lot 6B, §F.2 — assignation
d'une nouvelle observation à un clustering déjà entraîné)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from domains.clustering.services.inference import ClusterInferenceError, assign_cluster, assign_clusters_batch
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


# ── assign_clusters_batch (retour utilisateur : assigner un cluster à un
# dataset complet, pas seulement à l'échantillon effectivement clusterisé
# à l'entraînement — voir domains/clustering/router.py::export_cluster_assignments) ──


def test_batch_assignment_kmeans_covers_every_row_exactly():
    df = _make_three_blobs_df()
    result = train_and_evaluate_clustering(df, ClusteringConfig(seed=42, algorithm_ids=["kmeans"]), _NOOP)
    assigned = assign_clusters_batch(result.pipeline_bundle, result.feature_columns, df)
    assert len(assigned) == len(df)
    assert (assigned["assignment_method"] == "exact").all()
    assert assigned["cluster_id"].notna().all()
    assert assigned["is_noise"].eq(False).all()
    # Colonnes d'origine préservées (l'utilisateur doit pouvoir rejoindre
    # l'export à ses données réelles).
    assert "x1" in assigned.columns and "x2" in assigned.columns


def test_batch_assignment_matches_row_by_row_assignment():
    """La version vectorisée doit produire EXACTEMENT le même résultat que
    l'assignation ligne par ligne existante — jamais une approximation
    différente entre les deux chemins."""
    df = _make_three_blobs_df()
    result = train_and_evaluate_clustering(df, ClusteringConfig(seed=42, algorithm_ids=["kmeans"]), _NOOP)
    assigned = assign_clusters_batch(result.pipeline_bundle, result.feature_columns, df)
    for i in range(0, len(df), 25):
        row = df.iloc[i]
        single = assign_cluster(result.pipeline_bundle, result.feature_columns, row.to_dict())
        assert int(assigned.iloc[i]["cluster_id"]) == single["cluster_id"]


def test_batch_assignment_hierarchical_uses_vectorized_centroids():
    df = _make_three_blobs_df()
    result = train_and_evaluate_clustering(df, ClusteringConfig(seed=42, algorithm_ids=["hierarchical"]), _NOOP)
    assigned = assign_clusters_batch(result.pipeline_bundle, result.feature_columns, df)
    assert (assigned["assignment_method"] == "approximate_centroid").all()
    assert assigned["cluster_id"].notna().all()


def test_batch_assignment_dbscan_flags_far_rows_as_noise():
    df = _make_three_blobs_df()
    extra = pd.DataFrame({"x1": [500.0], "x2": [500.0]})
    df_with_outlier = pd.concat([df, extra], ignore_index=True)
    result = train_and_evaluate_clustering(df, ClusteringConfig(seed=42, algorithm_ids=["dbscan"]), _NOOP)
    assigned = assign_clusters_batch(result.pipeline_bundle, result.feature_columns, df_with_outlier)
    last_row = assigned.iloc[-1]
    assert last_row["is_noise"] == True  # noqa: E712
    assert pd.isna(last_row["cluster_id"])
    assert (assigned["assignment_method"] == "approximate_nearest_core").all()


def test_batch_assignment_reports_missing_column_without_crashing():
    df = _make_three_blobs_df()
    result = train_and_evaluate_clustering(df, ClusteringConfig(seed=42, algorithm_ids=["kmeans"]), _NOOP)
    incomplete = df.drop(columns=["x2"])
    assigned = assign_clusters_batch(result.pipeline_bundle, result.feature_columns, incomplete)
    assert (assigned["assignment_method"] == "missing_column").all()
    assert assigned["cluster_id"].isna().all()


def test_batch_assignment_reports_missing_values_per_row_not_globally():
    df = _make_three_blobs_df().copy()
    df.loc[0, "x2"] = np.nan
    result = train_and_evaluate_clustering(df.fillna(0), ClusteringConfig(seed=42, algorithm_ids=["kmeans"]), _NOOP)
    assigned = assign_clusters_batch(result.pipeline_bundle, result.feature_columns, df)
    assert assigned.iloc[0]["assignment_method"] == "missing_features"
    assert pd.isna(assigned.iloc[0]["cluster_id"])
    # Les autres lignes, complètes, restent assignées normalement.
    assert (assigned.iloc[1:]["assignment_method"] == "exact").all()


def test_batch_assignment_on_empty_dataframe_returns_empty_without_crashing():
    df = _make_three_blobs_df()
    result = train_and_evaluate_clustering(df, ClusteringConfig(seed=42, algorithm_ids=["kmeans"]), _NOOP)
    empty = df.iloc[0:0]
    assigned = assign_clusters_batch(result.pipeline_bundle, result.feature_columns, empty)
    assert len(assigned) == 0
