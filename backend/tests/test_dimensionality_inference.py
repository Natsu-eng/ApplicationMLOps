"""Tests de domains/dimensionality/services/inference.py (Lot 6B, §F.2 —
projeter une nouvelle observation à partir d'une projection déjà entraînée)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from domains.dimensionality.services.engine import DimensionalityConfig, train_and_evaluate_dimensionality
from domains.dimensionality.services.inference import (
    DimensionalityInferenceError,
    project_point,
    project_points_batch,
)

_NOOP = lambda step, pct: None  # noqa: E731


def _make_two_blobs_df(n_per_group: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    group = np.repeat([0, 1], n_per_group)
    signal = np.where(group == 0, 0.0, 20.0) + rng.normal(0, 0.5, len(group))
    noise = rng.normal(0, 0.5, len(group))
    return pd.DataFrame({"signal": signal, "noise": noise})


def _train(algorithm_id: str):
    df = _make_two_blobs_df()
    return train_and_evaluate_dimensionality(df, DimensionalityConfig(algorithm_id=algorithm_id, seed=42), _NOOP)


def test_pca_projects_a_new_point_exactly():
    result = _train("pca")
    projected = project_point(result.pipeline_bundle, result.feature_columns, {"signal": 0.0, "noise": 0.0})
    assert projected["projection_method"] == "exact"
    assert projected["x"] is not None
    assert projected["y"] is not None


def test_umap_projects_a_new_point_exactly():
    result = _train("umap")
    projected = project_point(result.pipeline_bundle, result.feature_columns, {"signal": 0.0, "noise": 0.0})
    assert projected["projection_method"] == "exact"
    assert projected["x"] is not None


def test_tsne_reports_unsupported_honestly():
    """t-SNE est transductif — sklearn n'expose aucun `.transform()`, jamais
    d'approximation inventée à sa place."""
    result = _train("tsne")
    projected = project_point(result.pipeline_bundle, result.feature_columns, {"signal": 0.0, "noise": 0.0})
    assert projected["projection_method"] == "unsupported"
    assert projected["x"] is None
    assert projected["y"] is None


def test_legacy_bundle_without_primary_model_degrades_to_unsupported():
    """Rétrocompatibilité par absence — un job entraîné AVANT ce correctif
    n'a pas la clé `primary_model` dans son pipeline_bundle."""
    result = _train("pca")
    legacy_bundle = {k: v for k, v in result.pipeline_bundle.items() if k != "primary_model"}
    projected = project_point(legacy_bundle, result.feature_columns, {"signal": 0.0, "noise": 0.0})
    assert projected["projection_method"] == "unsupported"


def test_missing_feature_raises_actionable_error():
    result = _train("pca")
    with pytest.raises(DimensionalityInferenceError):
        project_point(result.pipeline_bundle, result.feature_columns, {"signal": 0.0})  # noise manquant


def test_batch_projects_every_row_and_preserves_other_columns():
    result = _train("pca")
    input_df = pd.DataFrame({"signal": [0.0, 20.0], "noise": [0.0, 0.0], "id_ligne": ["a", "b"]})
    projected = project_points_batch(result.pipeline_bundle, result.feature_columns, input_df)
    assert list(projected["projection_status"]) == ["projected", "projected"]
    assert "id_ligne" in projected.columns


def test_batch_reports_unsupported_for_tsne_without_crashing():
    result = _train("tsne")
    input_df = pd.DataFrame({"signal": [0.0], "noise": [0.0]})
    projected = project_points_batch(result.pipeline_bundle, result.feature_columns, input_df)
    assert list(projected["projection_status"]) == ["unsupported"]
