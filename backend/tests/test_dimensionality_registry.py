"""Tests de services/dimensionality_registry.py (Lot 13 — ML non supervisé)."""
from __future__ import annotations

import pytest

from services.dimensionality_registry import DIMENSIONALITY_REGISTRY, spec_for


def test_registry_always_has_pca_and_tsne():
    ids = {s.id for s in DIMENSIONALITY_REGISTRY}
    assert {"pca", "tsne"} <= ids


def test_spec_for_unknown_id_raises():
    with pytest.raises(ValueError):
        spec_for("methode_magique")


def test_spec_for_returns_matching_spec():
    spec = spec_for("pca")
    assert spec.id == "pca"
    assert spec.family == "lineaire"


@pytest.mark.parametrize("algo_id", ["pca", "tsne"])
def test_build_estimator_clips_hyperparameters_for_small_datasets(algo_id):
    """Un dataset minuscule (5 lignes) ne doit jamais faire construire un
    estimateur avec un hyperparamètre >= n_samples (perplexity t-SNE,
    n_components PCA) — sklearn lèverait sinon une erreur brute."""
    spec = spec_for(algo_id)
    estimator = spec.build_estimator(5, 42)
    if algo_id == "tsne":
        assert estimator.perplexity < 5
    if algo_id == "pca":
        assert estimator.n_components < 5


def test_umap_present_only_if_importable():
    ids = {s.id for s in DIMENSIONALITY_REGISTRY}
    try:
        import umap  # noqa: F401

        assert "umap" in ids
    except ImportError:
        assert "umap" not in ids
