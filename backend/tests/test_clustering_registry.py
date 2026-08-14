"""Tests de services/clustering_registry.py (Lot 11 — ML non supervisé)."""
from __future__ import annotations

import numpy as np
import pytest

from services.clustering_registry import (
    CLUSTER_REGISTRY,
    DEFAULT_ALGORITHM_IDS,
    resolve_dbscan_eps,
    specs_for,
)


def test_registry_has_no_duplicate_ids():
    ids = [spec.id for spec in CLUSTER_REGISTRY]
    assert len(ids) == len(set(ids))


def test_default_subset_is_smaller_than_full_catalog():
    """Stratégie produit (comme le Lot 5 côté supervisé) : le sous-ensemble
    par défaut privilégie un temps de calcul raisonnable, le catalogue
    complet reste disponible en mode expert."""
    assert len(DEFAULT_ALGORITHM_IDS) < len(CLUSTER_REGISTRY)
    assert DEFAULT_ALGORITHM_IDS <= {s.id for s in CLUSTER_REGISTRY}


def test_specs_for_none_returns_default_subset():
    specs = specs_for(None)
    assert {s.id for s in specs} == DEFAULT_ALGORITHM_IDS


def test_specs_for_explicit_ids_filters_catalog():
    specs = specs_for(["kmeans", "dbscan"])
    assert {s.id for s in specs} == {"kmeans", "dbscan"}


def test_specs_for_unknown_id_returns_empty():
    assert specs_for(["algorithme_inexistant"]) == []


@pytest.mark.parametrize("spec", CLUSTER_REGISTRY, ids=lambda s: s.id)
def test_every_spec_builds_a_fittable_estimator(spec):
    """Chaque algorithme du registre doit produire un estimateur qui
    s'entraîne et prédit sans erreur sur un jeu de données jouet — preuve
    directe, pas seulement une assertion structurelle sur la déclaration."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    configs = spec.candidate_configs(X.shape[0], seed=42)
    assert len(configs) > 0, f"{spec.id} ne produit aucun candidat sur 60 échantillons"

    params = dict(configs[0].params)
    if spec.id == "dbscan":
        params["eps"] = resolve_dbscan_eps(X, params["min_samples"])

    estimator = spec.build_estimator(params, seed=42)
    labels = estimator.fit_predict(X)
    assert len(labels) == X.shape[0]


def test_kmeans_candidates_scale_with_n_samples():
    """Un k candidat ne doit jamais dépasser (ni égaler) le nombre
    d'échantillons — sklearn lèverait une erreur sinon."""
    from services.clustering_registry import _kmeans_candidates

    configs = _kmeans_candidates(n_samples=4, seed=42)
    assert all(c.params["n_clusters"] < 4 for c in configs)


def test_dbscan_candidates_empty_below_minimum_samples():
    from services.clustering_registry import _dbscan_candidates

    assert _dbscan_candidates(n_samples=5, seed=42) == []


def test_resolve_dbscan_eps_returns_positive_float():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 2))
    eps = resolve_dbscan_eps(X, min_samples=5)
    assert isinstance(eps, float)
    assert eps > 0


def test_resolve_dbscan_eps_larger_for_more_spread_data():
    """Sanity check directionnelle : des données plus dispersées doivent
    produire un eps résolu plus grand, pas une valeur magique fixe."""
    rng = np.random.default_rng(2)
    tight = rng.normal(scale=0.1, size=(50, 2))
    spread = rng.normal(scale=10.0, size=(50, 2))
    assert resolve_dbscan_eps(spread, min_samples=5) > resolve_dbscan_eps(tight, min_samples=5)
