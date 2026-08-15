"""Tests de services/anomaly_registry.py (Lot 14 — ML non supervisé)."""
from __future__ import annotations

from services.anomaly_registry import ANOMALY_REGISTRY


def test_registry_has_both_algorithms():
    ids = {s.id for s in ANOMALY_REGISTRY}
    assert ids == {"isolation_forest", "lof"}


def test_both_builders_use_contamination_auto():
    for spec in ANOMALY_REGISTRY:
        estimator = spec.build_estimator(100, 42)
        assert estimator.contamination == "auto"


def test_lof_n_neighbors_clipped_for_small_datasets():
    spec = next(s for s in ANOMALY_REGISTRY if s.id == "lof")
    estimator = spec.build_estimator(5, 42)
    assert estimator.n_neighbors < 5


def test_isolation_forest_seed_propagated_to_estimator():
    spec = next(s for s in ANOMALY_REGISTRY if s.id == "isolation_forest")
    estimator = spec.build_estimator(100, 777)
    assert estimator.random_state == 777
