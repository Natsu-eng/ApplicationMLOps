"""Tests de services/stats_utils.py (Lot B) — primitives d'association."""
from __future__ import annotations

import numpy as np
import pandas as pd

from services.stats_utils import correlation_ratio, cramers_v, sample_if_large, univariate_auc


def test_cramers_v_independence_is_near_zero():
    rng = np.random.default_rng(0)
    n = 2000
    x = pd.Series(rng.choice(["a", "b", "c"], size=n))
    y = pd.Series(rng.choice(["x", "y"], size=n))
    assert cramers_v(x, y) < 0.05


def test_cramers_v_perfect_dependence_is_near_one():
    x = pd.Series(["a", "b", "c"] * 200)
    y = x.map({"a": "x", "b": "y", "c": "z"})  # bijection exacte
    assert cramers_v(x, y) > 0.95


def test_cramers_v_independence_near_zero_even_at_high_cardinality():
    """Le cas qui piégeait le V de Cramér brut : beaucoup de catégories,
    variables indépendantes — la version corrigée (Bergsma) ne doit PAS
    afficher une fausse association élevée."""
    rng = np.random.default_rng(1)
    n = 3000
    x = pd.Series(rng.integers(0, 80, size=n).astype(str))  # 80 catégories
    y = pd.Series(rng.choice(["x", "y", "z"], size=n))
    assert cramers_v(x, y) < 0.15


def test_cramers_v_handles_constant_column():
    x = pd.Series(["a"] * 50)
    y = pd.Series(np.random.default_rng(2).choice(["x", "y"], size=50))
    assert cramers_v(x, y) == 0.0


def test_correlation_ratio_independence_is_low():
    rng = np.random.default_rng(3)
    cat = pd.Series(rng.choice(["a", "b", "c"], size=1000))
    num = pd.Series(rng.normal(size=1000))  # indépendant de cat
    assert correlation_ratio(cat, num) < 0.15


def test_correlation_ratio_perfect_group_determination_is_high():
    cat = pd.Series(["a", "b", "c"] * 100)
    num = cat.map({"a": 1.0, "b": 50.0, "c": 100.0})  # la catégorie détermine la valeur
    assert correlation_ratio(cat, num) > 0.95


def test_univariate_auc_random_feature_is_near_half():
    rng = np.random.default_rng(4)
    x = pd.Series(rng.normal(size=1000))
    y = pd.Series(rng.choice(["classe_a", "classe_b"], size=1000))
    auc = univariate_auc(x, y)
    assert 0.4 < auc < 0.6


def test_univariate_auc_perfect_separation_is_near_one():
    y = pd.Series(["a"] * 100 + ["b"] * 100)
    x = pd.Series([0.0] * 100 + [10.0] * 100)  # sépare parfaitement les deux classes
    assert univariate_auc(x, y) > 0.99


def test_sample_if_large_is_deterministic_and_bounded():
    df = pd.DataFrame({"x": range(1000)})
    sampled_1 = sample_if_large(df, max_rows=100, seed=42)
    sampled_2 = sample_if_large(df, max_rows=100, seed=42)
    assert len(sampled_1) == 100
    assert sampled_1["x"].tolist() == sampled_2["x"].tolist()


def test_sample_if_large_noop_when_dataset_already_small():
    df = pd.DataFrame({"x": range(10)})
    assert len(sample_if_large(df, max_rows=100, seed=42)) == 10
