"""Tests de services/dataset_eda.py (Lot 4b, enrichi Lot B) — logique pure."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from domains.shared.dataset_eda import (
    compute_categorical_correlation_matrix,
    compute_column_stats,
    compute_correlation_matrix,
    compute_feature_by_target,
    compute_histogram,
    compute_missing_summary,
    compute_outlier_boxplots,
    compute_top_correlated_pairs,
)


def _sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "age": [25, 30, np.nan, 40, 50, 22, 31, 45, 29, 60],
            "revenu": rng.normal(3000, 500, 10),
            "ville": ["Paris", "Lyon", "Paris", "Marseille", "Paris", "Lyon", "Paris", "Lyon", "Paris", "Nice"],
        }
    )


def test_column_stats_numeric_and_categorical():
    stats = compute_column_stats(_sample_df())
    by_name = {s["name"]: s for s in stats}

    assert by_name["age"]["kind"] == "numeric"
    assert by_name["age"]["missing_count"] == 1
    assert by_name["age"]["mean"] is not None

    assert by_name["ville"]["kind"] == "categorical"
    assert by_name["ville"]["n_unique"] == 4
    assert by_name["ville"]["top_values"][0]["value"] == "Paris"
    assert by_name["ville"]["top_values"][0]["count"] == 5  # "Paris" apparaît 5 fois dans _sample_df


def test_missing_summary_only_lists_columns_with_gaps():
    summary = compute_missing_summary(_sample_df())
    assert len(summary) == 1
    assert summary[0]["column"] == "age"
    assert summary[0]["missing_count"] == 1


def test_correlation_matrix_numeric_only():
    corr = compute_correlation_matrix(_sample_df())
    assert set(corr["columns"]) == {"age", "revenu"}
    assert len(corr["matrix"]) == 2
    # La diagonale d'une matrice de corrélation vaut toujours 1
    assert corr["matrix"][0][0] == 1.0


def test_correlation_matrix_requires_two_numeric_columns():
    df = pd.DataFrame({"ville": ["Paris", "Lyon"]})
    corr = compute_correlation_matrix(df)
    assert corr["matrix"] == []


def test_histogram_numeric_column():
    hist = compute_histogram(_sample_df(), "revenu", bins=5)
    assert hist["kind"] == "numeric"
    assert len(hist["counts"]) == 5
    assert len(hist["bin_edges"]) == 6
    assert sum(hist["counts"]) == 10  # aucune valeur manquante sur revenu


def test_histogram_categorical_column():
    hist = compute_histogram(_sample_df(), "ville")
    assert hist["kind"] == "categorical"
    assert "Paris" in hist["categories"]
    assert sum(hist["counts"]) == 10


# ── Lot B — corrélations catégorielles, outliers, paires corrélées ────────


def test_categorical_correlation_matrix_on_known_dependence():
    rng = np.random.default_rng(0)
    n = 1000
    x = pd.Series(rng.choice(["a", "b"], size=n))
    y = x.copy()  # dépendance parfaite
    z = pd.Series(rng.choice(["p", "q", "r"], size=n))  # indépendante de x
    df = pd.DataFrame({"x": x, "y": y, "z": z})
    result = compute_categorical_correlation_matrix(df)
    idx = {c: i for i, c in enumerate(result["columns"])}
    assert result["matrix"][idx["x"]][idx["y"]] > 0.95
    assert result["matrix"][idx["x"]][idx["z"]] < 0.15
    assert result["matrix"][idx["x"]][idx["x"]] == 1.0


def test_categorical_correlation_matrix_requires_two_categorical_columns():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    result = compute_categorical_correlation_matrix(df)
    assert result["matrix"] == []


def test_categorical_correlation_matrix_excludes_high_cardinality_columns():
    n = 200
    df = pd.DataFrame({"id": [f"id_{i}" for i in range(n)], "cat": ["a", "b"] * (n // 2)})
    result = compute_categorical_correlation_matrix(df)
    assert "id" not in result["columns"]


def test_outlier_boxplots_detects_extreme_values():
    values = list(np.random.default_rng(1).normal(50, 2, 100)) + [500.0]  # un outlier flagrant
    df = pd.DataFrame({"x": values})
    boxplots = compute_outlier_boxplots(df)
    assert len(boxplots) == 1
    assert 500.0 in boxplots[0]["outliers"]
    assert boxplots[0]["q1"] < boxplots[0]["median"] < boxplots[0]["q3"]


def test_outlier_boxplots_ignores_categorical_columns():
    df = pd.DataFrame({"cat": ["a", "b", "c"]})
    assert compute_outlier_boxplots(df) == []


def test_feature_by_target_produces_one_group_per_class_classification():
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {"valeur": rng.normal(size=90), "classe": ["a"] * 30 + ["b"] * 30 + ["c"] * 30}
    )
    result = compute_feature_by_target(df, "valeur", "classe")
    assert {g["class_name"] for g in result["groups"]} == {"a", "b", "c"}
    for g in result["groups"]:
        assert g["q1"] <= g["median"] <= g["q3"]


def test_feature_by_target_works_for_regression_target_too():
    """La cible régression est juste une autre variable groupable — le test
    vérifie l'absence d'erreur, pas une sémantique métier particulière."""
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"valeur": rng.normal(size=60), "cible_num": rng.integers(0, 3, 60)})
    result = compute_feature_by_target(df, "valeur", "cible_num")
    assert len(result["groups"]) > 0


def test_feature_by_target_rejects_non_numeric_feature():
    df = pd.DataFrame({"cat": ["a", "b"], "cible": ["x", "y"]})
    with pytest.raises(ValueError):
        compute_feature_by_target(df, "cat", "cible")


def test_feature_by_target_rejects_unknown_column():
    df = pd.DataFrame({"x": [1, 2], "cible": ["a", "b"]})
    with pytest.raises(KeyError):
        compute_feature_by_target(df, "inexistante", "cible")


def test_top_correlated_pairs_returns_strongest_pair_first():
    rng = np.random.default_rng(4)
    n = 300
    x1 = rng.normal(size=n)
    df = pd.DataFrame(
        {"x1": x1, "x2": x1 * 3 + rng.normal(0, 1e-6, n), "bruit": rng.normal(size=n)}
    )
    pairs = compute_top_correlated_pairs(df, top_n=2)
    assert pairs[0]["x_column"] == "x1" and pairs[0]["y_column"] == "x2"
    assert abs(pairs[0]["correlation"]) > 0.99
    assert len(pairs[0]["points"]) > 0


def test_top_correlated_pairs_requires_two_numeric_columns():
    df = pd.DataFrame({"cat": ["a", "b", "c"]})
    assert compute_top_correlated_pairs(df) == []
