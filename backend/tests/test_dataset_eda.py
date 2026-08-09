"""Tests de services/dataset_eda.py (Lot 4b) — logique pure."""
from __future__ import annotations

import numpy as np
import pandas as pd

from services.dataset_eda import (
    compute_column_stats,
    compute_correlation_matrix,
    compute_histogram,
    compute_missing_summary,
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
