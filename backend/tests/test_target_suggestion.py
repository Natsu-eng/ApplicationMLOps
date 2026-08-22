"""Tests de services/target_suggestion.py (Lot 7, §J.1) — suggestion de
colonne cible : un score de plausibilité, jamais un choix imposé."""
from __future__ import annotations

import numpy as np
import pandas as pd

from domains.datasets.services.target_suggestion import suggest_target_columns


def _make_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "id": range(1, n + 1),  # identifiant quasi-unique — jamais suggéré
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
            "categorie": rng.choice(["A", "B", "C"], n),
            "target": rng.choice([0, 1], n),
        }
    )


def test_identifier_column_is_never_suggested():
    df = _make_df()
    suggestions = suggest_target_columns(df)
    assert "id" not in [s.column for s in suggestions]


def test_constant_column_is_never_suggested():
    df = _make_df()
    df["constante"] = 1
    suggestions = suggest_target_columns(df)
    assert "constante" not in [s.column for s in suggestions]


def test_column_with_too_many_missing_values_is_never_suggested():
    df = _make_df()
    df["cible_trouee"] = [None] * 50 + [0, 1] * 25
    suggestions = suggest_target_columns(df)
    assert "cible_trouee" not in [s.column for s in suggestions]


def test_name_hint_ranks_a_column_above_a_plain_numeric_column_of_similar_shape():
    df = _make_df()
    df["mesure_brute"] = df["target"]  # même forme statistique que "target"
    suggestions = suggest_target_columns(df, max_suggestions=10)
    columns_ranked = [s.column for s in suggestions]
    assert columns_ranked.index("target") < columns_ranked.index("mesure_brute")


def test_every_suggestion_carries_concrete_reasons():
    df = _make_df()
    suggestions = suggest_target_columns(df)
    assert len(suggestions) > 0
    for s in suggestions:
        assert len(s.reasons) > 0
        assert all(isinstance(r, str) and len(r) > 0 for r in s.reasons)


def test_max_suggestions_is_respected():
    df = _make_df()
    for i in range(10):
        df[f"cible_possible_{i}"] = df["target"]
    suggestions = suggest_target_columns(df, max_suggestions=3)
    assert len(suggestions) == 3


def test_last_column_gets_a_small_bonus():
    """Une colonne en dernière position, sans autre indice, doit quand même
    apparaître (emplacement fréquent d'une cible dans un export tabulaire)."""
    df = pd.DataFrame(
        {
            "a": np.random.default_rng(0).normal(0, 1, 50),
            "b": np.random.default_rng(1).normal(0, 1, 50),
            "z": np.random.default_rng(2).choice(["oui", "non"], 50),
        }
    )
    suggestions = suggest_target_columns(df)
    assert "z" in [s.column for s in suggestions]
