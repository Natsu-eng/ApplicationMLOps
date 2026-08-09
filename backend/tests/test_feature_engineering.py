"""Tests de services/feature_engineering.py (Lot 4c)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from services.feature_engineering import (
    apply_datetime_decomposition,
    suggest_datetime_columns,
)


def _dates_df(n=120, seed=0):
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2022-01-01")
    dates = [base + pd.Timedelta(days=int(d)) for d in rng.integers(0, 700, n)]
    return pd.DataFrame({
        "date_inscription": [d.strftime("%Y-%m-%d") for d in dates],
        "montant": rng.normal(100, 20, n),
        "categorie": rng.choice(["a", "b", "c"], n),
    })


def test_suggest_datetime_columns_detects_text_date_not_others():
    df = _dates_df()
    suggestions = suggest_datetime_columns(df)
    columns_suggested = {s["columns"][0] for s in suggestions}
    assert columns_suggested == {"date_inscription"}
    assert suggestions[0]["transformation"] == {
        "type": "datetime_decompose",
        "source_column": "date_inscription",
    }


def test_suggest_datetime_columns_detects_native_datetime_dtype():
    df = pd.DataFrame({"d": pd.to_datetime(["2023-01-01", "2023-06-15", "2023-12-31"])})
    suggestions = suggest_datetime_columns(df)
    assert [s["columns"][0] for s in suggestions] == ["d"]


def test_suggest_datetime_columns_ignores_low_parse_ratio_text():
    df = pd.DataFrame({"commentaire": ["bof", "top produit", "12/3 étoiles", "rien à dire"] * 30})
    assert suggest_datetime_columns(df) == []


def test_apply_datetime_decomposition_produces_correct_parts():
    df = pd.DataFrame({"date": ["2023-03-15", "2024-11-02"], "target": [1, 0]})
    result, columns = apply_datetime_decomposition(
        df, ["date"], [{"type": "datetime_decompose", "source_column": "date"}]
    )
    assert "date" not in columns
    assert set(columns) == {"date_annee", "date_mois", "date_jour", "date_jour_semaine"}
    assert result.loc[0, "date_annee"] == 2023
    assert result.loc[0, "date_mois"] == 3
    assert result.loc[0, "date_jour"] == 15
    # Colonne source conservée dans le DataFrame (juste retirée des features) —
    # garantit qu'on ne risque jamais de supprimer par erreur une autre colonne.
    assert "date" in result.columns
    assert "target" in result.columns


def test_apply_datetime_decomposition_unparseable_value_becomes_nan_not_crash():
    df = pd.DataFrame({"date": ["2023-03-15", "n'importe quoi"]})
    result, _ = apply_datetime_decomposition(
        df, ["date"], [{"type": "datetime_decompose", "source_column": "date"}]
    )
    assert pd.isna(result.loc[1, "date_annee"])


def test_apply_datetime_decomposition_is_deterministic_single_row_vs_batch():
    """Preuve de rejouabilité train/inférence : la décomposition d'une ligne
    isolée doit produire exactement la même valeur que la même ligne au sein
    du dataset complet."""
    df = _dates_df(n=50, seed=1)
    spec = [{"type": "datetime_decompose", "source_column": "date_inscription"}]
    batch_result, _ = apply_datetime_decomposition(df, ["date_inscription"], spec)

    single_row = df.iloc[[7]].reset_index(drop=True)
    single_result, _ = apply_datetime_decomposition(single_row, ["date_inscription"], spec)

    for part in ("annee", "mois", "jour", "jour_semaine"):
        col = f"date_inscription_{part}"
        assert single_result.loc[0, col] == batch_result.loc[7, col]
