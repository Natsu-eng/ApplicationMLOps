"""Exploration de données (EDA) — logique pure, sans dépendance HTTP.

Reprend les analyses les plus utiles d'un notebook de référence partagé par
l'équipe (distributions, corrélations, valeurs manquantes), généralisées à
n'importe quel dataset tabulaire — voir `backend/workflow.md` (Lot 4b).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

MAX_TOP_CATEGORIES = 8
DEFAULT_HISTOGRAM_BINS = 20


def _clean_float(value: Any) -> Any:
    """NaN/inf ne sont pas JSON-sérialisables — convertis en None."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value
    return None if (np.isnan(f) or np.isinf(f)) else f


def compute_column_stats(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Statistiques par colonne — numériques (moyenne/écart-type/min/max/
    médiane) ou catégorielles (cardinalité, valeurs les plus fréquentes)."""
    n = len(df)
    stats: list[dict[str, Any]] = []
    for col in df.columns:
        series = df[col]
        missing = int(series.isna().sum())
        entry: dict[str, Any] = {
            "name": str(col),
            "dtype": str(series.dtype),
            "missing_count": missing,
            "missing_pct": _clean_float(missing / n * 100) if n else 0.0,
        }
        if pd.api.types.is_numeric_dtype(series):
            described = series.describe()
            entry.update(
                {
                    "kind": "numeric",
                    "mean": _clean_float(described.get("mean")),
                    "std": _clean_float(described.get("std")),
                    "min": _clean_float(described.get("min")),
                    "max": _clean_float(described.get("max")),
                    "median": _clean_float(series.median()),
                }
            )
        else:
            value_counts = series.value_counts().head(MAX_TOP_CATEGORIES)
            entry.update(
                {
                    "kind": "categorical",
                    "n_unique": int(series.nunique()),
                    "top_values": [
                        {"value": str(idx), "count": int(count)} for idx, count in value_counts.items()
                    ],
                }
            )
        stats.append(entry)
    return stats


def compute_missing_summary(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Colonnes triées par % de valeurs manquantes décroissant — ne garde
    que celles qui en ont, pour ne pas noyer le signal utile."""
    n = len(df)
    if n == 0:
        return []
    missing = df.isna().sum()
    result = [
        {"column": str(col), "missing_count": int(count), "missing_pct": _clean_float(count / n * 100)}
        for col, count in missing.items()
        if count > 0
    ]
    return sorted(result, key=lambda r: r["missing_pct"], reverse=True)


def compute_correlation_matrix(df: pd.DataFrame) -> dict[str, Any]:
    """Corrélation de Pearson entre colonnes numériques uniquement."""
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return {"columns": list(numeric_df.columns.astype(str)), "matrix": []}
    corr = numeric_df.corr()
    matrix = [[_clean_float(v) for v in row] for row in corr.to_numpy()]
    return {"columns": [str(c) for c in corr.columns], "matrix": matrix}


def compute_histogram(df: pd.DataFrame, column: str, bins: int = DEFAULT_HISTOGRAM_BINS) -> dict[str, Any]:
    """Histogramme d'une colonne — bins réguliers si numérique, comptage des
    catégories (les plus fréquentes) sinon."""
    if column not in df.columns:
        raise KeyError(f"Colonne '{column}' absente du dataset")
    series = df[column].dropna()

    if pd.api.types.is_numeric_dtype(series):
        counts, edges = np.histogram(series.to_numpy(), bins=min(bins, max(1, series.nunique())))
        return {
            "kind": "numeric",
            "bin_edges": [_clean_float(e) for e in edges],
            "counts": [int(c) for c in counts],
        }

    value_counts = series.value_counts()
    top = value_counts.head(MAX_TOP_CATEGORIES)
    other_count = int(value_counts.iloc[MAX_TOP_CATEGORIES:].sum())
    categories = [str(idx) for idx in top.index]
    counts = [int(c) for c in top.to_numpy()]
    if other_count > 0:
        categories.append("Autres")
        counts.append(other_count)
    return {"kind": "categorical", "categories": categories, "counts": counts}
