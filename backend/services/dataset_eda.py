"""Exploration de données (EDA) — logique pure, sans dépendance HTTP.

Reprend les analyses les plus utiles d'un notebook de référence partagé par
l'équipe (distributions, corrélations, valeurs manquantes), généralisées à
n'importe quel dataset tabulaire — voir `backend/workflow.md` (Lot 4b).

Enrichi au Lot B : corrélations catégorielles (Cramér's V corrigé),
boxplots IQR (détection d'outliers), nuages de points des paires de
features les plus corrélées, boxplots d'une feature par classe de la
cible — voir `services/stats_utils.py` pour les primitives statistiques
partagées avec les garde-fous de données (`services/data_quality.py`).
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from services.stats_utils import cramers_v, sample_if_large

MAX_TOP_CATEGORIES = 8
DEFAULT_HISTOGRAM_BINS = 20

# ── Bornage de coût (Lot B) ──────────────────────────────────────────────
MAX_ROWS_FOR_STATS = 20_000
# Au-delà, les calculs les plus coûteux de ce module (corrélations
# catégorielles, nuages de points) travaillent sur un échantillon
# déterministe (voir stats_utils.sample_if_large) plutôt que sur le dataset
# complet. Dupliqué intentionnellement depuis data_quality.py (même valeur,
# même justification) plutôt qu'importé, pour éviter un import circulaire
# (data_quality.py importe déjà compute_missing_summary depuis ce module).
CATEGORICAL_CORR_MAX_UNIQUE = 30
# Une colonne catégorielle à plus de 30 valeurs distinctes n'entre pas dans
# la matrice de Cramér's V : le tableau de contingence deviendrait coûteux
# et peu lisible, et une telle colonne ressemble de toute façon à un
# identifiant (déjà signalé par le garde-fou de cardinalité).
IQR_WHISKER_MULTIPLIER = 1.5
# Règle de Tukey standard pour délimiter les moustaches d'un boxplot :
# au-delà de Q1 - 1.5×IQR / Q3 + 1.5×IQR, un point est considéré outlier.
MAX_OUTLIERS_PER_COLUMN = 20
# Cap du nombre de points outliers renvoyés par colonne (les plus extrêmes
# des deux côtés) — un dataset bruité pourrait sinon renvoyer des milliers
# de points pour un graphique qui n'en affiche utilement qu'une poignée.
TOP_CORRELATED_PAIRS_COUNT = 3
# Nombre de paires de features numériques affichées en nuage de points —
# au-delà, le graphique devient plus difficile à lire qu'utile.
MAX_SCATTER_POINTS = 500
# Points échantillonnés par nuage de points — cohérent avec
# `shap_sample_size` (500) déjà utilisé comme ordre de grandeur "assez pour
# être représentatif, assez peu pour rester léger" dans `ml_training.py`.


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


def compute_categorical_correlation_matrix(df: pd.DataFrame) -> dict[str, Any]:
    """Association entre colonnes catégorielles — V de Cramér CORRIGÉ du
    biais (voir `stats_utils.cramers_v`), même logique que
    `compute_correlation_matrix` pour le numérique mais sur une échelle
    [0, 1] (jamais négatif, pas de notion de sens).

    Colonnes à plus de `CATEGORICAL_CORR_MAX_UNIQUE` valeurs distinctes
    exclues (quasi-identifiants, peu lisibles en matrice). Coût : O(k²)
    paires, chacune O(n) pour le tableau de contingence — échantillonné via
    `sample_if_large` au-delà de `MAX_ROWS_FOR_STATS` lignes.
    """
    categorical_cols = [
        c for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() <= CATEGORICAL_CORR_MAX_UNIQUE
    ]
    if len(categorical_cols) < 2:
        return {"columns": [str(c) for c in categorical_cols], "matrix": []}

    sampled = sample_if_large(df[categorical_cols], MAX_ROWS_FOR_STATS)
    n = len(categorical_cols)
    matrix = [
        [
            1.0 if i == j else _clean_float(cramers_v(sampled[categorical_cols[i]], sampled[categorical_cols[j]]))
            for j in range(n)
        ]
        for i in range(n)
    ]
    return {"columns": [str(c) for c in categorical_cols], "matrix": matrix}


def _boxplot_stats(series: pd.Series) -> Optional[dict[str, Any]]:
    """Statistiques d'un boxplot IQR (règle de Tukey) pour une série
    numérique — quartiles, moustaches, et outliers (valeurs au-delà des
    moustaches), plafonnés à `MAX_OUTLIERS_PER_COLUMN`. Coût : O(n log n)
    (calcul des quantiles)."""
    values = series.dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return None

    q1, median, q3 = (float(v) for v in np.percentile(values, [25, 50, 75]))
    iqr = q3 - q1
    whisker_low = q1 - IQR_WHISKER_MULTIPLIER * iqr
    whisker_high = q3 + IQR_WHISKER_MULTIPLIER * iqr

    inliers = values[(values >= whisker_low) & (values <= whisker_high)]
    outliers = np.sort(values[(values < whisker_low) | (values > whisker_high)])
    if len(outliers) > MAX_OUTLIERS_PER_COLUMN:
        half = MAX_OUTLIERS_PER_COLUMN // 2
        outliers = np.concatenate([outliers[:half], outliers[-(MAX_OUTLIERS_PER_COLUMN - half):]])

    actual_low = float(inliers.min()) if len(inliers) else float(values.min())
    actual_high = float(inliers.max()) if len(inliers) else float(values.max())

    return {
        "min": _clean_float(actual_low),
        "q1": _clean_float(q1),
        "median": _clean_float(median),
        "q3": _clean_float(q3),
        "max": _clean_float(actual_high),
        "outliers": [_clean_float(v) for v in outliers],
        "n": len(values),
    }


def compute_outlier_boxplots(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Boxplot IQR pour chaque colonne numérique — détection d'outliers
    visuelle. Coût : O(n log n) par colonne numérique, négligeable même sur
    un dataset large (pas d'échantillonnage nécessaire, contrairement aux
    calculs par paires ci-dessous)."""
    result = []
    for col in df.select_dtypes(include="number").columns:
        stats = _boxplot_stats(df[col])
        if stats is None:
            continue
        result.append({"column": str(col), **stats})
    return result


def compute_feature_by_target(df: pd.DataFrame, feature_column: str, target_column: str) -> dict[str, Any]:
    """Boxplot d'une feature NUMÉRIQUE, une boîte par classe de la cible —
    pouvoir discriminant visuel (classification uniquement : n'a de sens
    que pour un nombre fini de classes). Coût : O(n log n) au total (un
    boxplot par classe)."""
    if feature_column not in df.columns:
        raise KeyError(f"Colonne '{feature_column}' absente du dataset")
    if target_column not in df.columns:
        raise KeyError(f"Colonne '{target_column}' absente du dataset")
    if not pd.api.types.is_numeric_dtype(df[feature_column]):
        raise ValueError(f"La colonne '{feature_column}' doit être numérique pour ce graphique")

    groups = []
    for class_value, sub in df.groupby(target_column, dropna=True)[feature_column]:
        stats = _boxplot_stats(sub)
        if stats is None:
            continue
        groups.append({"class_name": str(class_value), **stats})
    return {"feature": str(feature_column), "target": str(target_column), "groups": groups}


def compute_top_correlated_pairs(
    df: pd.DataFrame, top_n: int = TOP_CORRELATED_PAIRS_COUNT, max_points: int = MAX_SCATTER_POINTS
) -> list[dict[str, Any]]:
    """Nuages de points des `top_n` paires de features numériques les plus
    corrélées (en valeur absolue) — repère visuellement la redondance déjà
    détectée par `compute_correlation_matrix`. Coût : O(k²) pour repérer les
    paires (k = nb colonnes numériques) + O(n) par paire retenue pour
    l'échantillonnage des points (`max_points`, cohérent avec le garde-fou
    de collinéarité qui utilise le même seuil de corrélation)."""
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return []

    sampled_for_corr = sample_if_large(numeric_df, MAX_ROWS_FOR_STATS)
    corr = sampled_for_corr.corr()
    cols = list(corr.columns)

    pairs: list[tuple[str, str, float]] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.notna(r):
                pairs.append((cols[i], cols[j], float(r)))
    pairs.sort(key=lambda p: -abs(p[2]))

    result = []
    for c1, c2, r in pairs[:top_n]:
        pair_df = df[[c1, c2]].dropna()
        pair_sample = sample_if_large(pair_df, max_points)
        result.append({
            "x_column": str(c1),
            "y_column": str(c2),
            "correlation": _clean_float(r),
            "points": [
                {"x": _clean_float(x), "y": _clean_float(y)}
                for x, y in zip(pair_sample[c1].to_numpy(), pair_sample[c2].to_numpy())
            ],
        })
    return result
