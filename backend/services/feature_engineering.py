"""Ingénierie de variables guidée (Lot 4c) — suggestions en langage clair,
approuvées par l'utilisateur, jamais une configuration technique imposée.

Principe anti-fuite central (voir `backend/workflow.md`, Lot A) : toute
transformation qui APPREND une statistique du jeu de données (une fréquence,
une moyenne, une catégorie vue...) doit être un step de `Pipeline` sklearn,
fit uniquement à l'intérieur de chaque fold de validation croisée — jamais
appliquée en amont sur tout le dataset. Ce module ne contient QUE les
transformations déterministes ligne-à-ligne (décomposition datetime, ratios
entre colonnes existantes), qui ne dépendent d'aucune statistique agrégée et
peuvent donc être appliquées une seule fois, en amont du split train/test.
Les transformations qui apprennent (encodage de fréquence, imputation
configurable) vivent dans `services/ml_preprocessing.py::build_preprocessor`,
à l'intérieur du `ColumnTransformer` fit par fold.

Chaque suggestion produite ici est reliée, quand c'est pertinent, à un
avertissement déjà détecté par `services/data_quality.py` (Lot B) — ce module
ne redétecte jamais un garde-fou existant, il propose l'action correspondante.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from services.stats_utils import sample_if_large

# ── Constantes nommées ──────────────────────────────────────────────────────

DATETIME_PARSE_SUCCESS_THRESHOLD = 0.95
# Part minimale de valeurs non nulles d'une colonne texte qui doit se parser
# comme une date pour la considérer comme une colonne datetime candidate —
# évite qu'une colonne texte ordinaire contenant occasionnellement un motif
# proche d'une date (ex. "12/3") déclenche une suggestion hors-sujet.

DATETIME_SAMPLE_MAX_ROWS = 5_000
# Bornage de coût pour le test de parsing (échantillon déterministe, voir
# `sample_if_large`) — inutile de parser des colonnes texte sur un dataset de
# centaines de milliers de lignes juste pour une suggestion.

DATETIME_PARTS = ("annee", "mois", "jour", "jour_semaine")
# Décomposition volontairement simple et fixe (pas de configuration par
# l'utilisateur, cohérent avec le principe "suggestion en langage clair, on
# approuve, on ne configure pas") — le sin/cos cyclique évoqué comme option
# dans le cadrage du lot est laissé de côté : les 3 boosters du catalogue
# (arbres) n'ont pas besoin d'un encodage cyclique pour exploiter la
# périodicité mois/jour, contrairement à un modèle linéaire.


class FeatureEngineeringSpecError(ValueError):
    """La spec de feature engineering (persistée) est absente, mal formée, ou
    d'une version que ce code ne sait pas rejouer — refus explicite plutôt
    qu'une interprétation best-effort qui produirait une prédiction fausse
    silencieusement."""


def _suggestion(
    code: str,
    title: str,
    explanation: str,
    action: str,
    columns: list[str],
    transformation: dict[str, Any],
    based_on_warning: Optional[str] = None,
    choice: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Gabarit commun à toutes les suggestions 4c — même forme que
    `data_quality._warning` (Lot B) pour rester cohérent côté API/UI, avec
    deux champs en plus : `based_on_warning` (code Lot B à l'origine de la
    suggestion, si applicable) et `transformation` (fragment de spec à
    renvoyer tel quel si l'utilisateur approuve)."""
    return {
        "code": code,
        "title": title,
        "explanation": explanation,
        "action": action,
        "columns": columns,
        "based_on_warning": based_on_warning,
        "transformation": transformation,
        "choice": choice,
    }


def _datetime_output_columns(source_column: str) -> list[str]:
    """Noms de colonnes dérivées — dérivés déterministiquement du nom de la
    colonne source (jamais lus depuis une spec externe) pour qu'il n'existe
    qu'une seule source de vérité sur le nommage, partagée entre la
    suggestion affichée et la transformation réellement appliquée."""
    return [f"{source_column}_{part}" for part in DATETIME_PARTS]


def _looks_like_datetime(series: pd.Series) -> bool:
    """Teste si une colonne texte ressemble à une date — sur un échantillon
    borné en coût (voir `DATETIME_SAMPLE_MAX_ROWS`)."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    sampled = sample_if_large(non_null.to_frame("v"), DATETIME_SAMPLE_MAX_ROWS)["v"]
    parsed = pd.to_datetime(sampled, errors="coerce", format="mixed")
    success_ratio = parsed.notna().mean()
    return bool(success_ratio >= DATETIME_PARSE_SUCCESS_THRESHOLD)


def suggest_datetime_columns(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Colonnes candidates à une décomposition datetime — déjà typées
    `datetime64`, ou colonnes texte dont un échantillon parse majoritairement
    comme une date (`read_dataframe` ne fait jamais de `parse_dates`, une
    date arrive donc systématiquement en texte, voir `services/datasets.py`).
    """
    suggestions: list[dict[str, Any]] = []
    for col in df.columns:
        series = df[col]
        is_candidate = pd.api.types.is_datetime64_any_dtype(series) or (
            not pd.api.types.is_numeric_dtype(series) and _looks_like_datetime(series)
        )
        if not is_candidate:
            continue
        parts = _datetime_output_columns(col)
        suggestions.append(_suggestion(
            code="decomposition_date",
            title=f"« {col} » ressemble à une date",
            explanation=(
                "Une date brute n'est pas directement exploitable par le modèle. La "
                "décomposer en composantes numériques (année, mois, jour, jour de la "
                "semaine) permet de capter des effets saisonniers ou calendaires."
            ),
            action=f"Décomposer « {col} » en {', '.join(parts)}.",
            columns=[col],
            transformation={"type": "datetime_decompose", "source_column": col},
        ))
    return suggestions


def _apply_one_datetime_decomposition(
    df: pd.DataFrame, feature_columns: list[str], transformation: dict[str, Any]
) -> tuple[pd.DataFrame, list[str]]:
    source = transformation["source_column"]
    if source not in df.columns:
        raise FeatureEngineeringSpecError(f"Colonne source '{source}' absente du dataset")

    parsed = pd.to_datetime(df[source], errors="coerce", format="mixed")
    result = df.copy()
    out_cols = _datetime_output_columns(source)
    result[out_cols[0]] = parsed.dt.year
    result[out_cols[1]] = parsed.dt.month
    result[out_cols[2]] = parsed.dt.day
    result[out_cols[3]] = parsed.dt.dayofweek
    # La colonne source (texte, non exploitable telle quelle) n'est jamais
    # retirée du DataFrame — seulement de la liste des colonnes vues par le
    # modèle, pour ne jamais risquer de supprimer par erreur la cible ou la
    # colonne de groupe si un appelant se trompait de liste.
    new_columns = [c for c in feature_columns if c != source] + out_cols
    return result, new_columns


def apply_datetime_decomposition(
    df: pd.DataFrame, feature_columns: list[str], transformations: list[dict[str, Any]]
) -> tuple[pd.DataFrame, list[str]]:
    """Applique, dans l'ordre, toutes les décompositions datetime approuvées
    (`type == "datetime_decompose"`) de la spec — fonction pure et
    déterministe, appelée à l'identique à l'entraînement et à l'inférence
    (voir `apply_upstream_feature_engineering`)."""
    result, columns = df, list(feature_columns)
    for transformation in transformations:
        if transformation.get("type") != "datetime_decompose":
            continue
        result, columns = _apply_one_datetime_decomposition(result, columns, transformation)
    return result, columns


# ── Ratios / interactions entre colonnes numériques ─────────────────────────
#
# Déterministe ligne-à-ligne (un ratio ne dépend d'aucune statistique
# agrégée) → applicable en amont du split, comme la décomposition datetime.
# Branché sur l'avertissement Lot B `collinearite_forte` : plutôt que de
# systématiquement retirer une des deux colonnes très corrélées (l'action
# actuellement recommandée par le garde-fou), un ratio entre les deux peut
# capturer une information relative que ni l'une ni l'autre ne porte seule
# (ex. "consommation / surface" plutôt que "consommation" et "surface" bruts).

RATIO_OUTPUT_SEPARATOR = "_sur_"


def _ratio_output_column(numerator: str, denominator: str) -> str:
    return f"{numerator}{RATIO_OUTPUT_SEPARATOR}{denominator}"


def suggest_ratio_features(quality_warnings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Suggestions de ratio, une par paire de colonnes signalée par le
    garde-fou Lot B `collinearite_forte` — ne redétecte rien, ne lit que les
    colonnes déjà remontées par `data_quality.analyze_data_quality`."""
    suggestions: list[dict[str, Any]] = []
    for warning in quality_warnings:
        if warning.get("code") != "collinearite_forte" or len(warning.get("columns", [])) != 2:
            continue
        c1, c2 = warning["columns"]
        output_column = _ratio_output_column(c1, c2)
        correlation = (warning.get("details") or {}).get("correlation")
        suggestions.append(_suggestion(
            code="ratio_colonnes_correlees",
            title=f"Créer un ratio entre « {c1} » et « {c2} »",
            explanation=(
                "Ces deux variables sont très corrélées"
                + (f" (corrélation {correlation:.2f})" if correlation is not None else "")
                + ". Plutôt que d'en retirer une, leur ratio peut capturer une "
                "information relative que ni l'une ni l'autre ne porte seule."
            ),
            action=f"Ajouter la variable « {output_column} » (= {c1} / {c2}).",
            columns=[c1, c2],
            based_on_warning="collinearite_forte",
            transformation={"type": "ratio", "numerator": c1, "denominator": c2},
        ))
    return suggestions


def _apply_one_ratio(
    df: pd.DataFrame, feature_columns: list[str], transformation: dict[str, Any]
) -> tuple[pd.DataFrame, list[str]]:
    numerator, denominator = transformation["numerator"], transformation["denominator"]
    for col in (numerator, denominator):
        if col not in df.columns:
            raise FeatureEngineeringSpecError(f"Colonne '{col}' absente du dataset")

    result = df.copy()
    output_column = _ratio_output_column(numerator, denominator)
    # Division par zéro → NaN plutôt qu'un +inf/-inf, pour rester géré par
    # l'imputation numérique déjà présente dans le préprocesseur en aval —
    # jamais un plantage ni une valeur numérique aberrante silencieuse.
    safe_denominator = result[denominator].replace(0, np.nan)
    result[output_column] = result[numerator].astype(float) / safe_denominator

    new_columns = list(feature_columns) if output_column in feature_columns else [*feature_columns, output_column]
    return result, new_columns


def apply_ratio_features(
    df: pd.DataFrame, feature_columns: list[str], transformations: list[dict[str, Any]]
) -> tuple[pd.DataFrame, list[str]]:
    """Applique, dans l'ordre, tous les ratios approuvés (`type == "ratio"`)
    de la spec — fonction pure et déterministe, symétrique de
    `apply_datetime_decomposition`."""
    result, columns = df, list(feature_columns)
    for transformation in transformations:
        if transformation.get("type") != "ratio":
            continue
        result, columns = _apply_one_ratio(result, columns, transformation)
    return result, columns
