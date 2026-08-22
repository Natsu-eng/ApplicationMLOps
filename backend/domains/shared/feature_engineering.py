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

from domains.shared.data_quality import analyze_data_quality, _try_parse_numeric_text
from domains.shared.stats_utils import sample_if_large

CURRENT_SPEC_VERSION = 1
# Versionnée dès la première spec (Lot 4c) : une spec persistée (bundle
# joblib + TrainingJob.feature_engineering_json) doit rester rejouable à
# l'identique à l'inférence, potentiellement des mois après l'entraînement.
# Une version inconnue/incompatible est un refus explicite (voir
# `validate_spec_version`), jamais une interprétation best-effort qui
# produirait une prédiction fausse silencieusement.

_UPSTREAM_TRANSFORMATION_TYPES = {"datetime_decompose", "ratio", "numeric_coerce"}
# Transformations déterministes (ne dépendent d'aucune statistique agrégée) —
# seules celles-ci sont rejouées par `apply_upstream_feature_engineering`.
# Les autres types de transformation (`frequency_encoding`, `imputation`)
# vivent dans `spec["pipeline"]`, consommé par
# `ml_preprocessing.build_preprocessor`, pas par ce module.
#
# `exclude_column` (Lot Nettoyage guidé des variables) n'y figure PAS : ce
# n'est pas une transformation de pipeline mais un pur repère UI (voir
# `_suggest_column_exclusion` plus bas) — approuver cette suggestion revient
# à décocher la colonne dans `TrainingJobCreate.feature_columns`, jamais à
# l'envoyer dans `spec["upstream"]`. Si elle y apparaissait quand même,
# `apply_upstream_feature_engineering` lèverait `FeatureEngineeringSpecError`
# (type inconnu) plutôt que de l'ignorer silencieusement.

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


# ── Conversion numérique (Lot Nettoyage guidé des variables) ────────────────
#
# Déterministe ligne-à-ligne (chaque valeur est reparsée indépendamment des
# autres, aucune statistique agrégée) → applicable en amont du split, comme
# la décomposition datetime et les ratios. Branché sur l'avertissement Lot B
# `numerique_mal_type` (`services/data_quality.py`) : réutilise EXACTEMENT le
# même parseur (`_try_parse_numeric_text`) que celui qui a servi à détecter
# la colonne, pour que la suggestion affichée et la conversion réellement
# appliquée ne puissent jamais diverger.


def suggest_numeric_coercion(quality_warnings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Suggestions de conversion, une par colonne signalée par le garde-fou
    Lot B `numerique_mal_type` — ne redétecte rien, ne lit que les colonnes
    déjà remontées par `data_quality.analyze_data_quality`."""
    suggestions: list[dict[str, Any]] = []
    for warning in quality_warnings:
        if warning.get("code") != "numerique_mal_type" or not warning.get("columns"):
            continue
        col = warning["columns"][0]
        suggestions.append(_suggestion(
            code="conversion_numerique",
            title=f"Convertir « {col} » en variable numérique",
            explanation=(
                "Cette colonne contient des nombres écrits avec une virgule décimale "
                "ou un séparateur de milliers, donc traités comme du texte. La "
                "convertir en nombre lui permet d'être exploitée comme une vraie "
                "variable numérique plutôt qu'une catégorie sans rapport d'ordre."
            ),
            action=f"Convertir « {col} » en nombre.",
            columns=[col],
            based_on_warning="numerique_mal_type",
            transformation={"type": "numeric_coerce", "column": col},
        ))
    return suggestions


def _apply_one_numeric_coercion(
    df: pd.DataFrame, feature_columns: list[str], transformation: dict[str, Any]
) -> tuple[pd.DataFrame, list[str]]:
    col = transformation["column"]
    if col not in df.columns:
        raise FeatureEngineeringSpecError(f"Colonne '{col}' absente du dataset")

    result = df.copy()
    # Remplace la colonne EN PLACE (même nom, même position dans
    # `feature_columns`) — contrairement à la décomposition datetime/ratio,
    # qui créent de nouvelles colonnes : convertir un texte en nombre ne
    # change pas ce que la colonne représente, seulement son type. Une valeur
    # qui ne parse pas devient NaN, jamais une exception — reprise ensuite
    # par l'imputation déjà présente dans le préprocesseur en aval, comme
    # toute autre valeur manquante.
    result[col] = result[col].map(lambda v: _try_parse_numeric_text(str(v)) if pd.notna(v) else np.nan)
    return result, list(feature_columns)


def apply_numeric_coercion(
    df: pd.DataFrame, feature_columns: list[str], transformations: list[dict[str, Any]]
) -> tuple[pd.DataFrame, list[str]]:
    """Applique, dans l'ordre, toutes les conversions numériques approuvées
    (`type == "numeric_coerce"`) de la spec — fonction pure et déterministe,
    même patron que `apply_datetime_decomposition`/`apply_ratio_features`."""
    result, columns = df, list(feature_columns)
    for transformation in transformations:
        if transformation.get("type") != "numeric_coerce":
            continue
        result, columns = _apply_one_numeric_coercion(result, columns, transformation)
    return result, columns


# ── Exclusion de variables — suggestion uniquement, jamais une transformation
#
# Contrairement à tout ce qui précède, exclure une colonne ne modifie aucune
# donnée : c'est une suggestion UI pure, consommée en décochant la colonne
# dans `TrainingJobCreate.feature_columns` (mécanisme qui existe depuis le
# Lot 3). Son `transformation` (`{"type": "exclude_column", ...}`) n'apparaît
# donc JAMAIS dans `spec["upstream"]` — voir la note sur
# `_UPSTREAM_TRANSFORMATION_TYPES` en tête de module.

_EXCLUSION_WARNING_CODES = {"colonne_constante", "cardinalite_excessive", "colonnes_dupliquees"}


def _suggest_column_exclusion(quality_warnings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Suggestions d'exclusion — une par colonne signalée comme sans valeur
    prédictive (constante/quasi-constante, cardinalité d'identifiant, ou
    doublon exact d'une autre colonne, Lot B). Répond à une lacune identifiée
    à l'usage : ces trois garde-fous recommandaient déjà textuellement de
    retirer la colonne, sans qu'aucune suggestion actionnable ne relie cette
    recommandation à la sélection de variables du formulaire d'entraînement."""
    suggestions: list[dict[str, Any]] = []
    for warning in quality_warnings:
        code = warning.get("code")
        if code not in _EXCLUSION_WARNING_CODES:
            continue
        columns = warning.get("columns") or []
        # Doublon exact : ne suggérer d'exclure que la SECONDE des deux
        # colonnes (garder au moins une des deux, contenu strictement
        # identique) — les deux autres codes ne portent qu'une seule colonne.
        target_cols = columns[1:] if code == "colonnes_dupliquees" and len(columns) == 2 else columns
        for col in target_cols:
            suggestions.append(_suggestion(
                code="exclusion_variable",
                title=f"Exclure « {col} » des variables utilisées",
                explanation=warning.get("explanation", ""),
                action=f"Ne pas utiliser « {col} » pour l'entraînement — elle n'apporte aucune valeur prédictive.",
                columns=[col],
                based_on_warning=code,
                transformation={"type": "exclude_column", "column": col},
            ))
    return suggestions


# ── Regroupement de fréquence & imputation — suggestions uniquement ─────────
#
# Les transformations elles-mêmes (step de `Pipeline` fold-safe) vivent dans
# `ml_preprocessing.build_preprocessor` (Lot 4c) — ce module ne fait ici que
# proposer les suggestions en langage clair, branchées sur les garde-fous
# Lot B déjà calculés, sans redétection.

_IMPUTATION_STRATEGIES_NUMERIC = ("mean", "median", "constant")
_IMPUTATION_STRATEGIES_CATEGORICAL = ("most_frequent", "constant")


def _suggest_frequency_encoding(quality_warnings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []
    for warning in quality_warnings:
        if warning.get("code") != "cardinalite_excessive" or not warning.get("columns"):
            continue
        col = warning["columns"][0]
        suggestions.append(_suggestion(
            code="regroupement_frequence",
            title=f"Regrouper les valeurs rares de « {col} » et encoder par fréquence",
            explanation=(
                "Cette colonne a trop de catégories différentes pour un encodage "
                "classique (one-hot). Regrouper les catégories rares sous une "
                "modalité commune, puis encoder chaque valeur par sa fréquence "
                "d'apparition, évite l'explosion du nombre de colonnes."
            ),
            action=f"Appliquer le regroupement + encodage de fréquence sur « {col} ».",
            columns=[col],
            based_on_warning="cardinalite_excessive",
            transformation={"type": "frequency_encoding", "column": col},
        ))
    return suggestions


def _suggest_imputation(quality_warnings: list[dict[str, Any]], df: pd.DataFrame) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []
    for warning in quality_warnings:
        if warning.get("code") != "valeurs_manquantes_elevees" or not warning.get("columns"):
            continue
        col = warning["columns"][0]
        is_numeric = pd.api.types.is_numeric_dtype(df[col]) if col in df.columns else True
        default_strategy = "median" if is_numeric else "most_frequent"
        options = _IMPUTATION_STRATEGIES_NUMERIC if is_numeric else _IMPUTATION_STRATEGIES_CATEGORICAL
        suggestions.append(_suggestion(
            code="imputation_configurable",
            title=f"Choisir la stratégie d'imputation de « {col} »",
            explanation=(
                "Beaucoup de valeurs manquent sur cette colonne — le remplacement "
                "par défaut (médiane pour une variable numérique, valeur la plus "
                "fréquente pour une catégorielle) n'est pas toujours le plus adapté."
            ),
            action=f"Choisir une stratégie d'imputation pour « {col} » (par défaut : {default_strategy}).",
            columns=[col],
            based_on_warning="valeurs_manquantes_elevees",
            transformation={"type": "imputation", "column": col, "strategy": default_strategy},
            choice={"type": "imputation_strategy", "options": list(options), "default": default_strategy},
        ))
    return suggestions


def suggest_feature_engineering(
    df: pd.DataFrame, target_column: str, group_column: Optional[str] = None
) -> list[dict[str, Any]]:
    """Point d'entrée — réutilise `analyze_data_quality` (Lot B) une seule
    fois et en dérive toutes les suggestions de ce lot, plutôt que de
    redétecter les mêmes problèmes. Même exclusion cible/groupe que
    `analyze_data_quality` pour la détection datetime (aucun sens à proposer
    de décomposer la cible ou la colonne de groupe)."""
    quality_warnings = analyze_data_quality(df, target_column, group_column)

    excluded = {target_column} | ({group_column} if group_column else set())
    feature_df = df[[c for c in df.columns if c not in excluded]]

    suggestions: list[dict[str, Any]] = []
    suggestions += _suggest_column_exclusion(quality_warnings)
    suggestions += suggest_datetime_columns(feature_df)
    suggestions += suggest_numeric_coercion(quality_warnings)
    suggestions += suggest_ratio_features(quality_warnings)
    suggestions += _suggest_frequency_encoding(quality_warnings)
    suggestions += _suggest_imputation(quality_warnings, df)
    return suggestions


def validate_spec_version(spec: dict[str, Any]) -> None:
    version = spec.get("version")
    if version != CURRENT_SPEC_VERSION:
        raise FeatureEngineeringSpecError(
            f"Version de spec de feature engineering non supportée : {version!r} "
            f"(version attendue : {CURRENT_SPEC_VERSION}). Un modèle entraîné avec "
            "une version différente ne peut pas être rejoué de façon fiable."
        )


def apply_upstream_feature_engineering(
    df: pd.DataFrame, feature_columns: list[str], spec: Optional[dict[str, Any]]
) -> tuple[pd.DataFrame, list[str]]:
    """Fonction UNIQUE et partagée, appelée à l'identique par le worker
    d'entraînement (`workers/training_worker.py`, avant `split_dataset`) et
    par l'inférence (`services/ml_inference.py`, avant
    `preprocessor.transform`) — c'est elle qui garantit que les
    transformations déterministes (datetime, ratio) produisent exactement
    les mêmes colonnes dans les deux contextes (voir docstring du module).

    `spec` absente ou vide → no-op, comportement historique inchangé
    (rétrocompatibilité totale). Sinon, la version est vérifiée avant toute
    application (voir `validate_spec_version`) et tout type de transformation
    `upstream` inconnu lève une erreur explicite plutôt que d'être ignoré
    silencieusement."""
    if not spec:
        return df, list(feature_columns)

    validate_spec_version(spec)
    upstream = spec.get("upstream") or []
    unknown_types = {t.get("type") for t in upstream} - _UPSTREAM_TRANSFORMATION_TYPES
    if unknown_types:
        raise FeatureEngineeringSpecError(f"Type(s) de transformation amont inconnu(s) : {sorted(unknown_types)}")

    # Coercion numérique appliquée EN PREMIER : un ratio ou une décomposition
    # datetime référençant une colonne "numérique mal typée" doit voir sa
    # forme déjà convertie, jamais le texte brut d'origine.
    result, columns = apply_numeric_coercion(df, feature_columns, upstream)
    result, columns = apply_datetime_decomposition(result, columns, upstream)
    result, columns = apply_ratio_features(result, columns, upstream)
    return result, columns
