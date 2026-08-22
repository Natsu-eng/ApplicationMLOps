"""Nettoyage, split anti-fuite et preprocessing — logique pure.

Le split groupé (`GroupShuffleSplit`/`GroupKFold`) et la vérification
explicite d'absence de fuite entre train et test sont directement inspirés
d'un notebook de référence partagé par l'équipe (méthodologie anti-fuite par
groupe, ex : plusieurs mesures répétées d'un même échantillon) — voir
`backend/workflow.md` (Lot 3) pour le détail de la méthodologie source.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── Lot 4c — regroupement des modalités rares + encodage de fréquence ──────
#
# Seuil FIXE (pas dérivé des fréquences observées) : un seuil qui dépendrait
# des données comptées sur le fold d'entraînement serait déjà fold-safe en
# soi, mais fixer sa VALEUR par une règle empirique évite tout raisonnement
# supplémentaire sur son origine — 1 % est la même intuition que le garde-fou
# Lot B `CARDINALITY_RATIO_THRESHOLD` (une modalité anecdotique n'apporte
# rien à un one-hot dédié), appliquée ici à l'échelle d'une seule modalité
# plutôt qu'à la colonne entière.
RARE_CATEGORY_FREQUENCY_THRESHOLD = 0.01


class DataLeakageError(RuntimeError):
    """Levée si le split groupé laisse un groupe présent à la fois en train et en test."""


class TrainingAbortedError(RuntimeError):
    """Erreur de pré-condition métier détectée AVANT tout calcul ML coûteux,
    avec un message déjà rédigé en français pour l'utilisateur (H7/H8,
    AUDIT_ROADMAP.md) — ex. dataset non prêt, classe absente du train après
    un split groupé. Type dédié (pas un `RuntimeError` nu) précisément pour
    que `workers/training_worker.py` puisse la distinguer avec certitude
    d'une exception technique de bibliothèque (elle aussi souvent un
    `RuntimeError`, ex. erreurs CatBoost) et la surfacer telle quelle sans
    risquer d'exposer un détail interne par erreur."""


@dataclass
class SplitResult:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    groups_train: Optional[np.ndarray]
    n_duplicates_removed: int
    # Exposé uniquement pour vérification indépendante (tests) que train et
    # test ne partagent aucun groupe — non utilisé par l'entraînement lui-même.
    groups_test: Optional[np.ndarray] = None


def remove_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Supprime les lignes strictement dupliquées — généralise le nettoyage
    "dédoublonnage" identifié comme étape préalable dans le notebook source
    (qui, lui, le faisait en amont sur une signature métier spécifique)."""
    before = len(df)
    deduped = df.drop_duplicates().reset_index(drop=True)
    return deduped, before - len(deduped)


def split_dataset(
    df: pd.DataFrame,
    target: str,
    feature_columns: list[str],
    task_type: str,
    group_column: Optional[str],
    test_size: float,
    seed: int,
) -> SplitResult:
    """Split train/test — groupé et vérifié anti-fuite si `group_column` est
    fourni (même échantillon jamais présent des deux côtés), sinon split
    classique (stratifié en classification)."""
    df, n_removed = remove_exact_duplicates(df)
    X = df[feature_columns]
    y = df[target]

    if group_column:
        groups = pd.factorize(df[group_column])[0]
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        overlap = set(groups[train_idx]) & set(groups[test_idx])
        if overlap:
            raise DataLeakageError(
                f"Fuite détectée entre train et test sur {len(overlap)} groupe(s) "
                f"de la colonne '{group_column}'"
            )
        return SplitResult(
            X_train=X.iloc[train_idx].reset_index(drop=True),
            X_test=X.iloc[test_idx].reset_index(drop=True),
            y_train=y.iloc[train_idx].reset_index(drop=True),
            y_test=y.iloc[test_idx].reset_index(drop=True),
            groups_train=groups[train_idx],
            n_duplicates_removed=n_removed,
            groups_test=groups[test_idx],
        )

    stratify = y if task_type == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )
    return SplitResult(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        groups_train=None,
        n_duplicates_removed=n_removed,
    )


class RareCategoryFrequencyEncoder(BaseEstimator, TransformerMixin):
    """Regroupe les modalités rares (fréquence < `rare_threshold` au fit)
    sous un bucket "_AUTRE_" implicite, puis encode chaque modalité par sa
    fréquence d'apparition — Lot 4c, alternative au one-hot pour une colonne
    à cardinalité excessive (garde-fou Lot B `cardinalite_excessive`).

    Fold-safe par construction : les fréquences ne sont calculées que dans
    `fit`, jamais recalculées dans `transform` — utilisé comme step de
    `Pipeline` à l'intérieur d'un `ColumnTransformer`, il est cloné et refit
    à l'intérieur de chaque fold de validation croisée comme le reste du
    préprocesseur (voir `ml_training.py`, Lot A), donc il ne voit jamais les
    fréquences du fold de validation.

    Cas limites explicitement pris en charge (voir tests dédiés) :
    - Modalité jamais vue au fit (absente du fold d'entraînement, présente en
      validation ou à l'inférence) → mappée sur la fréquence du bucket
      "_AUTRE_" (0.0 si aucune modalité rare n'a été regroupée au fit) —
      jamais un plantage, jamais un NaN silencieux.
    - Deux modalités de fréquence identique au fit → même valeur encodée.
      Choix conscient, pas un effet de bord : la fréquence est un résumé
      légitimement non injectif, acceptable pour des modèles à base d'arbres
      (seuils sur une valeur continue) qui ne dépendent pas d'un encodage
      bijectif comme le ferait un modèle linéaire.
    """

    def __init__(self, rare_threshold: float = RARE_CATEGORY_FREQUENCY_THRESHOLD):
        self.rare_threshold = rare_threshold

    def fit(self, X, y=None):
        values = pd.Series(np.asarray(X).ravel())
        frequencies = values.value_counts(normalize=True, dropna=True)
        rare_mask = frequencies < self.rare_threshold
        # Fréquence du bucket "_AUTRE_" — somme des fréquences regroupées, ou
        # 0.0 si aucune modalité rare n'existait au fit (cas limite : c'est
        # aussi la valeur de repli pour une modalité totalement inédite au
        # moment de `transform`, voir docstring de la classe).
        self.autre_frequency_ = float(frequencies[rare_mask].sum())
        self.frequency_map_ = frequencies[~rare_mask].to_dict()
        return self

    def transform(self, X):
        values = pd.Series(np.asarray(X).ravel())
        encoded = values.map(self.frequency_map_).fillna(self.autre_frequency_)
        return encoded.to_numpy(dtype=float).reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        base = str(input_features[0]) if input_features is not None and len(input_features) else "categorie"
        return np.asarray([f"{base}_frequence"], dtype=object)


def _resolve_imputation(col: str, is_numeric: bool, imputation_config: dict[str, Any]) -> tuple[str, Optional[Any]]:
    """Stratégie d'imputation effective pour une colonne — défaut historique
    (médiane pour le numérique, mode pour le catégoriel) sauf override
    explicite de l'utilisateur (Lot 4c, suggestion `imputation_configurable`,
    branchée sur le garde-fou Lot B `valeurs_manquantes_elevees`)."""
    override = imputation_config.get(col)
    if not override:
        return ("median", None) if is_numeric else ("most_frequent", None)
    strategy = override.get("strategy") or ("median" if is_numeric else "most_frequent")
    fill_value = override.get("fill_value") if strategy == "constant" else None
    return (strategy, fill_value)


def _group_by_imputation(
    cols: list[str], is_numeric: bool, imputation_config: dict[str, Any]
) -> dict[tuple[str, Optional[Any]], list[str]]:
    """Regroupe les colonnes partageant la même stratégie d'imputation dans
    un seul step de `ColumnTransformer` — minimise le nombre d'entrées quand
    peu de colonnes ont un override (cas courant : seules celles signalées
    par le garde-fou de valeurs manquantes en ont un)."""
    groups: dict[tuple[str, Optional[Any]], list[str]] = defaultdict(list)
    for col in cols:
        groups[_resolve_imputation(col, is_numeric, imputation_config)].append(col)
    return groups


def build_preprocessor(X: pd.DataFrame, feature_engineering_config: Optional[dict[str, Any]] = None) -> ColumnTransformer:
    """Imputation + normalisation (numérique) / one-hot (catégoriel) — fit
    uniquement sur le train, jamais sur test (appelé après le split).

    `feature_engineering_config` (Lot 4c, optionnel) — absent ou vide :
    comportement STRICTEMENT inchangé (rétrocompatibilité totale, voir test
    de non-régression structurelle dédié). Fourni, deux clés reconnues :
    - `frequency_encoding` : colonnes catégorielles à encoder par fréquence
      (`RareCategoryFrequencyEncoder`, ci-dessus) plutôt qu'en one-hot.
    - `imputation` : `{colonne: {"strategy": ..., "fill_value": ...}}`,
      stratégie par colonne (médiane/moyenne/mode/constante) plutôt que le
      défaut historique.
    """
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    if not feature_engineering_config:
        numeric_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        categorical_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers = []
        if numeric_cols:
            transformers.append(("num", numeric_pipe, numeric_cols))
        if categorical_cols:
            transformers.append(("cat", categorical_pipe, categorical_cols))
        return ColumnTransformer(transformers)

    freq_cols = [c for c in (feature_engineering_config.get("frequency_encoding") or []) if c in categorical_cols]
    onehot_cols = [c for c in categorical_cols if c not in freq_cols]
    imputation_config = feature_engineering_config.get("imputation") or {}

    transformers = []
    for i, ((strategy, fill_value), cols) in enumerate(_group_by_imputation(numeric_cols, True, imputation_config).items()):
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy=strategy, fill_value=fill_value)),
            ("scale", StandardScaler()),
        ])
        transformers.append((f"num_{i}", pipe, cols))

    for i, ((strategy, fill_value), cols) in enumerate(_group_by_imputation(onehot_cols, False, imputation_config).items()):
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy=strategy, fill_value=fill_value)),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append((f"cat_{i}", pipe, cols))

    for i, col in enumerate(freq_cols):
        strategy, fill_value = _resolve_imputation(col, False, imputation_config)
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy=strategy, fill_value=fill_value)),
            ("freq_encode", RareCategoryFrequencyEncoder()),
        ])
        transformers.append((f"freq_{i}", pipe, [col]))

    return ColumnTransformer(transformers)
