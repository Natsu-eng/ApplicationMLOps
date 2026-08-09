"""Nettoyage, split anti-fuite et preprocessing — logique pure.

Le split groupé (`GroupShuffleSplit`/`GroupKFold`) et la vérification
explicite d'absence de fuite entre train et test sont directement inspirés
d'un notebook de référence partagé par l'équipe (méthodologie anti-fuite par
groupe, ex : plusieurs mesures répétées d'un même échantillon) — voir
`backend/workflow.md` (Lot 3) pour le détail de la méthodologie source.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataLeakageError(RuntimeError):
    """Levée si le split groupé laisse un groupe présent à la fois en train et en test."""


@dataclass
class SplitResult:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    groups_train: Optional[np.ndarray]
    n_duplicates_removed: int


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


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Imputation + normalisation (numérique) / one-hot (catégoriel) — fit
    uniquement sur le train, jamais sur test (appelé après le split)."""
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

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
