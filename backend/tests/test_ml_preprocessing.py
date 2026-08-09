"""Tests de services/ml_preprocessing.py (Lot 3) — dédoublonnage, split anti-fuite."""
from __future__ import annotations

import numpy as np
import pandas as pd

from services.ml_preprocessing import remove_exact_duplicates, split_dataset


def test_remove_exact_duplicates_counts_correctly():
    df = pd.DataFrame({"x": [1, 1, 2, 3], "y": [10, 10, 20, 30]})
    deduped, n_removed = remove_exact_duplicates(df)
    assert n_removed == 1
    assert len(deduped) == 3


def test_group_split_has_zero_leakage():
    """Vérification indépendante de l'assertion interne à split_dataset :
    aucun groupe (échantillon répété) ne doit se retrouver des deux côtés."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {"x": rng.normal(size=n), "groupe": rng.integers(0, 40, n), "cible": rng.normal(size=n)}
    )
    split = split_dataset(
        df, target="cible", feature_columns=["x"], task_type="regression",
        group_column="groupe", test_size=0.2, seed=42,
    )
    assert split.groups_train is not None
    assert split.groups_test is not None
    assert set(split.groups_train).isdisjoint(set(split.groups_test))


def test_classification_split_is_stratified():
    rng = np.random.default_rng(1)
    n = 300
    df = pd.DataFrame(
        {"x": rng.normal(size=n), "cible": rng.choice(["a", "b", "c"], size=n, p=[0.6, 0.3, 0.1])}
    )
    split = split_dataset(
        df, target="cible", feature_columns=["x"], task_type="classification",
        group_column=None, test_size=0.2, seed=42,
    )
    train_ratio = split.y_train.value_counts(normalize=True)
    test_ratio = split.y_test.value_counts(normalize=True)
    for cls in ["a", "b", "c"]:
        assert abs(train_ratio[cls] - test_ratio[cls]) < 0.1
