"""Tests de services/ml_preprocessing.py (Lot 3 : dédoublonnage, split
anti-fuite — Lot 4c : encodeur de fréquence fold-safe, imputation configurable)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from domains.shared.ml_preprocessing import (
    RareCategoryFrequencyEncoder,
    build_preprocessor,
    remove_exact_duplicates,
    split_dataset,
)


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


# ── Lot 4c — RareCategoryFrequencyEncoder ────────────────────────────────


def test_frequency_encoder_basic_frequencies():
    # 6x "a" (60%), 3x "b" (30%), 1x "c" (10%) — aucune sous 1%, rien de rare.
    values = pd.Series(["a"] * 6 + ["b"] * 3 + ["c"] * 1).to_numpy().reshape(-1, 1)
    encoder = RareCategoryFrequencyEncoder(rare_threshold=0.01)
    encoder.fit(values)
    encoded = encoder.transform(values)
    assert np.isclose(encoded[0, 0], 0.6)
    assert np.isclose(encoded[6, 0], 0.3)
    assert np.isclose(encoded[9, 0], 0.1)


def test_frequency_encoder_groups_rare_categories_under_autre():
    # "rare" apparaît 1 fois sur 200 (0.5 %), sous le seuil de 1 %.
    values = np.array(["frequent"] * 199 + ["rare"], dtype=object).reshape(-1, 1)
    encoder = RareCategoryFrequencyEncoder(rare_threshold=0.01).fit(values)
    encoded = encoder.transform(values)
    assert np.isclose(encoded[-1, 0], encoder.autre_frequency_)
    assert np.isclose(encoder.autre_frequency_, 1 / 200)


def test_frequency_encoder_unseen_category_at_transform_maps_to_autre_not_crash():
    """Cas limite explicitement exigé : une modalité jamais vue au fit ne
    doit jamais planter ni produire un NaN silencieux."""
    train_values = np.array(["a"] * 8 + ["b"] * 2, dtype=object).reshape(-1, 1)
    encoder = RareCategoryFrequencyEncoder(rare_threshold=0.01).fit(train_values)

    unseen = np.array(["totalement_inedit"], dtype=object).reshape(-1, 1)
    encoded = encoder.transform(unseen)

    assert not np.isnan(encoded[0, 0])
    assert encoded[0, 0] == encoder.autre_frequency_
    assert encoder.autre_frequency_ == 0.0  # aucune modalité rare regroupée au fit


def test_frequency_encoder_frequency_collision_documented_as_accepted():
    """Deux modalités de même fréquence au fit reçoivent la même valeur
    encodée — choix conscient (voir docstring de la classe), pas un bug."""
    values = np.array(["a"] * 5 + ["b"] * 5, dtype=object).reshape(-1, 1)
    encoder = RareCategoryFrequencyEncoder(rare_threshold=0.01).fit(values)
    encoded = encoder.transform(values)
    assert np.isclose(encoded[0, 0], encoded[5, 0])  # "a" et "b" : même fréquence (50%)


def test_frequency_encoder_is_fold_safe_fit_on_subset_only():
    """Preuve d'absence de fuite : les fréquences apprises sur un sous-
    ensemble (fold d'entraînement) restent celles utilisées pour transformer
    un autre sous-ensemble (fold de validation), jamais recalculées dessus."""
    fold_train = np.array(["x"] * 9 + ["y"] * 1, dtype=object).reshape(-1, 1)  # x=90%, y=10%
    fold_valid = np.array(["x"] * 1 + ["y"] * 9, dtype=object).reshape(-1, 1)  # inverse

    encoder = RareCategoryFrequencyEncoder(rare_threshold=0.01).fit(fold_train)
    encoded_valid = encoder.transform(fold_valid)

    # Si l'encodeur avait été (re)fit sur fold_valid, "x" vaudrait 0.1 ici —
    # il doit garder la fréquence apprise sur fold_train (0.9).
    assert np.isclose(encoded_valid[0, 0], 0.9)
    assert np.isclose(encoded_valid[1, 0], 0.1)


# ── Lot 4c — build_preprocessor étendu ───────────────────────────────────


def test_build_preprocessor_without_config_is_structurally_unchanged():
    """Non-régression explicite : sans feature_engineering_config, le
    ColumnTransformer produit garde exactement la structure historique
    (2 blocs nommés "num"/"cat", mêmes colonnes)."""
    df = pd.DataFrame({"num1": [1.0, 2.0], "num2": [3.0, 4.0], "cat1": ["a", "b"]})
    pre_none = build_preprocessor(df)
    pre_empty = build_preprocessor(df, feature_engineering_config=None)
    for pre in (pre_none, pre_empty):
        names = [name for name, _, _ in pre.transformers]
        assert names == ["num", "cat"]
        assert dict((name, cols) for name, _, cols in pre.transformers) == {
            "num": ["num1", "num2"], "cat": ["cat1"],
        }


def test_build_preprocessor_frequency_encoding_is_fold_safe_in_column_transformer():
    """Reproduit le scénario réel : le ColumnTransformer complet (pas
    l'encodeur isolé) fit sur un sous-ensemble encode un autre sous-ensemble
    avec les fréquences apprises sur le premier."""
    df_fit = pd.DataFrame({"ville": ["paris"] * 9 + ["lyon"] * 1})
    df_other = pd.DataFrame({"ville": ["paris"] * 1 + ["lyon"] * 9})

    pre = build_preprocessor(df_fit, feature_engineering_config={"frequency_encoding": ["ville"]})
    pre.fit(df_fit)
    encoded_other = np.asarray(pre.transform(df_other))

    assert np.isclose(encoded_other[0, 0], 0.9)  # "paris" garde la fréquence apprise (90%)


def test_build_preprocessor_imputation_override_uses_configured_strategy():
    # Médiane (15) et moyenne (43.33) divergent nettement sur ce jeu — bon
    # cas pour distinguer les deux stratégies après imputation + mise à l'échelle.
    df = pd.DataFrame({"age": [10.0, np.nan, 20.0, 100.0]})
    median_pre = build_preprocessor(df)
    mean_pre = build_preprocessor(df, feature_engineering_config={"imputation": {"age": {"strategy": "mean"}}})
    median_val = median_pre.fit_transform(df)[1, 0]
    mean_val = mean_pre.fit_transform(df)[1, 0]
    assert not np.isclose(median_val, mean_val)


def test_build_preprocessor_constant_imputation_uses_fill_value():
    df = pd.DataFrame({"age": [10.0, np.nan, 20.0]})
    pre = build_preprocessor(df, feature_engineering_config={
        "imputation": {"age": {"strategy": "constant", "fill_value": 0.0}}
    })
    # Vérifie via le pipeline complet plutôt que l'implémentation interne :
    # la valeur imputée (0.0) doit se retrouver, une fois standardisée, à la
    # position de la valeur manquante.
    transformed = pre.fit_transform(df)
    raw_values = np.array([10.0, 0.0, 20.0])
    expected_scaled = (raw_values - raw_values.mean()) / raw_values.std()
    assert np.isclose(transformed[1, 0], expected_scaled[1])
