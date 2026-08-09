"""Tests de services/feature_engineering.py (Lot 4c)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from services.feature_engineering import (
    CURRENT_SPEC_VERSION,
    FeatureEngineeringSpecError,
    apply_datetime_decomposition,
    apply_ratio_features,
    apply_upstream_feature_engineering,
    suggest_datetime_columns,
    suggest_feature_engineering,
    suggest_ratio_features,
    validate_spec_version,
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


# ── Ratios / interactions ────────────────────────────────────────────────


def _collinearity_warning(c1="surface", c2="consommation", correlation=0.95):
    return {
        "level": "info",
        "code": "collinearite_forte",
        "title": f"« {c1} » et « {c2} » sont très corrélées",
        "explanation": "...",
        "action": "...",
        "columns": [c1, c2],
        "details": {"correlation": correlation},
    }


def test_suggest_ratio_features_branches_on_collinearity_warning():
    warnings = [_collinearity_warning(), {"code": "cardinalite_excessive", "columns": ["x"], "details": None}]
    suggestions = suggest_ratio_features(warnings)
    assert len(suggestions) == 1
    assert suggestions[0]["based_on_warning"] == "collinearite_forte"
    assert suggestions[0]["transformation"] == {
        "type": "ratio", "numerator": "surface", "denominator": "consommation",
    }


def test_suggest_ratio_features_ignores_unrelated_warnings():
    assert suggest_ratio_features([{"code": "valeurs_manquantes_elevees", "columns": ["x"]}]) == []


def test_apply_ratio_features_computes_division():
    df = pd.DataFrame({"a": [10.0, 20.0], "b": [2.0, 4.0]})
    result, columns = apply_ratio_features(df, ["a", "b"], [{"type": "ratio", "numerator": "a", "denominator": "b"}])
    assert columns == ["a", "b", "a_sur_b"]
    assert list(result["a_sur_b"]) == [5.0, 5.0]


def test_apply_ratio_features_division_by_zero_becomes_nan_not_inf():
    df = pd.DataFrame({"a": [10.0], "b": [0.0]})
    result, _ = apply_ratio_features(df, ["a", "b"], [{"type": "ratio", "numerator": "a", "denominator": "b"}])
    assert pd.isna(result.loc[0, "a_sur_b"])


def test_apply_ratio_features_is_deterministic_single_row_vs_batch():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"a": rng.normal(50, 10, 40), "b": rng.normal(5, 2, 40)})
    spec = [{"type": "ratio", "numerator": "a", "denominator": "b"}]
    batch_result, _ = apply_ratio_features(df, ["a", "b"], spec)
    single_result, _ = apply_ratio_features(df.iloc[[12]].reset_index(drop=True), ["a", "b"], spec)
    assert single_result.loc[0, "a_sur_b"] == batch_result.loc[12, "a_sur_b"]


# ── Orchestration : suggest_feature_engineering / apply_upstream_feature_engineering ──


def _rich_df(n=300, seed=5):
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2021-01-01")
    surface = rng.normal(100, 20, n)
    consommation = surface * 3 + rng.normal(0, 1, n)  # quasi colinéaire à surface
    ville = rng.choice([f"ville_{i}" for i in range(80)], n)  # cardinalité excessive
    revenu = rng.normal(2000, 500, n)
    revenu[rng.choice(n, size=int(n * 0.4), replace=False)] = np.nan  # >30% manquant
    cible = 2 * surface - consommation + rng.normal(0, 5, n)
    return pd.DataFrame({
        "date_signature": [(base + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d") for d in rng.integers(0, 900, n)],
        "surface": surface,
        "consommation": consommation,
        "ville": ville,
        "revenu": revenu,
        "cible": cible,
    })


def test_suggest_feature_engineering_covers_all_transformation_types():
    df = _rich_df()
    suggestions = suggest_feature_engineering(df, target_column="cible")
    codes = {s["code"] for s in suggestions}
    assert codes == {
        "decomposition_date", "ratio_colonnes_correlees",
        "regroupement_frequence", "imputation_configurable",
    }
    # Chaque suggestion branchée sur un garde-fou Lot B porte bien son code d'origine.
    by_code = {s["code"]: s for s in suggestions}
    assert by_code["regroupement_frequence"]["based_on_warning"] == "cardinalite_excessive"
    assert by_code["imputation_configurable"]["based_on_warning"] == "valeurs_manquantes_elevees"
    assert by_code["ratio_colonnes_correlees"]["based_on_warning"] == "collinearite_forte"
    assert by_code["decomposition_date"]["based_on_warning"] is None


def test_suggest_feature_engineering_excludes_target_and_group_from_datetime_scan():
    df = _rich_df()
    df["groupe_date"] = df["date_signature"]  # colonne de groupe elle-même une date
    suggestions = suggest_feature_engineering(df, target_column="cible", group_column="groupe_date")
    datetime_columns = {c for s in suggestions if s["code"] == "decomposition_date" for c in s["columns"]}
    assert "groupe_date" not in datetime_columns


def test_apply_upstream_feature_engineering_noop_on_empty_spec():
    df = _rich_df(n=10)
    result, columns = apply_upstream_feature_engineering(df, ["surface", "ville"], None)
    assert columns == ["surface", "ville"]
    assert result is df


def test_apply_upstream_feature_engineering_combines_datetime_and_ratio():
    df = _rich_df(n=10)
    spec = {
        "version": CURRENT_SPEC_VERSION,
        "upstream": [
            {"type": "datetime_decompose", "source_column": "date_signature"},
            {"type": "ratio", "numerator": "surface", "denominator": "consommation"},
        ],
        "pipeline": {"frequency_encoding": ["ville"]},
    }
    result, columns = apply_upstream_feature_engineering(
        df, ["date_signature", "surface", "consommation", "ville"], spec
    )
    assert "date_signature" not in columns
    assert "date_signature_annee" in columns
    assert "surface_sur_consommation" in columns
    assert "ville" in columns  # frequency_encoding est une clé pipeline, pas upstream — colonne inchangée ici


def test_apply_upstream_feature_engineering_rejects_wrong_version():
    df = _rich_df(n=5)
    spec = {"version": 999, "upstream": []}
    with pytest.raises(FeatureEngineeringSpecError):
        apply_upstream_feature_engineering(df, ["surface"], spec)


def test_apply_upstream_feature_engineering_rejects_unknown_transformation_type():
    df = _rich_df(n=5)
    spec = {"version": CURRENT_SPEC_VERSION, "upstream": [{"type": "target_encoding", "column": "ville"}]}
    with pytest.raises(FeatureEngineeringSpecError):
        apply_upstream_feature_engineering(df, ["ville"], spec)


def test_validate_spec_version_accepts_current_rejects_other():
    validate_spec_version({"version": CURRENT_SPEC_VERSION})
    with pytest.raises(FeatureEngineeringSpecError):
        validate_spec_version({"version": CURRENT_SPEC_VERSION + 1})
    with pytest.raises(FeatureEngineeringSpecError):
        validate_spec_version({})
