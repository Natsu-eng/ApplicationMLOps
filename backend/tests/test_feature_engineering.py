"""Tests de services/feature_engineering.py (Lot 4c)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from domains.shared.feature_engineering import (
    CURRENT_SPEC_VERSION,
    FeatureEngineeringSpecError,
    apply_datetime_decomposition,
    apply_numeric_coercion,
    apply_ratio_features,
    apply_upstream_feature_engineering,
    suggest_datetime_columns,
    suggest_feature_engineering,
    suggest_numeric_coercion,
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
        "exclusion_variable",  # "ville" (cardinalité excessive) — Lot Nettoyage guidé des variables
    }
    # Chaque suggestion branchée sur un garde-fou Lot B porte bien son code d'origine.
    by_code = {s["code"]: s for s in suggestions}
    assert by_code["regroupement_frequence"]["based_on_warning"] == "cardinalite_excessive"
    assert by_code["imputation_configurable"]["based_on_warning"] == "valeurs_manquantes_elevees"
    assert by_code["ratio_colonnes_correlees"]["based_on_warning"] == "collinearite_forte"
    assert by_code["exclusion_variable"]["based_on_warning"] == "cardinalite_excessive"
    assert by_code["exclusion_variable"]["columns"] == ["ville"]
    assert by_code["decomposition_date"]["based_on_warning"] is None


def test_suggest_feature_engineering_excludes_target_and_group_from_datetime_scan():
    df = _rich_df()
    df["groupe_date"] = df["date_signature"]  # colonne de groupe elle-même une date
    suggestions = suggest_feature_engineering(df, target_column="cible", group_column="groupe_date")
    datetime_columns = {c for s in suggestions if s["code"] == "decomposition_date" for c in s["columns"]}
    assert "groupe_date" not in datetime_columns


# ── excluded_columns (retour utilisateur direct — diagnostic de cohérence
# du wizard : "ref_complete exclue à l'étape 1, l'étape 3 propose encore de
# l'encoder") ────────────────────────────────────────────────────────────


def test_excluded_columns_removes_suggestions_about_them():
    """« ville » exclue à l'étape 1 : plus aucune suggestion ne doit la
    mentionner, ni l'exclusion elle-même (déjà faite) ni le regroupement
    par fréquence qui en dérivait — même si `date_signature` (colonne
    distincte, également à cardinalité excessive dans ce jeu de données)
    continue légitimement de déclencher ces deux mêmes codes pour
    elle-même : la vérification porte sur les COLONNES mentionnées, pas sur
    la simple présence du code dans l'ensemble des suggestions."""
    df = _rich_df()
    suggestions_before = suggest_feature_engineering(df, target_column="cible")
    ville_suggestions_before = [s for s in suggestions_before if "ville" in s["columns"]]
    assert ville_suggestions_before  # présentes avant exclusion (garde-fou du test)

    suggestions = suggest_feature_engineering(df, target_column="cible", excluded_columns={"ville"})
    mentioned_columns = {c for s in suggestions for c in s["columns"]}
    assert "ville" not in mentioned_columns
    # Les suggestions sur `date_signature` (indépendantes de « ville »)
    # doivent rester intactes.
    assert any(s["code"] == "exclusion_variable" and s["columns"] == ["date_signature"] for s in suggestions)
    assert any(s["code"] == "regroupement_frequence" and s["columns"] == ["date_signature"] for s in suggestions)


def test_excluded_columns_removes_datetime_suggestion_for_excluded_column():
    """La détection datetime ne dérive PAS des avertissements Lot B (seule
    exception dans `suggest_feature_engineering`) — vérifiée séparément."""
    df = _rich_df()
    suggestions = suggest_feature_engineering(df, target_column="cible", excluded_columns={"date_signature"})
    datetime_columns = {c for s in suggestions if s["code"] == "decomposition_date" for c in s["columns"]}
    assert "date_signature" not in datetime_columns


def test_excluded_columns_does_not_affect_still_included_suggestions():
    """Exclure « ville » ne doit jamais faire disparaître les suggestions
    sur les AUTRES colonnes toujours retenues (date, ratio, imputation)."""
    df = _rich_df()
    suggestions = suggest_feature_engineering(df, target_column="cible", excluded_columns={"ville"})
    codes = {s["code"] for s in suggestions}
    assert {"decomposition_date", "ratio_colonnes_correlees", "imputation_configurable"} <= codes


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


# ── Conversion numérique (Lot Nettoyage guidé des variables) ────────────────


def _mistyped_numeric_warning(col="prix"):
    return {
        "level": "attention",
        "code": "numerique_mal_type",
        "title": f"« {col} » ressemble à une variable numérique stockée en texte",
        "explanation": "...",
        "action": "...",
        "columns": [col],
        "details": {"parse_success_ratio": 0.99},
    }


def test_suggest_numeric_coercion_branches_on_mistyped_warning():
    suggestions = suggest_numeric_coercion([_mistyped_numeric_warning(), _collinearity_warning()])
    assert len(suggestions) == 1
    assert suggestions[0]["based_on_warning"] == "numerique_mal_type"
    assert suggestions[0]["transformation"] == {"type": "numeric_coerce", "column": "prix"}


def test_suggest_numeric_coercion_ignores_unrelated_warnings():
    assert suggest_numeric_coercion([{"code": "collinearite_forte", "columns": ["x", "y"]}]) == []


def test_apply_numeric_coercion_parses_comma_decimal_in_place():
    df = pd.DataFrame({"prix": ["1 234,50", "999,00"], "autre": ["a", "b"]})
    result, columns = apply_numeric_coercion(df, ["prix", "autre"], [{"type": "numeric_coerce", "column": "prix"}])
    assert columns == ["prix", "autre"]  # même colonne, pas de nouvelle colonne créée
    assert list(result["prix"]) == [1234.50, 999.00]
    assert pd.api.types.is_float_dtype(result["prix"])


def test_apply_numeric_coercion_unparseable_value_becomes_nan_not_crash():
    df = pd.DataFrame({"prix": ["1 234,50", "n'importe quoi"]})
    result, _ = apply_numeric_coercion(df, ["prix"], [{"type": "numeric_coerce", "column": "prix"}])
    assert pd.isna(result.loc[1, "prix"])


def test_apply_numeric_coercion_is_deterministic_single_row_vs_batch():
    df = pd.DataFrame({"prix": [f"1 {200 + i:03d},{i:02d}" for i in range(20)]})
    spec = [{"type": "numeric_coerce", "column": "prix"}]
    batch_result, _ = apply_numeric_coercion(df, ["prix"], spec)
    single_result, _ = apply_numeric_coercion(df.iloc[[5]].reset_index(drop=True), ["prix"], spec)
    assert single_result.loc[0, "prix"] == batch_result.loc[5, "prix"]


def test_apply_upstream_feature_engineering_applies_numeric_coercion_before_ratio():
    """La coercion numérique doit s'appliquer AVANT le ratio dans
    `apply_upstream_feature_engineering` : un ratio référençant une colonne
    mal typée doit voir sa forme déjà convertie."""
    df = pd.DataFrame({"prix": ["1 000,00", "2 000,00"], "surface": [10.0, 20.0]})
    spec = {
        "version": CURRENT_SPEC_VERSION,
        "upstream": [
            {"type": "numeric_coerce", "column": "prix"},
            {"type": "ratio", "numerator": "prix", "denominator": "surface"},
        ],
        "pipeline": {},
    }
    result, columns = apply_upstream_feature_engineering(df, ["prix", "surface"], spec)
    assert "prix_sur_surface" in columns
    assert list(result["prix_sur_surface"]) == [100.0, 100.0]


# ── Exclusion de variables (Lot Nettoyage guidé des variables) ──────────────


def test_suggest_feature_engineering_suggests_exclusion_for_constant_column():
    df = _rich_df()
    df["toujours_pareil"] = 42
    suggestions = suggest_feature_engineering(df, target_column="cible")
    exclusions = {s["columns"][0] for s in suggestions if s["code"] == "exclusion_variable"}
    assert "toujours_pareil" in exclusions
    assert "ville" in exclusions  # cardinalité excessive, déjà couverte plus haut


def test_suggest_feature_engineering_suggests_exclusion_for_duplicate_column():
    df = _rich_df()
    df["surface_bis"] = df["surface"]
    suggestions = suggest_feature_engineering(df, target_column="cible")
    exclusions = [s for s in suggestions if s["code"] == "exclusion_variable" and s["columns"] == ["surface_bis"]]
    assert len(exclusions) == 1
    assert exclusions[0]["based_on_warning"] == "colonnes_dupliquees"
    # "surface" (la première des deux) n'est jamais elle-même suggérée à l'exclusion.
    assert not any(s["columns"] == ["surface"] for s in suggestions if s["code"] == "exclusion_variable")


def test_exclude_column_transformation_never_reaches_upstream_apply():
    """`exclude_column` est un repère UI pur (voir docstring du module) —
    s'il apparaissait quand même dans `spec["upstream"]` (erreur du client),
    il doit être rejeté explicitement, jamais silencieusement ignoré."""
    df = pd.DataFrame({"x": [1, 2, 3]})
    spec = {"version": CURRENT_SPEC_VERSION, "upstream": [{"type": "exclude_column", "column": "x"}]}
    with pytest.raises(FeatureEngineeringSpecError):
        apply_upstream_feature_engineering(df, ["x"], spec)
