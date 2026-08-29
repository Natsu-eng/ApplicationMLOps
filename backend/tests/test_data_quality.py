"""Tests de services/data_quality.py (Lot B) — garde-fous de données."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from domains.shared.data_quality import TARGET_LEAKAGE_CRAMERS_V_THRESHOLD, analyze_data_quality


def _warnings_with_code(warnings, code):
    return [w for w in warnings if w["code"] == code]


# ── Fuite cible ──────────────────────────────────────────────────────────


def test_target_leakage_numeric_triggers_on_near_copy():
    rng = np.random.default_rng(1)
    n = 500
    target = rng.normal(100, 10, n)
    df = pd.DataFrame({"cible": target, "copie_bruitee": target + rng.normal(0, 0.01, n)})
    warnings = analyze_data_quality(df, "cible")
    leaks = _warnings_with_code(warnings, "fuite_cible")
    assert len(leaks) == 1
    assert leaks[0]["level"] == "critique"
    assert leaks[0]["columns"] == ["copie_bruitee"]


def test_target_leakage_numeric_does_not_trigger_on_independent_feature():
    rng = np.random.default_rng(2)
    n = 500
    df = pd.DataFrame({"cible": rng.normal(100, 10, n), "independante": rng.normal(0, 1, n)})
    warnings = analyze_data_quality(df, "cible")
    assert _warnings_with_code(warnings, "fuite_cible") == []


def test_target_leakage_categorical_triggers_on_bijection():
    df = pd.DataFrame({"cible": ["a", "b", "c"] * 100})
    df["copie"] = df["cible"].map({"a": "x", "b": "y", "c": "z"})
    warnings = analyze_data_quality(df, "cible")
    leaks = _warnings_with_code(warnings, "fuite_cible")
    assert any(w["columns"] == ["copie"] for w in leaks)


def test_target_leakage_categorical_does_not_trigger_on_independent_feature():
    rng = np.random.default_rng(3)
    n = 1500
    df = pd.DataFrame(
        {"cible": rng.choice(["a", "b", "c"], size=n), "independante": rng.choice(["x", "y"], size=n)}
    )
    warnings = analyze_data_quality(df, "cible")
    assert _warnings_with_code(warnings, "fuite_cible") == []


def test_target_leakage_partial_categorical_caught_by_metric_specific_threshold():
    """Fuite catégorielle réelle mais imparfaite (bruitée, pas une bijection) :
    Cramér's V corrigé attendu ~0.75-0.85. DOIT être capté par le seuil
    catégoriel dédié (0.70) — la preuve qu'un seuil unique à 0.95 (comme un
    seuil pensé pour Pearson/AUC) aurait raté cette fuite bien réelle."""
    rng = np.random.default_rng(42)
    n = 4000
    target = rng.choice(["oui", "non"], size=n)
    flip = rng.random(n) < 0.10  # ~10% de bruit -> V attendu ~1-2*0.10 = ~0.80
    feature = np.where(flip, np.where(target == "oui", "non", "oui"), target)
    df = pd.DataFrame({"cible": target, "presque_copie": feature, "bruit": rng.normal(size=n)})

    warnings = analyze_data_quality(df, "cible")
    leaks = [w for w in _warnings_with_code(warnings, "fuite_cible") if w["columns"] == ["presque_copie"]]
    assert len(leaks) == 1
    value = leaks[0]["details"]["value"]
    # Capté par le seuil catégoriel (0.70) tout en restant nettement sous 0.95
    # (le seuil qu'un seul seuil "générique" aurait imposé à tort).
    assert TARGET_LEAKAGE_CRAMERS_V_THRESHOLD < value < 0.95
    assert "bruit" not in [c for w in leaks for c in w["columns"]]


# ── Déséquilibre des classes ─────────────────────────────────────────────


def test_class_imbalance_triggers_on_skewed_classes():
    df = pd.DataFrame({"cible": ["rare"] * 5 + ["frequent"] * 200, "x": range(205)})
    warnings = analyze_data_quality(df, "cible")
    imbalance = _warnings_with_code(warnings, "desequilibre_classes")
    assert len(imbalance) == 1
    assert imbalance[0]["level"] == "attention"


def test_class_imbalance_details_expose_counts_and_balanced_weights_for_the_chart():
    """Retour utilisateur direct : "on détecte bien le déséquilibre mais on
    ne montre pas par un graphique... ce que ça donnera [avec le
    rééquilibrage]" — le frontend a besoin des comptages ET des poids
    effectifs (même formule "balanced" que `compute_sample_weight`
    réellement appliquée à l'entraînement), jamais juste le ratio."""
    df = pd.DataFrame({"cible": ["rare"] * 5 + ["frequent"] * 200, "x": range(205)})
    warnings = analyze_data_quality(df, "cible")
    details = _warnings_with_code(warnings, "desequilibre_classes")[0]["details"]

    assert details["class_counts"] == {"frequent": 200, "rare": 5}
    # Poids "balanced" = n_total / (n_classes * n_c) — la classe rare doit
    # recevoir un poids bien plus élevé que la classe fréquente.
    assert details["class_weights"]["rare"] > details["class_weights"]["frequent"]
    n_total, n_classes = 205, 2
    assert details["class_weights"]["rare"] == round(n_total / (n_classes * 5), 3)
    assert details["class_weights"]["frequent"] == round(n_total / (n_classes * 200), 3)


def test_class_imbalance_does_not_trigger_on_balanced_classes():
    df = pd.DataFrame({"cible": ["a"] * 100 + ["b"] * 100, "x": range(200)})
    warnings = analyze_data_quality(df, "cible")
    assert _warnings_with_code(warnings, "desequilibre_classes") == []


# ── Cardinalité excessive ────────────────────────────────────────────────


def test_high_cardinality_triggers_on_identifier_like_column():
    n = 300
    df = pd.DataFrame({"id_client": [f"C{i}" for i in range(n)], "cible": range(n)})
    warnings = analyze_data_quality(df, "cible")
    card = _warnings_with_code(warnings, "cardinalite_excessive")
    assert len(card) == 1
    assert card[0]["columns"] == ["id_client"]
    # Transparence (diagnostic "bad allocation") : le nombre de colonnes
    # qu'un one-hot produirait est exposé, informatif — plus une mise en
    # garde mémoire depuis que le moteur préserve le sparse (voir ml_training.py).
    assert card[0]["details"]["n_estimated_onehot_columns"] == n


def test_high_cardinality_does_not_trigger_on_low_cardinality_column():
    rng = np.random.default_rng(4)
    n = 300
    df = pd.DataFrame({"ville": rng.choice(["Paris", "Lyon", "Nice"], size=n), "cible": range(n)})
    warnings = analyze_data_quality(df, "cible")
    assert _warnings_with_code(warnings, "cardinalite_excessive") == []


# ── Colonnes constantes / quasi-constantes ───────────────────────────────


def test_constant_column_triggers_info():
    df = pd.DataFrame({"toujours_pareil": [42] * 100, "cible": range(100)})
    warnings = analyze_data_quality(df, "cible")
    constants = _warnings_with_code(warnings, "colonne_constante")
    assert len(constants) == 1
    assert constants[0]["level"] == "info"


def test_varying_column_does_not_trigger_constant_warning():
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"varie": rng.normal(size=100), "cible": range(100)})
    warnings = analyze_data_quality(df, "cible")
    assert _warnings_with_code(warnings, "colonne_constante") == []


# ── Dataset trop petit ────────────────────────────────────────────────────


def test_small_dataset_triggers_attention():
    df = pd.DataFrame({"x": range(30), "cible": range(30)})
    warnings = analyze_data_quality(df, "cible")
    assert _warnings_with_code(warnings, "dataset_trop_petit") != []


def test_large_enough_dataset_does_not_trigger():
    df = pd.DataFrame({"x": range(500), "y": range(500), "cible": range(500)})
    warnings = analyze_data_quality(df, "cible")
    assert _warnings_with_code(warnings, "dataset_trop_petit") == []
    assert _warnings_with_code(warnings, "ratio_lignes_variables_faible") == []


# ── Valeurs manquantes élevées ───────────────────────────────────────────


def test_high_missing_rate_triggers_attention():
    n = 200
    x = [np.nan] * 100 + list(range(100))
    df = pd.DataFrame({"x": x, "cible": range(n)})
    warnings = analyze_data_quality(df, "cible")
    missing = _warnings_with_code(warnings, "valeurs_manquantes_elevees")
    assert len(missing) == 1
    assert missing[0]["columns"] == ["x"]


def test_low_missing_rate_does_not_trigger():
    n = 200
    x = [np.nan] * 2 + list(range(198))
    df = pd.DataFrame({"x": x, "cible": range(n)})
    warnings = analyze_data_quality(df, "cible")
    assert _warnings_with_code(warnings, "valeurs_manquantes_elevees") == []


# ── Collinéarité ─────────────────────────────────────────────────────────


def test_collinearity_triggers_info_on_redundant_features():
    rng = np.random.default_rng(6)
    n = 300
    x1 = rng.normal(size=n)
    df = pd.DataFrame({"x1": x1, "x2": x1 * 2 + rng.normal(0, 1e-6, n), "cible": rng.normal(size=n)})
    warnings = analyze_data_quality(df, "cible")
    collin = _warnings_with_code(warnings, "collinearite_forte")
    assert len(collin) == 1
    assert set(collin[0]["columns"]) == {"x1", "x2"}


def test_collinearity_does_not_trigger_on_independent_features():
    rng = np.random.default_rng(7)
    n = 300
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n), "cible": rng.normal(size=n)})
    warnings = analyze_data_quality(df, "cible")
    assert _warnings_with_code(warnings, "collinearite_forte") == []


# ── Colonne de groupe (Décision 5/6) ──────────────────────────────────────


def test_group_column_excluded_from_all_feature_quality_detections():
    n = 200
    df = pd.DataFrame(
        {
            "groupe": [f"g{i}" for i in range(n)],  # cardinalité ~100% -> déclencherait si analysée
            "x": range(n),
            "cible": range(n),
        }
    )
    warnings = analyze_data_quality(df, "cible", group_column="groupe")
    # Aucune alerte de qualité de feature ne doit porter sur la colonne de groupe
    assert not any(
        "groupe" in w["columns"] and w["code"] != "colonne_groupe_exclue" for w in warnings
    )


def test_group_column_transparency_info_is_emitted():
    n = 100
    df = pd.DataFrame({"groupe": [f"g{i}" for i in range(n)], "x": range(n), "cible": range(n)})
    warnings = analyze_data_quality(df, "cible", group_column="groupe")
    transparency = _warnings_with_code(warnings, "colonne_groupe_exclue")
    assert len(transparency) == 1
    assert transparency[0]["level"] == "info"
    assert transparency[0]["columns"] == ["groupe"]


def test_no_group_column_means_no_transparency_info():
    n = 100
    df = pd.DataFrame({"x": range(n), "cible": range(n)})
    warnings = analyze_data_quality(df, "cible")
    assert _warnings_with_code(warnings, "colonne_groupe_exclue") == []


# ── Tri par niveau ────────────────────────────────────────────────────────


def test_warnings_sorted_critique_then_attention_then_info():
    rng = np.random.default_rng(8)
    n = 60  # < MIN_ROWS_THRESHOLD -> attention "dataset_trop_petit"
    target = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "cible": target,
            "copie": target + rng.normal(0, 0.001, n),  # critique : fuite
            "constante": [1] * n,  # info : colonne constante
        }
    )
    warnings = analyze_data_quality(df, "cible")
    levels = [w["level"] for w in warnings]
    assert levels == sorted(levels, key=lambda lv: {"critique": 0, "attention": 1, "info": 2}[lv])
    assert levels[0] == "critique"


# ── Robustesse ────────────────────────────────────────────────────────────


def test_single_column_dataset_no_crash():
    df = pd.DataFrame({"cible": list(range(50))})
    warnings = analyze_data_quality(df, "cible")
    assert isinstance(warnings, list)


def test_constant_target_column_no_crash():
    df = pd.DataFrame({"cible": [1] * 50, "x": range(50)})
    warnings = analyze_data_quality(df, "cible")
    assert isinstance(warnings, list)


def test_massive_nan_no_crash():
    rng = np.random.default_rng(9)
    n = 300
    df = pd.DataFrame(
        {
            "x": [np.nan] * 280 + list(rng.normal(size=20)),
            "cat": [None] * 290 + ["a"] * 10,
            "cible": rng.normal(size=n),
        }
    )
    warnings = analyze_data_quality(df, "cible")
    assert isinstance(warnings, list)


def test_unknown_target_column_raises_keyerror():
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(KeyError):
        analyze_data_quality(df, "inexistante")


# ── Colonnes dupliquées (Lot Nettoyage guidé des variables) ─────────────────


def test_duplicate_columns_triggers_attention_on_identical_content():
    df = pd.DataFrame({"a": range(200), "a_copie": range(200), "cible": range(200)})
    warnings = analyze_data_quality(df, "cible")
    dup = _warnings_with_code(warnings, "colonnes_dupliquees")
    assert len(dup) == 1
    assert dup[0]["level"] == "attention"
    assert set(dup[0]["columns"]) == {"a", "a_copie"}


def test_duplicate_columns_does_not_trigger_on_merely_correlated_columns():
    rng = np.random.default_rng(10)
    n = 300
    x1 = rng.normal(size=n)
    df = pd.DataFrame({"x1": x1, "x2": x1 * 2 + rng.normal(0, 1e-6, n), "cible": rng.normal(size=n)})
    warnings = analyze_data_quality(df, "cible")
    assert _warnings_with_code(warnings, "colonnes_dupliquees") == []


def test_duplicate_columns_handles_categorical_content():
    values = ["a", "b", "c"] * 50
    df = pd.DataFrame({"cat1": values, "cat2": values, "cible": range(150)})
    warnings = analyze_data_quality(df, "cible")
    dup = _warnings_with_code(warnings, "colonnes_dupliquees")
    assert set(dup[0]["columns"]) == {"cat1", "cat2"}


def test_duplicate_columns_reports_each_pair_once_for_three_identical_columns():
    df = pd.DataFrame({"a": range(100), "b": range(100), "c": range(100), "cible": range(100)})
    warnings = analyze_data_quality(df, "cible")
    dup = _warnings_with_code(warnings, "colonnes_dupliquees")
    # a/b/c strictement identiques : b et c signalés comme doublons de a,
    # jamais un doublon de doublon (chaque colonne au plus une fois "reported").
    reported_columns = {c for w in dup for c in w["columns"] if c != "a"}
    assert reported_columns == {"b", "c"}


# ── Numérique mal typé (Lot Nettoyage guidé des variables) ──────────────────


def test_mistyped_numeric_triggers_on_comma_decimal_column():
    n = 200
    df = pd.DataFrame({"prix": [f"{1000 + i},{i % 100:02d}" for i in range(n)], "cible": range(n)})
    warnings = analyze_data_quality(df, "cible")
    mistyped = _warnings_with_code(warnings, "numerique_mal_type")
    assert len(mistyped) == 1
    assert mistyped[0]["columns"] == ["prix"]
    assert mistyped[0]["level"] == "attention"


def test_mistyped_numeric_triggers_on_thousands_separator():
    n = 200
    df = pd.DataFrame({"montant": [f"1 {200 + i:03d}" for i in range(n)], "cible": range(n)})
    warnings = analyze_data_quality(df, "cible")
    assert _warnings_with_code(warnings, "numerique_mal_type") != []


def test_mistyped_numeric_does_not_trigger_on_already_numeric_column():
    df = pd.DataFrame({"prix": [1000.5 + i for i in range(100)], "cible": range(100)})
    warnings = analyze_data_quality(df, "cible")
    assert _warnings_with_code(warnings, "numerique_mal_type") == []


def test_mistyped_numeric_does_not_trigger_on_identifier_like_text():
    """Une colonne d'identifiants (aucun signe de formatage numérique) ne
    doit jamais être proposée à la conversion, même si elle ne contient que
    des chiffres (ex. codes postaux à zéro non significatif : "01234")."""
    n = 200
    df = pd.DataFrame({"code_postal": [f"{i:05d}" for i in range(n)], "cible": range(n)})
    warnings = analyze_data_quality(df, "cible")
    assert _warnings_with_code(warnings, "numerique_mal_type") == []


def test_mistyped_numeric_does_not_trigger_on_free_text():
    df = pd.DataFrame({"commentaire": ["bof", "top produit", "rien à dire", "5 étoiles"] * 30})
    warnings = analyze_data_quality(pd.concat([df, pd.DataFrame({"cible": range(len(df))})], axis=1), "cible")
    assert _warnings_with_code(warnings, "numerique_mal_type") == []


# ── target_column optionnel (Lot Nettoyage guidé des variables) ─────────────


def test_analyze_data_quality_without_target_runs_structural_detections_only():
    n = 200
    df = pd.DataFrame({
        "toujours_pareil": [42] * n,
        "id_client": [f"C{i}" for i in range(n)],
        "x": range(n),
    })
    warnings = analyze_data_quality(df)
    assert _warnings_with_code(warnings, "colonne_constante") != []
    assert _warnings_with_code(warnings, "cardinalite_excessive") != []


def test_analyze_data_quality_without_target_never_raises_keyerror():
    df = pd.DataFrame({"x": range(50)})
    warnings = analyze_data_quality(df, target_column=None)
    assert isinstance(warnings, list)


def test_analyze_data_quality_without_target_skips_leakage_and_imbalance():
    rng = np.random.default_rng(11)
    n = 500
    target = rng.normal(100, 10, n)
    df = pd.DataFrame({"cible": target, "copie_bruitee": target + rng.normal(0, 0.01, n)})
    warnings = analyze_data_quality(df)  # aucune cible fournie
    assert _warnings_with_code(warnings, "fuite_cible") == []
    assert _warnings_with_code(warnings, "desequilibre_classes") == []


def test_analyze_data_quality_with_target_unchanged_behavior():
    """Non-régression : fournir `target_column` produit exactement le même
    résultat qu'avant ce lot (fuite/déséquilibre inclus)."""
    rng = np.random.default_rng(1)
    n = 500
    target = rng.normal(100, 10, n)
    df = pd.DataFrame({"cible": target, "copie_bruitee": target + rng.normal(0, 0.01, n)})
    warnings = analyze_data_quality(df, "cible")
    assert _warnings_with_code(warnings, "fuite_cible") != []


# ── excluded_columns (retour utilisateur direct — diagnostic de cohérence
# du wizard : "une colonne exclue à l'étape 1 déclenche encore une alerte
# à l'étape 2") ─────────────────────────────────────────────────────────


def test_excluded_columns_produces_no_warning_about_them():
    """Une colonne déjà retirée de la sélection de variables (étape 1) ne
    doit plus jamais générer d'alerte — même règle que target_column/
    group_column, jamais une seconde catégorie de traitement."""
    df = pd.DataFrame({
        "identifiant": [f"REF-{i}" for i in range(200)],
        "valeur": np.random.default_rng(1).normal(0, 1, 200),
    })
    warnings_without_exclusion = analyze_data_quality(df, target_column=None)
    assert _warnings_with_code(warnings_without_exclusion, "cardinalite_excessive") != []

    warnings_with_exclusion = analyze_data_quality(df, target_column=None, excluded_columns={"identifiant"})
    assert _warnings_with_code(warnings_with_exclusion, "cardinalite_excessive") == []


def test_excluded_columns_does_not_affect_still_included_columns():
    """Exclure une colonne ne doit jamais masquer une alerte légitime sur
    une AUTRE colonne toujours retenue."""
    df = pd.DataFrame({
        "identifiant": [f"REF-{i}" for i in range(200)],
        "constante": [1] * 200,
        "valeur": np.random.default_rng(1).normal(0, 1, 200),
    })
    warnings = analyze_data_quality(df, target_column=None, excluded_columns={"identifiant"})
    assert _warnings_with_code(warnings, "cardinalite_excessive") == []
    assert _warnings_with_code(warnings, "colonne_constante") != []


def test_excluded_columns_combines_with_target_and_group_exclusion():
    """`excluded_columns` s'ajoute à l'exclusion cible/groupe existante,
    jamais à la place."""
    rng = np.random.default_rng(1)
    n = 200
    df = pd.DataFrame({
        "cible": rng.integers(0, 2, n),
        "groupe": rng.integers(0, 5, n),
        "identifiant": [f"REF-{i}" for i in range(n)],
        "valeur": rng.normal(0, 1, n),
    })
    warnings = analyze_data_quality(df, target_column="cible", group_column="groupe", excluded_columns={"identifiant"})
    assert _warnings_with_code(warnings, "cardinalite_excessive") == []
    # La transparence sur la colonne de groupe reste émise (comportement
    # inchangé, non affecté par excluded_columns).
    assert _warnings_with_code(warnings, "colonne_groupe_exclue") != []
