"""Tests de services/data_quality.py (Lot B) — garde-fous de données."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from services.data_quality import TARGET_LEAKAGE_CRAMERS_V_THRESHOLD, analyze_data_quality


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
