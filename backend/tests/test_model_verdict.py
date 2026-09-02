"""Tests de services/model_verdict.py (Lot 3, correctif I1,
AUDIT_DATALAB_2026-08-16.md §E.3/§P) — uniquement le point d'entrée public
`compute_verdict()`, jamais les fonctions `_assess_*` privées (même
convention que test_data_quality.py::analyze_data_quality). Données
synthétiques uniquement : ce module ne touche ni base ni entraînement réel."""
from __future__ import annotations

from domains.training.services.verdict import compute_verdict


def _codes(claims: list[dict]) -> set[str]:
    return {c["code"] for c in claims}


# ── Surapprentissage ──────────────────────────────────────────────────────

def test_regression_marked_overfitting_detected():
    verdict = compute_verdict(
        "regression",
        {"r2_train": 0.95, "r2_test": 0.70, "delta_r2": 0.25},
        {}, [],
    )
    assert "surapprentissage_marque" in _codes(verdict["claims"])
    claim = next(c for c in verdict["claims"] if c["code"] == "surapprentissage_marque")
    assert claim["level"] == "critique"
    assert claim["details"]["delta"] == 0.25


def test_regression_light_overfitting_detected():
    verdict = compute_verdict(
        "regression",
        {"r2_train": 0.85, "r2_test": 0.77, "delta_r2": 0.08},
        {}, [],
    )
    assert "surapprentissage_leger" in _codes(verdict["claims"])


def test_regression_no_overfitting():
    verdict = compute_verdict(
        "regression",
        {"r2_train": 0.81, "r2_test": 0.80, "delta_r2": 0.01},
        {}, [],
    )
    assert "pas_de_surapprentissage" in _codes(verdict["claims"])


def test_classification_overfitting_uses_delta_accuracy():
    verdict = compute_verdict(
        "classification",
        {"accuracy_train": 0.99, "accuracy": 0.75, "delta_accuracy": 0.24},
        {}, [],
    )
    assert "surapprentissage_marque" in _codes(verdict["claims"])


def test_missing_delta_metric_omits_overfitting_claim():
    """Job antérieur au correctif (delta absent) — jamais d'affirmation
    inventée, la claim est simplement omise."""
    verdict = compute_verdict("regression", {"r2_test": 0.8}, {}, [])
    assert "surapprentissage_marque" not in _codes(verdict["claims"])
    assert "pas_de_surapprentissage" not in _codes(verdict["claims"])


# ── Fiabilité (bootstrap CI) ───────────────────────────────────────────────

def test_wide_bootstrap_ci_flagged_unreliable():
    verdict = compute_verdict(
        "regression",
        {"r2_test": 0.8, "r2_bootstrap": {"mean": 0.8, "ci_low": 0.65, "ci_high": 0.90}},
        {}, [],
    )
    assert "fiabilite_faible" in _codes(verdict["claims"])
    claim = next(c for c in verdict["claims"] if c["code"] == "fiabilite_faible")
    assert claim["level"] == "critique"


def test_narrow_bootstrap_ci_flagged_reliable():
    verdict = compute_verdict(
        "classification",
        {"accuracy": 0.9, "accuracy_bootstrap": {"mean": 0.9, "ci_low": 0.885, "ci_high": 0.915}},
        {}, [],
    )
    assert "fiabilite_bonne" in _codes(verdict["claims"])


def test_missing_bootstrap_omits_reliability_claim():
    verdict = compute_verdict("regression", {"r2_test": 0.8}, {}, [])
    assert not any(c["code"].startswith("fiabilite_") for c in verdict["claims"])


# ── Choix de métrique (classification uniquement) ──────────────────────────

def test_imbalanced_classes_recommend_f1_over_accuracy():
    evaluation = {"confusion_matrix": [[90, 2], [3, 5]], "class_names": ["normal", "défaut"]}
    verdict = compute_verdict("classification", {"accuracy": 0.95}, evaluation, [])
    assert "classes_desequilibrees" in _codes(verdict["claims"])
    claim = next(c for c in verdict["claims"] if c["code"] == "classes_desequilibrees")
    assert claim["details"]["majority_class"] == "normal"


def test_balanced_classes_accuracy_trusted():
    evaluation = {"confusion_matrix": [[48, 4], [5, 43]], "class_names": ["a", "b"]}
    verdict = compute_verdict("classification", {"accuracy": 0.9}, evaluation, [])
    assert "classes_equilibrees" in _codes(verdict["claims"])


def test_regression_never_emits_metric_choice_claim():
    evaluation = {"confusion_matrix": [[90, 2], [3, 5]], "class_names": ["a", "b"]}
    verdict = compute_verdict("regression", {"r2_test": 0.8}, evaluation, [])
    assert not any(c["code"].startswith("classes_") for c in verdict["claims"])


# ── Écart au 2ᵉ (candidats) ─────────────────────────────────────────────────

def test_winner_margin_significant_when_gap_exceeds_fold_std():
    candidates = [
        {"algorithm": "LightGBM", "rank": 1, "selection_score": 0.90, "fold_scores": [0.89, 0.90, 0.91, 0.90]},
        {"algorithm": "XGBoost", "rank": 2, "selection_score": 0.80, "fold_scores": [0.79, 0.81]},
    ]
    verdict = compute_verdict("regression", {"r2_test": 0.9}, {}, candidates)
    assert "ecart_gagnant_significatif" in _codes(verdict["claims"])


def test_winner_margin_within_noise_when_gap_below_fold_std():
    candidates = [
        {"algorithm": "LightGBM", "rank": 1, "selection_score": 0.905, "fold_scores": [0.80, 0.85, 0.90, 0.95, 1.0]},
        {"algorithm": "XGBoost", "rank": 2, "selection_score": 0.900, "fold_scores": [0.89, 0.91]},
    ]
    verdict = compute_verdict("regression", {"r2_test": 0.9}, {}, candidates)
    assert "ecart_gagnant_dans_le_bruit" in _codes(verdict["claims"])
    claim = next(c for c in verdict["claims"] if c["code"] == "ecart_gagnant_dans_le_bruit")
    assert claim["level"] == "attention"


def test_winner_margin_unqualified_without_fold_scores():
    """Job antérieur au Lot D (pas de fold_scores) — écart signalé sans
    jamais affirmer une significativité qu'on ne peut pas vérifier."""
    candidates = [
        {"algorithm": "LightGBM", "rank": 1, "selection_score": 0.90, "fold_scores": None},
        {"algorithm": "XGBoost", "rank": 2, "selection_score": 0.80, "fold_scores": None},
    ]
    verdict = compute_verdict("regression", {"r2_test": 0.9}, {}, candidates)
    assert "ecart_gagnant_non_qualifie" in _codes(verdict["claims"])


def test_winner_margin_omitted_with_fewer_than_two_candidates():
    verdict = compute_verdict("regression", {"r2_test": 0.9}, {}, [])
    assert not any(c["code"].startswith("ecart_gagnant") for c in verdict["claims"])


# ── Calibration (classification uniquement) ─────────────────────────────────

def test_bad_calibration_flagged():
    calibration = {"global": {"mean_predicted": [0.9, 0.5, 0.1], "fraction_positive": [0.5, 0.5, 0.5]}}
    verdict = compute_verdict("classification", {"accuracy": 0.8}, {}, [], calibration=calibration)
    assert "calibration_mauvaise" in _codes(verdict["claims"])


def test_good_calibration_flagged():
    calibration = {"global": {"mean_predicted": [0.9, 0.5, 0.1], "fraction_positive": [0.91, 0.49, 0.11]}}
    verdict = compute_verdict("classification", {"accuracy": 0.8}, {}, [], calibration=calibration)
    assert "calibration_bonne" in _codes(verdict["claims"])


def test_regression_never_emits_calibration_claim():
    calibration = {"global": {"mean_predicted": [0.9], "fraction_positive": [0.1]}}
    verdict = compute_verdict("regression", {"r2_test": 0.8}, {}, [], calibration=calibration)
    assert not any(c["code"].startswith("calibration_") for c in verdict["claims"])


def test_missing_calibration_omits_claim():
    verdict = compute_verdict("classification", {"accuracy": 0.8}, {}, [])
    assert not any(c["code"].startswith("calibration_") for c in verdict["claims"])


# ── Plus de données utiles (courbe d'apprentissage) ─────────────────────────

def test_learning_curve_still_climbing():
    learning_curve = {"val_scores_mean": [0.5, 0.6, 0.65, 0.85], "metric_label": "R²"}
    verdict = compute_verdict("regression", {"r2_test": 0.85}, {}, [], learning_curve=learning_curve)
    assert "plus_de_donnees_utile" in _codes(verdict["claims"])


def test_learning_curve_plateaued():
    learning_curve = {"val_scores_mean": [0.5, 0.78, 0.799, 0.80], "metric_label": "R²"}
    verdict = compute_verdict("regression", {"r2_test": 0.8}, {}, [], learning_curve=learning_curve)
    assert "plateau_atteint" in _codes(verdict["claims"])


def test_missing_learning_curve_omits_claim():
    verdict = compute_verdict("regression", {"r2_test": 0.8}, {}, [])
    assert not any(c["code"] in ("plus_de_donnees_utile", "plateau_atteint") for c in verdict["claims"])


# ── Couverture CQR (régression uniquement) ──────────────────────────────────

def test_cqr_insufficient_coverage_flagged():
    cqr = {"target_coverage": 0.80, "empirical_coverage": 0.65}
    verdict = compute_verdict("regression", {"r2_test": 0.8}, {}, [], cqr=cqr)
    assert "couverture_insuffisante" in _codes(verdict["claims"])
    claim = next(c for c in verdict["claims"] if c["code"] == "couverture_insuffisante")
    assert claim["level"] == "critique"


def test_cqr_conforming_coverage_flagged():
    cqr = {"target_coverage": 0.80, "empirical_coverage": 0.81}
    verdict = compute_verdict("regression", {"r2_test": 0.8}, {}, [], cqr=cqr)
    assert "couverture_conforme" in _codes(verdict["claims"])


def test_classification_never_emits_cqr_claim():
    cqr = {"target_coverage": 0.80, "empirical_coverage": 0.5}
    verdict = compute_verdict("classification", {"accuracy": 0.8}, {}, [], cqr=cqr)
    assert not any(c["code"].startswith("couverture_") for c in verdict["claims"])


# ── Synthèse (next_actions) et tri ──────────────────────────────────────────
# Retour d'évaluation d'une maquette externe : "3 actions suivantes classées
# par ce que le diagnostic suggère" — enrichi depuis une simple phrase
# unique (`next_action`) vers une liste priorisée (`next_actions`).

def test_next_actions_prioritizes_overfitting_over_other_issues():
    cqr = {"target_coverage": 0.80, "empirical_coverage": 0.5}  # génère aussi une alerte
    verdict = compute_verdict(
        "regression",
        {"r2_train": 0.95, "r2_test": 0.70, "delta_r2": 0.25},
        {}, [], cqr=cqr,
    )
    first_action = verdict["next_actions"][0]["action"].lower()
    assert "surapprentissage" in first_action or "complexité" in first_action
    assert verdict["next_actions"][0]["code"] == "surapprentissage_marque"


def test_next_actions_default_when_no_issues():
    verdict = compute_verdict(
        "regression",
        {"r2_train": 0.81, "r2_test": 0.80, "delta_r2": 0.01, "r2_bootstrap": {"mean": 0.8, "ci_low": 0.79, "ci_high": 0.81}},
        {}, [],
    )
    assert len(verdict["next_actions"]) == 1
    assert verdict["next_actions"][0]["code"] == "aucune_alerte"
    action = verdict["next_actions"][0]["action"].lower()
    assert "prêt" in action or "production" in action


def test_next_actions_returns_up_to_three_ranked_by_priority():
    """Plusieurs signaux déclenchés simultanément — jusqu'à 3 actions,
    dans l'ordre de `_NEXT_ACTION_PRIORITY` (surapprentissage d'abord),
    jamais un remplissage artificiel au-delà des signaux réellement
    présents."""
    cqr = {"target_coverage": 0.80, "empirical_coverage": 0.5}  # couverture_insuffisante
    candidates = [
        {"algorithm": "A", "rank": 1, "selection_score": 0.80, "fold_scores": [0.79, 0.80, 0.81]},
        {"algorithm": "B", "rank": 2, "selection_score": 0.799, "fold_scores": [0.795, 0.80, 0.805]},
    ]  # écart_gagnant_dans_le_bruit
    verdict = compute_verdict(
        "regression",
        {"r2_train": 0.95, "r2_test": 0.70, "delta_r2": 0.25},  # surapprentissage_marque
        {}, candidates, cqr=cqr,
    )
    codes = [a["code"] for a in verdict["next_actions"]]
    assert len(codes) <= 3
    assert codes[0] == "surapprentissage_marque"  # priorité la plus haute en premier
    assert len(codes) == len(set(codes))  # jamais deux fois le même code


def test_next_actions_never_padded_below_three():
    """Un seul signal déclenché — une seule action renvoyée, jamais
    complétée artificiellement pour atteindre 3."""
    verdict = compute_verdict(
        "regression",
        {"r2_train": 0.85, "r2_test": 0.77, "delta_r2": 0.08},  # surapprentissage_leger, seul signal
        {}, [],
    )
    assert len(verdict["next_actions"]) == 1
    assert verdict["next_actions"][0]["code"] == "surapprentissage_leger"


def test_claims_sorted_critique_before_attention_before_info():
    cqr = {"target_coverage": 0.80, "empirical_coverage": 0.5}  # critique
    metrics = {
        "r2_train": 0.85, "r2_test": 0.77, "delta_r2": 0.08,  # attention (léger surapprentissage)
        "r2_bootstrap": {"mean": 0.77, "ci_low": 0.765, "ci_high": 0.775},  # info (fiable)
    }
    verdict = compute_verdict("regression", metrics, {}, [], cqr=cqr)
    levels = [c["level"] for c in verdict["claims"]]
    order = {"critique": 0, "attention": 1, "info": 2}
    assert levels == sorted(levels, key=lambda lv: order[lv])


def test_every_claim_carries_grounding_details():
    """Exigence explicite de l'audit : chaque affirmation doit être
    accompagnée de la donnée qui la fonde."""
    cqr = {"target_coverage": 0.80, "empirical_coverage": 0.5}
    verdict = compute_verdict(
        "regression",
        {"r2_train": 0.95, "r2_test": 0.70, "delta_r2": 0.25},
        {}, [], cqr=cqr,
    )
    assert len(verdict["claims"]) > 0
    for claim in verdict["claims"]:
        assert claim["details"], f"claim '{claim['code']}' sans donnée de justification"


# ── Fuite de données (retour utilisateur : maquette "les 6 questions" —
# "Y a-t-il eu fuite de données ?" manquait comme affirmation explicite du
# verdict, alors que la donnée existe déjà dans model_card) ────────────────


def test_leakage_claim_reports_duplicates_removed_and_grouping():
    verdict = compute_verdict(
        "regression", {"r2_test": 0.8}, {}, [], duplicates_removed=12, anti_leak_grouping=True
    )
    claim = next(c for c in verdict["claims"] if c["code"] == "fuite_verifiee_doublons_retires")
    assert claim["level"] == "info"
    assert claim["details"] == {"duplicates_removed": 12, "anti_leak_grouping": True}
    assert "12" in claim["explanation"]


def test_leakage_claim_reports_no_duplicates_and_no_grouping():
    verdict = compute_verdict(
        "regression", {"r2_test": 0.8}, {}, [], duplicates_removed=0, anti_leak_grouping=False
    )
    claim = next(c for c in verdict["claims"] if c["code"] == "fuite_verifiee_aucun_doublon")
    assert claim["level"] == "info"
    assert claim["details"] == {"duplicates_removed": 0, "anti_leak_grouping": False}
    assert "aucune colonne de regroupement" in claim["explanation"]


def test_missing_duplicates_removed_omits_leakage_claim():
    """Job antérieur à ce suivi (rétrocompatibilité par absence) — jamais
    une affirmation inventée sur des données absentes."""
    verdict = compute_verdict("regression", {"r2_test": 0.8}, {}, [])
    codes = {c["code"] for c in verdict["claims"]}
    assert "fuite_verifiee_doublons_retires" not in codes
    assert "fuite_verifiee_aucun_doublon" not in codes
