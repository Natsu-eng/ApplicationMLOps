"""Verdict post-entraînement (Lot 3, correctif I1, AUDIT_DATALAB_2026-08-16.md
§E.3/§P) — la promesse produit est « une aide à la décision », mais jusqu'ici
un modèle affichait des graphiques et des nombres bruts sans jamais dire à
l'utilisateur « ce modèle surapprend », « regardez plutôt cette métrique »,
« et maintenant ? ». Toutes les données nécessaires sont déjà calculées et
persistées par `ml_training.py` (`delta_r2`/`delta_accuracy`, bootstrap CI,
`fold_scores` des candidats, courbe de calibration, courbe d'apprentissage,
couverture CQR) — ce module n'entraîne rien et ne recalcule rien depuis les
données brutes : uniquement des règles déterministes sur des nombres déjà en
base, comme `services/data_quality.py` (mêmes conventions : `level`
"critique"/"attention"/"info", `code` stable, `title`/`explanation`/`action`
en français, toujours accompagnés de la donnée qui les fonde).

Prend des dictionnaires déjà désérialisés (jamais l'ORM ni le job) — pur et
testable avec des données synthétiques, sans base de données ni fixtures
d'entraînement réel."""
from __future__ import annotations

from typing import Any, Optional

# Écart train/test (delta_r2 en régression, delta_accuracy en
# classification — même échelle 0-1 dans les deux cas, mêmes seuils).
_OVERFIT_MARKED_THRESHOLD = 0.15
_OVERFIT_LIGHT_THRESHOLD = 0.05

# Largeur de l'intervalle de confiance bootstrap (ci_high - ci_low), même
# échelle 0-1 (R² ou accuracy).
_CI_WIDTH_UNRELIABLE_THRESHOLD = 0.15
_CI_WIDTH_MODERATE_THRESHOLD = 0.05

# Fraction de la classe majoritaire dans le jeu de test au-delà de laquelle
# l'accuracy brute devient trompeuse (ex. 90 % → un modèle qui prédit
# toujours la classe majoritaire obtient déjà 90 % d'accuracy).
_CLASS_IMBALANCE_THRESHOLD = 0.75

# Écart moyen absolu à la diagonale (mean_predicted vs fraction_positive)
# au-delà duquel les probabilités ne reflètent plus une confiance réelle.
_CALIBRATION_BAD_THRESHOLD = 0.15
_CALIBRATION_MODERATE_THRESHOLD = 0.05

# Tolérance sous `target_coverage` en dessous de laquelle la couverture CQR
# empirique est jugée insuffisante (les intervalles annoncés sont trop
# optimistes) plutôt qu'un simple bruit d'échantillonnage.
_CQR_COVERAGE_TOLERANCE = 0.03


def _claim(
    level: str, code: str, title: str, explanation: str, details: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    return {
        "level": level,
        "code": code,
        "title": title,
        "explanation": explanation,
        "details": details or {},
    }


def _assess_overfitting(task_type: str, metrics: dict[str, Any]) -> Optional[dict[str, Any]]:
    delta = metrics.get("delta_r2") if task_type == "regression" else metrics.get("delta_accuracy")
    if delta is None:
        return None
    metric_label = "R²" if task_type == "regression" else "l'exactitude"
    train_key = "r2_train" if task_type == "regression" else "accuracy_train"
    test_key = "r2_test" if task_type == "regression" else "accuracy"
    train_value, test_value = metrics.get(train_key), metrics.get(test_key)

    if delta > _OVERFIT_MARKED_THRESHOLD:
        return _claim(
            "critique", "surapprentissage_marque",
            "Ce modèle surapprend",
            f"{metric_label.capitalize()} passe de {train_value:.3f} sur l'entraînement à {test_value:.3f} sur le "
            f"test (écart de {delta:.3f}) — le modèle a en partie mémorisé les données d'entraînement plutôt "
            "qu'appris une règle générale.",
            {"delta": round(delta, 4), "train": train_value, "test": test_value},
        )
    if delta > _OVERFIT_LIGHT_THRESHOLD:
        return _claim(
            "attention", "surapprentissage_leger",
            "Léger surapprentissage",
            f"{metric_label.capitalize()} est un peu meilleure sur l'entraînement ({train_value:.3f}) que sur le "
            f"test ({test_value:.3f}) — écart de {delta:.3f}, à surveiller mais pas encore préoccupant.",
            {"delta": round(delta, 4), "train": train_value, "test": test_value},
        )
    return _claim(
        "info", "pas_de_surapprentissage",
        "Pas de signe de surapprentissage",
        f"{metric_label.capitalize()} sur l'entraînement ({train_value:.3f}) et sur le test ({test_value:.3f}) "
        f"sont proches (écart de {delta:.3f}) — le modèle généralise plutôt bien aux données qu'il n'a pas vues.",
        {"delta": round(delta, 4), "train": train_value, "test": test_value},
    )


def _assess_reliability(task_type: str, metrics: dict[str, Any]) -> Optional[dict[str, Any]]:
    bootstrap_key = "r2_bootstrap" if task_type == "regression" else "accuracy_bootstrap"
    bootstrap = metrics.get(bootstrap_key)
    if not bootstrap:
        return None
    width = bootstrap["ci_high"] - bootstrap["ci_low"]
    metric_label = "le R²" if task_type == "regression" else "l'exactitude"

    if width > _CI_WIDTH_UNRELIABLE_THRESHOLD:
        return _claim(
            "critique", "fiabilite_faible",
            "Résultat peu fiable",
            f"L'intervalle de confiance à 95 % de {metric_label} est large ([{bootstrap['ci_low']:.3f} ; "
            f"{bootstrap['ci_high']:.3f}], soit {width:.3f} de largeur) — le jeu de test est probablement trop "
            "petit pour estimer cette métrique précisément, la valeur affichée pourrait varier beaucoup avec "
            "d'autres données.",
            {"ci_low": bootstrap["ci_low"], "ci_high": bootstrap["ci_high"], "width": round(width, 4)},
        )
    if width > _CI_WIDTH_MODERATE_THRESHOLD:
        return _claim(
            "attention", "fiabilite_moderee",
            "Fiabilité modérée",
            f"L'intervalle de confiance à 95 % de {metric_label} est [{bootstrap['ci_low']:.3f} ; "
            f"{bootstrap['ci_high']:.3f}] — ni très large ni très étroit, la métrique affichée reste une bonne "
            "estimation mais avec une marge réelle.",
            {"ci_low": bootstrap["ci_low"], "ci_high": bootstrap["ci_high"], "width": round(width, 4)},
        )
    return _claim(
        "info", "fiabilite_bonne",
        "Résultat fiable",
        f"L'intervalle de confiance à 95 % de {metric_label} est étroit ([{bootstrap['ci_low']:.3f} ; "
        f"{bootstrap['ci_high']:.3f}]) — l'estimation est stable, un autre échantillon de test donnerait "
        "probablement un résultat proche.",
        {"ci_low": bootstrap["ci_low"], "ci_high": bootstrap["ci_high"], "width": round(width, 4)},
    )


def _assess_metric_choice(task_type: str, evaluation: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Classification uniquement — la régression n'a qu'une métrique de
    référence (R²), pas d'ambiguïté à lever."""
    if task_type != "classification":
        return None
    matrix = evaluation.get("confusion_matrix")
    class_names = evaluation.get("class_names")
    if not matrix or not class_names:
        return None
    class_counts = [sum(row) for row in matrix]
    total = sum(class_counts)
    if total == 0:
        return None
    majority_idx = max(range(len(class_counts)), key=lambda i: class_counts[i])
    majority_fraction = class_counts[majority_idx] / total

    if majority_fraction > _CLASS_IMBALANCE_THRESHOLD:
        return _claim(
            "attention", "classes_desequilibrees",
            "Regardez le F1 ou le ROC-AUC, pas seulement l'exactitude",
            f"La classe « {class_names[majority_idx]} » représente {majority_fraction:.0%} du jeu de test — un "
            "modèle qui prédirait toujours cette classe obtiendrait déjà une exactitude élevée sans avoir rien "
            "appris. Le F1 pondéré et le ROC-AUC restent significatifs même avec ce déséquilibre.",
            {"majority_class": class_names[majority_idx], "majority_fraction": round(majority_fraction, 4)},
        )
    return _claim(
        "info", "classes_equilibrees",
        "L'exactitude est une métrique fiable ici",
        f"Les classes du jeu de test sont raisonnablement équilibrées (la plus fréquente, « "
        f"{class_names[majority_idx]} », représente {majority_fraction:.0%}) — l'exactitude n'est pas trompeuse.",
        {"majority_class": class_names[majority_idx], "majority_fraction": round(majority_fraction, 4)},
    )


def _assess_winner_margin(candidates: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Compare le gagnant au 2ᵉ — l'écart de `selection_score` doit être jugé
    à l'aune de la variance inter-folds du gagnant (`fold_scores`), pas en
    valeur absolue : un écart de 0,003 avec un écart-type de folds de 0,04
    ne prouve rien (AUDIT_DATALAB_2026-08-16.md, §E.3)."""
    if len(candidates) < 2:
        return None
    by_rank = sorted(candidates, key=lambda c: c["rank"])
    winner, runner_up = by_rank[0], by_rank[1]
    fold_scores = winner.get("fold_scores")
    gap = winner["selection_score"] - runner_up["selection_score"]

    if not fold_scores or len(fold_scores) < 2:
        # Pas de fold_scores (job antérieur au Lot D) — écart en valeur
        # absolue seulement, jamais affirmer une significativité qu'on ne
        # peut pas vérifier.
        return _claim(
            "info", "ecart_gagnant_non_qualifie",
            f"« {winner['algorithm']} » devance « {runner_up['algorithm']} » de {gap:.4f}",
            "La variance entre plis de validation croisée n'est pas disponible pour ce modèle (entraîné avant "
            "que ce calcul n'existe) — impossible de dire si cet écart est significatif ou dans le bruit.",
            {"winner": winner["algorithm"], "runner_up": runner_up["algorithm"], "gap": round(gap, 4)},
        )

    std = float(_std(fold_scores))
    if std == 0 or gap > std:
        return _claim(
            "info", "ecart_gagnant_significatif",
            f"« {winner['algorithm']} » devance clairement « {runner_up['algorithm']} »",
            f"L'écart ({gap:.4f}) dépasse l'écart-type entre plis de validation croisée du gagnant ({std:.4f}) — "
            "ce n'est pas juste du bruit d'échantillonnage.",
            {"winner": winner["algorithm"], "runner_up": runner_up["algorithm"], "gap": round(gap, 4), "fold_std": round(std, 4)},
        )
    return _claim(
        "attention", "ecart_gagnant_dans_le_bruit",
        f"« {winner['algorithm']} » n'est pas clairement meilleur que « {runner_up['algorithm']} »",
        f"L'écart ({gap:.4f}) est plus petit que l'écart-type entre plis de validation croisée du gagnant "
        f"({std:.4f}) — les deux modèles sont statistiquement proches, le choix du gagnant tient en partie au "
        "hasard du découpage.",
        {"winner": winner["algorithm"], "runner_up": runner_up["algorithm"], "gap": round(gap, 4), "fold_std": round(std, 4)},
    )


def _std(values: list[float]) -> float:
    n = len(values)
    mean = sum(values) / n
    return (sum((v - mean) ** 2 for v in values) / n) ** 0.5


def _assess_calibration(task_type: str, calibration: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Régression : non applicable (pas de probabilités), toujours `None`."""
    if task_type != "classification" or not calibration:
        return None
    deviations = []
    for curve in calibration.values():
        mean_predicted, fraction_positive = curve["mean_predicted"], curve["fraction_positive"]
        deviations.extend(abs(p - f) for p, f in zip(mean_predicted, fraction_positive))
    if not deviations:
        return None
    mean_deviation = sum(deviations) / len(deviations)

    if mean_deviation > _CALIBRATION_BAD_THRESHOLD:
        return _claim(
            "attention", "calibration_mauvaise",
            "Les probabilités ne sont pas fiables telles quelles",
            f"L'écart moyen entre la confiance annoncée et la fréquence réelle est de {mean_deviation:.3f} — si "
            "vous utilisez les probabilités (pas seulement la classe prédite) pour prendre une décision, "
            "envisagez une recalibration avant de vous y fier.",
            {"mean_deviation": round(mean_deviation, 4)},
        )
    if mean_deviation > _CALIBRATION_MODERATE_THRESHOLD:
        return _claim(
            "info", "calibration_moderee",
            "Calibration correcte",
            f"L'écart moyen entre la confiance annoncée et la fréquence réelle est de {mean_deviation:.3f} — "
            "raisonnable, sans être parfait.",
            {"mean_deviation": round(mean_deviation, 4)},
        )
    return _claim(
        "info", "calibration_bonne",
        "Les probabilités reflètent une vraie confiance",
        f"L'écart moyen entre la confiance annoncée et la fréquence réelle observée n'est que de "
        f"{mean_deviation:.3f} — les probabilités prédites sont utilisables telles quelles.",
        {"mean_deviation": round(mean_deviation, 4)},
    )


def _assess_more_data(learning_curve: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not learning_curve:
        return None
    val_scores = learning_curve.get("val_scores_mean")
    if not val_scores or len(val_scores) < 2:
        return None
    total_range = max(val_scores) - min(val_scores)
    last_gain = val_scores[-1] - val_scores[-2]
    # Le dernier incrément représente plus du quart de la progression totale
    # observée sur la courbe → encore en pente, pas de plateau.
    still_climbing = total_range > 0 and last_gain > 0.25 * total_range

    if still_climbing:
        return _claim(
            "info", "plus_de_donnees_utile",
            "Plus de données amélioreraient probablement ce modèle",
            f"La courbe d'apprentissage progresse encore entre les deux plus grandes tailles d'échantillon "
            f"testées (+{last_gain:.4f} sur {learning_curve.get('metric_label', 'la métrique')}) — elle n'a pas "
            "atteint de plateau.",
            {"last_gain": round(last_gain, 4), "total_range": round(total_range, 4)},
        )
    return _claim(
        "info", "plateau_atteint",
        "Plus de données n'aideraient probablement pas beaucoup",
        f"La courbe d'apprentissage a atteint un plateau (+{last_gain:.4f} entre les deux plus grandes tailles "
        f"d'échantillon testées, sur {len(val_scores)} tailles) — le modèle a déjà extrait l'essentiel du signal "
        "disponible dans ce dataset.",
        {"last_gain": round(last_gain, 4), "total_range": round(total_range, 4)},
    )


def _assess_leakage(duplicates_removed: Optional[int], anti_leak_grouping: Optional[bool]) -> Optional[dict[str, Any]]:
    """Rend visible une mesure anti-fuite DÉJÀ appliquée par le pipeline
    (`ml_training.py` — doublons exacts retirés avant le découpage
    entraînement/test, regroupement optionnel par `group_column`), jamais
    une nouvelle vérification : ce claim affiche ce qui a déjà été fait,
    même esprit que les autres `_assess_*` (aucun recalcul depuis les
    données brutes). `duplicates_removed is None` = job antérieur à ce
    suivi (rétrocompatibilité par absence, jamais une affirmation inventée)."""
    if duplicates_removed is None:
        return None

    grouping_note = (
        "un regroupement (même identifiant physique) a aussi empêché qu'une variante de la même observation ne "
        "se retrouve à la fois dans l'entraînement et le test"
        if anti_leak_grouping
        else "aucune colonne de regroupement n'a été fournie — si vos lignes contiennent des variantes d'une "
        "même observation physique (mesures répétées, etc.), indiquez-en une au prochain entraînement pour "
        "renforcer ce contrôle"
    )
    if duplicates_removed > 0:
        return _claim(
            "info", "fuite_verifiee_doublons_retires",
            "Pas de fuite de données détectée",
            f"{duplicates_removed} ligne(s) strictement dupliquée(s) ont été retirées avant le découpage "
            f"entraînement/test (elles auraient sinon pu se retrouver des deux côtés) — {grouping_note}.",
            {"duplicates_removed": duplicates_removed, "anti_leak_grouping": bool(anti_leak_grouping)},
        )
    return _claim(
        "info", "fuite_verifiee_aucun_doublon",
        "Pas de fuite de données détectée",
        f"Aucune ligne strictement dupliquée trouvée dans vos données — {grouping_note}.",
        {"duplicates_removed": 0, "anti_leak_grouping": bool(anti_leak_grouping)},
    )


def _assess_cqr_coverage(task_type: str, cqr: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Classification : pas de CQR, toujours `None` (voir MLModel.cqr_json)."""
    if task_type != "regression" or not cqr:
        return None
    empirical, target = cqr["empirical_coverage"], cqr["target_coverage"]

    if empirical < target - _CQR_COVERAGE_TOLERANCE:
        return _claim(
            "critique", "couverture_insuffisante",
            "Les intervalles de prédiction sont trop optimistes",
            f"La couverture visée était de {target:.0%}, la couverture réellement observée n'est que de "
            f"{empirical:.0%} — la vraie valeur tombe hors de l'intervalle annoncé plus souvent que promis.",
            {"target_coverage": target, "empirical_coverage": empirical},
        )
    return _claim(
        "info", "couverture_conforme",
        "Les intervalles de prédiction tiennent leur promesse",
        f"La couverture visée était de {target:.0%}, la couverture réellement observée est de {empirical:.0%} — "
        "conforme.",
        {"target_coverage": target, "empirical_coverage": empirical},
    )


# Priorité décroissante — la première alerte "critique" ou "attention"
# rencontrée dans cet ordre pilote la recommandation de synthèse : le
# surapprentissage et la fiabilité conditionnent la confiance qu'on peut
# accorder à TOUTES les autres métriques, donc passent avant la calibration
# ou la couverture CQR, qui sont des raffinements.
_NEXT_ACTION_PRIORITY = [
    ("surapprentissage_marque", "Réduisez la complexité du modèle (moins de profondeur/d'essais Optuna) ou rassemblez plus de données avant de faire confiance à ce modèle."),
    ("surapprentissage_leger", "Surveillez ce modèle sur de nouvelles données avant de le promouvoir en production — l'écart entraînement/test reste modéré mais réel."),
    ("fiabilite_faible", "Collectez plus de données ou élargissez le jeu de test — la métrique affichée n'est pas assez précise pour trancher seule."),
    ("couverture_insuffisante", "Élargissez les intervalles ou revoyez le découpage en strates avant de communiquer ces intervalles à un tiers."),
    ("calibration_mauvaise", "Recalibrez les probabilités (Platt scaling ou isotonic regression) avant de les utiliser pour une décision automatisée."),
    ("ecart_gagnant_dans_le_bruit", "Le choix entre les deux meilleurs modèles est arbitraire statistiquement — le critère de départage (temps d'inférence, interprétabilité) peut légitimement primer sur le score."),
]

_DEFAULT_NEXT_ACTION = "Aucun signal d'alerte détecté — ce modèle est prêt à être utilisé ; envisagez de le promouvoir en production si les résultats vous conviennent."


def _synthesize_next_action(claims: list[dict[str, Any]]) -> str:
    codes_present = {c["code"] for c in claims}
    for code, action in _NEXT_ACTION_PRIORITY:
        if code in codes_present:
            return action
    return _DEFAULT_NEXT_ACTION


_LEVEL_ORDER = {"critique": 0, "attention": 1, "info": 2}


def compute_verdict(
    task_type: str,
    metrics: dict[str, Any],
    evaluation: dict[str, Any],
    candidates: list[dict[str, Any]],
    calibration: Optional[dict[str, Any]] = None,
    learning_curve: Optional[dict[str, Any]] = None,
    cqr: Optional[dict[str, Any]] = None,
    duplicates_removed: Optional[int] = None,
    anti_leak_grouping: Optional[bool] = None,
) -> dict[str, Any]:
    """Point d'entrée — calcule le verdict d'un modèle déjà entraîné à
    partir de données déjà persistées (`metrics_json`, `evaluation_json`,
    `ModelCandidate`, `calibration_json`, `learning_curve_json`, `cqr_json`,
    `model_card_json`). Chaque `claim` omise par une des fonctions
    `_assess_*` (donnée absente, ex. `cqr` toujours `None` en classification)
    est simplement omise du résultat — jamais remplacée par une affirmation
    inventée.

    Retourne `{"claims": [...], "next_action": "..."}`, `claims` triées
    critique > attention > info (même convention que
    `services/data_quality.py::analyze_data_quality`)."""
    assessors = [
        _assess_overfitting(task_type, metrics),
        _assess_reliability(task_type, metrics),
        _assess_metric_choice(task_type, evaluation),
        _assess_winner_margin(candidates),
        _assess_leakage(duplicates_removed, anti_leak_grouping),
        _assess_calibration(task_type, calibration),
        _assess_more_data(learning_curve),
        _assess_cqr_coverage(task_type, cqr),
    ]
    claims = [c for c in assessors if c is not None]
    claims.sort(key=lambda c: _LEVEL_ORDER.get(c["level"], 99))
    return {"claims": claims, "next_action": _synthesize_next_action(claims)}
