"""Entraînement ML supervisé — catalogue de modèles piloté par
`services/ml_registry.py` (9 modèles, 3 familles : arbres/ensembles,
linéaire, distance/noyau), recherche d'hyperparamètres Optuna, sélection du
meilleur modèle sur la validation croisée, explicabilité SHAP routée par
famille et intervalles de confiance conformes (CQR, variante Mondrian) pour
la régression.

Méthodologie reprise d'un notebook de référence partagé par l'équipe (voir
`backend/workflow.md`, Lot 3), étendue au catalogue complet au Lot 5 :
- Le moteur ne référence aucun nom d'algorithme en dur — il consomme
  `ml_registry.models_for_task()`. Par défaut (stratégie produit "B"), seul
  le sous-ensemble robuste/rapide tourne (boosters + RandomForest,
  `ModelSpec.is_default`) ; le reste du catalogue (ExtraTrees, régression
  régularisée, SVM, KNN, Naive Bayes) est disponible mais pas lancé tant que
  le mode expert (Lot E) n'expose pas son activation.
- sélection du meilleur modèle sur le score de VALIDATION CROISÉE, jamais
  sur le score test — le score test n'est qu'une estimation finale rapportée.
  En classification, le score (`_classification_selection_score`) reste
  calculable et comparable même pour un candidat sans `predict_proba` (ex.
  SVC construit sans calibration Platt pendant la recherche) : repli sur
  `decision_function`, puis sur l'accuracy (Lot 5).
- CV/split groupés (`GroupKFold`/`GroupShuffleSplit`) quand une colonne de
  groupe est fournie, pour rester anti-fuite jusque dans la recherche
  d'hyperparamètres.
- Préprocesseur (imputation, standardisation, one-hot) toujours refit
  À L'INTÉRIEUR de chaque fold de CV et de la portion fit du CQR — jamais
  fit sur des données qui serviront à valider ou à calibrer (Lot A). Il est
  déjà indifférencié par famille de modèle (le scaling est inconditionnel),
  ce qui satisfait sans changement le besoin des modèles à scaling requis.
- Explicabilité SHAP routée par famille (`explainer_kind` du registre) :
  TreeExplainer (arbres), LinearExplainer (linéaire), KernelExplainer
  (distance/noyau — borné et dégradé proprement au-delà d'une taille
  raisonnable, jamais un plantage silencieux, voir `_compute_explainability`).
- CQR Mondrian : la calibration conforme est faite par strate de prédiction
  plutôt qu'avec un quantile unique, pour corriger la sous-couverture aux
  valeurs extrêmes (défaut connu du split conformal simple). Le split
  fit/calibration est lui-même groupé (`GroupShuffleSplit`) quand une
  colonne de groupe est fournie (Lot A). Découplé du modèle gagnant (les
  régresseurs de quantile sont toujours des LightGBM dédiés) : fonctionne
  sans adaptation pour tout nouveau modèle de régression du catalogue.

Le clustering (non supervisé) est hors périmètre de ce module — voir
`services/ml_task.py`, qui ne détecte que classification/régression.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import optuna
import pandas as pd
import shap
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.utils.class_weight import compute_sample_weight

from services.ml_preprocessing import SplitResult, build_preprocessor
from services.ml_registry import ModelSpec, models_for_task

logger = logging.getLogger("datalab.ml_training")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Signature du callback de progression : (libellé de l'étape, pourcentage 0-100)
ProgressCallback = Callable[[str, int], None]


def _noop_progress(_step: str, _percent: int) -> None:
    pass


@dataclass
class TrainingConfig:
    test_size: float = 0.2
    seed: int = 42
    optuna_trials: int = 20
    cv_folds: int = 4
    cqr_alpha: float = 0.20
    cqr_n_strata: int = 5
    shap_sample_size: int = 500
    # Rééquilibrage des classes (lot déséquilibre), PROPOSÉ à l'utilisateur sur
    # la base du garde-fou Lot B (`services/data_quality.py::_detect_class_imbalance`),
    # jamais appliqué d'office — voir `train_and_evaluate`. Ignoré en régression
    # (concept propre à la classification). Absent/`False` : comportement
    # strictement inchangé (rétrocompatibilité totale).
    class_rebalancing: bool = False


@dataclass
class TrainedModelResult:
    algorithm: str
    pipeline_bundle: dict[str, Any]  # tout ce qu'il faut pour ré-inférer : model, preprocessor, cqr, features
    metrics: dict[str, Any]
    shap_summary: list[dict[str, Any]]
    cqr: Optional[dict[str, Any]]
    model_card: dict[str, Any]
    evaluation: dict[str, Any]  # matrice de confusion+ROC/PR (classif) ou prédit-vs-réel+résidus (régression)
    # TOUS les candidats comparés (Lot D, leaderboard) — pas seulement le
    # gagnant, triés par score de sélection décroissant. Chaque entrée :
    # algorithm, family, selection_score, is_winner, rank, fold_scores
    # (variance inter-folds, option A), secondary_metric/secondary_metric_label
    # (RMSE en régression, None en classification) — voir `_build_all_candidates`.
    all_candidates: list[dict[str, Any]]


@dataclass
class OptimizedCandidate:
    """Résultat de la recherche Optuna pour UN candidat du catalogue —
    porte, en plus du meilleur estimateur, ce qu'il faut pour l'afficher dans
    le leaderboard sans ré-entraîner (Lot D) : le score de sélection déjà
    calculé, la variance inter-folds capturée pendant la même recherche
    (option A du cadrage — `trial.set_user_attr`, aucun calcul
    supplémentaire), et l'erreur en unité réelle pour la régression."""

    model: Any
    cv_score: float
    fold_scores: Optional[list[float]]  # score de sélection par fold, essai gagnant — None si jamais capturé
    fold_error: Optional[float]  # RMSE moyen (validation croisée), régression uniquement


def selection_metric_label(task_type: str) -> str:
    """Libellé humain de LA métrique qui départage les candidats (Lot D) —
    partagé entre le moteur (model_card) et l'API (leaderboard) pour ne
    jamais afficher un nom incohérent avec le score réellement utilisé."""
    return "ROC-AUC (validation croisée)" if task_type == "classification" else "R² (validation croisée)"


def _make_cv(task_type: str, cv_folds: int, groups: Optional[np.ndarray]):
    """GroupKFold si une colonne de groupe est fournie (anti-fuite jusque dans
    la CV) — sinon StratifiedKFold (classification) ou KFold (régression)."""
    if groups is not None:
        return GroupKFold(n_splits=cv_folds)
    if task_type == "classification":
        return StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    return KFold(n_splits=cv_folds, shuffle=True, random_state=42)


def _classification_selection_score(estimator, X, y) -> float:
    """Score de sélection en classification — AUC (ovr, pondérée par classe)
    calculée à partir de `predict_proba` quand l'estimateur en dispose,
    sinon de `decision_function` (ex. SVC construit sans `probability=True`
    pendant la recherche Optuna — voir `ml_registry._build_svm` : la
    calibration Platt de `probability=True` est coûteuse et inutile pour un
    score de comparaison, elle n'est reconstruite que pour le candidat
    retenu, une seule fois). Repli sur l'accuracy seulement si l'estimateur
    n'a ni l'un ni l'autre, ou si une classe est absente du fold courant
    (roc_auc non calculable dans ce cas précis).

    Signature de scorer scikit-learn : `estimator` est déjà FIT sur le fold
    d'entraînement quand cette fonction est appelée par `cross_val_score`.
    Utilisée pour TOUS les candidats de classification du catalogue (Lot 5,
    correction 1) — condition nécessaire pour que "meilleur score CV parmi N
    modèles hétérogènes" compare des scores sur la même échelle."""
    y_arr = np.asarray(y)
    n_classes = len(np.unique(y_arr))
    try:
        if hasattr(estimator, "predict_proba"):
            proba = estimator.predict_proba(X)
            if n_classes == 2:
                return float(roc_auc_score(y_arr, proba[:, 1]))
            return float(roc_auc_score(y_arr, proba, multi_class="ovr", average="weighted"))
        if hasattr(estimator, "decision_function"):
            scores = estimator.decision_function(X)
            if n_classes == 2:
                return float(roc_auc_score(y_arr, scores))
            return float(roc_auc_score(y_arr, scores, multi_class="ovr", average="weighted"))
    except ValueError:
        pass  # ex. classe absente de ce fold — repli sur accuracy ci-dessous
    return float(accuracy_score(y_arr, estimator.predict(X)))


def _optimize_one_model(
    spec: ModelSpec,
    X_raw: pd.DataFrame,
    y: np.ndarray,
    task_type: str,
    cv,
    groups: Optional[np.ndarray],
    preprocessor_template: ColumnTransformer,
    config: TrainingConfig,
    progress_cb: ProgressCallback,
    progress_base: int,
    progress_span: int,
    sample_weight: Optional[np.ndarray] = None,
) -> OptimizedCandidate:
    """Recherche Optuna (TPE) pour un modèle du registre (Lot 5) — retourne
    le meilleur estimateur non-fit accompagné de tout ce qu'il faut pour le
    leaderboard (Lot D) : score de sélection moyen, variance inter-folds et
    (régression) erreur en unité réelle — voir `OptimizedCandidate`.

    `X_raw` est le train BRUT (non préprocessé) : chaque essai construit un
    `Pipeline(préprocesseur, modèle)` non-fit et le passe à `cross_validate`,
    qui le clone et le refit à l'intérieur de chaque fold — le préprocesseur
    ne voit donc jamais la portion de validation du fold (Lot A). Ce pattern
    est générique à tout `ModelSpec.build_estimator`, indépendamment du
    besoin de scaling du modèle (Lot 5, Phase 1 : déjà satisfait par
    `build_preprocessor`, aucune branche par famille nécessaire ici).

    `sample_weight` (lot déséquilibre, optionnel) : poids par échantillon du
    TRAIN COMPLET (déjà calculé une fois par `train_and_evaluate`), routé vers
    `model__sample_weight` — `cross_validate` le découpe lui-même selon les
    indices de chaque fold (mécanisme natif scikit-learn, vérifié), aucune
    fuite : c'est la même portion de données que celle vue par le fit, juste
    pondérée différemment. Ignoré si `spec.supports_rebalancing` est `False`
    (ex. KNN, voir `ml_registry.py`) — le modèle est alors entraîné sans
    pondération plutôt que de lever une erreur.

    Variance inter-folds + erreur réelle (Lot D, option A) : `cross_validate`
    remplace `cross_val_score` — MÊME calcul (un fit par fold, aucun fit
    supplémentaire), mais expose le détail par fold de chaque scorer au lieu
    de ne renvoyer que la moyenne. Le tableau par fold du score de sélection
    est stocké sur l'essai Optuna courant (`trial.set_user_attr`) : après la
    recherche, on relit celui de l'essai GAGNANT plutôt que de le recalculer
    (`study.best_trial.user_attrs`) — c'est le même calcul déjà fait pendant
    la recherche, jamais un ré-entraînement. En régression, un second scorer
    (RMSE) est évalué sur les MÊMES prédictions déjà produites par le fit de
    chaque fold — pas de coût supplémentaire non plus."""
    algo_name = spec.label(task_type)
    # Régression : "r2" (score de sélection) + RMSE (erreur en unité réelle,
    # Lot D précision 1 — le R² seul n'est pas lisible pour un BE) — les deux
    # scorers sont évalués sur le même modèle fit par fold, sans coût
    # supplémentaire. Classification : scorer maison
    # `_classification_selection_score`, robuste à l'absence de
    # `predict_proba` (Lot 5, correction 1) — pas de second scorer demandé
    # pour ce type de tâche (Lot D, cadrage).
    if task_type == "regression":
        scoring: dict[str, Any] = {"score": "r2", "error": "neg_root_mean_squared_error"}
    else:
        scoring = {"score": _classification_selection_score}

    apply_rebalancing = sample_weight is not None and spec.supports_rebalancing
    fit_params = {"model__sample_weight": sample_weight} if apply_rebalancing else None

    def objective(trial: optuna.Trial) -> float:
        params = spec.hyperparameter_space(trial)
        model = spec.build_estimator(task_type, config.seed, params, False)
        pipeline = Pipeline([("preprocess", clone(preprocessor_template)), ("model", model)])
        cv_results = cross_validate(
            pipeline, X_raw, y, cv=cv, groups=groups, scoring=scoring, n_jobs=1, fit_params=fit_params
        )
        fold_scores = cv_results["test_score"]
        trial.set_user_attr("fold_scores", [float(s) for s in fold_scores])
        if "test_error" in cv_results:
            trial.set_user_attr("fold_error_mean", float(np.mean(-cv_results["test_error"])))
        return float(np.mean(fold_scores))

    def on_trial_end(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        fraction = (trial.number + 1) / config.optuna_trials
        progress_cb(
            f"Optimisation {algo_name} — essai {trial.number + 1}/{config.optuna_trials}",
            progress_base + int(progress_span * fraction),
        )

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.seed))
    study.optimize(objective, n_trials=config.optuna_trials, callbacks=[on_trial_end], show_progress_bar=False)

    # `final_fit=True` : configuration de construction complète (ex. SVC
    # retrouve `probability=True`, nécessaire à `predict_proba` en évaluation
    # finale/inférence) — un seul objet construit ici, jamais fit tant que ce
    # candidat n'est pas le gagnant (voir `train_and_evaluate`).
    best_model = spec.build_estimator(task_type, config.seed, study.best_params, True)
    best_attrs = study.best_trial.user_attrs
    return OptimizedCandidate(
        model=best_model,
        cv_score=study.best_value,
        fold_scores=best_attrs.get("fold_scores"),
        fold_error=best_attrs.get("fold_error_mean"),
    )


def _regression_metrics(y_train, pred_train, y_test, pred_test) -> dict[str, float]:
    r2_train = r2_score(y_train, pred_train)
    r2_test = r2_score(y_test, pred_test)
    return {
        "r2_train": float(r2_train),
        "r2_test": float(r2_test),
        "delta_r2": float(r2_train - r2_test),
        "rmse": float(mean_squared_error(y_test, pred_test) ** 0.5),
        "mae": float(mean_absolute_error(y_test, pred_test)),
    }


def _classification_metrics(y_test, pred_test, proba_test) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred_test)),
        "f1": float(f1_score(y_test, pred_test, average="weighted", zero_division=0)),
        "precision": float(precision_score(y_test, pred_test, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, pred_test, average="weighted", zero_division=0)),
    }
    try:
        if proba_test.shape[1] == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_test, proba_test[:, 1]))
        else:
            metrics["roc_auc"] = float(roc_auc_score(y_test, proba_test, multi_class="ovr", average="weighted"))
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


def _bootstrap_ci(
    y_true: np.ndarray, y_pred: np.ndarray, metric_fn: Callable[[np.ndarray, np.ndarray], float], seed: int, n_boot: int = 500
) -> dict[str, float]:
    """Intervalle de confiance à 95 % par bootstrap — plus honnête qu'un
    chiffre nu pour une métrique calculée sur un seul jeu de test."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    y_true_arr, y_pred_arr = np.asarray(y_true), np.asarray(y_pred)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        values.append(metric_fn(y_true_arr[idx], y_pred_arr[idx]))
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
    }


# ── SHAP par famille (Lot 5) ────────────────────────────────────────────────
#
# Bornage de KernelExplainer — le seul des trois explainers dont le coût
# croît avec (taille du fond × taille de l'échantillon expliqué × nombre de
# variables) : sans borne, un modèle SVM/KNN/Naive Bayes sur un dataset large
# rendrait l'explicabilité impraticable. Constantes nommées (pas de valeur en
# dur perdue dans le code) — voir `_compute_explainability`.
_KERNEL_SHAP_MAX_FEATURES = 50  # au-delà : SHAP désactivé pour ce modèle, message explicite plutôt qu'un calcul trop long
_KERNEL_SHAP_BACKGROUND_SIZE = 50  # points de fond résumés par k-means (pas les données brutes)
_KERNEL_SHAP_SAMPLE_SIZE = 50  # échantillon expliqué — nettement plus bas que shap_sample_size (tree/linear, rapides)


def _build_explainer(explainer_kind: str, estimator: Any, X_train_background: np.ndarray):
    """Construit l'explainer SHAP adapté à la famille du modèle gagnant.

    Point critique (Lot 5, correction 2) : `X_train_background` DOIT être
    dans le même espace que celui vu par `estimator` — c'est-à-dire déjà
    passé par le préprocesseur (post-scaling, post-OHE), jamais les données
    brutes. Un `LinearExplainer`/`KernelExplainer` nourri de données brutes
    ne lève aucune erreur mais produit des valeurs SHAP fausses (les
    coefficients du modèle linéaire, eux, ont été appris dans l'espace
    préprocessé) — l'appelant (`_compute_explainability`) ne passe donc
    jamais autre chose que `X_train_proc`/`X_test_proc`.
    """
    if explainer_kind == "tree":
        return shap.TreeExplainer(estimator)
    if explainer_kind == "linear":
        return shap.LinearExplainer(estimator, X_train_background)
    background = shap.kmeans(X_train_background, min(_KERNEL_SHAP_BACKGROUND_SIZE, len(X_train_background)))
    predict_fn = estimator.predict_proba if hasattr(estimator, "predict_proba") else estimator.predict
    return shap.KernelExplainer(predict_fn, background)


def _compute_shap_summary(
    estimator: Any, explainer_kind: str, X_train_background: np.ndarray, X_sample: np.ndarray, feature_names: list[str]
) -> list[dict[str, Any]]:
    """Importance globale des features par SHAP — moyenne des valeurs
    absolues sur un échantillon du test, quel que soit le type d'explainer
    routé par `explainer_kind` (Lot 5 : tree/linear/kernel). Le détail par
    observation (dependence/waterfall) est laissé pour une itération future
    de la page d'évaluation (Lot 4).

    La forme de `shap_values` en classification multiclasse dépend de la
    version de SHAP/du backend et de l'explainer : soit une liste d'une
    matrice (n_échantillons, n_features) par classe (API historique), soit un
    seul tableau (n_échantillons, n_features, n_classes) (API unifiée
    récente) — `KernelExplainer` sur `predict_proba` produit des formes
    analogues à `TreeExplainer` en multiclasse. Les deux sont gérées — bug
    réel rencontré en test (classification 3 classes, Lot 3) : sans ce second
    cas, `mean_abs` restait 2D et l'indexation par une ligne entière au lieu
    d'un scalaire levait `only integer scalar arrays can be converted to a
    scalar index`.
    """
    explainer = _build_explainer(explainer_kind, estimator, X_train_background)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        abs_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 3:  # (n_échantillons, n_features, n_classes)
            abs_values = np.abs(shap_values).mean(axis=2)
        else:
            abs_values = np.abs(shap_values)
    mean_abs = abs_values.mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    return [{"feature": feature_names[i], "importance": float(mean_abs[i])} for i in order]


def _compute_explainability(
    estimator: Any,
    explainer_kind: str,
    X_train_proc: np.ndarray,
    X_test_proc: np.ndarray,
    feature_names: list[str],
    config: TrainingConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Calcule le résumé SHAP du modèle gagnant, ou dégrade proprement —
    jamais de plantage du job et jamais une section qui disparaît sans
    explication (Lot 5) : `explainability_status` (à afficher côté UI, pas
    seulement stocké) porte toujours un statut + un message clair en cas de
    dégradation.

    `KernelExplainer` (routage "kernel" — SVM/KNN/Naive Bayes) est borné
    (fond et échantillon réduits, voir constantes ci-dessus) et désactivé
    au-delà de `_KERNEL_SHAP_MAX_FEATURES` variables plutôt que de lancer un
    calcul trop long. Un dernier filet `try/except` couvre les cas non
    anticipés (ex. modèle dégénéré sur un petit jeu de données) : dans tous
    les cas, `train_and_evaluate` continue et produit un modèle utilisable.
    """
    n_features = len(feature_names)
    if explainer_kind == "kernel" and n_features > _KERNEL_SHAP_MAX_FEATURES:
        return [], {
            "status": "degraded",
            "message": (
                "L'explication détaillée des prédictions n'est pas disponible pour ce type de "
                f"modèle sur un jeu de données avec autant de variables ({n_features}) — "
                "le calcul serait trop long pour rester praticable."
            ),
        }

    sample_size = _KERNEL_SHAP_SAMPLE_SIZE if explainer_kind == "kernel" else config.shap_sample_size
    sample_size = min(sample_size, len(X_test_proc))
    rng = np.random.default_rng(config.seed)
    sample_idx = rng.choice(len(X_test_proc), size=sample_size, replace=False)

    try:
        shap_summary = _compute_shap_summary(
            estimator, explainer_kind, X_train_proc, X_test_proc[sample_idx], feature_names
        )
        return shap_summary, {"status": "ok", "message": None}
    except Exception as exc:  # dernier filet — l'explicabilité ne doit jamais faire échouer l'entraînement
        logger.warning("[Training] Calcul SHAP dégradé (explainer=%s) : %s", explainer_kind, exc)
        return [], {
            "status": "degraded",
            "message": "L'explication détaillée n'a pas pu être calculée pour ce modèle sur ce jeu de données.",
        }


def _downsample_curve(x: np.ndarray, y: np.ndarray, max_points: int = 100) -> tuple[list[float], list[float]]:
    """Les courbes ROC/PR de sklearn ont autant de points que d'échantillons
    test — inutile d'envoyer des milliers de points au frontend pour un
    graphe qui en affiche visuellement une centaine tout au plus."""
    if len(x) <= max_points:
        return [float(v) for v in x], [float(v) for v in y]
    idx = np.linspace(0, len(x) - 1, max_points).astype(int)
    return [float(x[i]) for i in idx], [float(y[i]) for i in idx]


def _compute_classification_evaluation(
    y_test: np.ndarray, pred_test: np.ndarray, proba_test: np.ndarray, class_names: list[str]
) -> dict[str, Any]:
    """Matrice de confusion + courbes ROC/PR (une par classe, one-vs-rest en
    multiclasse) — pour la page d'évaluation (Lot 4b), qui affiche des
    graphiques plutôt que des métriques brutes seules."""
    labels = list(range(len(class_names)))
    matrix = confusion_matrix(y_test, pred_test, labels=labels)

    roc_curves: dict[str, Any] = {}
    pr_curves: dict[str, Any] = {}

    if len(class_names) == 2:
        fpr, tpr, _ = roc_curve(y_test, proba_test[:, 1])
        precision, recall, _ = precision_recall_curve(y_test, proba_test[:, 1])
        fpr_s, tpr_s = _downsample_curve(fpr, tpr)
        prec_s, rec_s = _downsample_curve(precision, recall)
        roc_curves[class_names[1]] = {"fpr": fpr_s, "tpr": tpr_s}
        pr_curves[class_names[1]] = {"precision": prec_s, "recall": rec_s}
    else:
        y_bin = label_binarize(y_test, classes=labels)
        for i, name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_bin[:, i], proba_test[:, i])
            precision, recall, _ = precision_recall_curve(y_bin[:, i], proba_test[:, i])
            fpr_s, tpr_s = _downsample_curve(fpr, tpr)
            prec_s, rec_s = _downsample_curve(precision, recall)
            roc_curves[name] = {"fpr": fpr_s, "tpr": tpr_s}
            pr_curves[name] = {"precision": prec_s, "recall": rec_s}

    return {
        "confusion_matrix": matrix.tolist(),
        "class_names": class_names,
        "roc_curves": roc_curves,
        "pr_curves": pr_curves,
    }


def _compute_regression_evaluation(
    y_test: np.ndarray, pred_test: np.ndarray, seed: int, max_points: int = 300
) -> dict[str, Any]:
    """Échantillon prédit-vs-réel + résidus, pour les graphiques de diagnostic
    (détection d'hétéroscédasticité, biais systématique...)."""
    y_arr, pred_arr = np.asarray(y_test), np.asarray(pred_test)
    n = len(y_arr)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(max_points, n), replace=False)
    actual_sample = y_arr[idx]
    predicted_sample = pred_arr[idx]
    return {
        "actual": [float(v) for v in actual_sample],
        "predicted": [float(v) for v in predicted_sample],
        "residuals": [float(v) for v in (actual_sample - predicted_sample)],
    }


def _cqr_fit_calib_indices(
    n: int,
    groups_train: Optional[np.ndarray],
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Indices (fit, calibration) pour le split interne au CQR — groupé
    (`GroupShuffleSplit`) quand une colonne de groupe est fournie, pour
    qu'aucun groupe ne se retrouve à la fois dans le fit des régresseurs de
    quantile et dans la calibration conforme (Lot A). Sinon, split aléatoire
    classique (comportement historique)."""
    if groups_train is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        fit_idx, cal_idx = next(splitter.split(np.zeros(n), groups=groups_train))
    else:
        fit_idx, cal_idx = train_test_split(np.arange(n), test_size=test_size, random_state=seed)
    return fit_idx, cal_idx


def _compute_cqr(
    X_train_raw: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: Optional[np.ndarray],
    X_test_raw: pd.DataFrame,
    y_test: np.ndarray,
    config: TrainingConfig,
    feature_engineering_config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Split Conformal Quantile Regression, variante Mondrian (calibration
    par strate de prédiction) — voir le docstring du module pour la
    référence méthodologique.

    Les régresseurs de quantile sont volontairement indépendants de
    l'algorithme gagnant : ce sont des LightGBM dédiés (`objective="quantile"`),
    cohérent avec le fait que l'incertitude est une couche à part, pas une
    propriété de l'algorithme choisi pour la prédiction centrale.

    `X_train_raw`/`X_test_raw` sont BRUTS (non préprocessés) : le split
    fit/calibration se fait sur les données brutes, puis le préprocesseur du
    CQR est fit UNIQUEMENT sur la portion fit — jamais sur la portion de
    calibration ni sur le test (Lot A). Ce préprocesseur est distinct de
    celui du modèle principal (qui, lui, est fit sur tout le train) et doit
    être réutilisé tel quel en inférence pour rester cohérent avec les
    régresseurs de quantile qu'il a servi à alimenter.
    """
    alpha = config.cqr_alpha
    fit_idx, cal_idx = _cqr_fit_calib_indices(len(X_train_raw), groups_train, test_size=0.30, seed=config.seed)

    Xf_raw = X_train_raw.iloc[fit_idx].reset_index(drop=True)
    Xc_raw = X_train_raw.iloc[cal_idx].reset_index(drop=True)
    yf, yc = y_train[fit_idx], y_train[cal_idx]

    cqr_preprocessor = build_preprocessor(Xf_raw, feature_engineering_config)
    Xf = cqr_preprocessor.fit_transform(Xf_raw)
    Xf = np.asarray(Xf.todense()) if hasattr(Xf, "todense") else np.asarray(Xf)
    Xc = cqr_preprocessor.transform(Xc_raw)
    Xc = np.asarray(Xc.todense()) if hasattr(Xc, "todense") else np.asarray(Xc)
    X_test_proc = cqr_preprocessor.transform(X_test_raw)
    X_test_proc = np.asarray(X_test_proc.todense()) if hasattr(X_test_proc, "todense") else np.asarray(X_test_proc)

    q_lo = LGBMRegressor(objective="quantile", alpha=alpha / 2, n_estimators=300, verbose=-1, random_state=config.seed).fit(Xf, yf)
    q_hi = LGBMRegressor(objective="quantile", alpha=1 - alpha / 2, n_estimators=300, verbose=-1, random_state=config.seed).fit(Xf, yf)

    lo_cal, hi_cal = q_lo.predict(Xc), q_hi.predict(Xc)
    center_cal = (lo_cal + hi_cal) / 2

    K = config.cqr_n_strata
    bounds = np.quantile(center_cal, np.linspace(0, 1, K + 1))
    bounds[0], bounds[-1] = -np.inf, np.inf

    def strata_of(values: np.ndarray) -> np.ndarray:
        return np.clip(np.searchsorted(bounds, values, side="right") - 1, 0, K - 1)

    s_cal = strata_of(center_cal)
    yc_arr = np.asarray(yc)
    nonconformity = np.maximum(lo_cal - yc_arr, yc_arr - hi_cal)

    qhat = np.zeros(K)
    for k in range(K):
        mask = s_cal == k
        errors = nonconformity[mask] if mask.sum() > 0 else nonconformity
        n_e = len(errors)
        level = min(1.0, np.ceil((n_e + 1) * (1 - alpha)) / n_e)
        qhat[k] = np.quantile(errors, level)

    lo_test, hi_test = q_lo.predict(X_test_proc), q_hi.predict(X_test_proc)
    s_test = strata_of((lo_test + hi_test) / 2)
    lo_final = lo_test - qhat[s_test]
    hi_final = hi_test + qhat[s_test]

    # Borne physique : si toutes les valeurs observées sont positives, la
    # cible ne peut raisonnablement pas être négative (ex : durée, montant,
    # résistance...) — heuristique générique, pas une hypothèse métier figée.
    # Mémorisée (clip_negative) pour être réappliquée à l'identique en
    # inférence sur de nouvelles données (services/ml_inference.py).
    clip_negative = float(np.min(y_train)) >= 0
    if clip_negative:
        lo_final = np.clip(lo_final, 0, None)

    y_test_arr = np.asarray(y_test)
    coverage = float(np.mean((y_test_arr >= lo_final) & (y_test_arr <= hi_final)))

    return {
        "alpha": alpha,
        "target_coverage": 1 - alpha,
        "empirical_coverage": coverage,
        "mean_interval_width": float(np.mean(hi_final - lo_final)),
        "n_strata": K,
        "strata_bounds": [float(b) for b in bounds],
        "qhat_per_stratum": [float(q) for q in qhat],
        "clip_negative": clip_negative,
        # Régresseurs + préprocesseur dédié persistés dans le bundle joblib
        # pour une inférence future (Lot 4) — le préprocesseur DOIT rester
        # celui-ci (fit sur la seule portion fit) et non celui du modèle
        # principal, sous peine d'incohérence avec q_lo/q_hi (Lot A).
        "_q_lo_model": q_lo,
        "_q_hi_model": q_hi,
        "_preprocessor": cqr_preprocessor,
    }


def train_and_evaluate(
    split: SplitResult,
    task_type: str,
    config: TrainingConfig,
    progress_cb: ProgressCallback = _noop_progress,
    feature_engineering_config: Optional[dict[str, Any]] = None,
) -> TrainedModelResult:
    """Point d'entrée principal — compare LightGBM/XGBoost/CatBoost (Optuna),
    sélectionne le meilleur sur la CV, calcule métriques + SHAP + (en
    régression) CQR, et retourne un résultat prêt à persister.

    `feature_engineering_config` (Lot 4c, optionnel) : sous-partie "pipeline"
    de la spec de feature engineering (`{"frequency_encoding": [...],
    "imputation": {...}}`), transmise telle quelle à `build_preprocessor` à
    ses 3 points d'appel (gabarit Optuna, préprocesseur final, préprocesseur
    CQR) — les transformations déterministes (datetime, ratio) sont déjà
    appliquées en amont par l'appelant (voir `workers/training_worker.py`),
    `split` les contient donc déjà. Absent/`None` : comportement strictement
    inchangé (rétrocompatibilité totale)."""
    progress_cb("Préparation des données", 2)

    class_names: Optional[list[str]] = None
    y_train_raw, y_test_raw = split.y_train, split.y_test
    if task_type == "classification":
        encoder = LabelEncoder()
        y_train = encoder.fit_transform(y_train_raw)
        y_test = encoder.transform(y_test_raw)
        class_names = [str(c) for c in encoder.classes_]
    else:
        y_train = y_train_raw.to_numpy(dtype=float)
        y_test = y_test_raw.to_numpy(dtype=float)

    # Rééquilibrage des classes (lot déséquilibre) — PROPOSÉ à l'utilisateur
    # sur la base du garde-fou Lot B, jamais appliqué d'office (voir
    # `TrainingConfig.class_rebalancing`). Un seul poids par échantillon du
    # train, calculé une fois ici et réutilisé pour tous les candidats — pas
    # de rééchantillonnage, pas de modification du split : `sample_weight`
    # ne fait que repondérer la fonction de perte de chaque modèle pendant
    # son `.fit()` (Lot A : split train/test/CV strictement inchangé,
    # confirmé par `test_ml_training.py`). Ignoré en régression, concept
    # propre à la classification (cohérent avec `_detect_class_imbalance`).
    sample_weight: Optional[np.ndarray] = None
    if task_type == "classification" and config.class_rebalancing:
        sample_weight = compute_sample_weight("balanced", y_train)

    # Préprocesseur NON fit à ce stade — seule sa structure (colonnes
    # numériques/catégorielles) est déterminée ici. Il est cloné et refit à
    # l'intérieur de chaque fold de CV par `_optimize_one_model`, jamais fit
    # sur tout le train en amont de la validation croisée (Lot A : évite que
    # le préprocesseur ait vu la portion de validation de chaque fold).
    preprocessor_template = build_preprocessor(split.X_train, feature_engineering_config)

    cv = _make_cv(task_type, config.cv_folds, split.groups_train)
    # Stratégie produit "B" (Lot 5) : par défaut, seul le sous-ensemble
    # robuste/rapide tourne (boosters + RandomForest, `ModelSpec.is_default`)
    # — les modèles sensibles/lents (SVM, KNN, linéaire, Naive Bayes) restent
    # dans le registre, disponibles mais pas lancés tant que le mode expert
    # (Lot E) n'expose pas leur activation. Câblé ici en un seul mot
    # ("default" → "all") sans toucher au reste de la boucle : c'est la
    # mécanique que le Lot E n'aura qu'à rendre pilotable depuis l'API/l'UI.
    catalog = models_for_task(task_type, subset="default")

    candidates: list[tuple[str, ModelSpec, OptimizedCandidate]] = []
    n_models = len(catalog)
    span_per_model = 65 // n_models  # 5%→70% réparti entre les modèles du catalogue
    for i, spec in enumerate(catalog):
        algo_name = spec.label(task_type)
        base = 5 + i * span_per_model
        progress_cb(f"Optimisation {algo_name}", base)
        opt = _optimize_one_model(
            spec, split.X_train, y_train, task_type, cv, split.groups_train,
            preprocessor_template, config, progress_cb, base, span_per_model,
            sample_weight=sample_weight,
        )
        candidates.append((algo_name, spec, opt))
        logger.info("[Training] %s — score CV = %.4f", algo_name, opt.cv_score)

    # Sélection sur la CV, jamais sur le test (voir docstring du module).
    # `winning_spec` conservé pour router l'explainer SHAP sur la bonne
    # famille (Lot 5) — le reste du pipeline (métriques, CQR) reste
    # indépendant du type de modèle, comme avant.
    algo_name, winning_spec, winning_opt = max(candidates, key=lambda c: c[2].cv_score)
    best_model, cv_score = winning_opt.model, winning_opt.cv_score
    progress_cb(f"Modèle retenu : {algo_name} — entraînement final", 72)

    # Leaderboard (Lot D) — TOUS les candidats, triés par score de sélection
    # décroissant, jamais par accuracy (voir `_classification_selection_score`
    # et la correction de `_headline_metric`). Construit ici, avant le fit
    # final, à partir de ce que `_optimize_one_model` a déjà calculé — aucun
    # ré-entraînement pour produire ce classement.
    secondary_label = "RMSE (validation croisée)" if task_type == "regression" else None
    ranked_candidates = sorted(candidates, key=lambda c: c[2].cv_score, reverse=True)
    all_candidates = [
        {
            "algorithm": name,
            "family": spec.family,
            "selection_score": opt.cv_score,
            "is_winner": name == algo_name,
            "rank": rank,
            "fold_scores": opt.fold_scores,
            "secondary_metric": opt.fold_error,
            "secondary_metric_label": secondary_label if opt.fold_error is not None else None,
        }
        for rank, (name, spec, opt) in enumerate(ranked_candidates, start=1)
    ]

    # Artefact de production : fit sur TOUT le train, comme pour n'importe
    # quel modèle final destiné au déploiement (pas de fuite ici, contrairement
    # à la CV — il n'y a plus de portion de validation à préserver une fois le
    # modèle sélectionné).
    preprocessor = build_preprocessor(split.X_train, feature_engineering_config)
    X_train_proc = preprocessor.fit_transform(split.X_train)
    X_test_proc = preprocessor.transform(split.X_test)
    X_train_proc = np.asarray(X_train_proc.todense()) if hasattr(X_train_proc, "todense") else np.asarray(X_train_proc)
    X_test_proc = np.asarray(X_test_proc.todense()) if hasattr(X_test_proc, "todense") else np.asarray(X_test_proc)
    feature_names = list(preprocessor.get_feature_names_out())
    # Même règle que pendant la recherche Optuna : pondération appliquée
    # uniquement si demandée ET supportée par le modèle gagnant (`winning_spec`,
    # voir `ml_registry.py`) — sinon fit standard, silencieusement, jamais
    # d'erreur (cas KNN si jamais retenu en mode expert futur).
    apply_rebalancing = sample_weight is not None and winning_spec.supports_rebalancing
    if apply_rebalancing:
        best_model.fit(X_train_proc, y_train, sample_weight=sample_weight)
    else:
        best_model.fit(X_train_proc, y_train)

    pred_train = best_model.predict(X_train_proc)
    pred_test = best_model.predict(X_test_proc)
    if task_type == "regression":
        pred_train = np.clip(pred_train, 0, None) if float(np.min(y_train)) >= 0 else pred_train
        pred_test = np.clip(pred_test, 0, None) if float(np.min(y_train)) >= 0 else pred_test
        metrics = _regression_metrics(y_train, pred_train, y_test, pred_test)
        metrics["cv_score"] = float(cv_score)
        metrics["r2_bootstrap"] = _bootstrap_ci(y_test, pred_test, r2_score, config.seed)
        metrics["rmse_bootstrap"] = _bootstrap_ci(
            y_test, pred_test, lambda a, b: float(mean_squared_error(a, b) ** 0.5), config.seed
        )
        evaluation = _compute_regression_evaluation(y_test, pred_test, config.seed)
    else:
        proba_test = best_model.predict_proba(X_test_proc)
        metrics = _classification_metrics(y_test, pred_test, proba_test)
        metrics["cv_score"] = float(cv_score)
        metrics["accuracy_bootstrap"] = _bootstrap_ci(y_test, pred_test, accuracy_score, config.seed)
        evaluation = _compute_classification_evaluation(y_test, pred_test, proba_test, class_names or [])

    progress_cb("Calcul de l'explicabilité (SHAP)", 78)
    shap_summary, explainability_status = _compute_explainability(
        best_model, winning_spec.explainer_kind, X_train_proc, X_test_proc, feature_names, config
    )

    cqr_result: Optional[dict[str, Any]] = None
    cqr_artifacts: Optional[dict[str, Any]] = None
    if task_type == "regression":
        progress_cb("Calcul des intervalles de confiance (CQR)", 88)
        cqr_full = _compute_cqr(
            split.X_train, y_train, split.groups_train, split.X_test, y_test, config, feature_engineering_config
        )
        cqr_artifacts = {
            "q_lo": cqr_full.pop("_q_lo_model"),
            "q_hi": cqr_full.pop("_q_hi_model"),
            "preprocessor": cqr_full.pop("_preprocessor"),
        }
        cqr_result = cqr_full

    progress_cb("Constitution de la fiche modèle", 95)
    model_card = {
        "algorithm": algo_name,
        "task_type": task_type,
        "n_features": len(feature_names),
        "n_train": len(split.X_train),
        "n_test": len(split.X_test),
        "duplicates_removed": split.n_duplicates_removed,
        "anti_leak_grouping": split.groups_train is not None,
        "feature_engineering_active": bool(feature_engineering_config),
        # Lot déséquilibre — "requested" reflète le choix utilisateur,
        # "applied" son effet réel sur le modèle RETENU (peut différer si le
        # gagnant est un modèle sans support, ex. KNN en mode expert futur :
        # jamais une désactivation silencieuse, le frontend peut afficher les
        # deux pour rester transparent, comme `explainability_status`).
        "class_rebalancing_requested": bool(config.class_rebalancing),
        "class_rebalancing_applied": bool(apply_rebalancing),
        "cv_folds": config.cv_folds,
        "cv_score": float(cv_score),
        "optuna_trials": config.optuna_trials,
        "seed": config.seed,
        "class_names": class_names,
        "metrics": metrics,
        "cqr": cqr_result,
        "top_features": shap_summary[:10],
        # Statut d'explicabilité (Lot 5) — "ok" ou "degraded" + message FR
        # clair. Doit être AFFICHÉ côté frontend quand dégradé, pas
        # seulement stocké (voir ModelResultModal.tsx).
        "explainability": explainability_status,
    }

    bundle: dict[str, Any] = {
        "model": best_model,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "class_names": class_names,
        "task_type": task_type,
    }
    if cqr_artifacts:
        bundle["cqr"] = {**cqr_artifacts, **{k: v for k, v in (cqr_result or {}).items() if not k.startswith("_")}}

    progress_cb("Terminé", 100)
    return TrainedModelResult(
        algorithm=algo_name,
        pipeline_bundle=bundle,
        metrics=metrics,
        shap_summary=shap_summary,
        cqr=cqr_result,
        model_card=model_card,
        evaluation=evaluation,
        all_candidates=all_candidates,
    )
