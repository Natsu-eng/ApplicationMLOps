"""Entraînement ML supervisé — LightGBM / XGBoost / CatBoost, recherche
d'hyperparamètres Optuna, sélection du meilleur modèle sur la validation
croisée, explicabilité SHAP et intervalles de confiance conformes (CQR,
variante Mondrian) pour la régression.

Méthodologie reprise d'un notebook de référence partagé par l'équipe (voir
`backend/workflow.md`, Lot 3) :
- 3 algorithmes de gradient boosting comparés systématiquement, tous
  compatibles SHAP TreeExplainer et régression quantile (CQR) — d'où le
  choix de s'y limiter pour ce lot (le catalogue sklearn plus large arrive
  au Lot 5, sans cette profondeur d'explicabilité).
- sélection du meilleur modèle sur le score de VALIDATION CROISÉE, jamais
  sur le score test — le score test n'est qu'une estimation finale rapportée.
- CV/split groupés (`GroupKFold`/`GroupShuffleSplit`) quand une colonne de
  groupe est fournie, pour rester anti-fuite jusque dans la recherche
  d'hyperparamètres.
- CQR Mondrian : la calibration conforme est faite par strate de prédiction
  plutôt qu'avec un quantile unique, pour corriger la sous-couverture aux
  valeurs extrêmes (défaut connu du split conformal simple).

Le clustering (non supervisé) est hors périmètre de ce module — voir
`services/ml_task.py`, qui ne détecte que classification/régression ; le
non-supervisé arrive avec le catalogue ML complet (Lot 5).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import optuna
import pandas as pd
import shap
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import clone
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
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder, label_binarize
from xgboost import XGBClassifier, XGBRegressor

from services.ml_preprocessing import SplitResult, build_preprocessor

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


@dataclass
class TrainedModelResult:
    algorithm: str
    pipeline_bundle: dict[str, Any]  # tout ce qu'il faut pour ré-inférer : model, preprocessor, cqr, features
    metrics: dict[str, Any]
    shap_summary: list[dict[str, Any]]
    cqr: Optional[dict[str, Any]]
    model_card: dict[str, Any]
    evaluation: dict[str, Any]  # matrice de confusion+ROC/PR (classif) ou prédit-vs-réel+résidus (régression)


# ── Espaces de recherche Optuna — un par algorithme, régression et classification ──

def _lightgbm_space(trial: optuna.Trial) -> dict:
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 50, 500),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 20, 200),
        max_depth=trial.suggest_int("max_depth", 4, 14),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 80),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    )


def _xgboost_space(trial: optuna.Trial) -> dict:
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 50, 500),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    )


def _catboost_space(trial: optuna.Trial) -> dict:
    return dict(
        iterations=trial.suggest_int("iterations", 100, 600),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        depth=trial.suggest_int("depth", 4, 10),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-3, 10, log=True),
    )


_REGRESSORS: dict[str, tuple[type, Callable[[optuna.Trial], dict]]] = {
    "LightGBM": (LGBMRegressor, _lightgbm_space),
    "XGBoost": (XGBRegressor, _xgboost_space),
    "CatBoost": (CatBoostRegressor, _catboost_space),
}

_CLASSIFIERS: dict[str, tuple[type, Callable[[optuna.Trial], dict]]] = {
    "LightGBM": (LGBMClassifier, _lightgbm_space),
    "XGBoost": (XGBClassifier, _xgboost_space),
    "CatBoost": (CatBoostClassifier, _catboost_space),
}

_QUIET_KWARGS = {
    "LightGBM": {"verbose": -1},
    "XGBoost": {"verbosity": 0},
    "CatBoost": {"verbose": False},
}


def _make_cv(task_type: str, cv_folds: int, groups: Optional[np.ndarray]):
    """GroupKFold si une colonne de groupe est fournie (anti-fuite jusque dans
    la CV) — sinon StratifiedKFold (classification) ou KFold (régression)."""
    if groups is not None:
        return GroupKFold(n_splits=cv_folds)
    if task_type == "classification":
        return StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    return KFold(n_splits=cv_folds, shuffle=True, random_state=42)


def _optimize_one_model(
    algo_name: str,
    model_cls: type,
    space_fn: Callable[[optuna.Trial], dict],
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    cv,
    groups: Optional[np.ndarray],
    config: TrainingConfig,
    progress_cb: ProgressCallback,
    progress_base: int,
    progress_span: int,
) -> tuple[Any, float]:
    """Recherche Optuna (TPE) pour un algorithme — retourne (meilleurs
    hyperparamètres appliqués à un estimateur non-fit, score de CV moyen)."""
    scoring = "r2" if task_type == "regression" else "roc_auc_ovr_weighted"

    def objective(trial: optuna.Trial) -> float:
        params = space_fn(trial)
        model = model_cls(random_state=config.seed, **_QUIET_KWARGS[algo_name], **params)
        try:
            scores = cross_val_score(model, X, y, cv=cv, groups=groups, scoring=scoring, n_jobs=1)
        except ValueError:
            # ex: roc_auc_ovr_weighted indisponible (classe absente dans un fold) — repli sur accuracy
            scores = cross_val_score(model, X, y, cv=cv, groups=groups, scoring="accuracy", n_jobs=1)
        return float(np.mean(scores))

    def on_trial_end(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        fraction = (trial.number + 1) / config.optuna_trials
        progress_cb(
            f"Optimisation {algo_name} — essai {trial.number + 1}/{config.optuna_trials}",
            progress_base + int(progress_span * fraction),
        )

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.seed))
    study.optimize(objective, n_trials=config.optuna_trials, callbacks=[on_trial_end], show_progress_bar=False)

    best_model = model_cls(random_state=config.seed, **_QUIET_KWARGS[algo_name], **study.best_params)
    return best_model, study.best_value


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


def _compute_shap_summary(estimator: Any, X_sample: np.ndarray, feature_names: list[str]) -> list[dict[str, Any]]:
    """Importance globale des features par SHAP (TreeExplainer) — moyenne des
    valeurs absolues sur un échantillon du test. Le détail par observation
    (dependence/waterfall) est laissé pour une itération future de la page
    d'évaluation (Lot 4).

    La forme de `shap_values` en classification multiclasse dépend de la
    version de SHAP/du backend : soit une liste d'une matrice
    (n_échantillons, n_features) par classe (API historique), soit un seul
    tableau (n_échantillons, n_features, n_classes) (API unifiée récente).
    Les deux sont gérées — bug réel rencontré en test (classification 3
    classes) : sans ce second cas, `mean_abs` restait 2D et l'indexation
    par une ligne entière au lieu d'un scalaire levait
    `only integer scalar arrays can be converted to a scalar index`.
    """
    explainer = shap.TreeExplainer(estimator)
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


def _compute_cqr(
    X_train_proc: np.ndarray,
    y_train: np.ndarray,
    X_test_proc: np.ndarray,
    y_test: np.ndarray,
    config: TrainingConfig,
) -> dict[str, Any]:
    """Split Conformal Quantile Regression, variante Mondrian (calibration
    par strate de prédiction) — voir le docstring du module pour la
    référence méthodologique.

    Les régresseurs de quantile sont volontairement indépendants de
    l'algorithme gagnant : ce sont des LightGBM dédiés (`objective="quantile"`),
    cohérent avec le fait que l'incertitude est une couche à part, pas une
    propriété de l'algorithme choisi pour la prédiction centrale.
    """
    alpha = config.cqr_alpha
    Xf, Xc, yf, yc = train_test_split(X_train_proc, y_train, test_size=0.30, random_state=config.seed)

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
        # Régresseurs persistés dans le bundle joblib pour une inférence future (Lot 4)
        "_q_lo_model": q_lo,
        "_q_hi_model": q_hi,
    }


def train_and_evaluate(
    split: SplitResult,
    task_type: str,
    config: TrainingConfig,
    progress_cb: ProgressCallback = _noop_progress,
) -> TrainedModelResult:
    """Point d'entrée principal — compare LightGBM/XGBoost/CatBoost (Optuna),
    sélectionne le meilleur sur la CV, calcule métriques + SHAP + (en
    régression) CQR, et retourne un résultat prêt à persister."""
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

    preprocessor = build_preprocessor(split.X_train)
    X_train_proc = preprocessor.fit_transform(split.X_train)
    X_test_proc = preprocessor.transform(split.X_test)
    X_train_proc = np.asarray(X_train_proc.todense()) if hasattr(X_train_proc, "todense") else np.asarray(X_train_proc)
    X_test_proc = np.asarray(X_test_proc.todense()) if hasattr(X_test_proc, "todense") else np.asarray(X_test_proc)
    feature_names = list(preprocessor.get_feature_names_out())

    cv = _make_cv(task_type, config.cv_folds, split.groups_train)
    catalog = _REGRESSORS if task_type == "regression" else _CLASSIFIERS

    candidates: list[tuple[str, Any, float]] = []
    n_models = len(catalog)
    span_per_model = 65 // n_models  # 5%→70% réparti entre les 3 algos
    for i, (algo_name, (model_cls, space_fn)) in enumerate(catalog.items()):
        base = 5 + i * span_per_model
        progress_cb(f"Optimisation {algo_name}", base)
        best_model, cv_score = _optimize_one_model(
            algo_name, model_cls, space_fn, X_train_proc, y_train, task_type, cv, split.groups_train,
            config, progress_cb, base, span_per_model,
        )
        candidates.append((algo_name, best_model, cv_score))
        logger.info("[Training] %s — score CV = %.4f", algo_name, cv_score)

    # Sélection sur la CV, jamais sur le test (voir docstring du module).
    algo_name, best_model, cv_score = max(candidates, key=lambda c: c[2])
    progress_cb(f"Modèle retenu : {algo_name} — entraînement final", 72)
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
    sample_size = min(config.shap_sample_size, len(X_test_proc))
    rng = np.random.default_rng(config.seed)
    sample_idx = rng.choice(len(X_test_proc), size=sample_size, replace=False)
    shap_summary = _compute_shap_summary(best_model, X_test_proc[sample_idx], feature_names)

    cqr_result: Optional[dict[str, Any]] = None
    cqr_artifacts: Optional[dict[str, Any]] = None
    if task_type == "regression":
        progress_cb("Calcul des intervalles de confiance (CQR)", 88)
        cqr_full = _compute_cqr(X_train_proc, y_train, X_test_proc, y_test, config)
        cqr_artifacts = {"q_lo": cqr_full.pop("_q_lo_model"), "q_hi": cqr_full.pop("_q_hi_model")}
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
        "cv_folds": config.cv_folds,
        "cv_score": float(cv_score),
        "optuna_trials": config.optuna_trials,
        "seed": config.seed,
        "class_names": class_names,
        "metrics": metrics,
        "cqr": cqr_result,
        "top_features": shap_summary[:10],
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
    )
