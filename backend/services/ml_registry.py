"""Registre des modèles ML supervisés — catalogue de classification/régression
(Lot 5).

Chaque entrée (`ModelSpec`) déclare, au même endroit, tout ce qui distingue un
modèle du reste : sa famille, les tâches qu'il supporte, comment le construire
(chaque estimateur scikit-learn/booster a ses propres particularités —
`random_state` absent pour KNN/SVR, `probability=True` coûteux pour SVC et
inutile pendant la recherche Optuna...), son espace de recherche Optuna, et le
type d'explainer SHAP à lui appliquer (`services/ml_training.py`, Lot 5).

Ajouter un modèle = ajouter une entrée à `MODEL_REGISTRY` (+ sa fonction
d'espace Optuna et son constructeur si besoin d'une particularité). Le moteur
d'entraînement ne référence plus aucun nom d'algorithme en dur — il consomme
`models_for_task()`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import optuna
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

TaskType = Literal["classification", "regression"]
Family = Literal["arbre_ensemble", "lineaire", "distance_noyau"]
ExplainerKind = Literal["tree", "linear", "kernel"]

# Signature commune à tous les constructeurs d'estimateur : (tâche, graine,
# hyperparamètres, construction_finale) -> estimateur non-fit.
# `construction_finale` distingue la configuration utilisée pendant la
# recherche Optuna (rapide, répétée à chaque essai × chaque fold) de celle
# utilisée pour le seul candidat retenu par algorithme après la recherche —
# ex. SVC : `probability=False` (rapide, `decision_function`) pendant la
# recherche, `probability=True` seulement pour le candidat final (un seul
# objet construit, jamais fit à ce stade — voir `ml_training.py`,
# `_optimize_one_model`). La plupart des modèles ignorent ce paramètre.
EstimatorBuilder = Callable[[TaskType, int, dict[str, Any], bool], BaseEstimator]
HyperparameterSpace = Callable[[optuna.Trial], dict[str, Any]]


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: Callable[[TaskType], str]  # nom lisible, potentiellement différent par tâche (ex. Ridge/LogisticRegression)
    family: Family
    supported_tasks: frozenset[TaskType]
    build_estimator: EstimatorBuilder
    hyperparameter_space: HyperparameterSpace
    explainer_kind: ExplainerKind
    requires_scaling: bool  # déclaratif — build_preprocessor scale déjà tout inconditionnellement (Lot 5, Phase 1)
    is_default: bool  # fait partie du sous-ensemble lancé par défaut (stratégie produit "B")


# ── Famille arbres/ensembles — LightGBM / XGBoost / CatBoost ──────────────
# Espaces Optuna et kwargs de silence repris à l'identique du Lot 3
# (services/ml_training.py avant Lot 5) — aucun changement de comportement.

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


def _build_lightgbm(task_type: TaskType, seed: int, params: dict, final_fit: bool) -> BaseEstimator:
    cls = LGBMClassifier if task_type == "classification" else LGBMRegressor
    return cls(random_state=seed, verbose=-1, **params)


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


def _build_xgboost(task_type: TaskType, seed: int, params: dict, final_fit: bool) -> BaseEstimator:
    cls = XGBClassifier if task_type == "classification" else XGBRegressor
    return cls(random_state=seed, verbosity=0, **params)


def _catboost_space(trial: optuna.Trial) -> dict:
    return dict(
        iterations=trial.suggest_int("iterations", 100, 600),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        depth=trial.suggest_int("depth", 4, 10),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-3, 10, log=True),
    )


def _build_catboost(task_type: TaskType, seed: int, params: dict, final_fit: bool) -> BaseEstimator:
    cls = CatBoostClassifier if task_type == "classification" else CatBoostRegressor
    return cls(random_state=seed, verbose=False, **params)


# ── Famille arbres/ensembles — RandomForest / ExtraTrees ───────────────────
# `n_jobs=1` explicite : évite la sur-souscription CPU pendant la recherche
# Optuna (déjà exécutée avec `cross_val_score(..., n_jobs=1)`, voir
# `ml_training.py`) — les boosters gèrent leur propre parallélisme interne
# différemment, ce paramètre est spécifique aux forêts scikit-learn.

def _random_forest_space(trial: optuna.Trial) -> dict:
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 500),
        max_depth=trial.suggest_int("max_depth", 3, 20),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        max_features=trial.suggest_float("max_features", 0.3, 1.0),
    )


def _build_random_forest(task_type: TaskType, seed: int, params: dict, final_fit: bool) -> BaseEstimator:
    cls = RandomForestClassifier if task_type == "classification" else RandomForestRegressor
    return cls(random_state=seed, n_jobs=1, **params)


def _extra_trees_space(trial: optuna.Trial) -> dict:
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 500),
        max_depth=trial.suggest_int("max_depth", 3, 20),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        max_features=trial.suggest_float("max_features", 0.3, 1.0),
    )


def _build_extra_trees(task_type: TaskType, seed: int, params: dict, final_fit: bool) -> BaseEstimator:
    cls = ExtraTreesClassifier if task_type == "classification" else ExtraTreesRegressor
    return cls(random_state=seed, n_jobs=1, **params)


# ── Famille linéaire — régression régularisée (Ridge / LogisticRegression) ─
# Un seul spec pour les deux : Ridge et LogisticRegression n'ont pas le même
# nom d'hyperparamètre pour un concept équivalent (`alpha` vs `C`, et `C` est
# l'INVERSE de la force de régularisation) — l'espace Optuna reste générique
# (`regularization_strength`), c'est `build_estimator` qui connaît la
# traduction propre à chaque tâche, sans exposer cette différence au moteur.

def _linear_reg_space(trial: optuna.Trial) -> dict:
    return dict(regularization_strength=trial.suggest_float("regularization_strength", 1e-3, 100, log=True))


def _build_linear_reg(task_type: TaskType, seed: int, params: dict, final_fit: bool) -> BaseEstimator:
    strength = params.get("regularization_strength", 1.0)
    if task_type == "classification":
        return LogisticRegression(C=1.0 / strength, max_iter=1000, random_state=seed)
    return Ridge(alpha=strength, random_state=seed)


# ── Famille distance/noyau — SVM / KNN / Naive Bayes ────────────────────────

def _svm_space(trial: optuna.Trial) -> dict:
    return dict(
        C=trial.suggest_float("C", 1e-2, 100, log=True),
        kernel=trial.suggest_categorical("kernel", ["rbf", "linear"]),
        gamma=trial.suggest_categorical("gamma", ["scale", "auto"]),
    )


def _build_svm(task_type: TaskType, seed: int, params: dict, final_fit: bool) -> BaseEstimator:
    if task_type == "classification":
        # `probability=False` pendant la recherche Optuna — évite la
        # calibration Platt interne (5-fold), coûteuse et inutile pour un
        # score de comparaison (voir `ml_training._classification_selection_score`,
        # qui route sur `decision_function` en son absence). Reconstruit avec
        # `probability=True` uniquement pour le candidat final retenu
        # (`final_fit=True`) — nécessaire à `predict_proba` en évaluation et
        # en inférence, cohérent avec tous les autres modèles du catalogue.
        return SVC(random_state=seed, probability=final_fit, **params)
    return SVR(**params)  # SVR : solveur déterministe, pas de random_state


def _knn_space(trial: optuna.Trial) -> dict:
    return dict(
        n_neighbors=trial.suggest_int("n_neighbors", 3, 30),
        weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
        p=trial.suggest_int("p", 1, 2),  # 1 = distance de Manhattan, 2 = euclidienne
    )


def _build_knn(task_type: TaskType, seed: int, params: dict, final_fit: bool) -> BaseEstimator:
    cls = KNeighborsClassifier if task_type == "classification" else KNeighborsRegressor
    return cls(n_jobs=1, **params)  # pas de random_state : algorithme déterministe à hyperparamètres fixés


def _naive_bayes_space(trial: optuna.Trial) -> dict:
    return dict(var_smoothing=trial.suggest_float("var_smoothing", 1e-12, 1e-6, log=True))


def _build_naive_bayes(task_type: TaskType, seed: int, params: dict, final_fit: bool) -> BaseEstimator:
    return GaussianNB(**params)  # classification uniquement — pas de composante stochastique, pas de random_state


_BOTH_TASKS = frozenset({"classification", "regression"})
_CLASSIFICATION_ONLY = frozenset({"classification"})

MODEL_REGISTRY: dict[str, ModelSpec] = {
    "lightgbm": ModelSpec(
        id="lightgbm",
        label=lambda _t: "LightGBM",
        family="arbre_ensemble",
        supported_tasks=_BOTH_TASKS,
        build_estimator=_build_lightgbm,
        hyperparameter_space=_lightgbm_space,
        explainer_kind="tree",
        requires_scaling=False,
        is_default=True,
    ),
    "xgboost": ModelSpec(
        id="xgboost",
        label=lambda _t: "XGBoost",
        family="arbre_ensemble",
        supported_tasks=_BOTH_TASKS,
        build_estimator=_build_xgboost,
        hyperparameter_space=_xgboost_space,
        explainer_kind="tree",
        requires_scaling=False,
        is_default=True,
    ),
    "catboost": ModelSpec(
        id="catboost",
        label=lambda _t: "CatBoost",
        family="arbre_ensemble",
        supported_tasks=_BOTH_TASKS,
        build_estimator=_build_catboost,
        hyperparameter_space=_catboost_space,
        explainer_kind="tree",
        requires_scaling=False,
        is_default=True,
    ),
    "random_forest": ModelSpec(
        id="random_forest",
        label=lambda _t: "Forêt aléatoire (Random Forest)",
        family="arbre_ensemble",
        supported_tasks=_BOTH_TASKS,
        build_estimator=_build_random_forest,
        hyperparameter_space=_random_forest_space,
        explainer_kind="tree",
        requires_scaling=False,
        is_default=True,  # sous-ensemble par défaut, stratégie produit "B" — boosters + RandomForest
    ),
    "extra_trees": ModelSpec(
        id="extra_trees",
        label=lambda _t: "Arbres extrêmement aléatoires (Extra Trees)",
        family="arbre_ensemble",
        supported_tasks=_BOTH_TASKS,
        build_estimator=_build_extra_trees,
        hyperparameter_space=_extra_trees_space,
        explainer_kind="tree",
        requires_scaling=False,
        is_default=False,  # disponible, activable en mode expert (Lot E)
    ),
    "linear_reg": ModelSpec(
        id="linear_reg",
        label=lambda t: "Régression Ridge (régularisée)" if t == "regression" else "Régression logistique",
        family="lineaire",
        supported_tasks=_BOTH_TASKS,
        build_estimator=_build_linear_reg,
        hyperparameter_space=_linear_reg_space,
        explainer_kind="linear",
        requires_scaling=True,
        is_default=False,
    ),
    "svm": ModelSpec(
        id="svm",
        label=lambda t: "Machine à vecteurs de support (SVR)" if t == "regression" else "Machine à vecteurs de support (SVC)",
        family="distance_noyau",
        supported_tasks=_BOTH_TASKS,
        build_estimator=_build_svm,
        hyperparameter_space=_svm_space,
        explainer_kind="kernel",
        requires_scaling=True,
        is_default=False,
    ),
    "knn": ModelSpec(
        id="knn",
        label=lambda _t: "K plus proches voisins (KNN)",
        family="distance_noyau",
        supported_tasks=_BOTH_TASKS,
        build_estimator=_build_knn,
        hyperparameter_space=_knn_space,
        explainer_kind="kernel",
        requires_scaling=True,
        is_default=False,
    ),
    "naive_bayes": ModelSpec(
        id="naive_bayes",
        label=lambda _t: "Naive Bayes gaussien",
        # Rattaché à "distance_noyau" par CONVENTION DE ROUTAGE (Lot 5, Phase
        # 1/2) — pas par nature mathématique : Naive Bayes n'est ni un arbre
        # ni un modèle linéaire au sens SHAP (pas de coef_ exploitable par
        # LinearExplainer). Il route vers KernelExplainer et bénéficie du
        # scaling comme SVM/KNN, d'où ce bucket plutôt qu'une 4e famille.
        family="distance_noyau",
        supported_tasks=_CLASSIFICATION_ONLY,
        build_estimator=_build_naive_bayes,
        hyperparameter_space=_naive_bayes_space,
        explainer_kind="kernel",
        requires_scaling=True,
        is_default=False,
    ),
}


def models_for_task(task_type: TaskType, subset: Literal["all", "default"] = "all") -> list[ModelSpec]:
    """Entrées du registre supportant `task_type`, dans l'ordre de déclaration.

    `subset="default"` : sous-ensemble lancé automatiquement (stratégie
    produit "B" — boosters + RandomForest, voir `ModelSpec.is_default`).
    `subset="all"` : catalogue complet — c'est la mécanique qu'activera le
    mode expert (Lot E) sans toucher au moteur d'entraînement.
    """
    specs = [spec for spec in MODEL_REGISTRY.values() if task_type in spec.supported_tasks]
    if subset == "default":
        specs = [spec for spec in specs if spec.is_default]
    return specs
