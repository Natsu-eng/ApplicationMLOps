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


_BOTH_TASKS = frozenset({"classification", "regression"})

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
