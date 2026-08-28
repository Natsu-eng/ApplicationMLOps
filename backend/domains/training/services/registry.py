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
from typing import Any, Callable, Literal, Optional

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
# `overrides` (mode expert, retour utilisateur direct : "laisser le choix sur
# les hyperparamètres, profondeur des arbres etc.") — optionnel, toujours en
# dernier paramètre pour rester compatible avec tout appelant existant à un
# seul argument (`spec.hyperparameter_space(trial)`, comportement strictement
# inchangé). Voir `_sg` ci-dessous pour le mécanisme de substitution.
HyperparameterSpace = Callable[[optuna.Trial, "Optional[dict[str, Any]]"], dict[str, Any]]


def _sg(overrides: dict[str, Any], name: str, suggest: Callable[[], Any]) -> Any:
    """Un seul mécanisme de surcharge pour les 8 espaces d'hyperparamètres du
    catalogue (mode expert, Lot Hyperparamètres) — si l'utilisateur a fixé
    `name`, retourne SA valeur sans jamais appeler Optuna (donc sans jamais
    l'exposer à `trial.suggest_*`, qui échouerait sur `study.best_params` s'il
    n'était jamais réellement demandé) ; sinon délègue à Optuna via le thunk
    `suggest` (ex. `lambda: trial.suggest_int("max_depth", 4, 14)`) — jamais
    deux chemins de calcul différents pour le même hyperparamètre."""
    return overrides[name] if name in overrides else suggest()


@dataclass(frozen=True)
class HyperparamMeta:
    """Décrit UN hyperparamètre réglable d'un modèle pour le mode expert —
    consommé par `GET /training/models-catalog` pour que le frontend rende le
    bon type de contrôle (curseur entier/flottant, liste déroulante) SANS
    connaître le catalogue de modèles en dur. Les bornes reprennent
    EXACTEMENT celles du `trial.suggest_*` correspondant dans le `_xxx_space`
    du même modèle ci-dessous — vérifié par
    `test_ml_registry.py::test_tunable_hyperparameters_match_the_actual_search_space`
    (garde-fou anti-dérive : jamais des bornes affichées différentes de
    celles réellement utilisées)."""

    name: str
    label: str
    kind: Literal["int", "float", "categorical"]
    low: Optional[float] = None
    high: Optional[float] = None
    log: bool = False
    choices: Optional[tuple[str, ...]] = None
    help: str = ""


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: Callable[[TaskType], str]  # nom lisible, potentiellement différent par tâche (ex. Ridge/LogisticRegression)
    family: Family
    supported_tasks: frozenset[TaskType]
    build_estimator: EstimatorBuilder
    hyperparameter_space: HyperparameterSpace
    explainer_kind: ExplainerKind
    is_default: bool  # fait partie du sous-ensemble lancé par défaut (stratégie produit "B")
    # Rééquilibrage des classes (lot déséquilibre) — déclaratif, un seul mécanisme
    # pour tout le catalogue : `sample_weight` passé à `.fit()`, mathématiquement
    # équivalent à `class_weight="balanced"` (c'est son implémentation interne
    # côté sklearn) mais supporté de façon quasi uniforme par sklearn, LightGBM,
    # XGBoost et CatBoost — pas besoin d'un paramètre différent par famille. Seul
    # KNN n'a aucune notion de pondération d'échantillon (vote par plus proches
    # voisins) : False pour lui uniquement (voir `services/ml_training.py`).
    supports_rebalancing: bool
    # Bug réel trouvé en production (retour direct, adult.csv en mode expert) —
    # GaussianNB (seul modèle concerné) ne supporte PAS les matrices creuses
    # (sklearn lève un TypeError net) : le préprocesseur produit du sparse dès
    # qu'un one-hot est présent (voir engine.py, docstring du module), ce qui
    # cassait tout essai Optuna de ce modèle. Défaut à False : les 8 autres
    # modèles du catalogue acceptent le sparse nativement (vérifié
    # empiriquement) — ne pas densifier pour eux reste la bonne valeur par
    # défaut (mémoire).
    requires_dense_input: bool = False
    # Mode expert (Lot Hyperparamètres, retour utilisateur direct : "laisser
    # le choix sur les hyperparamètres, profondeur des arbres etc.") —
    # catalogue des hyperparamètres réglables de CE modèle, pour le sélecteur
    # du mode expert. Vide par défaut (rétrocompatibilité : un modèle sans
    # entrée ici reste utilisable normalement, juste sans réglage manuel).
    tunable_hyperparameters: tuple[HyperparamMeta, ...] = ()


# ── Famille arbres/ensembles — LightGBM / XGBoost / CatBoost ──────────────
# Espaces Optuna et kwargs de silence repris à l'identique du Lot 3
# (services/ml_training.py avant Lot 5) — aucun changement de comportement.

def _lightgbm_space(trial: optuna.Trial, overrides: Optional[dict[str, Any]] = None) -> dict:
    o = overrides or {}
    return dict(
        n_estimators=_sg(o, "n_estimators", lambda: trial.suggest_int("n_estimators", 50, 500)),
        learning_rate=_sg(o, "learning_rate", lambda: trial.suggest_float("learning_rate", 0.01, 0.2, log=True)),
        num_leaves=_sg(o, "num_leaves", lambda: trial.suggest_int("num_leaves", 20, 200)),
        max_depth=_sg(o, "max_depth", lambda: trial.suggest_int("max_depth", 4, 14)),
        min_child_samples=_sg(o, "min_child_samples", lambda: trial.suggest_int("min_child_samples", 5, 80)),
        subsample=_sg(o, "subsample", lambda: trial.suggest_float("subsample", 0.6, 1.0)),
        colsample_bytree=_sg(o, "colsample_bytree", lambda: trial.suggest_float("colsample_bytree", 0.6, 1.0)),
        reg_alpha=_sg(o, "reg_alpha", lambda: trial.suggest_float("reg_alpha", 1e-3, 10, log=True)),
        reg_lambda=_sg(o, "reg_lambda", lambda: trial.suggest_float("reg_lambda", 1e-3, 10, log=True)),
    )


_LIGHTGBM_HYPERPARAMS = (
    HyperparamMeta(
        "n_estimators", "Nombre d'arbres", "int", 50, 500,
        help="Plus d'arbres = modèle plus précis mais entraînement plus long, risque de surapprentissage accru.",
    ),
    HyperparamMeta(
        "learning_rate", "Taux d'apprentissage", "float", 0.01, 0.2, log=True,
        help="Plus bas = apprentissage prudent (plus d'arbres requis), plus haut = convergence rapide mais instable.",
    ),
    HyperparamMeta(
        "num_leaves", "Nombre de feuilles par arbre", "int", 20, 200,
        help="Complexité de chaque arbre — plus élevé = arbres plus expressifs, risque de surapprentissage accru.",
    ),
    HyperparamMeta(
        "max_depth", "Profondeur maximale des arbres", "int", 4, 14,
        help="Plus profond = interactions plus complexes capturées, risque de surapprentissage plus élevé.",
    ),
    HyperparamMeta(
        "min_child_samples", "Observations minimales par feuille", "int", 5, 80,
        help="Plus élevé = arbres plus prudents (moins de surapprentissage), moins élevé = arbres plus fins.",
    ),
    HyperparamMeta(
        "subsample", "Proportion de lignes par arbre", "float", 0.6, 1.0,
        help="Sous-échantillonnage des lignes à chaque arbre — réduit la variance, ajoute de la diversité.",
    ),
    HyperparamMeta(
        "colsample_bytree", "Proportion de variables par arbre", "float", 0.6, 1.0,
        help="Sous-échantillonnage des variables à chaque arbre — réduit la corrélation entre arbres.",
    ),
    HyperparamMeta(
        "reg_alpha", "Régularisation L1", "float", 1e-3, 10, log=True,
        help="Pénalise les poids non nuls — plus élevé, plus le modèle simplifie (certaines variables ignorées).",
    ),
    HyperparamMeta(
        "reg_lambda", "Régularisation L2", "float", 1e-3, 10, log=True,
        help="Pénalise les poids élevés — plus élevé, plus le modèle lisse ses prédictions.",
    ),
)


def _build_lightgbm(task_type: TaskType, seed: int, params: dict, final_fit: bool) -> BaseEstimator:
    cls = LGBMClassifier if task_type == "classification" else LGBMRegressor
    return cls(random_state=seed, verbose=-1, **params)


def _xgboost_space(trial: optuna.Trial, overrides: Optional[dict[str, Any]] = None) -> dict:
    o = overrides or {}
    return dict(
        n_estimators=_sg(o, "n_estimators", lambda: trial.suggest_int("n_estimators", 50, 500)),
        learning_rate=_sg(o, "learning_rate", lambda: trial.suggest_float("learning_rate", 0.01, 0.2, log=True)),
        max_depth=_sg(o, "max_depth", lambda: trial.suggest_int("max_depth", 3, 12)),
        subsample=_sg(o, "subsample", lambda: trial.suggest_float("subsample", 0.6, 1.0)),
        colsample_bytree=_sg(o, "colsample_bytree", lambda: trial.suggest_float("colsample_bytree", 0.6, 1.0)),
        reg_alpha=_sg(o, "reg_alpha", lambda: trial.suggest_float("reg_alpha", 1e-3, 10, log=True)),
        reg_lambda=_sg(o, "reg_lambda", lambda: trial.suggest_float("reg_lambda", 1e-3, 10, log=True)),
    )


_XGBOOST_HYPERPARAMS = (
    HyperparamMeta(
        "n_estimators", "Nombre d'arbres", "int", 50, 500,
        help="Plus d'arbres = modèle plus précis mais entraînement plus long, risque de surapprentissage accru.",
    ),
    HyperparamMeta(
        "learning_rate", "Taux d'apprentissage", "float", 0.01, 0.2, log=True,
        help="Plus bas = apprentissage prudent (plus d'arbres requis), plus haut = convergence rapide mais instable.",
    ),
    HyperparamMeta(
        "max_depth", "Profondeur maximale des arbres", "int", 3, 12,
        help="Plus profond = interactions plus complexes capturées, risque de surapprentissage plus élevé.",
    ),
    HyperparamMeta(
        "subsample", "Proportion de lignes par arbre", "float", 0.6, 1.0,
        help="Sous-échantillonnage des lignes à chaque arbre — réduit la variance.",
    ),
    HyperparamMeta(
        "colsample_bytree", "Proportion de variables par arbre", "float", 0.6, 1.0,
        help="Sous-échantillonnage des variables à chaque arbre — réduit la corrélation entre arbres.",
    ),
    HyperparamMeta(
        "reg_alpha", "Régularisation L1", "float", 1e-3, 10, log=True,
        help="Pénalise les poids non nuls — plus élevé, plus le modèle simplifie.",
    ),
    HyperparamMeta(
        "reg_lambda", "Régularisation L2", "float", 1e-3, 10, log=True,
        help="Pénalise les poids élevés — plus élevé, plus le modèle lisse ses prédictions.",
    ),
)


def _build_xgboost(task_type: TaskType, seed: int, params: dict, final_fit: bool) -> BaseEstimator:
    cls = XGBClassifier if task_type == "classification" else XGBRegressor
    return cls(random_state=seed, verbosity=0, **params)


def _catboost_space(trial: optuna.Trial, overrides: Optional[dict[str, Any]] = None) -> dict:
    o = overrides or {}
    return dict(
        iterations=_sg(o, "iterations", lambda: trial.suggest_int("iterations", 100, 600)),
        learning_rate=_sg(o, "learning_rate", lambda: trial.suggest_float("learning_rate", 0.01, 0.2, log=True)),
        depth=_sg(o, "depth", lambda: trial.suggest_int("depth", 4, 10)),
        l2_leaf_reg=_sg(o, "l2_leaf_reg", lambda: trial.suggest_float("l2_leaf_reg", 1e-3, 10, log=True)),
    )


_CATBOOST_HYPERPARAMS = (
    HyperparamMeta(
        "iterations", "Nombre d'itérations", "int", 100, 600,
        help="Plus d'itérations = modèle plus précis mais entraînement plus long, risque de surapprentissage accru.",
    ),
    HyperparamMeta(
        "learning_rate", "Taux d'apprentissage", "float", 0.01, 0.2, log=True,
        help="Plus bas = apprentissage prudent, plus haut = convergence rapide mais instable.",
    ),
    HyperparamMeta(
        "depth", "Profondeur des arbres", "int", 4, 10,
        help="Plus profond = interactions plus complexes capturées, risque de surapprentissage plus élevé.",
    ),
    HyperparamMeta(
        "l2_leaf_reg", "Régularisation L2", "float", 1e-3, 10, log=True,
        help="Pénalise les poids élevés — plus élevé, plus le modèle lisse ses prédictions.",
    ),
)


def _build_catboost(task_type: TaskType, seed: int, params: dict, final_fit: bool) -> BaseEstimator:
    cls = CatBoostClassifier if task_type == "classification" else CatBoostRegressor
    return cls(random_state=seed, verbose=False, **params)


# ── Famille arbres/ensembles — RandomForest / ExtraTrees ───────────────────
# `n_jobs=1` explicite : évite la sur-souscription CPU pendant la recherche
# Optuna (déjà exécutée avec `cross_val_score(..., n_jobs=1)`, voir
# `ml_training.py`) — les boosters gèrent leur propre parallélisme interne
# différemment, ce paramètre est spécifique aux forêts scikit-learn.

def _random_forest_space(trial: optuna.Trial, overrides: Optional[dict[str, Any]] = None) -> dict:
    o = overrides or {}
    return dict(
        n_estimators=_sg(o, "n_estimators", lambda: trial.suggest_int("n_estimators", 100, 500)),
        max_depth=_sg(o, "max_depth", lambda: trial.suggest_int("max_depth", 3, 20)),
        min_samples_split=_sg(o, "min_samples_split", lambda: trial.suggest_int("min_samples_split", 2, 20)),
        min_samples_leaf=_sg(o, "min_samples_leaf", lambda: trial.suggest_int("min_samples_leaf", 1, 10)),
        max_features=_sg(o, "max_features", lambda: trial.suggest_float("max_features", 0.3, 1.0)),
    )


_RANDOM_FOREST_HYPERPARAMS = (
    HyperparamMeta(
        "n_estimators", "Nombre d'arbres", "int", 100, 500,
        help="Plus d'arbres = estimation plus stable, entraînement plus long — peu de risque de surapprentissage.",
    ),
    HyperparamMeta(
        "max_depth", "Profondeur maximale des arbres", "int", 3, 20,
        help="Plus profond = interactions plus complexes capturées, risque de surapprentissage plus élevé.",
    ),
    HyperparamMeta(
        "min_samples_split", "Observations minimales pour diviser un nœud", "int", 2, 20,
        help="Plus élevé = arbres plus prudents (moins de surapprentissage).",
    ),
    HyperparamMeta(
        "min_samples_leaf", "Observations minimales par feuille", "int", 1, 10,
        help="Plus élevé = arbres plus prudents, prédictions plus lissées.",
    ),
    HyperparamMeta(
        "max_features", "Proportion de variables par division", "float", 0.3, 1.0,
        help="Moins de variables considérées à chaque division = arbres plus diversifiés (moins corrélés entre eux).",
    ),
)


def _build_random_forest(task_type: TaskType, seed: int, params: dict, final_fit: bool) -> BaseEstimator:
    cls = RandomForestClassifier if task_type == "classification" else RandomForestRegressor
    return cls(random_state=seed, n_jobs=1, **params)


def _extra_trees_space(trial: optuna.Trial, overrides: Optional[dict[str, Any]] = None) -> dict:
    o = overrides or {}
    return dict(
        n_estimators=_sg(o, "n_estimators", lambda: trial.suggest_int("n_estimators", 100, 500)),
        max_depth=_sg(o, "max_depth", lambda: trial.suggest_int("max_depth", 3, 20)),
        min_samples_split=_sg(o, "min_samples_split", lambda: trial.suggest_int("min_samples_split", 2, 20)),
        min_samples_leaf=_sg(o, "min_samples_leaf", lambda: trial.suggest_int("min_samples_leaf", 1, 10)),
        max_features=_sg(o, "max_features", lambda: trial.suggest_float("max_features", 0.3, 1.0)),
    )


_EXTRA_TREES_HYPERPARAMS = (
    HyperparamMeta(
        "n_estimators", "Nombre d'arbres", "int", 100, 500,
        help="Plus d'arbres = estimation plus stable, au prix d'un entraînement plus long.",
    ),
    HyperparamMeta(
        "max_depth", "Profondeur maximale des arbres", "int", 3, 20,
        help="Plus profond = interactions plus complexes capturées, risque de surapprentissage plus élevé.",
    ),
    HyperparamMeta(
        "min_samples_split", "Observations minimales pour diviser un nœud", "int", 2, 20,
        help="Plus élevé = arbres plus prudents (moins de surapprentissage).",
    ),
    HyperparamMeta(
        "min_samples_leaf", "Observations minimales par feuille", "int", 1, 10,
        help="Plus élevé = arbres plus prudents, prédictions plus lissées.",
    ),
    HyperparamMeta(
        "max_features", "Proportion de variables par division", "float", 0.3, 1.0,
        help="Moins de variables par division = arbres plus diversifiés (moins corrélés entre eux).",
    ),
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

def _linear_reg_space(trial: optuna.Trial, overrides: Optional[dict[str, Any]] = None) -> dict:
    o = overrides or {}
    return dict(
        regularization_strength=_sg(
            o, "regularization_strength", lambda: trial.suggest_float("regularization_strength", 1e-3, 100, log=True)
        )
    )


_LINEAR_REG_HYPERPARAMS = (
    HyperparamMeta(
        "regularization_strength", "Force de régularisation", "float", 1e-3, 100, log=True,
        help="Plus élevé = coefficients plus petits (moins de surapprentissage) ; plus bas = modèle plus libre.",
    ),
)


def _build_linear_reg(task_type: TaskType, seed: int, params: dict, final_fit: bool) -> BaseEstimator:
    strength = params.get("regularization_strength", 1.0)
    if task_type == "classification":
        return LogisticRegression(C=1.0 / strength, max_iter=1000, random_state=seed)
    return Ridge(alpha=strength, random_state=seed)


# ── Famille distance/noyau — SVM / KNN / Naive Bayes ────────────────────────

def _svm_space(trial: optuna.Trial, overrides: Optional[dict[str, Any]] = None) -> dict:
    o = overrides or {}
    return dict(
        C=_sg(o, "C", lambda: trial.suggest_float("C", 1e-2, 100, log=True)),
        kernel=_sg(o, "kernel", lambda: trial.suggest_categorical("kernel", ["rbf", "linear"])),
        gamma=_sg(o, "gamma", lambda: trial.suggest_categorical("gamma", ["scale", "auto"])),
    )


_SVM_HYPERPARAMS = (
    HyperparamMeta(
        "C", "Pénalité de régularisation (C)", "float", 1e-2, 100, log=True,
        help="Plus élevé = tolère moins d'erreurs sur l'entraînement (surapprentissage) ; plus bas = marge plus large.",
    ),
    HyperparamMeta(
        "kernel", "Noyau", "categorical", choices=("rbf", "linear"),
        help="rbf capture des frontières non linéaires ; linear est plus simple si les classes sont séparables.",
    ),
    HyperparamMeta(
        "gamma", "Portée du noyau (gamma)", "categorical", choices=("scale", "auto"),
        help="Influence de chaque point d'entraînement pour le noyau « rbf » — sans effet avec « linear ».",
    ),
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


def _knn_space(trial: optuna.Trial, overrides: Optional[dict[str, Any]] = None) -> dict:
    o = overrides or {}
    return dict(
        n_neighbors=_sg(o, "n_neighbors", lambda: trial.suggest_int("n_neighbors", 3, 30)),
        weights=_sg(o, "weights", lambda: trial.suggest_categorical("weights", ["uniform", "distance"])),
        p=_sg(o, "p", lambda: trial.suggest_int("p", 1, 2)),  # 1 = distance de Manhattan, 2 = euclidienne
    )


_KNN_HYPERPARAMS = (
    HyperparamMeta(
        "n_neighbors", "Nombre de voisins", "int", 3, 30,
        help="Plus élevé = décision plus lissée, plus bas = plus sensible aux détails locaux (surapprentissage).",
    ),
    HyperparamMeta(
        "weights", "Pondération des voisins", "categorical", choices=("uniform", "distance"),
        help="« uniform » : tous les voisins comptent pareil. « distance » : les plus proches comptent plus.",
    ),
    HyperparamMeta(
        "p", "Distance utilisée (1=Manhattan, 2=euclidienne)", "int", 1, 2,
        help="1 = distance de Manhattan, 2 = distance euclidienne (la plus courante).",
    ),
)


def _build_knn(task_type: TaskType, seed: int, params: dict, final_fit: bool) -> BaseEstimator:
    cls = KNeighborsClassifier if task_type == "classification" else KNeighborsRegressor
    return cls(n_jobs=1, **params)  # pas de random_state : algorithme déterministe à hyperparamètres fixés


def _naive_bayes_space(trial: optuna.Trial, overrides: Optional[dict[str, Any]] = None) -> dict:
    o = overrides or {}
    return dict(
        var_smoothing=_sg(o, "var_smoothing", lambda: trial.suggest_float("var_smoothing", 1e-12, 1e-6, log=True))
    )


_NAIVE_BAYES_HYPERPARAMS = (
    HyperparamMeta(
        "var_smoothing", "Lissage de variance", "float", 1e-12, 1e-6, log=True,
        help="Stabilise le calcul si une variable est presque constante dans une classe — rarement utile à changer.",
    ),
)


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
        is_default=True,
        supports_rebalancing=True,
        tunable_hyperparameters=_LIGHTGBM_HYPERPARAMS,
    ),
    "xgboost": ModelSpec(
        id="xgboost",
        label=lambda _t: "XGBoost",
        family="arbre_ensemble",
        supported_tasks=_BOTH_TASKS,
        build_estimator=_build_xgboost,
        hyperparameter_space=_xgboost_space,
        explainer_kind="tree",
        is_default=True,
        supports_rebalancing=True,
        tunable_hyperparameters=_XGBOOST_HYPERPARAMS,
    ),
    "catboost": ModelSpec(
        id="catboost",
        label=lambda _t: "CatBoost",
        family="arbre_ensemble",
        supported_tasks=_BOTH_TASKS,
        build_estimator=_build_catboost,
        hyperparameter_space=_catboost_space,
        explainer_kind="tree",
        is_default=True,
        supports_rebalancing=True,
        tunable_hyperparameters=_CATBOOST_HYPERPARAMS,
    ),
    "random_forest": ModelSpec(
        id="random_forest",
        label=lambda _t: "Forêt aléatoire (Random Forest)",
        family="arbre_ensemble",
        supported_tasks=_BOTH_TASKS,
        build_estimator=_build_random_forest,
        hyperparameter_space=_random_forest_space,
        explainer_kind="tree",
        is_default=True,  # sous-ensemble par défaut, stratégie produit "B" — boosters + RandomForest
        supports_rebalancing=True,
        tunable_hyperparameters=_RANDOM_FOREST_HYPERPARAMS,
    ),
    "extra_trees": ModelSpec(
        id="extra_trees",
        label=lambda _t: "Arbres extrêmement aléatoires (Extra Trees)",
        family="arbre_ensemble",
        supported_tasks=_BOTH_TASKS,
        build_estimator=_build_extra_trees,
        hyperparameter_space=_extra_trees_space,
        explainer_kind="tree",
        is_default=False,  # disponible, activable en mode expert (Lot E)
        supports_rebalancing=True,
        tunable_hyperparameters=_EXTRA_TREES_HYPERPARAMS,
    ),
    "linear_reg": ModelSpec(
        id="linear_reg",
        label=lambda t: "Régression Ridge (régularisée)" if t == "regression" else "Régression logistique",
        family="lineaire",
        supported_tasks=_BOTH_TASKS,
        build_estimator=_build_linear_reg,
        hyperparameter_space=_linear_reg_space,
        explainer_kind="linear",
        is_default=False,
        supports_rebalancing=True,
        tunable_hyperparameters=_LINEAR_REG_HYPERPARAMS,
    ),
    "svm": ModelSpec(
        id="svm",
        label=lambda t: "Machine à vecteurs de support (SVR)" if t == "regression" else "Machine à vecteurs de support (SVC)",
        family="distance_noyau",
        supported_tasks=_BOTH_TASKS,
        build_estimator=_build_svm,
        hyperparameter_space=_svm_space,
        explainer_kind="kernel",
        is_default=False,
        supports_rebalancing=True,
        tunable_hyperparameters=_SVM_HYPERPARAMS,
    ),
    "knn": ModelSpec(
        id="knn",
        label=lambda _t: "K plus proches voisins (KNN)",
        family="distance_noyau",
        supported_tasks=_BOTH_TASKS,
        build_estimator=_build_knn,
        hyperparameter_space=_knn_space,
        explainer_kind="kernel",
        is_default=False,
        # Seul modèle du catalogue sans notion de pondération d'échantillon : le
        # vote par plus proches voisins n'a pas de paramètre `sample_weight` en
        # `.fit()` (contrairement à tous les autres, voir docstring `ModelSpec`).
        supports_rebalancing=False,
        tunable_hyperparameters=_KNN_HYPERPARAMS,
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
        is_default=False,
        supports_rebalancing=True,
        requires_dense_input=True,
        tunable_hyperparameters=_NAIVE_BAYES_HYPERPARAMS,
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
