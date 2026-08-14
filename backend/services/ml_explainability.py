"""Construction d'explainers SHAP par famille de modèle — logique PARTAGÉE
entre l'entraînement (résumé global : barres + beeswarm, `services/
ml_training.py`) et l'inférence (explication locale par prédiction, `services/
ml_inference.py`, Lot Explicabilité locale). Isolée ici pour qu'aucune des
deux consommatrices ne puisse diverger sur la normalisation de la sortie
SHAP — bug réel rencontré au Lot 3 (`explainer.shap_values` renvoie soit une
liste d'une matrice par classe, soit un seul tableau 3D `(n_échantillons,
n_features, n_classes)` selon la version de SHAP/le backend d'arbre) : une
seule fonction de normalisation, jamais deux copies qui pourraient dériver.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import shap

# Bornage du fond pour Linear/KernelExplainer (les deux seuls qui en ont
# besoin — TreeExplainer n'en exige aucun) — même ordre de grandeur que
# `_KERNEL_SHAP_BACKGROUND_SIZE` historique de `ml_training.py`, dupliqué en
# constante ici pour que ce module reste autonome (pas de dépendance vers
# ml_training.py, qui importe déjà des primitives d'autres services).
EXPLAINER_BACKGROUND_SIZE = 50


def build_explainer(explainer_kind: str, estimator: Any, background: Optional[np.ndarray] = None) -> Any:
    """Construit l'explainer SHAP adapté à la famille du modèle
    (`explainer_kind` du registre — voir `services/ml_registry.py`).

    `background` DOIT être dans le même espace que celui vu par `estimator`
    (post-préprocesseur), déjà dense — jamais les données brutes ni un
    tableau sparse : un `LinearExplainer`/`KernelExplainer` nourri du mauvais
    espace produit des valeurs SHAP fausses sans lever d'erreur. Ignoré pour
    `explainer_kind == "tree"` (non nécessaire)."""
    if explainer_kind == "tree":
        return shap.TreeExplainer(estimator)
    if explainer_kind == "linear":
        return shap.LinearExplainer(estimator, background)
    kmeans_background = shap.kmeans(background, min(EXPLAINER_BACKGROUND_SIZE, len(background)))
    predict_fn = estimator.predict_proba if hasattr(estimator, "predict_proba") else estimator.predict
    return shap.KernelExplainer(predict_fn, kmeans_background)


def shap_values_per_class(shap_values: Any) -> list[np.ndarray] | np.ndarray:
    """Normalise la sortie brute de `explainer.shap_values` en une liste de
    matrices `(n_échantillons, n_features)` (une par classe), ou une matrice
    2D unique en régression/sortie unique. Voir la docstring du module pour
    le bug historique que cette normalisation évite de reproduire."""
    if isinstance(shap_values, list):
        return [np.asarray(sv) for sv in shap_values]
    arr = np.asarray(shap_values)
    if arr.ndim == 3:  # (n_échantillons, n_features, n_classes)
        return [arr[:, :, k] for k in range(arr.shape[2])]
    return arr  # 2D — régression ou classification à une seule sortie


def select_class_matrix(class_matrices: list[np.ndarray] | np.ndarray, class_index: Optional[int]) -> np.ndarray:
    """Sélectionne la matrice `(n_échantillons, n_features)` d'UNE classe à
    partir de la sortie normalisée de `shap_values_per_class` — pour
    l'explicabilité locale, où l'on veut expliquer la classe PRÉDITE, pas
    toutes à la fois. Si `shap_values_per_class` a renvoyé une matrice 2D
    unique (régression, ou classification à sortie unique selon la
    version de SHAP), elle est déjà la seule disponible — retournée telle
    quelle, `class_index` sans effet."""
    if isinstance(class_matrices, list):
        if not class_matrices:
            raise ValueError("Aucune matrice SHAP disponible")
        idx = class_index if class_index is not None and 0 <= class_index < len(class_matrices) else -1
        return class_matrices[idx]
    return class_matrices


def normalize_base_value(expected_value: Any, class_index: Optional[int]) -> float:
    """`explainer.expected_value` porte la même ambiguïté de forme que
    `shap_values` (scalaire, liste, ou tableau — un par classe ou une seule
    valeur selon la version de SHAP/le type d'explainer) — même principe de
    normalisation défensive, jamais une hypothèse sur la forme."""
    if isinstance(expected_value, (list, np.ndarray)):
        arr = np.asarray(expected_value).ravel()
        if len(arr) == 0:
            return 0.0
        idx = class_index if class_index is not None and 0 <= class_index < len(arr) else -1
        return float(arr[idx])
    return float(expected_value)
