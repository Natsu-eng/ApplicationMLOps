"""Inférence clustering — assigne une nouvelle observation à un groupe déjà
découvert, à partir du pipeline persisté par `workers/clustering_worker.py`
(Lot 6B, §F.2 — jusqu'ici, un clustering entraîné ne pouvait jamais être
réutilisé sur une nouvelle observation, contrairement au pilier supervisé,
voir `services/ml_inference.py`).

Module séparé de `ml_inference.py` — même raisonnement de fond que
`clustering_training.py`/`ml_training.py` (aucune notion de cible), à
l'exception de `load_bundle`, réutilisé tel quel : chargement joblib
générique, zéro logique spécifique au supervisé.

Trois cas selon la famille de l'algorithme retenu (voir
`services/clustering_registry.py`) :
- partitionnement (KMeans/MiniBatchKMeans) : `.predict()` natif, assignation
  EXACTE (même critère que l'entraînement — distance au centroïde le plus
  proche).
- hiérarchique (AgglomerativeClustering) : pas de `.predict()` en sklearn
  (modèle transductif) — approximation par centroïde le plus proche,
  centroïdes calculés une seule fois à l'entraînement
  (`clustering_training.py::train_and_evaluate_clustering`) et persistés
  dans le pipeline_bundle.
- densité (DBSCAN) : pas de `.predict()` non plus — approximation standard
  de la littérature (assignation au cluster du point cœur ("core sample")
  le plus proche SI cette distance est ≤ eps, sinon atypique/bruit — même
  règle que celle appliquée par DBSCAN à ses propres points d'entraînement).

Chaque cas retourne `assignment_method` ("exact" / "approximate_centroid" /
"approximate_nearest_core" / "unsupported") — jamais silencieux sur la
nature de l'assignation, cohérent avec le principe de dégradation honnête
du produit. "unsupported" couvre aussi la rétrocompatibilité par absence :
un clustering entraîné AVANT ce lot n'a pas les clés `centroids`/
`core_points` dans son pipeline_bundle."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_EXACT_PREDICT_ALGORITHM_IDS = {"kmeans", "minibatch_kmeans"}


class ClusterInferenceError(ValueError):
    """La donnée fournie ne peut pas être assignée à un cluster (colonne
    manquante, valeur non convertible...)."""


def _build_input_frame(row: dict[str, Any], feature_columns: list[str]) -> pd.DataFrame:
    missing = [c for c in feature_columns if c not in row or row[c] in (None, "")]
    if missing:
        raise ClusterInferenceError(f"Valeur manquante pour : {', '.join(missing)}")
    ordered = {c: [row[c]] for c in feature_columns}
    df = pd.DataFrame(ordered)
    # Conversion best-effort en numérique — laisse les colonnes non
    # convertibles telles quelles (catégorielles), le préprocesseur gère les
    # deux (même logique que ml_inference.py::_build_input_frame).
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if not converted.isna().any():
            df[col] = converted
    return df


def assign_cluster(bundle: dict[str, Any], feature_columns: list[str], row: dict[str, Any]) -> dict[str, Any]:
    """Retourne `{"cluster_id": int | None, "is_noise": bool, "assignment_method": str}`."""
    df = _build_input_frame(row, feature_columns)
    preprocessor = bundle["preprocessor"]
    try:
        X_proc = preprocessor.transform(df)
    except Exception as exc:
        raise ClusterInferenceError(
            "Valeurs incompatibles avec les variables utilisées lors de l'entraînement de ce clustering"
        ) from exc
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()
    point = X_proc[0]

    algorithm_id = bundle.get("algorithm_id")
    model = bundle.get("model")

    if algorithm_id in _EXACT_PREDICT_ALGORITHM_IDS and model is not None:
        cluster_id = int(model.predict(X_proc)[0])
        return {"cluster_id": cluster_id, "is_noise": False, "assignment_method": "exact"}

    if algorithm_id == "hierarchical":
        centroids: dict[int, np.ndarray] | None = bundle.get("centroids")
        if not centroids:
            return {"cluster_id": None, "is_noise": False, "assignment_method": "unsupported"}
        nearest = min(centroids, key=lambda cid: float(np.linalg.norm(point - centroids[cid])))
        return {"cluster_id": nearest, "is_noise": False, "assignment_method": "approximate_centroid"}

    if algorithm_id == "dbscan":
        core_points = bundle.get("core_points")
        core_labels = bundle.get("core_labels")
        eps = bundle.get("eps")
        if core_points is None or eps is None or len(core_points) == 0:
            return {"cluster_id": None, "is_noise": False, "assignment_method": "unsupported"}
        distances = np.linalg.norm(np.asarray(core_points) - point, axis=1)
        nearest_idx = int(np.argmin(distances))
        if distances[nearest_idx] <= eps:
            return {
                "cluster_id": int(core_labels[nearest_idx]),
                "is_noise": False,
                "assignment_method": "approximate_nearest_core",
            }
        return {"cluster_id": None, "is_noise": True, "assignment_method": "approximate_nearest_core"}

    return {"cluster_id": None, "is_noise": False, "assignment_method": "unsupported"}
