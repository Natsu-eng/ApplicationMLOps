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
from sklearn.metrics import pairwise_distances_argmin_min

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


def assign_clusters_batch(bundle: dict[str, Any], feature_columns: list[str], df: pd.DataFrame) -> pd.DataFrame:
    """Version vectorisée de `assign_cluster` pour un dataset ENTIER — répond
    au retour utilisateur direct : "l'entreprise vient avec 50 000 lignes,
    l'entraînement n'en clusterise que 5 000 (plafond mémoire, voir
    `services/clustering_training.py::MAX_ROWS_FOR_CLUSTERING`) — comment
    obtenir un cluster pour les 45 000 autres ?". Applique le modèle déjà
    entraîné à TOUTES les lignes de `df` en une seule passe vectorisée
    (jamais une boucle Python ligne par ligne, prohibitif à cette échelle —
    `pairwise_distances_argmin_min` pour les 2 méthodes approximatives).

    Retourne une COPIE de `df` avec 3 colonnes ajoutées :
    - `cluster_id` (Int64 nullable — `pd.NA` si non assignable)
    - `is_noise` (bool)
    - `assignment_method` (même vocabulaire que `assign_cluster`, plus
      `missing_column`/`missing_features`/`incompatible_values` pour les cas
      qu'une inférence ligne par ligne rejetterait — ici on ne peut pas faire
      échouer tout l'export pour quelques lignes incomplètes, chaque ligne
      garde une trace honnête de pourquoi elle n'a pas pu être assignée)."""
    result = df.copy()
    result["cluster_id"] = pd.array([None] * len(result), dtype="Int64")
    result["is_noise"] = False
    result["assignment_method"] = "unsupported"

    missing_cols = [c for c in feature_columns if c not in df.columns]
    if missing_cols:
        result["assignment_method"] = "missing_column"
        return result
    if len(result) == 0:
        return result

    X_raw = df[feature_columns].replace("", np.nan)
    valid_mask = ~X_raw.isna().any(axis=1)
    result.loc[~valid_mask, "assignment_method"] = "missing_features"
    if not valid_mask.any():
        return result

    X_valid = X_raw.loc[valid_mask].copy()
    for col in X_valid.columns:
        converted = pd.to_numeric(X_valid[col], errors="coerce")
        if not converted.isna().any():
            X_valid[col] = converted

    preprocessor = bundle["preprocessor"]
    try:
        X_proc = preprocessor.transform(X_valid)
    except Exception:
        result.loc[valid_mask, "assignment_method"] = "incompatible_values"
        return result
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()

    algorithm_id = bundle.get("algorithm_id")
    model = bundle.get("model")
    valid_idx = result.index[valid_mask]

    if algorithm_id in _EXACT_PREDICT_ALGORITHM_IDS and model is not None:
        cluster_ids = model.predict(X_proc)
        result.loc[valid_idx, "cluster_id"] = cluster_ids.astype("int64")
        result.loc[valid_idx, "assignment_method"] = "exact"
        return result

    if algorithm_id == "hierarchical":
        centroids: dict[int, np.ndarray] | None = bundle.get("centroids")
        if not centroids:
            return result
        cluster_ids_sorted = sorted(centroids)
        centroid_matrix = np.stack([centroids[cid] for cid in cluster_ids_sorted])
        nearest_idx, _ = pairwise_distances_argmin_min(X_proc, centroid_matrix)
        result.loc[valid_idx, "cluster_id"] = np.asarray(cluster_ids_sorted)[nearest_idx].astype("int64")
        result.loc[valid_idx, "assignment_method"] = "approximate_centroid"
        return result

    if algorithm_id == "dbscan":
        core_points = bundle.get("core_points")
        core_labels = bundle.get("core_labels")
        eps = bundle.get("eps")
        if core_points is None or eps is None or len(core_points) == 0:
            return result
        nearest_idx, nearest_dist = pairwise_distances_argmin_min(X_proc, np.asarray(core_points))
        is_core_reachable = nearest_dist <= eps
        assigned_labels = np.asarray(core_labels)[nearest_idx]
        result.loc[valid_idx, "assignment_method"] = "approximate_nearest_core"
        reachable_idx = valid_idx[is_core_reachable]
        noise_idx = valid_idx[~is_core_reachable]
        result.loc[reachable_idx, "cluster_id"] = assigned_labels[is_core_reachable].astype("int64")
        result.loc[noise_idx, "is_noise"] = True
        return result

    return result
