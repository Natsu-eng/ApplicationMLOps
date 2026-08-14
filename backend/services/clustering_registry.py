"""Registre des algorithmes de clustering — Lot 11 (ML non supervisé).

Même pattern que `services/ml_registry.py` (Lot 5) : chaque entrée déclare
tout ce qui distingue un algorithme du reste, ajouter un algorithme = ajouter
une entrée, jamais toucher le moteur (`clustering_training.py`).

Module VOLONTAIREMENT séparé de `ml_registry.py`/`ml_training.py` — confirmé
par l'audit du 2026-08-14 (AUDIT_ROADMAP.md) : `ml_training.py` est truffé
d'hypothèses binaires classification/régression (encodage de `y`, CQR, score
de sélection supervisé) qui n'ont pas de sens pour un algorithme sans cible.
Injecter un 3ᵉ `task_type="clustering"` dans ce module aurait cassé
silencieusement plusieurs branches.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

Family = Literal["partitionnement", "densite", "hierarchique"]

# Nombres de clusters candidats comparés automatiquement pour les algorithmes
# qui prennent k en paramètre — même esprit que le leaderboard du Lot D
# (plusieurs candidats comparés, jamais un seul lancé à l'aveugle). Borné à
# 8 : au-delà, des segments trop fins perdent leur intérêt métier pour un
# usage exploratoire de type "profils similaires".
CANDIDATE_K_VALUES = [2, 3, 4, 5, 6, 8]


@dataclass(frozen=True)
class ClusterCandidateConfig:
    """Une configuration concrète à évaluer — ex. KMeans avec k=4."""

    algorithm_id: str
    label: str
    params: dict[str, Any]


@dataclass(frozen=True)
class ClusterSpec:
    id: str
    label: str
    family: Family
    # True pour les algorithmes qui prennent un nombre de clusters explicite
    # (KMeans, hiérarchique) — génère un candidat par valeur de
    # `CANDIDATE_K_VALUES`. False pour DBSCAN, qui n'a pas de k mais un
    # voisinage (eps/min_samples), généré différemment (voir
    # `_dbscan_candidates`).
    needs_k: bool
    build_estimator: Callable[[dict[str, Any], int], BaseEstimator]
    candidate_configs: Callable[[int, int], list[ClusterCandidateConfig]]  # (n_samples, seed) -> configs


def _kmeans_candidates(n_samples: int, seed: int) -> list[ClusterCandidateConfig]:
    return [
        ClusterCandidateConfig("kmeans", f"K-Means (k={k})", {"n_clusters": k})
        for k in CANDIDATE_K_VALUES
        if k < n_samples
    ]


def _build_kmeans(params: dict[str, Any], seed: int) -> BaseEstimator:
    return KMeans(n_clusters=params["n_clusters"], random_state=seed, n_init=10)


def _minibatch_kmeans_candidates(n_samples: int, seed: int) -> list[ClusterCandidateConfig]:
    return [
        ClusterCandidateConfig("minibatch_kmeans", f"K-Means rapide (k={k})", {"n_clusters": k})
        for k in CANDIDATE_K_VALUES
        if k < n_samples
    ]


def _build_minibatch_kmeans(params: dict[str, Any], seed: int) -> BaseEstimator:
    return MiniBatchKMeans(n_clusters=params["n_clusters"], random_state=seed, n_init=10, batch_size=256)


def _hierarchical_candidates(n_samples: int, seed: int) -> list[ClusterCandidateConfig]:
    return [
        ClusterCandidateConfig("hierarchical", f"Clustering hiérarchique (k={k})", {"n_clusters": k})
        for k in CANDIDATE_K_VALUES
        if k < n_samples
    ]


def _build_hierarchical(params: dict[str, Any], seed: int) -> BaseEstimator:
    # Pas de `random_state` — l'agglomératif (linkage de Ward) est
    # déterministe par construction, aucune notion de graine.
    return AgglomerativeClustering(n_clusters=params["n_clusters"], linkage="ward")


def _dbscan_candidates(n_samples: int, seed: int) -> list[ClusterCandidateConfig]:
    """DBSCAN n'a pas de k — `eps` (rayon de voisinage) est dérivé de la
    distribution réelle des distances au k-ème plus proche voisin (heuristique
    du "k-distance plot", méthode usuelle pour DBSCAN), pas d'une valeur
    magique fixe. Combiné à quelques valeurs de `min_samples` pour couvrir un
    petit espace de recherche raisonnable plutôt qu'un unique essai à
    l'aveugle."""
    if n_samples < 10:
        return []
    configs: list[ClusterCandidateConfig] = []
    for min_samples in (5, 10):
        k = min(min_samples, n_samples - 1)
        try:
            neighbors = NearestNeighbors(n_neighbors=k).fit(np.zeros((n_samples, 1)))
        except ValueError:
            continue
        # Placeholder — la vraie distance est calculée dans clustering_training.py
        # une fois les données préprocessées disponibles (cette fonction ne
        # connaît que n_samples, pas X) ; eps est donc résolu plus tard via
        # `resolve_dbscan_eps`, ce candidat porte juste min_samples pour
        # l'instant, eps=None signale "à résoudre".
        configs.append(
            ClusterCandidateConfig(
                "dbscan", f"DBSCAN (voisinage ≥ {min_samples})", {"eps": None, "min_samples": min_samples}
            )
        )
    return configs


def resolve_dbscan_eps(X_processed: np.ndarray, min_samples: int) -> float:
    """Résout `eps` pour un `min_samples` donné à partir des données
    réellement préprocessées — percentile 90 de la distance au `min_samples`-
    ème plus proche voisin (heuristique du k-distance plot : le "coude" de
    cette courbe indique un bon eps ; le percentile 90 en est une
    approximation robuste et déterministe, sans tracé interactif)."""
    n_neighbors = min(min_samples + 1, X_processed.shape[0])  # +1 : un point est son propre plus proche voisin
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(X_processed)
    distances, _ = nn.kneighbors(X_processed)
    k_distances = distances[:, -1]
    return float(np.percentile(k_distances, 90))


def _build_dbscan(params: dict[str, Any], seed: int) -> BaseEstimator:
    # Pas de `random_state` — DBSCAN est déterministe par construction.
    return DBSCAN(eps=params["eps"], min_samples=params["min_samples"])


CLUSTER_REGISTRY: list[ClusterSpec] = [
    ClusterSpec(
        id="kmeans",
        label="K-Means",
        family="partitionnement",
        needs_k=True,
        build_estimator=_build_kmeans,
        candidate_configs=_kmeans_candidates,
    ),
    ClusterSpec(
        id="minibatch_kmeans",
        label="K-Means rapide (gros volumes)",
        family="partitionnement",
        needs_k=True,
        build_estimator=_build_minibatch_kmeans,
        candidate_configs=_minibatch_kmeans_candidates,
    ),
    ClusterSpec(
        id="hierarchical",
        label="Clustering hiérarchique",
        family="hierarchique",
        needs_k=True,
        build_estimator=_build_hierarchical,
        candidate_configs=_hierarchical_candidates,
    ),
    ClusterSpec(
        id="dbscan",
        label="DBSCAN (détection par densité)",
        family="densite",
        needs_k=False,
        build_estimator=_build_dbscan,
        candidate_configs=_dbscan_candidates,
    ),
]

# Gros volumes (plan Lot 11, A) : MiniBatchKMeans est réservé au mode expert
# — le sous-ensemble par défaut privilégie la qualité (K-Means classique,
# hiérarchique, DBSCAN) sur un temps de calcul raisonnable pour un dataset de
# bureau d'études typique, cohérent avec la stratégie produit "B" déjà
# adoptée pour le catalogue supervisé (Lot 5).
DEFAULT_ALGORITHM_IDS = {"kmeans", "hierarchical", "dbscan"}


def specs_for(algorithm_ids: list[str] | None = None) -> list[ClusterSpec]:
    """Sous-ensemble du registre — `None` : sous-ensemble par défaut (mode
    guidé) ; liste explicite : catalogue complet filtré (mode expert), même
    pattern que `ml_registry.models_for_task(subset=...)`."""
    if algorithm_ids is None:
        return [s for s in CLUSTER_REGISTRY if s.id in DEFAULT_ALGORITHM_IDS]
    selected = set(algorithm_ids)
    return [s for s in CLUSTER_REGISTRY if s.id in selected]
