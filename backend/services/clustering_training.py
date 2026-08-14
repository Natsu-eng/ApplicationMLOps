"""Moteur d'entraînement du clustering — Lot 11 (ML non supervisé).

Module séparé de `ml_training.py` par construction (voir
`clustering_registry.py` pour le raisonnement complet) — aucune donnée ni
fonction partagée au-delà de `services/ml_preprocessing.py::build_preprocessor`,
générique et déjà indépendant de toute notion de cible.

Suit le même principe que le Lot D (leaderboard supervisé) : plusieurs
configurations sont comparées automatiquement (plusieurs k, plusieurs
algorithmes), jamais un seul essai lancé à l'aveugle — voir
`services/data-science.md` du skill senior-ai-saas-engineer, section
"Sélection du nombre de clusters : assistée... comparer plusieurs
configurations avant de choisir"."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from services.clustering_registry import CLUSTER_REGISTRY, resolve_dbscan_eps, specs_for
from services.ml_preprocessing import TrainingAbortedError, build_preprocessor

ProgressCallback = Callable[[str, int], None]


@dataclass
class ClusteringConfig:
    # `None` = mode guidé (sous-ensemble par défaut du registre) ; liste
    # explicite = mode expert, même pattern que `TrainingConfig.model_ids`
    # côté supervisé (Lot E2).
    algorithm_ids: Optional[list[str]] = None
    seed: int = 42


@dataclass
class ClusterCandidateResult:
    algorithm_id: str
    label: str
    family: str
    params: dict[str, Any]
    n_clusters: int
    silhouette: Optional[float]
    davies_bouldin: Optional[float]
    calinski_harabasz: Optional[float]
    noise_ratio: float
    is_winner: bool
    rank: int


@dataclass
class ClusterProfile:
    cluster_id: int
    size: int
    size_pct: float
    numeric_summary: dict[str, dict[str, float]]
    categorical_summary: dict[str, dict[str, Any]]
    # Variables numériques triées par |z-score| décroissant — ce qui
    # distingue le plus ce cluster de la population globale, pas seulement
    # ses propres statistiques (voir skill senior-ai-saas-engineer,
    # data-science.md : "variables différenciantes... pas seulement les
    # stats propres au cluster").
    differentiating_variables: list[str]


@dataclass
class ClusteringResult:
    winning_algorithm_id: str
    winning_label: str
    all_candidates: list[ClusterCandidateResult]
    labels: list[int]  # cluster assigné par ligne du dataset d'origine (-1 = bruit, DBSCAN)
    noise_count: int
    cluster_profiles: list[ClusterProfile]
    feature_columns: list[str]
    model_card: dict[str, Any]
    pipeline_bundle: dict[str, Any] = field(repr=False)


def _compute_cluster_metrics(X_processed: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Silhouette / Davies-Bouldin / Calinski-Harabasz calculés en EXCLUANT
    le bruit DBSCAN (label -1) — un point de bruit n'appartient à aucun
    cluster réel, l'inclure fausserait les trois métriques. `None` si moins
    de 2 clusters distincts restent (dégénéré, jamais une exception —
    filtré ensuite par l'appelant, même logique de dégradation propre que
    `services/ml_training.py`)."""
    labels = np.asarray(labels)
    noise_mask = labels == -1
    noise_ratio = float(noise_mask.mean())
    core_labels = labels[~noise_mask]
    core_X = X_processed[~noise_mask]
    n_clusters = len(np.unique(core_labels))

    if n_clusters < 2 or n_clusters >= len(core_labels):
        return {
            "n_clusters": n_clusters,
            "silhouette": None,
            "davies_bouldin": None,
            "calinski_harabasz": None,
            "noise_ratio": noise_ratio,
        }
    return {
        "n_clusters": n_clusters,
        "silhouette": float(silhouette_score(core_X, core_labels)),
        "davies_bouldin": float(davies_bouldin_score(core_X, core_labels)),
        "calinski_harabasz": float(calinski_harabasz_score(core_X, core_labels)),
        "noise_ratio": noise_ratio,
    }


def _build_cluster_profiles(X: pd.DataFrame, labels: np.ndarray) -> list[ClusterProfile]:
    """Profil interprétable par cluster, calculé sur les données D'ORIGINE
    (pas préprocessées — une moyenne sur une colonne one-hot ne veut rien
    dire pour un utilisateur non-DS). Jamais de profil pour le bruit DBSCAN
    (-1) : ce n'est pas un segment, juste des points non rattachés."""
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    global_mean = X[numeric_cols].mean() if numeric_cols else pd.Series(dtype=float)
    global_std = X[numeric_cols].std() if numeric_cols else pd.Series(dtype=float)

    n_total = len(X)
    unique_labels = sorted(int(c) for c in set(labels.tolist()) if c != -1)
    profiles: list[ClusterProfile] = []

    for cluster_id in unique_labels:
        mask = labels == cluster_id
        subset = X[mask]
        size = int(mask.sum())

        numeric_summary: dict[str, dict[str, float]] = {}
        z_scores: dict[str, float] = {}
        for col in numeric_cols:
            mean = float(subset[col].mean()) if size > 0 else 0.0
            median = float(subset[col].median()) if size > 0 else 0.0
            std = global_std.get(col, 0.0)
            z = (mean - global_mean.get(col, 0.0)) / std if std and not np.isnan(std) and std != 0 else 0.0
            numeric_summary[col] = {"mean": mean, "median": median, "z_score": float(z)}
            z_scores[col] = abs(float(z))

        categorical_summary: dict[str, dict[str, Any]] = {}
        for col in categorical_cols:
            counts = subset[col].value_counts()
            if len(counts) > 0 and size > 0:
                categorical_summary[col] = {
                    "top_category": str(counts.index[0]),
                    "top_pct": float(counts.iloc[0] / size * 100),
                }

        differentiating = sorted(z_scores, key=lambda c: z_scores[c], reverse=True)[:5]

        profiles.append(
            ClusterProfile(
                cluster_id=cluster_id,
                size=size,
                size_pct=float(size / n_total * 100) if n_total else 0.0,
                numeric_summary=numeric_summary,
                categorical_summary=categorical_summary,
                differentiating_variables=differentiating,
            )
        )
    return profiles


def train_and_evaluate_clustering(
    X: pd.DataFrame,
    config: ClusteringConfig,
    progress_cb: ProgressCallback,
) -> ClusteringResult:
    """Point d'entrée principal — compare plusieurs configurations
    (algorithme × hyperparamètres) du registre, sélectionne la meilleure sur
    le score de silhouette, calcule les profils de segments du candidat
    retenu. Jamais de `y` : le clustering n'a pas de cible, contrairement au
    supervisé (`ml_training.py::train_and_evaluate`)."""
    progress_cb("Préparation des données", 5)
    preprocessor = build_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)
    if hasattr(X_processed, "toarray"):
        # Contrairement au supervisé (Lot 3, sparse préservé pour les gros
        # volumes), tous les algorithmes de ce registre (KMeans, DBSCAN,
        # hiérarchique) exigent une entrée dense — un usage exploratoire de
        # clustering porte typiquement sur des jeux de données plus modestes
        # que l'entraînement supervisé, le risque mémoire est bien moindre.
        X_processed = X_processed.toarray()

    specs = specs_for(config.algorithm_ids)
    if not specs:
        # Garde-fou défensif seulement — l'API valide déjà les ids en amont
        # (même pattern que `ml_training.py`), ne devrait jamais arriver.
        specs = specs_for(None)

    all_configs = [
        (spec, cand)
        for spec in specs
        for cand in spec.candidate_configs(X_processed.shape[0], config.seed)
    ]
    if not all_configs:
        raise TrainingAbortedError(
            "Aucune configuration de clustering n'est applicable à ce jeu de données "
            "(trop peu de lignes). Essayez avec un jeu de données plus grand."
        )

    progress_cb(f"Évaluation de {len(all_configs)} configurations", 15)

    evaluated: list[dict[str, Any]] = []
    total = len(all_configs)
    for i, (spec, cand) in enumerate(all_configs):
        params = dict(cand.params)
        if spec.id == "dbscan" and params.get("eps") is None:
            eps = resolve_dbscan_eps(X_processed, params["min_samples"])
            if eps <= 0:
                # Dégénéré (beaucoup de points strictement identiques après
                # préprocessing — DBSCAN rejette `eps<=0`) : candidat ignoré
                # plutôt qu'un plantage, les autres configurations du
                # catalogue restent évaluées normalement.
                progress_cb(
                    f"Évaluation de {len(all_configs)} configurations",
                    15 + int(70 * (i + 1) / total),
                )
                continue
            params["eps"] = eps

        estimator = spec.build_estimator(params, config.seed)
        labels = np.asarray(estimator.fit_predict(X_processed))
        metrics = _compute_cluster_metrics(X_processed, labels)

        evaluated.append(
            {
                "spec": spec,
                "cand_label": cand.label,
                "params": params,
                "labels": labels,
                "estimator": estimator,
                **metrics,
            }
        )
        progress_cb(
            f"Évaluation de {len(all_configs)} configurations",
            15 + int(70 * (i + 1) / total),
        )

    valid = [c for c in evaluated if c["silhouette"] is not None]
    if not valid:
        raise TrainingAbortedError(
            "Aucune configuration testée n'a produit de regroupement exploitable sur ce jeu "
            "de données (un seul groupe détecté, ou tous les points classés atypiques). "
            "Essayez avec d'autres variables, ou vérifiez qu'il existe une vraie structure "
            "de groupes dans vos données."
        )

    # Score de sélection = silhouette (borné [-1, 1], seule métrique des 3
    # dont l'échelle est directement interprétable ["proche de 1" = bon] —
    # Davies-Bouldin/Calinski-Harabasz servent à recouper, pas à classer,
    # même logique que le SHAP/permutation qui recoupent sans remplacer le
    # score de sélection principal côté supervisé).
    valid.sort(key=lambda c: c["silhouette"], reverse=True)

    progress_cb("Sélection du meilleur regroupement", 88)

    candidates_result: list[ClusterCandidateResult] = []
    for rank, c in enumerate(valid, start=1):
        candidates_result.append(
            ClusterCandidateResult(
                algorithm_id=c["spec"].id,
                label=c["cand_label"],
                family=c["spec"].family,
                params={k: v for k, v in c["params"].items()},
                n_clusters=c["n_clusters"],
                silhouette=c["silhouette"],
                davies_bouldin=c["davies_bouldin"],
                calinski_harabasz=c["calinski_harabasz"],
                noise_ratio=c["noise_ratio"],
                is_winner=(rank == 1),
                rank=rank,
            )
        )

    winner = valid[0]
    progress_cb("Calcul des profils de segments", 92)
    profiles = _build_cluster_profiles(X.reset_index(drop=True), winner["labels"])

    model_card = {
        "algorithm": winner["spec"].label,
        "algorithm_id": winner["spec"].id,
        "family": winner["spec"].family,
        "n_clusters": winner["n_clusters"],
        "n_samples": int(X.shape[0]),
        "silhouette": winner["silhouette"],
        "davies_bouldin": winner["davies_bouldin"],
        "calinski_harabasz": winner["calinski_harabasz"],
        "noise_ratio": winner["noise_ratio"],
        "n_candidates_evaluated": len(all_configs),
        "seed": config.seed,
    }

    progress_cb("Terminé", 100)

    return ClusteringResult(
        winning_algorithm_id=winner["spec"].id,
        winning_label=winner["cand_label"],
        all_candidates=candidates_result,
        labels=[int(v) for v in winner["labels"]],
        noise_count=int((winner["labels"] == -1).sum()),
        cluster_profiles=profiles,
        feature_columns=list(X.columns),
        model_card=model_card,
        pipeline_bundle={
            "preprocessor": preprocessor,
            "model": winner["estimator"],
            "algorithm_id": winner["spec"].id,
        },
    )
