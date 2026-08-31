"""Moteur de détection d'anomalies — Lot 14 (ML non supervisé).

Module séparé de `clustering_training.py`/`dimensionality_training.py` —
même raisonnement que les deux lots précédents du pilier non supervisé :
aucune notion de cible, aucun code partagé au-delà de
`services/ml_preprocessing.py::build_preprocessor` et
`services/stats_utils.py::sample_if_large`, déjà génériques.

Isolation Forest et LOF tournent TOUJOURS ensemble (jamais un seul essai
lancé à l'aveugle) — sans vérité terrain, il n'y a pas de "gagnant" à élire
comme au clustering. À la place, un score de CONSENSUS continu (moyenne des
rangs percentiles de chaque algorithme — les scores bruts d'Isolation Forest
et de LOF ne sont pas sur la même échelle, seul le rang est comparable) et
un drapeau d'accord catégoriel (`agreement`) construit à partir des deux
résultats déjà calculés — jamais un nombre inventé."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from domains.anomalies.services.registry import ANOMALY_REGISTRY, build_lof_novelty_from
from domains.shared.ml_preprocessing import TrainingAbortedError, build_preprocessor
from domains.shared.stats_utils import sample_if_large

ProgressCallback = Callable[[str, int], None]

# Isolation Forest/LOF restent efficaces à cette échelle (contrairement à
# t-SNE, quasi-quadratique) — cohérent avec les autres caps du projet
# (MAX_ROWS_FOR_STATS, services/stats_utils.py).
MAX_ROWS_FOR_ANOMALY = 20_000
MIN_ROWS_FOR_ANOMALY = 10
DEFAULT_TOP_N = 50
MAX_TOP_N = 200
_ROW_INDEX_COLUMN = "__an_row_index__"
_RARE_CATEGORY_THRESHOLD = 0.05  # même seuil d'esprit que RARE_CATEGORY_FREQUENCY_THRESHOLD (ml_preprocessing.py)


@dataclass
class AnomalyConfig:
    top_n: int = DEFAULT_TOP_N
    seed: int = 42
    # "auto" (défaut, formule du papier original) ou une fraction explicite
    # de la population attendue comme anomalie — jusqu'ici codé en dur dans
    # `anomaly_registry.py`, jamais exposé (Lot 6B, §F.2). Validé côté API
    # (0, 0.5], voir `api/routers/anomalies.py::AnomalyJobCreate`.
    contamination: str | float = "auto"


@dataclass
class AnomalyObservation:
    row_index: int
    rank: int
    consensus_score: float
    score_isolation_forest: float
    score_lof: float
    is_anomaly_isolation_forest: bool
    is_anomaly_lof: bool
    agreement: str  # "both" | "isolation_forest_only" | "lof_only" | "none"
    numeric_deviations: dict[str, dict[str, float]]  # {col: {"value":, "z_score":}}
    categorical_flags: dict[str, dict[str, Any]]  # {col: {"value":, "population_pct":}} — colonnes rares seulement


@dataclass
class AnomalyResult:
    n_samples_total: int
    n_samples_used: int
    sampled: bool
    n_anomalies_isolation_forest: int
    n_anomalies_lof: int
    n_anomalies_consensus: int
    anomaly_rate_isolation_forest: float
    anomaly_rate_lof: float
    anomaly_rate_consensus: float
    score_histogram: dict[str, Any]
    top_observations: list[AnomalyObservation]
    feature_columns: list[str]
    model_card: dict[str, Any]
    pipeline_bundle: dict[str, Any] = field(repr=False)


def agreement_label(is_if: bool, is_lof: bool) -> str:
    if is_if and is_lof:
        return "both"
    if is_if:
        return "isolation_forest_only"
    if is_lof:
        return "lof_only"
    return "none"


def _build_numeric_deviations(row: pd.Series, global_mean: pd.Series, global_std: pd.Series, numeric_cols: list[str]) -> dict[str, dict[str, float]]:
    z_scores: dict[str, float] = {}
    for col in numeric_cols:
        std = global_std.get(col, 0.0)
        value = float(row[col]) if pd.notna(row[col]) else 0.0
        z = (value - global_mean.get(col, 0.0)) / std if std and not np.isnan(std) and std != 0 else 0.0
        z_scores[col] = float(z)
    top_cols = sorted(z_scores, key=lambda c: abs(z_scores[c]), reverse=True)[:5]
    return {col: {"value": float(row[col]) if pd.notna(row[col]) else 0.0, "z_score": z_scores[col]} for col in top_cols}


def _build_categorical_flags(row: pd.Series, rarity_by_col: dict[str, dict[str, float]], categorical_cols: list[str]) -> dict[str, dict[str, Any]]:
    flags: dict[str, dict[str, Any]] = {}
    for col in categorical_cols:
        value = row[col]
        if pd.isna(value):
            continue
        pct = rarity_by_col.get(col, {}).get(str(value))
        if pct is not None and pct < _RARE_CATEGORY_THRESHOLD:
            flags[col] = {"value": str(value), "population_pct": float(pct * 100)}
    return flags


def train_and_evaluate_anomalies(
    X: pd.DataFrame,
    config: AnomalyConfig,
    progress_cb: ProgressCallback,
) -> AnomalyResult:
    """Point d'entrée principal — Isolation Forest et LOF évalués ensemble,
    jamais un seul isolément. Score de consensus = moyenne des rangs
    percentiles (voir docstring du module)."""
    progress_cb("Préparation des données", 5)
    n_samples_total = int(len(X))
    if n_samples_total < MIN_ROWS_FOR_ANOMALY:
        raise TrainingAbortedError(
            f"Il faut au moins {MIN_ROWS_FOR_ANOMALY} observations pour une détection d'anomalies "
            f"exploitable (dataset actuel : {n_samples_total})."
        )

    X_indexed = X.reset_index(drop=True).copy()
    X_indexed[_ROW_INDEX_COLUMN] = np.arange(n_samples_total)
    sampled_flag = n_samples_total > MAX_ROWS_FOR_ANOMALY
    X_sampled = sample_if_large(X_indexed, MAX_ROWS_FOR_ANOMALY, config.seed)
    row_indices = X_sampled[_ROW_INDEX_COLUMN].to_numpy()
    X_used = X_sampled.drop(columns=[_ROW_INDEX_COLUMN]).reset_index(drop=True)
    n_used = int(len(X_used))

    progress_cb("Préparation des données", 15)
    preprocessor = build_preprocessor(X_used)
    X_processed = preprocessor.fit_transform(X_used)
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    if X_processed.shape[1] < 1:
        raise TrainingAbortedError("Aucune variable exploitable après préparation des données.")

    specs = {s.id: s for s in ANOMALY_REGISTRY}

    progress_cb("Isolation Forest", 25)
    if_estimator = specs["isolation_forest"].build_estimator(n_used, config.seed, config.contamination)
    if_estimator.fit(X_processed)
    labels_if = np.asarray(if_estimator.predict(X_processed))
    scores_if = np.asarray(if_estimator.score_samples(X_processed))  # plus bas = plus atypique

    progress_cb("Local Outlier Factor", 55)
    lof_estimator = specs["lof"].build_estimator(n_used, config.seed, config.contamination)
    labels_lof = np.asarray(lof_estimator.fit_predict(X_processed))
    scores_lof = np.asarray(lof_estimator.negative_outlier_factor_)  # plus bas = plus atypique
    # Instance dédiée à la notation de NOUVELLES observations (Lot 6B, §F.2
    # — voir `registry.py::build_lof_novelty_from` pour le raisonnement :
    # `lof_estimator` ci-dessus reste `novelty=False`, jamais réutilisée
    # pour prédire hors du jeu d'entraînement).
    lof_novelty_estimator = build_lof_novelty_from(lof_estimator, X_processed)

    progress_cb("Calcul du consensus", 75)
    # Rangs percentiles (pas les scores bruts, non comparables entre IF et
    # LOF) — négation car "plus bas = plus atypique" pour les deux : le
    # point le plus atypique doit obtenir le rang le plus ÉLEVÉ.
    rank_if = pd.Series(-scores_if).rank(pct=True).to_numpy()
    rank_lof = pd.Series(-scores_lof).rank(pct=True).to_numpy()
    consensus = (rank_if + rank_lof) / 2.0

    is_anomaly_if = labels_if == -1
    is_anomaly_lof = labels_lof == -1
    is_anomaly_consensus = is_anomaly_if & is_anomaly_lof

    n_if = int(is_anomaly_if.sum())
    n_lof = int(is_anomaly_lof.sum())
    n_consensus = int(is_anomaly_consensus.sum())

    counts, edges = np.histogram(consensus, bins=20)
    score_histogram = {"bin_edges": [float(e) for e in edges], "counts": [int(c) for c in counts]}

    progress_cb("Classement des observations atypiques", 88)
    top_n = max(1, min(config.top_n, MAX_TOP_N, n_used))
    order = np.argsort(-consensus)[:top_n]

    numeric_cols = X_used.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in X_used.columns if c not in numeric_cols]
    global_mean = X_used[numeric_cols].mean() if numeric_cols else pd.Series(dtype=float)
    global_std = X_used[numeric_cols].std() if numeric_cols else pd.Series(dtype=float)
    rarity_by_col: dict[str, dict[str, float]] = {}
    for col in categorical_cols:
        counts_by_value = X_used[col].value_counts(normalize=True)
        rarity_by_col[col] = {str(idx): float(pct) for idx, pct in counts_by_value.items()}

    top_observations: list[AnomalyObservation] = []
    for rank, i in enumerate(order, start=1):
        row = X_used.iloc[i]
        top_observations.append(
            AnomalyObservation(
                row_index=int(row_indices[i]),
                rank=rank,
                consensus_score=float(consensus[i]),
                score_isolation_forest=float(rank_if[i]),
                score_lof=float(rank_lof[i]),
                is_anomaly_isolation_forest=bool(is_anomaly_if[i]),
                is_anomaly_lof=bool(is_anomaly_lof[i]),
                agreement=agreement_label(bool(is_anomaly_if[i]), bool(is_anomaly_lof[i])),
                numeric_deviations=_build_numeric_deviations(row, global_mean, global_std, numeric_cols),
                categorical_flags=_build_categorical_flags(row, rarity_by_col, categorical_cols),
            )
        )

    progress_cb("Terminé", 100)

    model_card = {
        "n_samples_total": n_samples_total,
        "n_samples_used": n_used,
        "sampled": sampled_flag,
        "seed": config.seed,
        "contamination": config.contamination,
        "n_anomalies_isolation_forest": n_if,
        "n_anomalies_lof": n_lof,
        "n_anomalies_consensus": n_consensus,
        "anomaly_rate_isolation_forest": n_if / n_used if n_used else 0.0,
        "anomaly_rate_lof": n_lof / n_used if n_used else 0.0,
        "anomaly_rate_consensus": n_consensus / n_used if n_used else 0.0,
        "top_n": top_n,
    }

    return AnomalyResult(
        n_samples_total=n_samples_total,
        n_samples_used=n_used,
        sampled=sampled_flag,
        n_anomalies_isolation_forest=n_if,
        n_anomalies_lof=n_lof,
        n_anomalies_consensus=n_consensus,
        anomaly_rate_isolation_forest=model_card["anomaly_rate_isolation_forest"],
        anomaly_rate_lof=model_card["anomaly_rate_lof"],
        anomaly_rate_consensus=model_card["anomaly_rate_consensus"],
        score_histogram=score_histogram,
        top_observations=top_observations,
        feature_columns=list(X.columns),
        model_card=model_card,
        pipeline_bundle={
            "preprocessor": preprocessor,
            "isolation_forest": if_estimator,
            "lof": lof_estimator,
            # Notation de nouvelles observations (Lot 6B, §F.2 — voir
            # `services/inference.py`) : instance LOF dédiée + scores BRUTS
            # d'entraînement des deux algorithmes, nécessaires pour situer le
            # score d'une nouvelle observation au même rang percentile que le
            # consensus calculé ci-dessus (jamais recalculé à partir de rien —
            # un rang percentile n'a de sens que relatif à une distribution
            # de référence, ici celle du jeu d'entraînement).
            "lof_novelty": lof_novelty_estimator,
            "scores_if_train": scores_if,
            "scores_lof_train": scores_lof,
        },
    )
