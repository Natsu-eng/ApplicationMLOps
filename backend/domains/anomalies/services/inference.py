"""Inférence détection d'anomalies — note une NOUVELLE observation à partir
du pipeline persisté par `workers/anomaly_worker.py` (Lot 6B, §F.2 — jusqu'ici,
comme le clustering avant lui, une détection entraînée ne pouvait jamais être
réutilisée sur une nouvelle observation, seulement consultée sur le jeu de
données d'entraînement).

Isolation Forest est nativement inductif (`.predict()`/`.score_samples()`
fonctionnent directement sur une observation jamais vue). LOF ne l'est PAS en
mode d'entraînement (`novelty=False`, transductif — voir `registry.py`) : une
instance dédiée `novelty=True`, entraînée sur les MÊMES données, est persistée
dans le bundle spécifiquement pour ce cas (`lof_novelty`).

Le score de consensus d'une nouvelle observation doit rester sur la MÊME
échelle (rang percentile) que celui calculé à l'entraînement pour être
comparable — `services/engine.py::train_and_evaluate_anomalies` persiste donc
aussi les scores BRUTS d'entraînement (`scores_if_train`/`scores_lof_train`)
dans le bundle : le rang percentile d'une nouvelle observation se calcule
comme sa position dans CETTE distribution de référence (jamais recalculé sur
un seul point, un percentile n'a de sens que relatif à une population)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from domains.anomalies.services.engine import agreement_label


class AnomalyInferenceError(ValueError):
    """La donnée fournie ne peut pas être notée (colonne manquante, valeur
    non convertible...)."""


def _build_input_frame(row: dict[str, Any], feature_columns: list[str]) -> pd.DataFrame:
    missing = [c for c in feature_columns if c not in row or row[c] in (None, "")]
    if missing:
        raise AnomalyInferenceError(f"Valeur manquante pour : {', '.join(missing)}")
    ordered = {c: [row[c]] for c in feature_columns}
    df = pd.DataFrame(ordered)
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if not converted.isna().any():
            df[col] = converted
    return df


def _percentile_rank(train_scores: np.ndarray, new_score: float) -> float:
    """Position de `new_score` dans la distribution `train_scores` — mêmes
    conventions que `engine.py` (`pd.Series(-scores).rank(pct=True)`) : "plus
    bas = plus atypique", le rang le plus élevé doit revenir au point le plus
    atypique. Équivalent, pour un point HORS échantillon, à la fraction de
    points d'entraînement au moins aussi atypiques."""
    return float((train_scores >= new_score).mean())


def score_anomaly(bundle: dict[str, Any], feature_columns: list[str], row: dict[str, Any]) -> dict[str, Any]:
    """Retourne le score de consensus (et le détail par algorithme) d'une
    nouvelle observation — même structure que `AnomalyObservation`, sans les
    champs propres au classement (rang, déviations détaillées)."""
    df = _build_input_frame(row, feature_columns)
    preprocessor = bundle["preprocessor"]
    try:
        X_proc = preprocessor.transform(df)
    except Exception as exc:
        raise AnomalyInferenceError(
            "Valeurs incompatibles avec les variables utilisées lors de l'entraînement de cette détection"
        ) from exc
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()

    if_estimator = bundle["isolation_forest"]
    lof_novelty = bundle["lof_novelty"]
    scores_if_train = np.asarray(bundle["scores_if_train"])
    scores_lof_train = np.asarray(bundle["scores_lof_train"])

    score_if = float(if_estimator.score_samples(X_proc)[0])
    is_anomaly_if = bool(if_estimator.predict(X_proc)[0] == -1)

    score_lof = float(lof_novelty.score_samples(X_proc)[0])
    is_anomaly_lof = bool(lof_novelty.predict(X_proc)[0] == -1)

    rank_if = _percentile_rank(scores_if_train, score_if)
    rank_lof = _percentile_rank(scores_lof_train, score_lof)
    consensus = (rank_if + rank_lof) / 2.0

    return {
        "consensus_score": consensus,
        "score_isolation_forest": rank_if,
        "score_lof": rank_lof,
        "is_anomaly_isolation_forest": is_anomaly_if,
        "is_anomaly_lof": is_anomaly_lof,
        "is_anomaly_consensus": is_anomaly_if and is_anomaly_lof,
        "agreement": agreement_label(is_anomaly_if, is_anomaly_lof),
    }


def score_anomalies_batch(bundle: dict[str, Any], feature_columns: list[str], df: pd.DataFrame) -> pd.DataFrame:
    """Version vectorisée de `score_anomaly` pour un dataset ENTIER — même
    retour utilisateur que `clustering/services/inference.py::assign_clusters_batch`
    ("l'entraînement échantillonne, comment noter le reste ?"). Retourne une
    COPIE de `df` avec les colonnes de score ajoutées ; une ligne incomplète
    ou incompatible garde une trace honnête (`score_status`) plutôt que de
    faire échouer tout l'export."""
    result = df.copy()
    for col in (
        "consensus_score",
        "score_isolation_forest",
        "score_lof",
        "is_anomaly_isolation_forest",
        "is_anomaly_lof",
        "is_anomaly_consensus",
        "agreement",
    ):
        result[col] = None
    result["score_status"] = "unscored"

    missing_cols = [c for c in feature_columns if c not in df.columns]
    if missing_cols:
        result["score_status"] = "missing_column"
        return result
    if len(result) == 0:
        return result

    X_raw = df[feature_columns].replace("", np.nan)
    valid_mask = ~X_raw.isna().any(axis=1)
    result.loc[~valid_mask, "score_status"] = "missing_features"
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
        result.loc[valid_mask, "score_status"] = "incompatible_values"
        return result
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()

    if_estimator = bundle["isolation_forest"]
    lof_novelty = bundle["lof_novelty"]
    scores_if_train = np.asarray(bundle["scores_if_train"])
    scores_lof_train = np.asarray(bundle["scores_lof_train"])
    valid_idx = result.index[valid_mask]

    scores_if = if_estimator.score_samples(X_proc)
    is_anomaly_if = if_estimator.predict(X_proc) == -1
    scores_lof = lof_novelty.score_samples(X_proc)
    is_anomaly_lof = lof_novelty.predict(X_proc) == -1

    # Rang percentile vectorisé — équivalent à `_percentile_rank` appliqué
    # ligne à ligne (broadcast plutôt qu'une boucle Python, prohibitif à
    # grande échelle, même principe que `assign_clusters_batch`).
    rank_if = (scores_if_train[None, :] >= scores_if[:, None]).mean(axis=1)
    rank_lof = (scores_lof_train[None, :] >= scores_lof[:, None]).mean(axis=1)
    consensus = (rank_if + rank_lof) / 2.0
    is_anomaly_consensus = is_anomaly_if & is_anomaly_lof

    result.loc[valid_idx, "consensus_score"] = consensus
    result.loc[valid_idx, "score_isolation_forest"] = rank_if
    result.loc[valid_idx, "score_lof"] = rank_lof
    result.loc[valid_idx, "is_anomaly_isolation_forest"] = is_anomaly_if
    result.loc[valid_idx, "is_anomaly_lof"] = is_anomaly_lof
    result.loc[valid_idx, "is_anomaly_consensus"] = is_anomaly_consensus
    result.loc[valid_idx, "agreement"] = [
        agreement_label(bool(a), bool(b)) for a, b in zip(is_anomaly_if, is_anomaly_lof, strict=True)
    ]
    result.loc[valid_idx, "score_status"] = "scored"
    return result
