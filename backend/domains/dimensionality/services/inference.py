"""Inférence réduction de dimension — projette une NOUVELLE observation à
partir du pipeline persisté par `workers/dimensionality_worker.py` (Lot 6B,
§F.2 — jusqu'ici, comme le clustering et la détection d'anomalies avant
elle, une projection entraînée ne pouvait jamais être réutilisée sur une
nouvelle observation, seulement consultée sur le jeu de données
d'entraînement).

Contrairement au clustering (toujours une approximation possible pour les
familles sans `.predict()` natif), t-SNE est un modèle TRANSDUCTIF au sens
strict : sklearn n'expose aucune méthode `.transform()` sur `TSNE`, il
n'existe littéralement AUCUNE façon de situer un nouveau point dans
l'embedding déjà calculé sans ré-entraîner sur l'ensemble des données (ce
que ce module ne fait jamais — un `services/inference.py` ne réentraîne
rien, principe déjà établi par les autres piliers). PCA et UMAP, eux,
supportent nativement `.transform()` — jamais d'approximation inventée ici,
uniquement une projection EXACTE ou un statut honnête `"unsupported"`."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class DimensionalityInferenceError(ValueError):
    """La donnée fournie ne peut pas être projetée (colonne manquante,
    valeur non convertible...)."""


def _build_input_frame(row: dict[str, Any], feature_columns: list[str]) -> pd.DataFrame:
    missing = [c for c in feature_columns if c not in row or row[c] in (None, "")]
    if missing:
        raise DimensionalityInferenceError(f"Valeur manquante pour : {', '.join(missing)}")
    ordered = {c: [row[c]] for c in feature_columns}
    df = pd.DataFrame(ordered)
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if not converted.isna().any():
            df[col] = converted
    return df


def _supports_new_point_projection(primary_model: Any) -> bool:
    return primary_model is not None and hasattr(primary_model, "transform")


def project_point(bundle: dict[str, Any], feature_columns: list[str], row: dict[str, Any]) -> dict[str, Any]:
    """Retourne `{"x": float | None, "y": float | None, "projection_method": str}`
    — `"exact"` (PCA/UMAP, `.transform()` natif) ou `"unsupported"` (t-SNE,
    ou bundle antérieur au correctif Lot 6B, §F.2 sans `primary_model`)."""
    primary_model = bundle.get("primary_model")
    if not _supports_new_point_projection(primary_model):
        return {"x": None, "y": None, "projection_method": "unsupported"}
    assert primary_model is not None  # narrowing pour mypy — déjà vérifié ci-dessus

    df = _build_input_frame(row, feature_columns)
    preprocessor = bundle["preprocessor"]
    try:
        X_proc = preprocessor.transform(df)
    except Exception as exc:
        raise DimensionalityInferenceError(
            "Valeurs incompatibles avec les variables utilisées lors de l'entraînement de cette projection"
        ) from exc
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()

    embedding = np.asarray(primary_model.transform(X_proc))
    x = float(embedding[0, 0])
    y = float(embedding[0, 1]) if embedding.shape[1] > 1 else 0.0
    return {"x": x, "y": y, "projection_method": "exact"}


def project_points_batch(bundle: dict[str, Any], feature_columns: list[str], df: pd.DataFrame) -> pd.DataFrame:
    """Version vectorisée de `project_point` pour un dataset ENTIER — même
    retour utilisateur que les autres piliers non supervisés ("l'entraînement
    échantillonne, comment projeter le reste ?"). Retourne une COPIE de `df`
    avec `x`/`y`/`projection_status` ajoutées."""
    result = df.copy()
    result["x"] = None
    result["y"] = None
    result["projection_status"] = "unscored"

    primary_model = bundle.get("primary_model")
    if not _supports_new_point_projection(primary_model):
        result["projection_status"] = "unsupported"
        return result
    assert primary_model is not None  # narrowing pour mypy — déjà vérifié ci-dessus

    missing_cols = [c for c in feature_columns if c not in df.columns]
    if missing_cols:
        result["projection_status"] = "missing_column"
        return result
    if len(result) == 0:
        return result

    X_raw = df[feature_columns].replace("", np.nan)
    valid_mask = ~X_raw.isna().any(axis=1)
    result.loc[~valid_mask, "projection_status"] = "missing_features"
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
        result.loc[valid_mask, "projection_status"] = "incompatible_values"
        return result
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()

    embedding = np.asarray(primary_model.transform(X_proc))
    valid_idx = result.index[valid_mask]
    result.loc[valid_idx, "x"] = embedding[:, 0]
    result.loc[valid_idx, "y"] = embedding[:, 1] if embedding.shape[1] > 1 else 0.0
    result.loc[valid_idx, "projection_status"] = "projected"
    return result
