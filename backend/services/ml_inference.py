"""Inférence — charge un bundle de modèle entraîné (Lot 3) et produit une
prédiction sur une nouvelle observation fournie par l'utilisateur.

Logique pure (aucune dépendance HTTP), le bundle est le fichier joblib
produit par `services/ml_training.py::train_and_evaluate` — modèle +
preprocessor + (en régression) les régresseurs de quantile CQR.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd


class InferenceError(ValueError):
    """La donnée fournie ne peut pas être utilisée pour prédire (colonne
    manquante, valeur non convertible...)."""


def load_bundle(file_path: str) -> dict[str, Any]:
    if not Path(file_path).exists():
        raise InferenceError("Artefact du modèle introuvable sur le serveur")
    return joblib.load(file_path)


def _build_input_frame(row: dict[str, Any], feature_columns: list[str]) -> pd.DataFrame:
    missing = [c for c in feature_columns if c not in row or row[c] in (None, "")]
    if missing:
        raise InferenceError(f"Valeur manquante pour : {', '.join(missing)}")
    ordered = {c: [row[c]] for c in feature_columns}
    df = pd.DataFrame(ordered)
    # Conversion best-effort en numérique — laisse les colonnes non convertibles
    # telles quelles (catégorielles), le preprocessor gère les deux.
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if not converted.isna().any():
            df[col] = converted
    return df


def _cqr_interval(cqr: dict[str, Any], X_proc: np.ndarray, point_prediction: float) -> dict[str, float]:
    q_lo, q_hi = cqr["q_lo"], cqr["q_hi"]
    lo = float(q_lo.predict(X_proc)[0])
    hi = float(q_hi.predict(X_proc)[0])
    center = (lo + hi) / 2

    bounds = np.array(cqr["strata_bounds"])
    qhat = np.array(cqr["qhat_per_stratum"])
    stratum = int(np.clip(np.searchsorted(bounds, center, side="right") - 1, 0, len(qhat) - 1))

    lo_final = lo - qhat[stratum]
    hi_final = hi + qhat[stratum]
    if cqr.get("clip_negative"):
        lo_final = max(lo_final, 0.0)
    # L'intervalle doit toujours contenir la prédiction centrale du modèle
    # retenu, même si les régresseurs de quantile (indépendants) divergent
    # légèrement sur un point hors distribution.
    lo_final = min(lo_final, point_prediction)
    hi_final = max(hi_final, point_prediction)

    return {"low": float(lo_final), "high": float(hi_final), "confidence": float(cqr["target_coverage"])}


def predict_one(bundle: dict[str, Any], feature_columns: list[str], row: dict[str, Any]) -> dict[str, Any]:
    """Prédit sur une seule observation (dict colonne → valeur brute saisie
    par l'utilisateur). Retourne un résultat prêt à sérialiser en JSON."""
    df = _build_input_frame(row, feature_columns)

    try:
        X_proc = bundle["preprocessor"].transform(df)
    except Exception as exc:  # colonnes/types incompatibles avec le preprocessor entraîné
        raise InferenceError(f"Donnée incompatible avec le modèle : {exc}") from exc
    X_proc = np.asarray(X_proc.todense()) if hasattr(X_proc, "todense") else np.asarray(X_proc)

    model = bundle["model"]
    task_type = bundle["task_type"]

    if task_type == "classification":
        pred_index = int(model.predict(X_proc)[0])
        class_names = bundle.get("class_names") or []
        label = class_names[pred_index] if pred_index < len(class_names) else str(pred_index)
        result: dict[str, Any] = {"prediction": label}
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_proc)[0]
            result["probabilities"] = {
                (class_names[i] if i < len(class_names) else str(i)): float(p) for i, p in enumerate(proba)
            }
        return result

    point = float(model.predict(X_proc)[0])
    if bundle.get("cqr", {}).get("clip_negative"):
        point = max(point, 0.0)
    result = {"prediction": point}
    if bundle.get("cqr"):
        # Le CQR a son propre préprocesseur (fit uniquement sur sa portion
        # fit, distincte de celle du modèle principal — voir ml_training.py,
        # Lot A) : on ne peut pas réutiliser X_proc, calculé avec le
        # préprocesseur du modèle principal, pour les régresseurs de quantile.
        cqr_preprocessor = bundle["cqr"]["preprocessor"]
        X_proc_cqr = cqr_preprocessor.transform(df)
        X_proc_cqr = np.asarray(X_proc_cqr.todense()) if hasattr(X_proc_cqr, "todense") else np.asarray(X_proc_cqr)
        result["interval"] = _cqr_interval(bundle["cqr"], X_proc_cqr, point)
    return result
