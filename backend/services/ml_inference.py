"""Inférence — charge un bundle de modèle entraîné (Lot 3) et produit une
prédiction sur une nouvelle observation fournie par l'utilisateur.

Logique pure (aucune dépendance HTTP), le bundle est le fichier joblib
produit par `services/ml_training.py::train_and_evaluate` — modèle +
preprocessor + (en régression) les régresseurs de quantile CQR + (Lot
Explicabilité locale) de quoi construire l'explainer SHAP à la demande.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from services.feature_engineering import FeatureEngineeringSpecError, apply_upstream_feature_engineering
from services.ml_explainability import build_explainer, normalize_base_value, select_class_matrix, shap_values_per_class
from services.model_bundle import InferenceError, load_bundle

logger = logging.getLogger("datalab.ml_inference")

# Contributions individuelles affichées dans le waterfall local — le reste
# est agrégé sous "Autres" pour rester lisible même sur un dataset à
# beaucoup de variables (colonnes one-hot comprises, qui gonflent
# `feature_names` bien au-delà du nombre de colonnes saisies par
# l'utilisateur).
LOCAL_EXPLAIN_TOP_FEATURES = 10

# Réexportés pour compatibilité — `InferenceError`/`load_bundle` vivent
# désormais dans `services/model_bundle.py` (Lot 8, §Phase 0 : chargement
# générique, sans raison d'être un import training-domain pour clustering).
__all__ = ["InferenceError", "load_bundle", "explain_one", "predict_one"]


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


def explain_one(bundle: dict[str, Any], X_proc_row: np.ndarray, class_index: Optional[int]) -> dict[str, Any]:
    """Explication locale (waterfall) — pourquoi CETTE prédiction précise,
    contribution signée par variable, plutôt que la seule importance
    moyenne déjà disponible sur la fiche modèle (explicabilité globale, Lot
    5/Explicabilité globale). Réutilise le MÊME routage d'explainer par
    famille que l'entraînement (`services/ml_explainability.py`) — jamais un
    second calcul de SHAP qui pourrait diverger dans sa forme.

    `class_index` : classe prédite (classification, voir `predict_one`) à
    expliquer — `None` en régression. Sans effet si l'explainer ne renvoie
    qu'une seule sortie (selon la version de SHAP, voir
    `ml_explainability.select_class_matrix`).

    Dégrade proprement (jamais d'exception propagée à l'appelant) : un job
    entraîné avant ce lot n'a pas `explainer_kind` dans son bundle — statut
    `"degraded"` avec message clair plutôt qu'un crash de la prédiction
    elle-même (l'explication est un COMPLÉMENT, jamais une condition pour
    obtenir la prédiction). Contributions bornées à
    `LOCAL_EXPLAIN_TOP_FEATURES`, le reste agrégé sous "Autres" pour rester
    lisible même avec beaucoup de colonnes (one-hot compris)."""
    explainer_kind = bundle.get("explainer_kind")
    feature_names: list[str] = bundle.get("feature_names") or []
    if explainer_kind is None:
        return {
            "status": "degraded",
            "message": "Ce modèle a été entraîné avant l'explicabilité locale — réentraînez-le pour l'obtenir.",
            "base_value": None,
            "contributions": [],
            "other_contribution": None,
        }
    try:
        explainer = build_explainer(explainer_kind, bundle["model"], bundle.get("local_explain_background"))
        shap_values = explainer.shap_values(X_proc_row)
        class_matrices = shap_values_per_class(shap_values)

        # Cas binaire non listé (bug réel trouvé en test, vérifié empiriquement
        # sur un modèle réel) : certaines versions de SHAP/backends d'arbre
        # renvoient, pour une classification à 2 classes, UN SEUL tableau —
        # celui de `class_names[1]` (la "classe positive"), jamais celui de
        # `class_names[0]`, quelle que soit la classe réellement prédite pour
        # cette observation. Sans correction, expliquer une observation
        # prédite classe 0 afficherait des contributions au signe inversé
        # (une variable qui pousse VERS la classe prédite s'afficherait comme
        # la poussant CONTRE). Les classes ≥ 3 (liste de matrices, une par
        # classe) n'ont pas cette ambiguïté — chaque entrée correspond déjà
        # directement à son índice de classe (même hypothèse que le beeswarm
        # global, `ml_training.py`, en production depuis le Lot Explicabilité
        # globale sans anomalie rapportée).
        class_names = bundle.get("class_names") or []
        sign = -1.0 if (
            not isinstance(class_matrices, list) and class_index == 0 and len(class_names) == 2
        ) else 1.0

        row_values = sign * np.asarray(select_class_matrix(class_matrices, class_index))[0]
        base_value = sign * normalize_base_value(explainer.expected_value, class_index)

        order = np.argsort(-np.abs(row_values))
        top_idx = order[:LOCAL_EXPLAIN_TOP_FEATURES]
        rest_idx = order[LOCAL_EXPLAIN_TOP_FEATURES:]
        row_dense = X_proc_row[0]

        contributions = [
            {
                "feature": feature_names[i] if i < len(feature_names) else f"variable_{i}",
                "value": float(row_dense[i]),
                "contribution": float(row_values[i]),
            }
            for i in top_idx
        ]
        other_contribution = float(np.sum(row_values[rest_idx])) if len(rest_idx) else 0.0

        return {
            "status": "ok",
            "message": None,
            "base_value": base_value,
            "contributions": contributions,
            "other_contribution": other_contribution,
        }
    except Exception as exc:  # dernier filet — l'explication ne doit jamais faire échouer la prédiction
        logger.warning("[Inference] Explication locale dégradée (explainer=%s) : %s", explainer_kind, exc)
        return {
            "status": "degraded",
            "message": "L'explication détaillée n'a pas pu être calculée pour cette prédiction.",
            "base_value": None,
            "contributions": [],
            "other_contribution": None,
        }


def predict_one(bundle: dict[str, Any], feature_columns: list[str], row: dict[str, Any]) -> dict[str, Any]:
    """Prédit sur une seule observation (dict colonne → valeur brute saisie
    par l'utilisateur). Retourne un résultat prêt à sérialiser en JSON.

    `feature_columns` désigne les colonnes SAISIES par l'utilisateur (ex.
    "date"), pas les colonnes dérivées vues par le préprocesseur (ex.
    "date_annee") — voir `workers/training_worker.py`, précision 3 du Lot 4c.
    Si le bundle porte une spec de feature engineering (Lot 4c), elle est
    rejouée ici via la MÊME fonction pure que celle utilisée à l'entraînement
    (`apply_upstream_feature_engineering`), pour produire des colonnes
    identiques dans les deux contextes."""
    df = _build_input_frame(row, feature_columns)

    feature_engineering_spec = bundle.get("feature_engineering_spec")
    if feature_engineering_spec:
        try:
            df, _ = apply_upstream_feature_engineering(df, feature_columns, feature_engineering_spec)
        except FeatureEngineeringSpecError as exc:
            # Refus explicite (précision 2 du cadrage) plutôt qu'une prédiction
            # calculée sur des colonnes potentiellement incorrectes.
            raise InferenceError(f"Ingénierie de variables du modèle non rejouable : {exc}") from exc

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
        # Explicabilité locale (Lot Explicabilité locale) — explique la
        # classe PRÉDITE, celle qui intéresse réellement l'utilisateur pour
        # cette observation précise, pas les K classes à la fois.
        result["explanation"] = explain_one(bundle, X_proc, pred_index)
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
    result["explanation"] = explain_one(bundle, X_proc, None)
    return result
