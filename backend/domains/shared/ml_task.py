"""Détection du type de tâche ML à partir de la colonne cible — logique pure.

Portage simplifié de `src/data/data_analysis.py::get_target_and_task` (app
Streamlit historique, voir le diagnostic de migration section C).
"""
from __future__ import annotations

import pandas as pd

MAX_CLASSES_FOR_CLASSIFICATION = 20
LOW_CARDINALITY_RATIO = 0.05


def detect_task_type(series: pd.Series) -> str:
    """Retourne "classification" ou "regression" à partir de la colonne cible.

    Le non-supervisé (clustering, pas de cible) est hors périmètre de ce
    module — voir la page Training du frontend, qui ne propose cette
    détection qu'après sélection d'une colonne cible.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return "classification"
    n_unique = series.nunique()
    n_total = max(len(series), 1)
    if n_unique <= MAX_CLASSES_FOR_CLASSIFICATION or (n_unique / n_total) < LOW_CARDINALITY_RATIO:
        return "classification"
    return "regression"
