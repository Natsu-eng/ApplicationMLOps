"""Suggestion de colonne cible — Lot 7, §J.1 : "l'utilisateur doit deviner
quoi prédire alors que le backend dispose déjà de la cardinalité, du type et
des corrélations de chaque colonne."

Un score de plausibilité + les raisons concrètes qui le justifient — jamais
un choix fait à la place de l'utilisateur (voir `TargetSuggestionOut.reasons`,
toujours peuplé de faits réellement calculés sur CE dataset, jamais un texte
générique). Réutilise les mêmes seuils que `services/data_quality.py`
(cardinalité excessive) et `services/ml_task.py` (cardinalité faible ->
classification) plutôt que d'en inventer de nouveaux."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from services.data_quality import CARDINALITY_ABSOLUTE_THRESHOLD, CARDINALITY_RATIO_THRESHOLD
from services.ml_task import MAX_CLASSES_FOR_CLASSIFICATION

# Mots-clés fréquents dans le nom d'une colonne cible — indice, jamais une
# certitude à lui seul (voir le score combiné ci-dessous). Français et
# anglais, l'un ou l'autre selon l'origine du dataset importé.
_TARGET_NAME_HINTS = [
    "target", "cible", "label", "libelle", "class", "classe", "outcome",
    "resultat", "résultat", "result", "price", "prix", "score", "status",
    "statut", "defaut", "défaut", "defect", "churn", "fraud", "fraude",
    "risk", "risque", "conforme", "success", "reussite", "réussite", "y",
]

# Une cible avec plus de 20 % de valeurs manquantes n'est de toute façon pas
# exploitable telle quelle (voir data_quality.py, garde-fou valeurs
# manquantes) — jamais suggérée en tête de liste.
MAX_MISSING_RATIO_FOR_TARGET = 0.2

DEFAULT_MAX_SUGGESTIONS = 3


@dataclass
class TargetSuggestion:
    column: str
    score: float
    reasons: list[str] = field(default_factory=list)


def suggest_target_columns(df: pd.DataFrame, max_suggestions: int = DEFAULT_MAX_SUGGESTIONS) -> list[TargetSuggestion]:
    n_total = len(df)
    candidates: list[TargetSuggestion] = []

    for col in df.columns:
        series = df[col]
        n_unique = int(series.nunique(dropna=True))

        # Jamais suggérée : constante (aucune information) ou quasi-identifiant
        # (même seuil que le garde-fou "cardinalité excessive").
        if n_unique <= 1:
            continue
        if n_total > 0 and (n_unique / n_total) > CARDINALITY_RATIO_THRESHOLD and n_unique > CARDINALITY_ABSOLUTE_THRESHOLD:
            continue

        missing_ratio = float(series.isna().mean()) if n_total else 0.0
        if missing_ratio > MAX_MISSING_RATIO_FOR_TARGET:
            continue

        score = 0.0
        reasons: list[str] = []

        name_lower = str(col).lower()
        if any(hint in name_lower for hint in _TARGET_NAME_HINTS):
            score += 3.0
            reasons.append(f"le nom de la colonne « {col} » évoque une cible")

        if pd.api.types.is_numeric_dtype(series):
            if n_unique <= MAX_CLASSES_FOR_CLASSIFICATION:
                score += 1.0
                reasons.append(f"{n_unique} valeurs distinctes — cohérent avec une classification")
            elif n_total and (n_unique / n_total) > 0.3:
                score += 0.5
                reasons.append("valeurs numériques continues — cohérent avec une régression")
        elif 2 <= n_unique <= MAX_CLASSES_FOR_CLASSIFICATION:
            score += 1.0
            reasons.append(f"{n_unique} catégories distinctes — cohérent avec une classification")

        if len(df.columns) > 0 and col == df.columns[-1]:
            score += 0.5
            reasons.append("dernière colonne du jeu de données — emplacement fréquent d'une cible")

        if score > 0:
            candidates.append(TargetSuggestion(column=str(col), score=score, reasons=reasons))

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:max_suggestions]
