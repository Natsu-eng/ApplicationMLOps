"""Router exploration/analyse d'un dataset — EDA, qualité, suggestions.

Toutes les routes sont en LECTURE SEULE et portent sur un dataset déjà
importé : statistiques et corrélations (`/eda`), histogramme d'une
colonne, contrôle qualité avant entraînement, suggestions de cible et de
feature engineering, croisement variable×cible. Aucune n'écrit quoi que
ce soit — elles rechargent le fichier du dataset et calculent.

Extrait de `router.py` lors du découpage (734 lignes) : ce fichier
mélangeait deux responsabilités sans rapport — la gestion du cycle de vie
d'un dataset (import, liste, aperçu, usage, suppression) et son
exploration statistique. Cette seconde famille, la plus volumineuse,
part ici ; `router.py` garde le CRUD.

Même préfixe `/datasets` et même tag OpenAPI que `router.py` : le
découpage est interne, aucune URL ne change. `_get_org_dataset` (contrôle
d'appartenance à l'organisation) reste défini dans `router.py` et est
importé ici — import à sens unique, comme partout ailleurs dans le projet.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.core.database import get_db
from api.core.error_codes import ErrorCode
from api.core.models import User
from domains.auth.router import get_current_user
from domains.datasets.router import _get_org_dataset
from domains.datasets.services.target_suggestion import suggest_target_columns
from domains.shared.data_quality import analyze_data_quality
from domains.shared.dataset_eda import (
    compute_categorical_correlation_matrix,
    compute_column_stats,
    compute_correlation_matrix,
    compute_feature_by_target,
    compute_histogram,
    compute_missing_summary,
    compute_outlier_boxplots,
    compute_top_correlated_pairs,
)
from domains.shared.dataset_io import DatasetParsingError, read_dataset_dataframe
from domains.shared.feature_engineering import suggest_feature_engineering

router = APIRouter(prefix="/datasets", tags=["datasets"])


# ── Schémas ──────────────────────────────────────────────────────────────────

class ColumnStat(BaseModel):
    name: str
    dtype: str
    kind: str  # "numeric" | "categorical"
    missing_count: int
    missing_pct: float
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    median: Optional[float] = None
    n_unique: Optional[int] = None
    top_values: Optional[List[dict]] = None


class MissingSummaryEntry(BaseModel):
    column: str
    missing_count: int
    missing_pct: float


class CorrelationMatrix(BaseModel):
    columns: List[str]
    matrix: List[List[Optional[float]]]


class HistogramResponse(BaseModel):
    kind: str  # "numeric" | "categorical"
    bin_edges: Optional[List[float]] = None
    counts: List[int]
    categories: Optional[List[str]] = None


class BoxplotStat(BaseModel):
    column: str
    min: Optional[float] = None
    q1: Optional[float] = None
    median: Optional[float] = None
    q3: Optional[float] = None
    max: Optional[float] = None
    outliers: List[float] = []
    n: int


class ScatterPoint(BaseModel):
    x: Optional[float] = None
    y: Optional[float] = None


class ScatterPair(BaseModel):
    x_column: str
    y_column: str
    correlation: Optional[float] = None
    points: List[ScatterPoint]


class FeatureByTargetGroup(BaseModel):
    class_name: str
    min: Optional[float] = None
    q1: Optional[float] = None
    median: Optional[float] = None
    q3: Optional[float] = None
    max: Optional[float] = None
    outliers: List[float] = []
    n: int


class FeatureByTargetResponse(BaseModel):
    feature: str
    target: str
    groups: List[FeatureByTargetGroup]


class EdaResponse(BaseModel):
    row_count: int
    column_stats: List[ColumnStat]
    missing_summary: List[MissingSummaryEntry]
    correlation_matrix: CorrelationMatrix
    categorical_correlation_matrix: CorrelationMatrix
    outlier_summary: List[BoxplotStat]
    top_correlated_pairs: List[ScatterPair]
    target_distribution: Optional[HistogramResponse] = None


class DataWarning(BaseModel):
    level: str  # "info" | "attention" | "critique"
    code: str
    title: str
    explanation: str
    # Question métier à se poser avant de décider (Lot UI — refonte
    # visuelle, Données et qualité) — distincte de `explanation` (le
    # pourquoi) et `action` (la recommandation), SPEC-UI.md §7 règle n°3.
    question: str
    action: str
    columns: List[str] = []
    details: Optional[dict] = None


class DataQualityResponse(BaseModel):
    warnings: List[DataWarning]


class TargetSuggestionOut(BaseModel):
    column: str
    score: float
    reasons: List[str]


class TargetSuggestionsResponse(BaseModel):
    suggestions: List[TargetSuggestionOut]


class FeatureEngineeringSuggestion(BaseModel):
    code: str
    title: str
    explanation: str
    action: str
    columns: List[str] = []
    based_on_warning: Optional[str] = None
    transformation: dict
    choice: Optional[dict] = None


class FeatureEngineeringSuggestionsResponse(BaseModel):
    suggestions: List[FeatureEngineeringSuggestion]


# ── Aides ────────────────────────────────────────────────────────────────────

def _excluded_columns_from_allowlist(df: pd.DataFrame, feature_columns: Optional[str]) -> Optional[set[str]]:
    """Convertit le paramètre `feature_columns` (liste blanche CSV, étape 1
    du formulaire) en ensemble de colonnes à EXCLURE — c'est ce dernier
    format qu'attendent `analyze_data_quality`/`suggest_feature_engineering`
    (même mécanisme que l'exclusion cible/groupe déjà en place). `None`
    (paramètre absent) : aucune restriction, comportement historique
    inchangé — appelants qui ne connaissent pas encore de sélection de
    variables (ex. `GET /datasets/{id}/eda`)."""
    if feature_columns is None:
        return None
    included = {c.strip() for c in feature_columns.split(",") if c.strip()}
    return {c for c in df.columns if c not in included}


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/{dataset_id}/eda", response_model=EdaResponse)
def get_dataset_eda(
    dataset_id: int,
    target_column: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Exploration de données (EDA) — stats par colonne, corrélations
    (numériques ET catégorielles), valeurs manquantes, outliers, paires de
    features corrélées, et (Lot B) distribution de la cible si
    `target_column` est fourni. Calculé à la demande (résultat jamais
    stocké — seule la lecture du fichier source est mise en cache, voir
    `read_dataset_dataframe`, Lot 4/I4).

    `target_column` est optionnel et rétrocompatible : sans lui, l'EDA
    fonctionne comme avant (exploration autonome d'un dataset, sans contexte
    d'entraînement) — seul `target_distribution` reste absent."""
    dataset = _get_org_dataset(dataset_id, current_user, db)
    if dataset.status != "ready":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": ErrorCode.DATASET_NON_PRET, "message": "Ce dataset n'a pas pu être analysé"},
        )
    try:
        df = read_dataset_dataframe(Path(dataset.file_path), Path(dataset.file_path).suffix)
        target_distribution = compute_histogram(df, target_column) if target_column else None
    except DatasetParsingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": ErrorCode.DATASET_LECTURE_ECHEC, "message": str(exc)},
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": ErrorCode.COLONNE_INTROUVABLE, "message": str(exc)},
        )
    return EdaResponse(
        row_count=len(df),
        column_stats=[ColumnStat(**c) for c in compute_column_stats(df)],
        missing_summary=[MissingSummaryEntry(**m) for m in compute_missing_summary(df)],
        correlation_matrix=CorrelationMatrix(**compute_correlation_matrix(df)),
        categorical_correlation_matrix=CorrelationMatrix(**compute_categorical_correlation_matrix(df)),
        outlier_summary=[BoxplotStat(**b) for b in compute_outlier_boxplots(df)],
        top_correlated_pairs=[ScatterPair(**p) for p in compute_top_correlated_pairs(df)],
        target_distribution=HistogramResponse(**target_distribution) if target_distribution else None,
    )


@router.get("/{dataset_id}/histogram", response_model=HistogramResponse)
def get_dataset_histogram(
    dataset_id: int,
    column: str,
    bins: int = 20,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Histogramme d'une seule colonne, à la demande — évite de calculer les
    histogrammes de toutes les colonnes d'un coup sur un dataset large."""
    dataset = _get_org_dataset(dataset_id, current_user, db)
    if dataset.status != "ready":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": ErrorCode.DATASET_NON_PRET, "message": "Ce dataset n'a pas pu être analysé"},
        )
    try:
        df = read_dataset_dataframe(Path(dataset.file_path), Path(dataset.file_path).suffix)
        histogram = compute_histogram(df, column, bins=max(5, min(bins, 100)))
    except DatasetParsingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": ErrorCode.DATASET_LECTURE_ECHEC, "message": str(exc)},
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": ErrorCode.COLONNE_INTROUVABLE, "message": str(exc)},
        )
    return HistogramResponse(**histogram)


@router.get("/{dataset_id}/quality-check", response_model=DataQualityResponse)
def get_dataset_quality_check(
    dataset_id: int,
    target_column: Optional[str] = None,
    group_column: Optional[str] = None,
    feature_columns: Optional[str] = Query(
        None,
        description=(
            "Colonnes CSV déjà retenues à l'étape 1 du formulaire — si fourni, seules ces colonnes "
            "peuvent générer une alerte, jamais une colonne déjà exclue. Absent = comportement "
            "historique (toutes les colonnes du dataset analysées)."
        ),
    ),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Garde-fous de données (Lot B) — avertissements actionnables sur le
    dataset par rapport à la colonne cible choisie (fuite, déséquilibre,
    cardinalité...). Calculé à la demande, au moment du choix dataset+cible,
    avant le lancement de l'entraînement — jamais bloquant : on informe,
    on n'empêche pas.

    `target_column` optionnel (Lot Nettoyage guidé des variables) : absent,
    seules les détections structurelles (colonnes constantes, cardinalité
    excessive, doublons exacts, numérique mal typé, valeurs manquantes,
    colinéarité) sont renvoyées — permet d'appeler cet endpoint dès
    l'exploration d'un dataset (page Données/EDA), avant même de choisir une
    cible pour un entraînement.

    `feature_columns` (retour utilisateur direct — diagnostic de cohérence
    du wizard : "une colonne exclue à l'étape 1 déclenche encore une alerte
    à l'étape 2") : liste blanche des variables encore retenues. Une colonne
    absente de cette liste (déjà exclue par l'utilisateur) n'est plus
    analysée du tout, quel que soit son problème — voir
    `services/data_quality.py::analyze_data_quality`."""
    dataset = _get_org_dataset(dataset_id, current_user, db)
    if dataset.status != "ready":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": ErrorCode.DATASET_NON_PRET, "message": "Ce dataset n'a pas pu être analysé"},
        )
    try:
        df = read_dataset_dataframe(Path(dataset.file_path), Path(dataset.file_path).suffix)
        excluded_columns = _excluded_columns_from_allowlist(df, feature_columns)
        warnings = analyze_data_quality(df, target_column, group_column, excluded_columns=excluded_columns)
    except DatasetParsingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": ErrorCode.DATASET_LECTURE_ECHEC, "message": str(exc)},
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": ErrorCode.COLONNE_INTROUVABLE, "message": str(exc)},
        )
    return DataQualityResponse(warnings=[DataWarning(**w) for w in warnings])


@router.get("/{dataset_id}/target-suggestions", response_model=TargetSuggestionsResponse)
def get_dataset_target_suggestions(
    dataset_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Suggestion de colonne cible (Lot 7, §J.1) — un score de plausibilité
    et les raisons concrètes qui le justifient, jamais un choix fait à la
    place de l'utilisateur (voir `services/target_suggestion.py`)."""
    dataset = _get_org_dataset(dataset_id, current_user, db)
    if dataset.status != "ready":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": ErrorCode.DATASET_NON_PRET, "message": "Ce dataset n'a pas pu être analysé"},
        )
    try:
        df = read_dataset_dataframe(Path(dataset.file_path), Path(dataset.file_path).suffix)
    except DatasetParsingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": ErrorCode.DATASET_LECTURE_ECHEC, "message": str(exc)},
        )
    suggestions = suggest_target_columns(df)
    return TargetSuggestionsResponse(
        suggestions=[TargetSuggestionOut(column=s.column, score=s.score, reasons=s.reasons) for s in suggestions]
    )


@router.get("/{dataset_id}/feature-engineering-suggestions", response_model=FeatureEngineeringSuggestionsResponse)
def get_dataset_feature_engineering_suggestions(
    dataset_id: int,
    target_column: str,
    group_column: Optional[str] = None,
    feature_columns: Optional[str] = Query(
        None,
        description=(
            "Colonnes CSV déjà retenues à l'étape 1 du formulaire — si fourni, aucune suggestion "
            "n'est générée pour une colonne déjà exclue. Absent = comportement historique."
        ),
    ),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Suggestions d'ingénierie de variables (Lot 4c) — en langage clair,
    reliées quand c'est pertinent à un garde-fou déjà détecté (Lot B).
    Jamais bloquant, jamais appliqué automatiquement : l'utilisateur approuve
    explicitement au moment de lancer l'entraînement
    (`POST /training/jobs`, champ `feature_engineering`).

    `feature_columns` (retour utilisateur direct — diagnostic de cohérence
    du wizard : "ref_complete exclue à l'étape 1, l'étape 3 propose encore
    de l'encoder") : voir `_excluded_columns_from_allowlist` /
    `services/feature_engineering.py::suggest_feature_engineering`."""
    dataset = _get_org_dataset(dataset_id, current_user, db)
    if dataset.status != "ready":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": ErrorCode.DATASET_NON_PRET, "message": "Ce dataset n'a pas pu être analysé"},
        )
    try:
        df = read_dataset_dataframe(Path(dataset.file_path), Path(dataset.file_path).suffix)
        excluded_columns = _excluded_columns_from_allowlist(df, feature_columns)
        suggestions = suggest_feature_engineering(df, target_column, group_column, excluded_columns=excluded_columns)
    except DatasetParsingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": ErrorCode.DATASET_LECTURE_ECHEC, "message": str(exc)},
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": ErrorCode.COLONNE_INTROUVABLE, "message": str(exc)},
        )
    return FeatureEngineeringSuggestionsResponse(
        suggestions=[FeatureEngineeringSuggestion(**s) for s in suggestions]
    )


@router.get("/{dataset_id}/feature-by-target", response_model=FeatureByTargetResponse)
def get_dataset_feature_by_target(
    dataset_id: int,
    feature: str,
    target: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Boxplot d'une feature numérique par classe de la cible (Lot B) — à la
    demande, comme `/histogram`, pour ne pas calculer ce graphique pour
    toutes les combinaisons feature×cible d'un coup."""
    dataset = _get_org_dataset(dataset_id, current_user, db)
    if dataset.status != "ready":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": ErrorCode.DATASET_NON_PRET, "message": "Ce dataset n'a pas pu être analysé"},
        )
    try:
        df = read_dataset_dataframe(Path(dataset.file_path), Path(dataset.file_path).suffix)
        result = compute_feature_by_target(df, feature, target)
    except DatasetParsingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": ErrorCode.DATASET_LECTURE_ECHEC, "message": str(exc)},
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": ErrorCode.COLONNE_INTROUVABLE, "message": str(exc)},
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "FEATURE_NON_NUMERIQUE", "message": str(exc)},
        )
    return FeatureByTargetResponse(**result)
