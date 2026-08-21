"""Router datasets — upload, liste, aperçu, suppression.

Isolation : toute opération est filtrée par
`Dataset.organization_id == current_user.organization_id` — jamais par un
identifiant fourni par le client (même principe que
`api/routers/auth.py::list_team_members`). Accessible à tout membre de
l'organisation, pas réservé au owner : un dataset appartient à l'équipe.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.orm import Session, joinedload

from api.core.config import get_settings
from api.core.database import get_db
from api.core.models import AnomalyJob, ClusteringJob, Dataset, DimensionalityJob, TrainingJob, User
from api.core.pagination import paginate_by_id
from api.core.rate_limit import rate_limit_dependency
from api.core.storage import dataset_file_path, delete_dataset_file
from api.routers.auth import get_current_user
from services.audit import log_action
from services.data_quality import analyze_data_quality
from services.target_suggestion import suggest_target_columns
from services.dataset_eda import (
    compute_categorical_correlation_matrix,
    compute_column_stats,
    compute_correlation_matrix,
    compute_feature_by_target,
    compute_histogram,
    compute_missing_summary,
    compute_outlier_boxplots,
    compute_top_correlated_pairs,
)
from services.datasets import (
    DatasetParsingError,
    UnsupportedFileType,
    extract_schema,
    read_dataframe,
    read_dataset_dataframe,
    sample_rows,
    validate_extension,
)
from services.feature_engineering import suggest_feature_engineering

logger = logging.getLogger("datalab.datasets")
router = APIRouter(prefix="/datasets", tags=["datasets"])

_settings = get_settings()
_MAX_UPLOAD_BYTES = _settings.max_upload_size_mb * 1024 * 1024


# ── Schémas ──────────────────────────────────────────────────────────────────

class DatasetSummary(BaseModel):
    id: int
    name: str
    file_size_bytes: int
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    status: str
    error_message: Optional[str] = None
    uploaded_by: Optional[str] = None
    created_at: datetime


class ColumnSchema(BaseModel):
    name: str
    dtype: str


class DatasetDetail(DatasetSummary):
    columns: List[ColumnSchema] = []
    # Lot 5 (correctif P2) — content_hash absent (None) pour tout dataset
    # uploadé avant ce lot, jamais recalculé a posteriori.
    content_hash: Optional[str] = None
    duplicate_of_dataset_id: Optional[int] = None


class PreviewResponse(BaseModel):
    columns: List[str]
    rows: List[dict]
    sample_size: int
    row_count: Optional[int] = None


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


class DatasetUsageResponse(BaseModel):
    # Toutes les tables de job référencent `datasets.id` en `ON DELETE
    # CASCADE` (voir api/core/models.py) — la suppression est donc déjà
    # sûre côté base ; ce décompte sert uniquement à avertir l'utilisateur
    # AVANT de confirmer (Lot 7, §J.3 — avertissement de suppression en
    # cascade, absent jusqu'ici : seule une confirmation générique à deux
    # clics existait).
    training_jobs: int
    clustering_jobs: int
    dimensionality_jobs: int
    anomaly_jobs: int
    total: int


def _to_summary(dataset: Dataset) -> DatasetSummary:
    # Construction explicite plutôt que model_validate(dataset, from_attributes=True) :
    # le champ Pydantic `uploaded_by` (str) entre en collision de nom avec la relation
    # SQLAlchemy `Dataset.uploaded_by` (objet User) — from_attributes essaierait de
    # valider l'objet User comme une string et lèverait une ValidationError.
    return DatasetSummary(
        id=dataset.id,
        name=dataset.name,
        file_size_bytes=dataset.file_size_bytes,
        row_count=dataset.row_count,
        column_count=dataset.column_count,
        status=dataset.status,
        error_message=dataset.error_message,
        uploaded_by=dataset.uploaded_by.nom if dataset.uploaded_by else None,
        created_at=dataset.created_at,
    )


def _to_detail(dataset: Dataset) -> DatasetDetail:
    columns = [ColumnSchema(**c) for c in json.loads(dataset.columns_json)] if dataset.columns_json else []
    return DatasetDetail(
        **_to_summary(dataset).model_dump(),
        columns=columns,
        content_hash=dataset.content_hash,
        duplicate_of_dataset_id=dataset.duplicate_of_dataset_id,
    )


def _get_org_dataset(dataset_id: int, current_user: User, db: Session) -> Dataset:
    dataset = (
        db.query(Dataset)
        .filter(Dataset.id == dataset_id, Dataset.organization_id == current_user.organization_id)
        .first()
    )
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "DATASET_INTROUVABLE", "message": "Dataset introuvable"},
        )
    return dataset


# ── Endpoints ────────────────────────────────────────────────────────────────

_upload_rate_limit = rate_limit_dependency(
    "dataset_upload", _settings.upload_rate_limit_max_attempts, _settings.upload_rate_limit_window_seconds
)


@router.post(
    "", response_model=DatasetDetail, status_code=status.HTTP_201_CREATED, dependencies=[Depends(_upload_rate_limit)]
)
async def upload_dataset(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Upload d'un dataset tabulaire (csv/xlsx/xls/parquet/json) — visible par toute l'organisation."""
    try:
        extension = validate_extension(file.filename or "")
    except UnsupportedFileType as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "DATASET_FORMAT_NON_SUPPORTE", "message": str(exc)},
        )

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "DATASET_FICHIER_VIDE", "message": "Le fichier est vide"},
        )
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "code": "DATASET_TROP_VOLUMINEUX",
                "message": f"Fichier trop volumineux (max {_settings.max_upload_size_mb} Mo)",
            },
        )

    # Lot 5 (correctif P2) — SHA-256 du contenu brut, jamais du fichier une
    # fois écrit sur disque (mêmes octets de toute façon, mais évite une
    # relecture disque : `content` est déjà en mémoire ci-dessus).
    content_hash = hashlib.sha256(content).hexdigest()
    duplicate = (
        db.query(Dataset)
        .filter(Dataset.organization_id == current_user.organization_id, Dataset.content_hash == content_hash)
        .first()
    )

    # La ligne DB est créée avant l'écriture sur disque pour obtenir un id
    # (utilisé dans le nom du fichier stocké — voir api/core/storage.py).
    dataset = Dataset(
        organization_id=current_user.organization_id,
        uploaded_by_id=current_user.id,
        name=file.filename or "dataset",
        file_path="",
        file_size_bytes=len(content),
        status="processing",
        content_hash=content_hash,
        # Jamais bloquant (voir docstring du modèle) : signalé, l'upload
        # aboutit quand même — l'utilisateur décide si c'est voulu.
        duplicate_of_dataset_id=duplicate.id if duplicate else None,
    )
    db.add(dataset)
    db.flush()

    target_path = dataset_file_path(current_user.organization_id, dataset.id, extension)
    target_path.write_bytes(content)
    dataset.file_path = str(target_path)

    try:
        df = read_dataframe(target_path, extension)
        dataset.row_count = int(len(df))
        dataset.column_count = int(len(df.columns))
        dataset.columns_json = json.dumps(extract_schema(df))
        dataset.status = "ready"
    except DatasetParsingError as exc:
        dataset.status = "error"
        dataset.error_message = str(exc)
        logger.warning("[Datasets] Échec de parsing pour %s : %s", file.filename, exc)

    db.commit()
    db.refresh(dataset)
    return _to_detail(dataset)


@router.get("", response_model=List[DatasetSummary])
def list_datasets(
    response: Response,
    limit: Optional[int] = Query(None, ge=1, le=500),
    cursor: Optional[int] = Query(None, description="id de la dernière ligne de la page précédente"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # joinedload (Lot 4, correctif I3, même motif que les 6 listes de jobs
    # — training.py::list_training_jobs) : _to_summary accède à
    # dataset.uploaded_by, sans quoi 1 requête SQL par dataset (N+1).
    query = (
        db.query(Dataset)
        .options(joinedload(Dataset.uploaded_by))
        .filter(Dataset.organization_id == current_user.organization_id)
        # id DESC, pas created_at DESC — voir anomalies.py::list_anomaly_jobs.
        .order_by(Dataset.id.desc())
    )
    datasets = paginate_by_id(query, Dataset.id, response, cursor, limit)
    return [_to_summary(d) for d in datasets]


@router.get("/{dataset_id}", response_model=DatasetDetail)
def get_dataset(dataset_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return _to_detail(_get_org_dataset(dataset_id, current_user, db))


@router.get("/{dataset_id}/preview", response_model=PreviewResponse)
def preview_dataset(
    dataset_id: int,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    dataset = _get_org_dataset(dataset_id, current_user, db)
    if dataset.status != "ready":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "DATASET_NON_PRET", "message": "Ce dataset n'a pas pu être analysé"},
        )
    extension = Path(dataset.file_path).suffix
    try:
        df = read_dataset_dataframe(Path(dataset.file_path), extension)
    except DatasetParsingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "DATASET_LECTURE_ECHEC", "message": str(exc)},
        )
    limit = max(1, min(limit, 500))
    rows = sample_rows(df, limit)
    return PreviewResponse(
        columns=list(df.columns.astype(str)),
        rows=rows,
        sample_size=len(rows),
        row_count=dataset.row_count,
    )


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
            detail={"code": "DATASET_NON_PRET", "message": "Ce dataset n'a pas pu être analysé"},
        )
    try:
        df = read_dataset_dataframe(Path(dataset.file_path), Path(dataset.file_path).suffix)
        target_distribution = compute_histogram(df, target_column) if target_column else None
    except DatasetParsingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "DATASET_LECTURE_ECHEC", "message": str(exc)},
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "COLONNE_INTROUVABLE", "message": str(exc)},
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
            detail={"code": "DATASET_NON_PRET", "message": "Ce dataset n'a pas pu être analysé"},
        )
    try:
        df = read_dataset_dataframe(Path(dataset.file_path), Path(dataset.file_path).suffix)
        histogram = compute_histogram(df, column, bins=max(5, min(bins, 100)))
    except DatasetParsingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "DATASET_LECTURE_ECHEC", "message": str(exc)},
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "COLONNE_INTROUVABLE", "message": str(exc)},
        )
    return HistogramResponse(**histogram)


@router.get("/{dataset_id}/quality-check", response_model=DataQualityResponse)
def get_dataset_quality_check(
    dataset_id: int,
    target_column: Optional[str] = None,
    group_column: Optional[str] = None,
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
    cible pour un entraînement."""
    dataset = _get_org_dataset(dataset_id, current_user, db)
    if dataset.status != "ready":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "DATASET_NON_PRET", "message": "Ce dataset n'a pas pu être analysé"},
        )
    try:
        df = read_dataset_dataframe(Path(dataset.file_path), Path(dataset.file_path).suffix)
        warnings = analyze_data_quality(df, target_column, group_column)
    except DatasetParsingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "DATASET_LECTURE_ECHEC", "message": str(exc)},
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "COLONNE_INTROUVABLE", "message": str(exc)},
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
            detail={"code": "DATASET_NON_PRET", "message": "Ce dataset n'a pas pu être analysé"},
        )
    try:
        df = read_dataset_dataframe(Path(dataset.file_path), Path(dataset.file_path).suffix)
    except DatasetParsingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "DATASET_LECTURE_ECHEC", "message": str(exc)},
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
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Suggestions d'ingénierie de variables (Lot 4c) — en langage clair,
    reliées quand c'est pertinent à un garde-fou déjà détecté (Lot B).
    Jamais bloquant, jamais appliqué automatiquement : l'utilisateur approuve
    explicitement au moment de lancer l'entraînement
    (`POST /training/jobs`, champ `feature_engineering`)."""
    dataset = _get_org_dataset(dataset_id, current_user, db)
    if dataset.status != "ready":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "DATASET_NON_PRET", "message": "Ce dataset n'a pas pu être analysé"},
        )
    try:
        df = read_dataset_dataframe(Path(dataset.file_path), Path(dataset.file_path).suffix)
        suggestions = suggest_feature_engineering(df, target_column, group_column)
    except DatasetParsingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "DATASET_LECTURE_ECHEC", "message": str(exc)},
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "COLONNE_INTROUVABLE", "message": str(exc)},
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
            detail={"code": "DATASET_NON_PRET", "message": "Ce dataset n'a pas pu être analysé"},
        )
    try:
        df = read_dataset_dataframe(Path(dataset.file_path), Path(dataset.file_path).suffix)
        result = compute_feature_by_target(df, feature, target)
    except DatasetParsingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "DATASET_LECTURE_ECHEC", "message": str(exc)},
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "COLONNE_INTROUVABLE", "message": str(exc)},
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "FEATURE_NON_NUMERIQUE", "message": str(exc)},
        )
    return FeatureByTargetResponse(**result)


@router.get("/{dataset_id}/usage", response_model=DatasetUsageResponse)
def get_dataset_usage(dataset_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Décompte des jobs qui référencent ce dataset — appelé côté frontend
    avant de confirmer une suppression (Lot 7), pour que l'avertissement de
    suppression en cascade soit chiffré, pas générique."""
    dataset = _get_org_dataset(dataset_id, current_user, db)
    org_id = current_user.organization_id
    training_jobs = db.query(TrainingJob).filter(
        TrainingJob.dataset_id == dataset.id, TrainingJob.organization_id == org_id
    ).count()
    clustering_jobs = db.query(ClusteringJob).filter(
        ClusteringJob.dataset_id == dataset.id, ClusteringJob.organization_id == org_id
    ).count()
    dimensionality_jobs = db.query(DimensionalityJob).filter(
        DimensionalityJob.dataset_id == dataset.id, DimensionalityJob.organization_id == org_id
    ).count()
    anomaly_jobs = db.query(AnomalyJob).filter(
        AnomalyJob.dataset_id == dataset.id, AnomalyJob.organization_id == org_id
    ).count()
    return DatasetUsageResponse(
        training_jobs=training_jobs,
        clustering_jobs=clustering_jobs,
        dimensionality_jobs=dimensionality_jobs,
        anomaly_jobs=anomaly_jobs,
        total=training_jobs + clustering_jobs + dimensionality_jobs + anomaly_jobs,
    )


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(dataset_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    dataset = _get_org_dataset(dataset_id, current_user, db)
    if dataset.file_path:
        delete_dataset_file(Path(dataset.file_path))
    log_action(
        db, current_user.organization_id, current_user.id, "dataset.deleted",
        target_type="dataset", target_id=dataset.id, details={"name": dataset.name},
    )
    db.delete(dataset)
    db.commit()
