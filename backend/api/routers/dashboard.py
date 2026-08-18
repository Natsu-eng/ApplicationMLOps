"""Router tableau de bord — endpoint agrégé (Lot 4, correctif I3,
AUDIT_DATALAB_2026-08-16.md §C.2.4) : `Dashboard.tsx` appelait jusqu'ici 8
endpoints de liste COMPLETS à chaque montage (membres, datasets, et les 6
listes de jobs) pour n'en tirer que des compteurs et les 6 activités les
plus récentes — remplacé par UN seul aller-retour.

Réutilise les fonctions `_to_summary` et schémas `XJobSummary` déjà
définis dans chaque router de job plutôt que de dupliquer leur forme —
un seul endroit fait foi sur "à quoi ressemble un résumé de job
supervisé/clustering/...". Réutilise aussi `count_active_jobs`/
`ALL_JOB_MODELS` (`services/job_quota.py`), déjà le point de vérité sur
"tous les types de job confondus" pour le quota — même liste, même
raisonnement, pas une seconde définition qui pourrait diverger."""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session, joinedload

from api.core.database import get_db
from api.core.models import (
    AnomalyJob,
    ClusteringJob,
    Dataset,
    DimensionalityJob,
    TrainingJob,
    User,
    VisionAnomalyJob,
    VisionClassificationJob,
)
from api.routers.anomalies import AnomalyJobSummary, _to_summary as _anomaly_summary
from api.routers.auth import get_current_user
from api.routers.clustering import ClusteringJobSummary, _to_summary as _clustering_summary
from api.routers.datasets import DatasetSummary, _to_summary as _dataset_summary
from api.routers.dimensionality import DimensionalityJobSummary, _to_summary as _dimensionality_summary
from api.routers.training import TrainingJobSummary, _to_summary as _training_summary
from api.routers.vision_anomalies import VisionAnomalyJobSummary, _to_summary as _vision_anomaly_summary
from api.routers.vision_classification import VisionClassificationJobSummary, _to_summary as _vision_classification_summary
from services.job_quota import ALL_JOB_MODELS, count_active_jobs

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

# Nombre d'activités récentes remontées PAR TYPE — jamais moins que ce que
# `Dashboard.tsx` affiche au final (6, tous types fusionnés puis triés par
# date) : si un seul pilier concentrait les 6 plus récents, les 5 autres
# listes à 6 chacune couvrent quand même largement de quoi les dominer dans
# le tri final, sans jamais ramener des milliers de lignes juste pour en
# garder 6.
_RECENT_PER_PILLAR = 6
_RECENT_DATASETS = 5


class DashboardSummary(BaseModel):
    members_count: int
    datasets_count: int
    recent_datasets: List[DatasetSummary]
    supervised_count: int
    unsupervised_count: int
    vision_count: int
    active_count: int
    recent_supervised: List[TrainingJobSummary]
    recent_clustering: List[ClusteringJobSummary]
    recent_dimensionality: List[DimensionalityJobSummary]
    recent_anomalies: List[AnomalyJobSummary]
    recent_vision_classification: List[VisionClassificationJobSummary]
    recent_vision_anomalies: List[VisionAnomalyJobSummary]


@router.get("/summary", response_model=DashboardSummary)
def get_dashboard_summary(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    org_id = current_user.organization_id

    members_count = db.query(User).filter(User.organization_id == org_id).count()
    datasets_count = db.query(Dataset).filter(Dataset.organization_id == org_id).count()
    recent_datasets = (
        db.query(Dataset)
        .options(joinedload(Dataset.uploaded_by))
        .filter(Dataset.organization_id == org_id)
        # id DESC, pas created_at DESC — voir
        # anomalies.py::list_anomaly_jobs (précision seconde de SQLite).
        .order_by(Dataset.id.desc())
        .limit(_RECENT_DATASETS)
        .all()
    )

    def _count(model) -> int:
        return db.query(model).filter(model.organization_id == org_id).count()

    supervised_count = _count(TrainingJob)
    unsupervised_count = _count(ClusteringJob) + _count(DimensionalityJob) + _count(AnomalyJob)
    vision_count = _count(VisionClassificationJob) + _count(VisionAnomalyJob)
    active_count = count_active_jobs(db, org_id, ALL_JOB_MODELS)

    def _recent(model, *relationships):
        return (
            db.query(model)
            .options(*[joinedload(rel) for rel in relationships])
            .filter(model.organization_id == org_id)
            .order_by(model.id.desc())
            .limit(_RECENT_PER_PILLAR)
            .all()
        )

    return DashboardSummary(
        members_count=members_count,
        datasets_count=datasets_count,
        recent_datasets=[_dataset_summary(d) for d in recent_datasets],
        supervised_count=supervised_count,
        unsupervised_count=unsupervised_count,
        vision_count=vision_count,
        active_count=active_count,
        recent_supervised=[
            _training_summary(j) for j in _recent(TrainingJob, TrainingJob.dataset, TrainingJob.created_by, TrainingJob.model)
        ],
        recent_clustering=[
            _clustering_summary(j)
            for j in _recent(ClusteringJob, ClusteringJob.dataset, ClusteringJob.created_by, ClusteringJob.result)
        ],
        recent_dimensionality=[
            _dimensionality_summary(j)
            for j in _recent(
                DimensionalityJob, DimensionalityJob.dataset, DimensionalityJob.created_by, DimensionalityJob.result
            )
        ],
        recent_anomalies=[
            _anomaly_summary(j) for j in _recent(AnomalyJob, AnomalyJob.dataset, AnomalyJob.created_by, AnomalyJob.result)
        ],
        recent_vision_classification=[
            _vision_classification_summary(j)
            for j in _recent(
                VisionClassificationJob,
                VisionClassificationJob.vision_dataset,
                VisionClassificationJob.created_by,
                VisionClassificationJob.result,
            )
        ],
        recent_vision_anomalies=[
            _vision_anomaly_summary(j)
            for j in _recent(
                VisionAnomalyJob, VisionAnomalyJob.vision_dataset, VisionAnomalyJob.created_by, VisionAnomalyJob.result
            )
        ],
    )
