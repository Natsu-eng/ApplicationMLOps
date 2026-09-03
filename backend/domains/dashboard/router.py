"""Router tableau de bord — endpoint agrégé (Lot 4, correctif I3,
AUDIT_DATALAB_2026-08-16.md §C.2.4) : `Dashboard.tsx` appelait jusqu'ici 8
endpoints de liste COMPLETS à chaque montage (membres, datasets, et les 6
listes de jobs) pour n'en tirer que des compteurs et les 6 activités les
plus récentes — remplacé par UN seul aller-retour.

Réutilise les fonctions `to_summary` et schémas `XJobSummary` déjà
définis dans chaque router de job plutôt que de dupliquer leur forme —
un seul endroit fait foi sur "à quoi ressemble un résumé de job
supervisé/clustering/...". `to_summary` est délibérément publique dans
chaque router de job (Lot 8, §Phase 0 — corrige une fuite d'encapsulation :
ce module importait jusqu'ici les fonctions PRIVÉES `_to_summary` de 6
autres routers). Réutilise aussi `count_active_jobs`/
`ALL_JOB_MODELS` (`services/job_quota.py`), déjà le point de vérité sur
"tous les types de job confondus" pour le quota — même liste, même
raisonnement, pas une seconde définition qui pourrait diverger."""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session, joinedload

from api.core.database import get_db
from api.core.models import (
    AnomalyJob,
    ClusteringJob,
    Dataset,
    DimensionalityJob,
    MLModel,
    TrainingJob,
    User,
    VisionAnomalyJob,
    VisionClassificationJob,
)
from domains.auth.router import get_current_user
from domains.anomalies.router import AnomalyJobSummary, to_summary as _anomaly_summary
from domains.clustering.router import ClusteringJobSummary, to_summary as _clustering_summary
from domains.datasets.router import DatasetSummary, to_summary as _dataset_summary
from domains.dimensionality.router import DimensionalityJobSummary, to_summary as _dimensionality_summary
from domains.training.model_registry_router import to_model_detail
from domains.training.router import TrainingJobSummary, to_summary as _training_summary
from domains.vision.anomalies.router import VisionAnomalyJobSummary, to_summary as _vision_anomaly_summary
from domains.vision.classification.router import VisionClassificationJobSummary, to_summary as _vision_classification_summary
from domains.shared.job_quota import ALL_JOB_MODELS, count_active_jobs

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
    # Retour utilisateur (maquette de refonte) : "part des modèles dont le
    # verdict est utilisable" — sur les modèles ML tabulaire réellement en
    # jeu (staging + production), jamais TOUS les entraînements historiques
    # (un essai abandonné qui a échoué depuis longtemps n'a pas sa place
    # dans un indicateur "puis-je faire confiance à ce qui est actif ?").
    # `None` = aucun modèle en staging/production (jamais un faux 0 %).
    active_models_reliability_pct: Optional[float] = None
    recent_supervised: List[TrainingJobSummary]
    recent_clustering: List[ClusteringJobSummary]
    recent_dimensionality: List[DimensionalityJobSummary]
    recent_anomalies: List[AnomalyJobSummary]
    recent_vision_classification: List[VisionClassificationJobSummary]
    recent_vision_anomalies: List[VisionAnomalyJobSummary]


def _active_models_reliability(db: Session, org_id: int) -> Optional[float]:
    """Part des modèles ML tabulaire en staging/production dont le verdict
    ne comporte aucune affirmation "critique" (`services/verdict.py`) —
    réutilise `training/router.py::to_model_detail` (déjà le point de vérité
    pour construire un verdict, rendu public ici même correctif que
    `to_summary`, Lot 8 §Phase 0 — jamais réimporter une fonction privée
    d'un autre router), aucun second calcul de règles. Population
    volontairement
    petite (staging + production seulement, pas tout l'historique
    d'entraînement) : `compute_verdict` interroge les candidats en base par
    modèle, coûteux à refaire pour des dizaines d'essais abandonnés sur un
    endpoint agrégé pensé pour un seul aller-retour (voir docstring du
    module)."""
    models = (
        db.query(MLModel)
        .filter(MLModel.organization_id == org_id, MLModel.stage.in_(["staging", "production"]))
        .all()
    )
    if not models:
        return None
    usable = sum(
        1
        for m in models
        if not any(c["level"] == "critique" for c in to_model_detail(m, db).verdict.get("claims", []))
    )
    return usable / len(models)


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
    active_models_reliability_pct = _active_models_reliability(db, org_id)

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
        active_models_reliability_pct=active_models_reliability_pct,
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
