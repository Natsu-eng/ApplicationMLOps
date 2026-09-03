"""Dépendances et aides partagées du domaine training.

`get_org_training_job` est le point d'accès UNIQUE à un `TrainingJob` par
son id : il applique systématiquement le filtrage par `organization_id` et
renvoie 404 (jamais 403) quand le job appartient à une autre organisation
— voir `tests/test_idor_regression.py`, qui traite ce helper comme le
proxy fidèle de tous les endpoints du domaine.

Extraits de `router.py` lors du découpage du router training : ces éléments
sont partagés entre `router.py`, `batch_prediction_router.py` et
`model_registry_router.py`, des modules dont aucun ne doit importer l'autre.
"""
from __future__ import annotations

from typing import Any

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from api.core.error_codes import ErrorCode
from api.core.models import TrainingJob, User


def get_org_training_job(job_id: int, current_user: User, db: Session) -> TrainingJob:
    job = (
        db.query(TrainingJob)
        .filter(TrainingJob.id == job_id, TrainingJob.organization_id == current_user.organization_id)
        .first()
    )
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": ErrorCode.TRAINING_JOB_INTROUVABLE, "message": "Entraînement introuvable"},
        )
    return job


def headline_metric(task_type: str, metrics: dict[str, Any]) -> dict[str, Any]:
    """Métrique mise en avant sur la carte d'historique (`JobCard.tsx`).

    En classification, `accuracy` est trompeuse sur un dataset déséquilibré
    (un modèle qui ignore la classe rare peut afficher 95 % d'exactitude) —
    on affiche donc `cv_score`, la métrique RÉELLEMENT utilisée pour
    départager les candidats (ROC-AUC pondérée, voir
    `services/ml_training._classification_selection_score`), jamais
    l'accuracy brute (bug trouvé lors de l'audit leaderboard, Lot D)."""
    if task_type == "regression":
        return {"name": "r2_test", "value": metrics.get("r2_test")}
    return {"name": "cv_score", "value": metrics.get("cv_score")}
