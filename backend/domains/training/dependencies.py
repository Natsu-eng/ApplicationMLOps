"""Dépendances partagées du domaine training.

`get_org_training_job` est le point d'accès UNIQUE à un `TrainingJob` par
son id : il applique systématiquement le filtrage par `organization_id` et
renvoie 404 (jamais 403) quand le job appartient à une autre organisation
— voir `tests/test_idor_regression.py`, qui traite ce helper comme le
proxy fidèle de tous les endpoints du domaine.

Extrait de `router.py` lors du découpage du router training : il est
désormais partagé entre `router.py` et `batch_prediction_router.py`, deux
modules dont aucun ne doit importer l'autre.
"""
from __future__ import annotations

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
