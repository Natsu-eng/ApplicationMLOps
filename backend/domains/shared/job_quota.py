"""Quota de jobs concurrents partagé entre tous les types d'entraînement
(AUDIT_ROADMAP.md, Lot 10 puis généralisé au Lot 13/14).

Un nombre de workers RQ toujours limité traite tous les types de job
(supervisé, clustering, réduction de dimension, détection d'anomalies,
vision — voir `docker-compose.yml` et `api/core/job_queue.py`, 3 files
depuis le correctif I6) : sans compter tous les types ensemble, une
organisation pourrait saturer la capacité disponible en cumulant
plusieurs types de jobs actifs, chacun restant sous la limite pris
isolément.

Extrait en helper partagé plutôt que dupliqué dans chaque router : le
Lot 11+12 avait ajouté le comptage combiné côté `clustering.py` sans jamais
mettre à jour `training.py` en retour (un `TrainingJob` ne comptait alors
que les autres `TrainingJob`, jamais les `ClusteringJob` actifs) — un oubli
qui rendait le quota contournable depuis le côté supervisé. Corrigé ici en
même temps que l'ajout d'une 3ᵉ puis 4ᵉ table, pour ne plus jamais dupliquer
ce bloc de comptage à la main.

`ALL_JOB_MODELS` (Lot 15 sous-lot B) va plus loin dans le même sens : avant,
la liste `[TrainingJob, ClusteringJob, DimensionalityJob, AnomalyJob]` était
recopiée telle quelle dans CHAQUE router (`training.py`, `clustering.py`,
`dimensionality.py`, `anomalies.py`) — ajouter un 5ᵉ type de job (vision)
obligeait à modifier les 4 en même temps, exactement le genre d'oubli déjà
documenté ci-dessus. Un seul point de vérité désormais : tout router
d'entraînement importe `ALL_JOB_MODELS` plutôt que de reconstruire la liste."""
from __future__ import annotations

from typing import Type

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from api.core.models import (
    AnomalyJob,
    ClusteringJob,
    DimensionalityJob,
    TrainingJob,
    VisionAnomalyJob,
    VisionClassificationJob,
)

ALL_JOB_MODELS: list[Type] = [
    TrainingJob,
    ClusteringJob,
    DimensionalityJob,
    AnomalyJob,
    VisionClassificationJob,
    VisionAnomalyJob,
]


def count_active_jobs(db: Session, organization_id: int, models: list[Type]) -> int:
    """Additionne les jobs `queued`/`running` de l'organisation, tous types
    de job confondus (`models`) — chaque modèle partage les colonnes
    `organization_id`/`status` standard du projet."""
    total = 0
    for model in models:
        total += (
            db.query(model)
            .filter(model.organization_id == organization_id, model.status.in_(("queued", "running")))
            .count()
        )
    return total


def raise_if_quota_exceeded(db: Session, organization_id: int, models: list[Type], limit: int) -> None:
    """Lève une 429 `QUOTA_ENTRAINEMENTS_ATTEINT` si le nombre de jobs actifs
    (tous types confondus) atteint ou dépasse `limit`. À appeler APRÈS
    `services/job_watchdog.py::reconcile_stale_jobs` pour chaque type de job
    concerné — un job orphelin ne doit jamais bloquer indéfiniment un slot."""
    active_count = count_active_jobs(db, organization_id, models)
    if active_count >= limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "code": "QUOTA_ENTRAINEMENTS_ATTEINT",
                "message": (
                    f"Trop d'entraînements en cours ({active_count}/{limit}, tous types confondus — "
                    "supervisé, clustering, réduction de dimension, détection d'anomalies) — attendez "
                    "qu'un entraînement se termine, ou supprimez-en un depuis l'historique."
                ),
            },
        )
