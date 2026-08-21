"""Estimation de la durée d'un entraînement AVANT lancement (Lot 7, §J.1) —
jamais un chiffre inventé : dérivée des durées RÉELLES des entraînements
déjà terminés de la même organisation (`TrainingJob.started_at`/
`finished_at`), jamais d'une constante de calibration à l'aveugle.

Dégradation honnête si l'historique est insuffisant (`status: "degraded"`,
même motif que le reste du produit) — pas d'organisation "type", pas de
moyenne inventée pour combler l'absence de données."""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from typing import Optional

from sqlalchemy.orm import Session

from api.core.models import TrainingJob
from services.ml_registry import models_for_task

# En dessous de ce nombre d'entraînements terminés, le taux (durée / unité
# de travail) n'est pas assez stable pour être montré — mieux vaut dire
# "pas encore assez d'historique" que projeter un chiffre sur 1-2 points.
MIN_COMPLETED_JOBS_FOR_ESTIMATE = 3


@dataclass
class DurationEstimate:
    status: str  # "estimated" | "degraded"
    estimated_seconds: Optional[float]
    based_on_n_jobs: int
    message: Optional[str] = None


def _unit_of_work(n_rows: int, n_models: int, n_trials: int, n_folds: int) -> float:
    """Grandeur proportionnelle au travail réellement effectué (lignes ×
    modèles comparés × essais Optuna × folds de validation croisée) — voir
    `services/ml_training.py::train_and_evaluate`, qui boucle exactement sur
    ces 4 dimensions."""
    return max(1, n_rows) * max(1, n_models) * max(1, n_trials) * max(1, n_folds)


def _n_models_of(job: TrainingJob) -> int:
    config = json.loads(job.config_json or "{}")
    model_ids = config.get("model_ids")
    if model_ids:
        return len(model_ids)
    return len(models_for_task(job.task_type, "default")) or 1


def estimate_training_duration(
    db: Session,
    organization_id: int,
    n_rows: int,
    n_models: int,
    n_trials: int,
    n_folds: int,
) -> DurationEstimate:
    completed = (
        db.query(TrainingJob)
        .filter(
            TrainingJob.organization_id == organization_id,
            TrainingJob.status == "completed",
            TrainingJob.started_at.isnot(None),
            TrainingJob.finished_at.isnot(None),
        )
        .order_by(TrainingJob.id.desc())
        .limit(50)  # historique récent seulement — un taux d'il y a 6 mois
        # sur une machine différente n'est pas plus pertinent qu'aucun historique.
        .all()
    )

    rates: list[float] = []
    for job in completed:
        duration = (job.finished_at - job.started_at).total_seconds()
        n_rows_job = job.dataset.row_count if job.dataset else None
        if not n_rows_job or duration <= 0:
            continue
        config = json.loads(job.config_json or "{}")
        unit = _unit_of_work(
            n_rows_job,
            _n_models_of(job),
            config.get("optuna_trials") or 20,
            config.get("cv_folds") or 4,
        )
        rates.append(duration / unit)

    if len(rates) < MIN_COMPLETED_JOBS_FOR_ESTIMATE:
        return DurationEstimate(
            status="degraded",
            estimated_seconds=None,
            based_on_n_jobs=len(rates),
            message="Pas encore assez d'entraînements terminés pour estimer une durée.",
        )

    # Médiane plutôt que moyenne — un unique entraînement anormalement long
    # (ex. dataset avec beaucoup de colonnes catégorielles à forte
    # cardinalité) ne doit pas à lui seul décaler l'estimation.
    rate = statistics.median(rates)
    estimated_seconds = rate * _unit_of_work(n_rows, n_models, n_trials, n_folds)
    return DurationEstimate(status="estimated", estimated_seconds=estimated_seconds, based_on_n_jobs=len(rates))
