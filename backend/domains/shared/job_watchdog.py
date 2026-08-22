"""Réconciliation des entraînements orphelins (AUDIT_ROADMAP.md, H2).

Constat de l'audit du 2026-08-14 : `SimpleWorker` (nécessaire sous Windows,
voir workers/run_worker.py) exécute chaque job dans le process du worker,
sans l'isolation process-par-job qu'utilise normalement RQ pour détecter un
worker mort. Si le process worker s'arrête brutalement (OOM, coupure) pendant
un entraînement, la ligne `TrainingJob` reste `status="running"` pour
toujours — sans réconciliation, elle consomme indéfiniment un des
`max_concurrent_jobs_per_org` slots de quota (voir
api/routers/training.py::create_training_job).

Approche volontairement simple, cohérente avec le reste du projet (polling
REST plutôt que WebSocket, pas de scheduler dédié type Celery beat) :
appelée à la demande, au moment où le quota est vérifié — pas de process
séparé ni de dépendance nouvelle. Un job vraiment bloqué est donc détecté au
plus tard à la prochaine tentative de lancement d'entraînement de son
organisation, ce qui est précisément le moment où son statut orphelin
importe (il bloque un slot de quota)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Type, TypeVar, Union

from sqlalchemy.orm import Session

from api.core.models import ClusteringJob, TrainingJob

# Généralisé (Lot 11+) pour couvrir aussi `ClusteringJob`, même risque
# d'orphelin qu'un `TrainingJob` (même worker, même absence d'isolation
# process-par-job) — les deux tables partagent exactement les mêmes
# colonnes de progression (`status`/`progress_updated_at`/`started_at`/
# `error_message`/`finished_at`), donc une seule fonction générique plutôt
# qu'une copie quasi identique par type de job.
JobModel = TypeVar("JobModel", bound=Union[TrainingJob, ClusteringJob])


def _as_aware_utc(dt: datetime | None) -> datetime | None:
    """SQLite (dev) ne conserve pas le fuseau horaire des colonnes
    `DateTime(timezone=True)` — une valeur relue est naïve alors qu'elle a
    toujours été écrite en UTC (voir `training_worker.py`,
    `datetime.now(timezone.utc)` partout). PostgreSQL (prod) conserve le
    fuseau : `dt.tzinfo` est déjà renseigné, ne rien changer."""
    if dt is None or dt.tzinfo is not None:
        return dt
    return dt.replace(tzinfo=timezone.utc)


def reconcile_stale_jobs(
    db: Session, organization_id: int, stale_after_minutes: int, model: Type[JobModel] = TrainingJob
) -> int:
    """Marque `failed` tout job `running` de l'organisation (par défaut
    `TrainingJob` — passer `model=ClusteringJob` pour le pilier non
    supervisé) sans signal de vie depuis plus de `stale_after_minutes`.
    Retourne le nombre de jobs reclassés.

    Référence de fraîcheur = `progress_updated_at` si présent, sinon
    `started_at` (job resté bloqué avant la toute première mise à jour de
    progression) — jamais `created_at`, qui ne reflète que l'enfilement, pas
    l'exécution réelle."""
    threshold = datetime.now(timezone.utc) - timedelta(minutes=stale_after_minutes)
    stale_jobs = (
        db.query(model)
        .filter(
            model.organization_id == organization_id,
            model.status == "running",
        )
        .all()
    )
    reconciled = 0
    for job in stale_jobs:
        last_seen = _as_aware_utc(job.progress_updated_at or job.started_at)
        if last_seen is None or last_seen < threshold:
            job.status = "failed"
            job.error_message = (
                "L'entraînement a été interrompu de façon inattendue (aucune "
                "activité détectée) et a été marqué en échec automatiquement. "
                "Relancez l'entraînement."
            )
            job.finished_at = datetime.now(timezone.utc)
            reconciled += 1
    if reconciled:
        db.commit()
    return reconciled
