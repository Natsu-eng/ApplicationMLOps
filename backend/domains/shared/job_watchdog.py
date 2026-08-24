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


def _queued_rq_job_is_gone(rq_job_id: str | None) -> bool:
    """Correctif Phase 2 (AUDIT_BACKEND_2026-08-23.md §F3) — avant ce
    correctif, un job `"queued"` n'était JAMAIS reconsidéré par ce module
    (seul `"running"` l'était), alors qu'un job RQ peut disparaître de Redis
    sans jamais avoir démarré (expiration, `FLUSHALL`, incident Redis) —
    invisible jusqu'à intervention manuelle. `rq_job_id is None` compte
    comme "disparu" : depuis le correctif F5
    (`domains/shared/job_creation.py::enqueue_or_mark_failed`), un job
    `"queued"` a TOUJOURS un `rq_job_id` non nul en fonctionnement normal —
    None ici signale un résidu d'avant ce correctif, ou un bug, jamais un
    état transitoire légitime."""
    if not rq_job_id:
        return True
    from rq.exceptions import NoSuchJobError
    from rq.job import Job

    from api.core.job_queue import redis_conn

    try:
        Job.fetch(rq_job_id, connection=redis_conn)
        return False  # encore dans Redis — en attente légitime d'un worker libre, ou en cours
    except NoSuchJobError:
        # Bug réel trouvé en testant (test_job_watchdog.py) — un premier
        # essai attrapait `NoSuchJobError` dans le même `except Exception`
        # générique que "Redis injoignable", renvoyant `False` (donc "pas
        # disparu") dans les deux cas : exactement l'inverse de ce que ce
        # cas précis (le job n'existe vraiment plus) doit produire.
        return True
    except Exception:
        # Toute AUTRE exception (Redis injoignable, etc.) reste en échec
        # ouvert délibéré : mieux vaut ne pas déclarer un job disparu à
        # tort qu'affirmer une perte non prouvée pendant une panne Redis
        # passagère (voir is_rate_limited pour le même principe).
        return False


def reconcile_stale_jobs(
    db: Session, organization_id: int, stale_after_minutes: int, model: Type[JobModel] = TrainingJob
) -> int:
    """Marque `failed` tout job de l'organisation (par défaut `TrainingJob`
    — passer `model=ClusteringJob` pour le pilier non supervisé) qui ne
    progressera plus jamais. Retourne le nombre de jobs reclassés. Deux cas
    distincts :
    - `"running"` sans signal de vie depuis plus de `stale_after_minutes`
      (référence = `progress_updated_at` si présent, sinon `started_at` —
      jamais `created_at`, qui ne reflète que l'enfilement, pas l'exécution
      réelle) — worker mort pendant l'exécution.
    - `"queued"` dont le job RQ sous-jacent n'existe plus dans Redis (voir
      `_queued_rq_job_is_gone`) — job perdu avant même d'avoir démarré."""
    threshold = datetime.now(timezone.utc) - timedelta(minutes=stale_after_minutes)
    stale_jobs = (
        db.query(model)
        .filter(
            model.organization_id == organization_id,
            model.status.in_(("running", "queued")),
        )
        .all()
    )
    reconciled = 0
    for job in stale_jobs:
        if job.status == "running":
            last_seen = _as_aware_utc(job.progress_updated_at or job.started_at)
            is_stale = last_seen is None or last_seen < threshold
            message = (
                "L'entraînement a été interrompu de façon inattendue (aucune "
                "activité détectée) et a été marqué en échec automatiquement. "
                "Relancez l'entraînement."
            )
        else:  # "queued"
            # Délai de grâce identique au cas "running" (`threshold`, basé sur
            # `created_at` — un job en attente n'a pas encore de
            # `started_at`) : sans lui, un job enfilé l'INSTANT d'avant (le
            # temps qu'un worker le prenne, normal sous charge) serait
            # vérifié dans Redis avant même d'y être pleinement visible et
            # risquerait un faux positif. Bug réel trouvé en testant (suite
            # complète, pas le fichier isolé) : `test_saas_hardening.py`
            # crée `limit` jobs d'affilée avec une file RQ simulée
            # (`mock_queue.enqueue.return_value.id = "fake-rq-id"`, jamais
            # réellement enfilé dans Redis) — sans ce délai, CHAQUE job
            # nouvellement créé était immédiatement réconcilié `"failed"` à
            # la création du suivant (`rq_job_id` introuvable dans le VRAI
            # Redis), le quota ne se déclenchait donc jamais. Le délai de
            # grâce est la correction juste dans les deux cas : un job
            # simulé en test n'est jamais "réellement" perdu avant son délai
            # de grâce, et un job réel tout juste enfilé en production ne
            # doit pas non plus être jugé perdu avant d'avoir eu le temps
            # d'être visible dans Redis.
            created_at = _as_aware_utc(job.created_at)
            is_stale = (created_at is None or created_at < threshold) and _queued_rq_job_is_gone(job.rq_job_id)
            message = (
                "Ce job a été perdu avant son démarrage (incident infrastructure) "
                "et a été marqué en échec automatiquement. Relancez l'entraînement."
            )
        if is_stale:
            job.status = "failed"
            job.error_message = message
            job.finished_at = datetime.now(timezone.utc)
            reconciled += 1
    if reconciled:
        db.commit()
    return reconciled
