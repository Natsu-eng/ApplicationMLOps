"""Idempotence de création de job + échec propre à l'enfilage — Phase 2
(AUDIT_BACKEND_2026-08-23.md, Axe F).

Deux garde-fous distincts, souvent nécessaires ensemble mais indépendants :
- **F4 — idempotence** : avant ce correctif, un double-clic ou une requête
  retentée après un timeout réseau (le serveur avait pourtant déjà traité
  la première) créait deux jobs identiques, consommant deux fois le quota
  d'entraînements concurrents. `Idempotency-Key` (en-tête fourni par le
  client, jamais généré côté serveur — c'est au client de garantir que la
  MÊME tentative logique réutilise la MÊME clé) mémorisé dans Redis,
  scopé par organisation.
- **F5 — échec propre à l'enfilage** : avant ce correctif, si Redis tombait
  entre le `db.commit()` de création et l'appel `queue.enqueue()`, le job
  restait en base avec `status="queued"` et `rq_job_id=NULL` pour
  toujours — `job_watchdog.py` ne détecte que les jobs `"running"`, jamais
  `"queued"`. Cette fonction ne laisse plus jamais ce cas se produire :
  soit l'enfilage réussit et le job est correctement lié à son job RQ,
  soit il échoue et le job est marqué `"failed"` DANS LA MÊME REQUÊTE,
  jamais orphelin."""

from __future__ import annotations

import logging
from typing import Callable, Optional, Protocol

from fastapi import HTTPException, Request, status
from redis import Redis
from rq import Queue
from sqlalchemy.orm import Session

logger = logging.getLogger("datalab.job_creation")


class _JobRecord(Protocol):
    """Forme structurelle commune aux 6 modèles de job (TrainingJob,
    ClusteringJob, DimensionalityJob, AnomalyJob, VisionClassificationJob,
    VisionAnomalyJob) — pas de classe de base commune en héritage (chacun
    est une table indépendante), donc un `Protocol` structurel plutôt qu'un
    `TypeVar` non borné (que mypy ne peut pas typer sans lui prêter
    d'attributs)."""

    id: int
    status: str
    error_message: Optional[str]
    rq_job_id: Optional[str]


# 10 min — assez large pour absorber un double-clic ou une reprise réseau
# (l'utilisateur ne clique pas deux fois à 9 minutes d'intervalle en
# pensant à la même tentative), jamais assez pour dédupliquer deux jobs
# VRAIMENT distincts lancés plus tard avec la même configuration.
_IDEMPOTENCY_TTL_SECONDS = 600


def resolve_idempotent_job_id(redis_conn: Redis, organization_id: int, request: Request) -> Optional[int]:
    """Retourne l'id du job DÉJÀ créé pour cette clé si `Idempotency-Key`
    (en-tête HTTP, optionnel) a déjà été vue pour cette organisation dans
    la fenêtre de dédoublonnage — l'appelant doit alors renvoyer ce job
    existant plutôt que d'en créer un nouveau. `None` si l'en-tête est
    absent, jamais vue, ou si Redis est indisponible (échec ouvert : mieux
    vaut risquer un doublon rare qu'un service d'entraînement en panne à
    cause d'une infra annexe).

    Scope par organisation, jamais global : une clé fournie par un client
    de l'organisation A ne doit jamais pouvoir résoudre vers un job de
    l'organisation B, même en cas de collision de valeur (peu probable
    avec un UUID côté client, mais le principe de moindre confiance
    s'applique à un en-tête entièrement contrôlé par l'appelant)."""
    key = request.headers.get("idempotency-key")
    if not key:
        return None
    try:
        raw = redis_conn.get(f"idempotency:{organization_id}:{key}")
    except Exception:
        logger.warning("[Idempotency] Redis indisponible — clé %r ignorée (création normale)", key, exc_info=True)
        return None
    if raw is None:
        return None
    try:
        # Le stub redis-py type `.get()` en `Awaitable[Any] | Any` (couvre
        # aussi le client asynchrone) — client toujours synchrone ici
        # (`Redis.from_url`, api/core/job_queue.py). Même correctif que
        # api/core/token_store.py::get_refresh_jti_owner.
        return int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def remember_idempotent_job_id(redis_conn: Redis, organization_id: int, request: Request, job_id: int) -> None:
    """À appeler juste après la création réussie d'un job — mémorise la
    correspondance clé → job pour la fenêtre de dédoublonnage. No-op si
    aucune `Idempotency-Key` n'a été fournie."""
    key = request.headers.get("idempotency-key")
    if not key:
        return
    try:
        redis_conn.setex(f"idempotency:{organization_id}:{key}", _IDEMPOTENCY_TTL_SECONDS, str(job_id))
    except Exception:
        logger.warning("[Idempotency] Redis indisponible — clé %r non mémorisée", key, exc_info=True)


#  Correctif Phase 2 (AUDIT_BACKEND_2026-08-23.md §F1) — le worker avale
# presque toutes les exceptions métier (traduites en `status="failed"` DANS
# la table applicative, jamais via un échec RQ), mais une exception qui
# survient hors de cette zone protégée (ex. `db.commit()` échoue parce que
# Postgres est momentanément indisponible) part réellement en échec RQ et
# tombe dans le `FailedJobRegistry` de RQ, dont la rétention par défaut
# (`failure_ttl`) est d'environ un an — jamais purgé, jamais surveillé.
# Resserré à 30 jours : assez pour investiguer un incident réel (le même
# ordre de grandeur que `prediction_retention_days`), sans accumuler
# indéfiniment des entrées mortes dans Redis. Une vraie file de rebut
# surveillée (alerting, tableau de bord) reste hors périmètre de cette
# phase — voir _backend/RAPPORT-FINAL.md, "ce qui a été laissé de côté".
_FAILED_JOB_TTL_SECONDS = 30 * 24 * 3600


def enqueue_or_mark_failed(db: Session, job: _JobRecord, queue: Queue, task_fn: Callable, job_timeout: int) -> None:
    """Enfile `job` sur `queue`. Si l'enfilage lève (Redis indisponible à
    cet instant précis), le job est marqué `"failed"` IMMÉDIATEMENT — dans
    la même requête, la même transaction logique — plutôt que laissé
    `"queued"` avec `rq_job_id=NULL` pour toujours (voir docstring du
    module). Lève ensuite un 503 explicite : le client sait que la
    création a échoué, au lieu de recevoir un 201 mensonger pour un job
    qui ne démarrera jamais."""
    try:
        rq_job = queue.enqueue(task_fn, job.id, job_timeout=job_timeout, failure_ttl=_FAILED_JOB_TTL_SECONDS)
    except Exception as exc:
        job.status = "failed"
        job.error_message = "Service de traitement temporairement indisponible — réessayez dans quelques instants."
        db.commit()
        logger.error(
            "[JobCreation] Échec d'enfilage pour job %s (%s) : %s", job.id, type(job).__name__, exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "FILE_INDISPONIBLE",
                "message": "Service de traitement temporairement indisponible — réessayez dans quelques instants.",
            },
        ) from exc
    job.rq_job_id = rq_job.id
    db.commit()
    db.refresh(job)
