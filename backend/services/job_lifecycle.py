"""Aides partagées sur le cycle de vie d'un job de fond (RQ) — Lot 7, §J.2.

Extrait de la logique d'annulation RQ best-effort, jusqu'ici dupliquée à
l'identique dans les 6 routers de job (`training.py`, `clustering.py`,
`dimensionality.py`, `anomalies.py`, `vision_classification.py`,
`vision_anomalies.py`) à la fois dans `DELETE /jobs/{id}` et, à partir de
ce lot, dans le nouvel endpoint `POST /jobs/{id}/cancel`. Extraction
justifiée par CE lot (sert directement le nouvel endpoint, pas un
refactoring opportuniste hors périmètre) — sans elle, la même logique
existerait en 12 exemplaires au lieu de 6."""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("datalab.job_lifecycle")

ACTIVE_STATUSES = ("queued", "running")

# Message utilisateur commun aux 6 types de job — jamais un détail
# technique (voir règle produit sur les messages français actionnables).
CANCELLED_MESSAGE = "Annulé par l'utilisateur."


def try_cancel_rq_job(rq_job_id: Optional[str], queue: Any) -> None:
    """Annule et supprime le job RQ sous-jacent, best-effort — un job déjà
    terminé, déjà absent de Redis, ou une erreur de connexion ne doit
    jamais empêcher l'opération côté base de données (suppression ou
    annulation) de réussir."""
    if not rq_job_id:
        return
    try:
        from rq.job import Job as RQJob

        rq_job = RQJob.fetch(rq_job_id, connection=queue.connection)
        rq_job.cancel()
        rq_job.delete()
    except Exception as exc:
        logger.info("Annulation RQ best-effort échouée pour %s : %s", rq_job_id, exc)
