"""Point d'entrée du worker RQ.

Utilise `SimpleWorker` (exécution dans le process courant, pas de fork —
`os.fork()` n'existe pas sur Windows) et `TimerDeathPenalty` (timeout par
thread) plutôt que le défaut `UnixSignalDeathPenalty` (`signal.SIGALRM`, lui
aussi absent sur Windows). Ce choix est appliqué partout, pas seulement en
dev Windows : un seul point d'entrée identique en local et en conteneur
(voir docker-compose.yml, service `worker`) plutôt qu'un cas spécial caché.

Lancer :
    cd backend && python -m workers.run_worker
"""
from __future__ import annotations

import logging

from rq import SimpleWorker
from rq.timeouts import TimerDeathPenalty

from api.core.job_queue import redis_conn, training_queue

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class PortableWorker(SimpleWorker):
    """SimpleWorker + timeout thread-based — compatible Windows et Linux."""

    death_penalty_class = TimerDeathPenalty


def main() -> None:
    worker = PortableWorker([training_queue], connection=redis_conn)
    worker.work()


if __name__ == "__main__":
    main()
