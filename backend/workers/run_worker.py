"""Point d'entrée du worker RQ.

Lot 1.3 (correctif C7, AUDIT_DATALAB_2026-08-16.md) — avant ce lot,
`SimpleWorker` (exécution dans le process courant, sans fork) était utilisé
PARTOUT, y compris en production Linux : un seul job à la fois pour TOUTES
les organisations, et un `MemoryError`/segfault (LightGBM, torch) tuait le
worker entier plutôt que le seul job en cours.

Sous Linux, `Worker` classique (fork par job, isolation réelle — un crash
ne tue que l'enfant) est maintenant le défaut. `SimpleWorker` reste
disponible en développement Windows (`os.fork()` n'existe pas sur
Windows), piloté par la variable d'environnement `RQ_WORKER_MODE` :
- "fork" (défaut sous Linux/macOS) : `Worker` + `UnixSignalDeathPenalty`
  (SIGALRM) — interrompt RÉELLEMENT du code natif bloquant (boucle C de
  LightGBM, forward torch), contrairement au thread-based ci-dessous.
- "simple" (défaut sous Windows, et le SEUL mode utilisable sous Windows) :
  `SimpleWorker` + `TimerDeathPenalty` (timeout par thread — ne peut PAS
  interrompre du code natif bloquant, théorique sur les cas qui en
  auraient le plus besoin : préservé uniquement pour que le développement
  local sous Windows reste possible, jamais recommandé en production).

Lancer :
    cd backend && python -m workers.run_worker
Pour plusieurs workers en parallèle (isolation par organisation) :
    voir docker-compose.yml, service `worker`, `docker compose up --scale worker=N`.
"""
from __future__ import annotations

import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("datalab.worker")


def _resolve_worker_mode() -> str:
    """`RQ_WORKER_MODE=fork|simple` explicite en priorité ; sinon détecté
    depuis la plateforme — "simple" est le SEUL mode fonctionnel sous
    Windows (`os.fork()` inexistant), jamais un choix arbitraire."""
    override = os.environ.get("RQ_WORKER_MODE", "").strip().lower()
    if override in ("fork", "simple"):
        return override
    if override:
        logger.warning("[Worker] RQ_WORKER_MODE=%r inconnu (attendu 'fork' ou 'simple') — ignoré", override)
    return "simple" if sys.platform.startswith("win") else "fork"


def _build_worker():
    from api.core.job_queue import redis_conn, training_queue

    mode = _resolve_worker_mode()

    if mode == "simple":
        from rq import SimpleWorker
        from rq.timeouts import TimerDeathPenalty

        class PortableWorker(SimpleWorker):
            """SimpleWorker + timeout thread-based — compatible Windows,
            jamais la production (voir docstring du module)."""

            death_penalty_class = TimerDeathPenalty

        logger.warning(
            "[Worker] Mode 'simple' (SimpleWorker, sans fork) — un seul job à la fois pour "
            "toutes les organisations, timeout non fiable sur code natif bloquant. "
            "Réservé au développement Windows, jamais recommandé en production."
        )
        return PortableWorker([training_queue], connection=redis_conn)

    from rq import Worker

    logger.info("[Worker] Mode 'fork' (Worker classique, isolation par job, UnixSignalDeathPenalty)")
    return Worker([training_queue], connection=redis_conn)


def main() -> None:
    worker = _build_worker()
    worker.work()


if __name__ == "__main__":
    main()
