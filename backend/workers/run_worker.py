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

Lot 4 (correctif I6) — `RQ_QUEUES` (liste séparée par des virgules parmi
"training", "vision", "analysis") choisit les files écoutées par CE
process ; absente ou vide, les 3 sont écoutées (comportement historique,
pratique pour un unique worker de développement). En production
(`docker-compose.yml`), chaque service `worker*` fixe `RQ_QUEUES` pour
dédier de la capacité aux jobs courts (clustering/dimensionnalité/
anomalies) — voir `api/core/job_queue.py` pour le détail des 3 files.
"""
from __future__ import annotations

import logging
import os
import sys

from api.core.config import get_settings
from api.core.observability import configure_logging

# Phase 3 (AUDIT_BACKEND_2026-08-23.md, Axe I) — avant ce correctif, ce
# process appelait `logging.basicConfig` avec un format texte libre,
# indépendant de `api/core/observability.py::configure_logging` (JSON,
# `request_id` corrélé) utilisé côté API : les logs du worker RQ étaient
# illisibles par un collecteur JSON (ELK, Loki...) et jamais corrélables à
# la requête HTTP d'origine, même quand `domains/*/worker.py` peuplait
# déjà le ContextVar `request_id_var` (`configure_logging` est ce qui lit
# ce ContextVar pour l'écrire dans chaque ligne — sans lui, le travail des
# workers n'avait aucun effet visible). Même fonction, même format, que
# le process API — un seul format de log dans tout le système.
configure_logging(get_settings().log_level)
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


def _resolve_queues() -> list:
    """`RQ_QUEUES=training,analysis` par ex. ; absente/vide → les 3 files
    (comportement historique, un seul process qui écoute tout)."""
    from api.core.job_queue import ALL_QUEUES

    names = [n.strip().lower() for n in os.environ.get("RQ_QUEUES", "").split(",") if n.strip()]
    if not names:
        return list(ALL_QUEUES.values())
    unknown = [n for n in names if n not in ALL_QUEUES]
    if unknown:
        raise ValueError(
            f"RQ_QUEUES contient des files inconnues : {', '.join(unknown)} "
            f"(attendu parmi {', '.join(ALL_QUEUES)})"
        )
    return [ALL_QUEUES[n] for n in names]


def _build_worker():
    from api.core.job_queue import redis_conn

    mode = _resolve_worker_mode()
    queues = _resolve_queues()
    logger.info("[Worker] Files écoutées : %s", ", ".join(q.name for q in queues))

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
        return PortableWorker(queues, connection=redis_conn)

    from rq import Worker

    logger.info("[Worker] Mode 'fork' (Worker classique, isolation par job, UnixSignalDeathPenalty)")
    return Worker(queues, connection=redis_conn)


def main() -> None:
    worker = _build_worker()
    worker.work()


if __name__ == "__main__":
    main()
