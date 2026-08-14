"""Limitation de débit par fenêtre glissante — Redis (H11, AUDIT_ROADMAP.md).

Aucune dépendance nouvelle (pas de `slowapi`) : un simple `INCR` + `EXPIRE`
Redis suffit pour une fenêtre glissante approximative, cohérent avec le
reste du projet (Redis déjà utilisé pour la file RQ, voir job_queue.py).
Échec ouvert si Redis est indisponible — jamais bloquant pour une connexion
légitime à cause d'une infra annexe en panne, même principe que le reste du
projet (voir api/main.py::lifespan pour la DB)."""
from __future__ import annotations

import logging

from redis import Redis

logger = logging.getLogger("datalab.rate_limit")


def is_rate_limited(redis_conn: Redis, key: str, max_attempts: int, window_seconds: int) -> bool:
    """Incrémente le compteur `key` et retourne `True` si `max_attempts` est
    dépassé dans la fenêtre `window_seconds`. Ne lève jamais d'exception :
    une erreur Redis journalisée compte comme "non limité" (échec ouvert)."""
    try:
        count = redis_conn.incr(key)
        if count == 1:
            redis_conn.expire(key, window_seconds)
        return count > max_attempts
    except Exception:
        logger.warning("[RateLimit] Redis indisponible — limite ignorée pour %s", key, exc_info=True)
        return False


def reset_rate_limit(redis_conn: Redis, key: str) -> None:
    """Efface le compteur — appelé après une connexion réussie, pour qu'un
    utilisateur légitime qui a tapé son mot de passe deux fois de travers ne
    reste pas pénalisé après avoir enfin réussi."""
    try:
        redis_conn.delete(key)
    except Exception:
        logger.warning("[RateLimit] Redis indisponible — reset ignoré pour %s", key, exc_info=True)
