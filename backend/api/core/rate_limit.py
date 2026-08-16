"""Limitation de débit par fenêtre glissante — Redis (H11, AUDIT_ROADMAP.md).

Aucune dépendance nouvelle (pas de `slowapi`) : un simple `INCR` + `EXPIRE`
Redis suffit pour une fenêtre glissante approximative, cohérent avec le
reste du projet (Redis déjà utilisé pour la file RQ, voir job_queue.py).
Échec ouvert si Redis est indisponible — jamais bloquant pour une connexion
légitime à cause d'une infra annexe en panne, même principe que le reste du
projet (voir api/main.py::lifespan pour la DB)."""
from __future__ import annotations

import logging
from typing import Callable

from fastapi import HTTPException, Request, status
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


def rate_limit_dependency(action: str, max_attempts: int, window_seconds: int) -> Callable[[Request], None]:
    """Fabrique une dépendance FastAPI de limitation de débit par IP
    cliente — Lot 1.4 (§C.2.7/§D.4, AUDIT_DATALAB_2026-08-16.md).

    Avant ce lot, seul `POST /auth/login` était limité (H11,
    AUDIT_ROADMAP.md) : `/register`, les uploads et `/explain` (charge un
    modèle torch à chaque appel) n'avaient aucune limite. Même mécanisme
    (fenêtre glissante Redis, échec ouvert), généralisé plutôt que
    dupliqué — `action` distingue les compteurs entre endpoints (une IP qui
    épuise sa limite d'upload ne doit pas être bloquée sur l'inscription)."""

    def _dependency(request: Request) -> None:
        from api.core.job_queue import redis_conn  # import local — évite un cycle avec job_queue au chargement du module

        client_ip = request.client.host if request.client else "inconnu"
        key = f"rate_limit:{action}:{client_ip}"
        if is_rate_limited(redis_conn, key, max_attempts, window_seconds):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "code": "TROP_DE_REQUETES",
                    "message": "Trop de requêtes — réessayez dans quelques minutes.",
                },
            )

    return _dependency
