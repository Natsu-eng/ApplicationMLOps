"""Limitation de débit par fenêtre glissante — Redis (H11, AUDIT_ROADMAP.md).

Aucune dépendance nouvelle (pas de `slowapi`) : un simple `INCR` + `EXPIRE`
Redis suffit pour une fenêtre glissante approximative, cohérent avec le
reste du projet (Redis déjà utilisé pour la file RQ, voir job_queue.py).
Échec ouvert si Redis est indisponible — jamais bloquant pour une connexion
légitime à cause d'une infra annexe en panne, même principe que le reste du
projet (voir api/main.py::lifespan pour la DB)."""
from __future__ import annotations

import ipaddress
import logging
from functools import lru_cache
from typing import Callable, List

from fastapi import HTTPException, Request, status
from redis import Redis

from api.core.config import get_settings

logger = logging.getLogger("datalab.rate_limit")


@lru_cache
def _trusted_proxy_networks() -> List["ipaddress._BaseNetwork"]:
    """Réseaux dont on accepte de faire confiance à `X-Real-IP` (voir
    `Settings.trusted_proxy_cidrs`). Mis en cache : lu à chaque requête sinon
    (une fois par worker gunicorn, la config ne change jamais en cours de
    vie du process)."""
    raw = get_settings().trusted_proxy_cidrs
    networks: List["ipaddress._BaseNetwork"] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            networks.append(ipaddress.ip_network(chunk, strict=False))
        except ValueError:
            logger.warning("[RateLimit] CIDR de confiance invalide ignoré : %r", chunk)
    return networks


def get_client_ip(request: Request) -> str:
    """IP cliente réelle, résistante à la topologie nginx→backend
    (AUDIT_BACKEND_2026-08-23.md §A.6) : `request.client.host` est l'IP du
    pair TCP direct — infalsifiable par le client, mais vaut l'IP du
    conteneur nginx pour TOUTE requête passée par le reverse proxy. On ne
    fait confiance à `X-Real-IP` (fixé par nginx à `$remote_addr`, jamais
    dérivé d'un en-tête client — voir nginx/templates/default.conf.template)
    QUE si ce pair TCP direct appartient à `trusted_proxy_cidrs` — sinon un
    client qui parlerait directement au backend (contournement du proxy)
    pourrait usurper n'importe quelle IP via cet en-tête.

    Ne consulte jamais `X-Forwarded-For` : nginx l'alimente avec
    `$proxy_add_x_forwarded_for`, qui AJOUTE à une valeur déjà fournie par
    le client — un en-tête à plusieurs valeurs où la plus fiable n'est pas
    toujours la première ni la dernière selon le nombre de sauts. Un seul
    reverse proxy dans cette topologie : `X-Real-IP` n'a pas cette
    ambiguïté."""
    peer = request.client.host if request.client else None
    if peer is None:
        return "inconnu"
    try:
        peer_addr = ipaddress.ip_address(peer)
    except ValueError:
        return peer
    if any(peer_addr in network for network in _trusted_proxy_networks()):
        forwarded = request.headers.get("x-real-ip")
        if forwarded:
            return forwarded.strip()
    return peer


class RateLimitBackendUnavailable(Exception):
    """Levée par `is_rate_limited` quand Redis est indisponible ET que
    l'appelant a demandé un échec FERMÉ (`fail_open=False`) — distincte
    d'un simple `True` (limite réellement atteinte), pour que l'appelant
    puisse répondre 503 (« on ne peut pas vérifier ») plutôt que 429
    (« vous avez dépassé la limite »), deux situations différentes pour
    l'utilisateur légitime."""


def is_rate_limited(
    redis_conn: Redis, key: str, max_attempts: int, window_seconds: int, fail_open: bool = True
) -> bool:
    """Incrémente le compteur `key` et retourne `True` si `max_attempts` est
    dépassé dans la fenêtre `window_seconds`.

    `fail_open` distingue deux politiques quand Redis est indisponible
    (Phase 1, AUDIT_BACKEND_2026-08-23.md §4) — un mode global unique était
    défendable pour `/login` (ne jamais bloquer une connexion légitime à
    cause d'une infra annexe en panne) mais dangereux pour `/explain`
    (charge un modèle torch à chaque appel) : qui met Redis à genoux
    déverrouillait l'endpoint le plus coûteux.
    - `fail_open=True` (authentification — login/register) : Redis down ⇒
      compte comme "non limité", jamais bloquant pour un utilisateur
      légitime.
    - `fail_open=False` (endpoints coûteux — upload, `/explain`) : Redis
      down ⇒ lève `RateLimitBackendUnavailable`, jamais confondu avec une
      limite réellement atteinte (voir la classe ci-dessus)."""
    try:
        # Le stub redis-py type `.incr()` en `Awaitable[Any] | Any` (couvre
        # aussi le client asynchrone) — client toujours synchrone ici
        # (`Redis.from_url`, api/core/job_queue.py).
        count: int = redis_conn.incr(key)  # type: ignore[assignment]
        if count == 1:
            redis_conn.expire(key, window_seconds)
        return count > max_attempts
    except RateLimitBackendUnavailable:
        raise
    except Exception:
        if fail_open:
            logger.warning("[RateLimit] Redis indisponible — limite ignorée (échec ouvert) pour %s", key, exc_info=True)
            return False
        logger.warning("[RateLimit] Redis indisponible — requête refusée (échec fermé) pour %s", key, exc_info=True)
        raise RateLimitBackendUnavailable(key) from None


def reset_rate_limit(redis_conn: Redis, key: str) -> None:
    """Efface le compteur — appelé après une connexion réussie, pour qu'un
    utilisateur légitime qui a tapé son mot de passe deux fois de travers ne
    reste pas pénalisé après avoir enfin réussi."""
    try:
        redis_conn.delete(key)
    except Exception:
        logger.warning("[RateLimit] Redis indisponible — reset ignoré pour %s", key, exc_info=True)


def rate_limit_dependency(
    action: str, max_attempts: int, window_seconds: int, fail_open: bool = True
) -> Callable[[Request], None]:
    """Fabrique une dépendance FastAPI de limitation de débit par IP
    cliente — Lot 1.4 (§C.2.7/§D.4, AUDIT_DATALAB_2026-08-16.md).

    Avant ce lot, seul `POST /auth/login` était limité (H11,
    AUDIT_ROADMAP.md) : `/register`, les uploads et `/explain` (charge un
    modèle torch à chaque appel) n'avaient aucune limite. Même mécanisme
    (fenêtre glissante Redis), généralisé plutôt que dupliqué — `action`
    distingue les compteurs entre endpoints (une IP qui épuise sa limite
    d'upload ne doit pas être bloquée sur l'inscription). `fail_open` : voir
    `is_rate_limited`."""

    def _dependency(request: Request) -> None:
        from api.core.job_queue import (
            redis_conn,  # import local — évite un cycle avec job_queue au chargement du module
        )

        client_ip = get_client_ip(request)
        key = f"rate_limit:{action}:{client_ip}"
        try:
            limited = is_rate_limited(redis_conn, key, max_attempts, window_seconds, fail_open=fail_open)
        except RateLimitBackendUnavailable as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": "LIMITE_INDISPONIBLE",
                    "message": "Service momentanément indisponible — réessayez dans quelques instants.",
                },
            ) from exc
        if limited:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "code": "TROP_DE_REQUETES",
                    "message": "Trop de requêtes — réessayez dans quelques minutes.",
                },
            )

    return _dependency
