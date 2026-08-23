"""Révocation de jetons — Redis (Phase 1, AUDIT_BACKEND_2026-08-23.md, Axe A).

Deux registres, tous deux dans Redis (déjà une dépendance du projet — voir
`job_queue.py`, même principe que `rate_limit.py`) :
- `revoked_jti:<jti>` — un jeton ACCESS révoqué individuellement (logout
  d'une session précise). TTL = durée de vie restante du jeton : inutile de
  le garder en mémoire plus longtemps qu'il n'aurait de toute façon été
  valide.
- `refresh_jti:<jti>` → id utilisateur — source de vérité pour un jeton
  REFRESH : sa seule présence en Redis le rend utilisable. Supprimé à la
  rotation (usage unique) ou à la révocation explicite.
- `refresh_jtis_by_user:<user_id>` — SET des `jti` refresh actuellement
  valides pour un utilisateur, uniquement pour la révocation en masse
  (changement de mot de passe). Ne fait jamais foi individuellement — un
  membre du set déjà expiré individuellement (`refresh_jti:<jti>` disparu)
  est simplement un no-op à la révocation, jamais une erreur.

Échec ouvert si Redis est indisponible pour la VÉRIFICATION d'une
révocation (mieux vaut un jeton pas encore expiré accepté à tort qu'un
service d'authentification qui tombe en panne parce que Redis est
indisponible — même principe que `rate_limit.py`) ; mais échec **loggé bruyamment**
si l'ÉCRITURE d'une révocation échoue (un logout qui échoue silencieusement
donnerait une fausse impression de sécurité à l'utilisateur)."""
from __future__ import annotations

import logging
from typing import Optional

from redis import Redis

logger = logging.getLogger("datalab.token_store")

_ACCESS_REVOKED_PREFIX = "revoked_jti:"
_REFRESH_JTI_PREFIX = "refresh_jti:"
_REFRESH_SET_PREFIX = "refresh_jtis_by_user:"


def revoke_access_jti(redis_conn: Redis, jti: str, ttl_seconds: int) -> None:
    if ttl_seconds <= 0:
        return  # déjà expiré naturellement, rien à révoquer
    try:
        redis_conn.setex(f"{_ACCESS_REVOKED_PREFIX}{jti}", ttl_seconds, "1")
    except Exception:
        logger.error("[TokenStore] Échec de révocation du jeton access %s", jti, exc_info=True)


def is_access_jti_revoked(redis_conn: Redis, jti: str) -> bool:
    try:
        return bool(redis_conn.exists(f"{_ACCESS_REVOKED_PREFIX}{jti}"))
    except Exception:
        logger.warning("[TokenStore] Redis indisponible — révocation ignorée pour %s", jti, exc_info=True)
        return False


def store_refresh_jti(redis_conn: Redis, user_id: int, jti: str, ttl_seconds: int) -> None:
    try:
        redis_conn.setex(f"{_REFRESH_JTI_PREFIX}{jti}", ttl_seconds, str(user_id))
        redis_conn.sadd(f"{_REFRESH_SET_PREFIX}{user_id}", jti)
    except Exception:
        logger.error("[TokenStore] Échec d'enregistrement du refresh token %s", jti, exc_info=True)


def get_refresh_jti_owner(redis_conn: Redis, jti: str) -> Optional[int]:
    """Retourne l'id utilisateur propriétaire si le refresh jti est encore
    valide, sinon None (expiré, jamais émis, ou déjà consommé/révoqué)."""
    try:
        raw = redis_conn.get(f"{_REFRESH_JTI_PREFIX}{jti}")
    except Exception:
        logger.warning("[TokenStore] Redis indisponible — refresh token %s traité comme invalide", jti, exc_info=True)
        return None
    if raw is None:
        return None
    try:
        # Le stub redis-py type `.get()` en `Awaitable[Any] | Any` pour
        # couvrir aussi le client asynchrone — ce projet n'utilise que le
        # client synchrone (`Redis.from_url`, api/core/job_queue.py),
        # `raw` est donc toujours `bytes` en pratique à ce point.
        return int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def revoke_refresh_jti(redis_conn: Redis, user_id: int, jti: str) -> None:
    try:
        redis_conn.delete(f"{_REFRESH_JTI_PREFIX}{jti}")
        redis_conn.srem(f"{_REFRESH_SET_PREFIX}{user_id}", jti)
    except Exception:
        logger.error("[TokenStore] Échec de révocation du refresh token %s", jti, exc_info=True)


def revoke_all_refresh_tokens(redis_conn: Redis, user_id: int) -> None:
    """Révocation en masse — changement de mot de passe (Phase 1B) ou
    réinitialisation. Ne révoque QUE les refresh tokens ; les jetons access
    déjà émis sont eux invalidés par `User.token_valid_after` (voir
    security.py), pas par cette fonction — les deux mécanismes sont
    complémentaires et doivent être appelés ensemble par l'appelant."""
    try:
        members = redis_conn.smembers(f"{_REFRESH_SET_PREFIX}{user_id}")
        for jti in members:  # type: ignore[union-attr]  # client synchrone — voir get_refresh_jti_owner ci-dessus
            jti_str = jti.decode("utf-8") if isinstance(jti, bytes) else jti
            redis_conn.delete(f"{_REFRESH_JTI_PREFIX}{jti_str}")
        redis_conn.delete(f"{_REFRESH_SET_PREFIX}{user_id}")
    except Exception:
        logger.error("[TokenStore] Échec de révocation en masse pour l'utilisateur %s", user_id, exc_info=True)
