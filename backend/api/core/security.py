"""JWT (HS256) + hashing des mots de passe (bcrypt) — pattern repris de CIAM.

`sub` est toujours sérialisé en string dans le token (RFC 7519, python-jose
valide strictement) ; `exp`/`iat` en timestamp Unix pour éviter les pièges de
comparaison de datetimes timezone-aware entre versions de python-jose.

Cycle de vie des jetons (Phase 1, AUDIT_BACKEND_2026-08-23.md, Axe A) — avant
ce correctif, un seul JWT stateless de 24h sans `jti` ne pouvait jamais être
révoqué : un jeton volé restait valide jusqu'à expiration, `POST
/auth/logout` ne faisait rien côté serveur, et changer de mot de passe
n'invalidait aucune session existante. Deux jetons désormais :
- **access** : `_ACCESS_TOKEN_TTL_MINUTES` (court, 20 min) — c'est lui qui
  autorise chaque requête (`get_current_user`).
- **refresh** : `_REFRESH_TOKEN_TTL_DAYS` (14 jours), rotatif — sert
  uniquement à obtenir un nouveau couple access/refresh via `POST
  /auth/refresh`, jamais à autoriser une requête métier directement.

Révocation (voir `api/core/token_store.py`, Redis) :
- `jti` sur CHAQUE jeton (access et refresh) — permet de révoquer un jeton
  précis (logout d'UNE session) sans affecter les autres.
- `User.token_valid_after` (colonne DB, pas seulement Redis — survit à un
  `FLUSHALL` Redis) — tout jeton ACCESS dont `iat` est antérieur à cette
  date est rejeté, quel que soit son `jti` individuel. C'est le mécanisme de
  « toutes les sessions » (changement de mot de passe, réinitialisation) :
  un seul UPDATE, pas besoin d'énumérer chaque jeton émis.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Literal, Optional, Tuple

import bcrypt
from jose import JWTError, jwt

from api.core.config import get_settings

logger = logging.getLogger("datalab.security")

_settings = get_settings()
_ALGORITHM = "HS256"

# Correctif Phase 1 (AUDIT_BACKEND_2026-08-23.md §A.1) — 24h remplacé par 20
# min : un jeton d'accès volé (XSS, log, MITM) n'a plus qu'une fenêtre
# d'exploitation courte, le renouvellement transparent (refresh token) rend
# ce raccourcissement invisible pour un utilisateur légitime (voir
# frontend/src/api/client.ts).
_ACCESS_TOKEN_TTL_MINUTES = 20
_REFRESH_TOKEN_TTL_DAYS = 14

_DEV_DEFAULT_KEY = "changez-cette-cle-en-production"
if _settings.jwt_secret_key == _DEV_DEFAULT_KEY:
    if _settings.environment == "production":
        # Bloquant, pas juste journalisé : une clé par défaut en production
        # permettrait à quiconque de forger un token valide pour n'importe
        # quel utilisateur (la clé par défaut est publique, présente dans le
        # dépôt). Constaté lors de l'audit du 2026-08-14 (AUDIT_ROADMAP.md,
        # H3) : seul un avertissement journalisé existait, jamais bloquant.
        raise RuntimeError(
            "JWT_SECRET_KEY non définie en environnement de production — "
            "démarrage refusé. Générer une clé : "
            "python -c \"import secrets; print(secrets.token_hex(64))\""
        )
    logger.warning(
        "JWT_SECRET_KEY non définie — clé de développement utilisée. "
        "NE PAS utiliser cette clé en production !"
    )


def hash_password(plain: str) -> str:
    """Hash bcrypt (12 rounds) — un mot de passe en clair n'est jamais stocké."""
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


def access_token_ttl_seconds() -> int:
    return _ACCESS_TOKEN_TTL_MINUTES * 60


def refresh_token_ttl_seconds() -> int:
    return _REFRESH_TOKEN_TTL_DAYS * 24 * 60 * 60


def create_access_token(user_id: int, role: str, organization_id: int) -> Tuple[str, str]:
    """Retourne (token, jti) — l'appelant (router) stocke le `jti` s'il a
    besoin de le révoquer individuellement (logout d'une session précise)."""
    jti = uuid.uuid4().hex
    now = int(time.time())
    payload = {
        "sub": str(user_id),
        "role": role,
        "org": organization_id,
        "type": "access",
        "jti": jti,
        "iat": now,
        "exp": now + access_token_ttl_seconds(),
    }
    return jwt.encode(payload, _settings.jwt_secret_key, algorithm=_ALGORITHM), jti


def create_refresh_token(user_id: int) -> Tuple[str, str]:
    """Retourne (token, jti) — le jti est la clé de vérité côté Redis
    (`token_store.py::store_refresh_jti`) : le contenu du JWT lui-même n'est
    JAMAIS suffisant pour accepter un refresh, sa présence dans Redis fait
    foi (permet la révocation ET la rotation à usage unique)."""
    jti = uuid.uuid4().hex
    now = int(time.time())
    payload = {
        "sub": str(user_id),
        "type": "refresh",
        "jti": jti,
        "iat": now,
        "exp": now + refresh_token_ttl_seconds(),
    }
    return jwt.encode(payload, _settings.jwt_secret_key, algorithm=_ALGORITHM), jti


def decode_token(token: str, expected_type: Optional[Literal["access", "refresh"]] = None) -> Optional[dict]:
    """Décode et valide signature + expiration. `expected_type` empêche
    qu'un jeton refresh (longue durée, usage restreint) soit accepté là où
    un jeton access est attendu, et inversement — avant ce correctif,
    `decode_token` ne distinguait pas les deux types de jeton."""
    try:
        payload = jwt.decode(token, _settings.jwt_secret_key, algorithms=[_ALGORITHM])
    except JWTError:
        return None
    if expected_type is not None and payload.get("type") != expected_type:
        return None
    return payload
