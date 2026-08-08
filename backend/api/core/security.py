"""JWT (HS256) + hashing des mots de passe (bcrypt) — pattern repris de CIAM.

`sub` est toujours sérialisé en string dans le token (RFC 7519, python-jose
valide strictement) ; `exp` en timestamp Unix pour éviter les pièges de
comparaison de datetimes timezone-aware entre versions de python-jose.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import bcrypt
from jose import JWTError, jwt

from api.core.config import get_settings

logger = logging.getLogger("datalab.security")

_settings = get_settings()
_ALGORITHM = "HS256"
_TOKEN_TTL_MINUTES = 60 * 24  # 24 h

_DEV_DEFAULT_KEY = "changez-cette-cle-en-production"
if _settings.jwt_secret_key == _DEV_DEFAULT_KEY:
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


def create_access_token(data: dict, expires_minutes: int = _TOKEN_TTL_MINUTES) -> str:
    to_encode = data.copy()
    if "sub" in to_encode:
        to_encode["sub"] = str(to_encode["sub"])
    to_encode["exp"] = int(time.time()) + expires_minutes * 60
    return jwt.encode(to_encode, _settings.jwt_secret_key, algorithm=_ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, _settings.jwt_secret_key, algorithms=[_ALGORITHM])
    except JWTError:
        return None
