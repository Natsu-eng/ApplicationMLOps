"""Configuration centralisée de l'API.

Toutes les valeurs sont surchargeables par variable d'environnement ou par le
fichier `backend/.env` (jamais commité — voir `backend/.env.example`).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/.env — deux niveaux au-dessus de ce fichier (backend/api/core/config.py)
_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"


class Settings(BaseSettings):
    """Paramètres applicatifs. Un seul point de vérité pour toute la config backend."""

    # Identité de l'application (affichée dans /api/health et la doc Swagger)
    app_name: str = "DataLab Pro"
    app_version: str = "0.1.0"
    environment: str = "development"  # development | production

    # Réseau — utilisé pour le CORS (voir api/main.py)
    frontend_url: str = "http://localhost:5173"

    # Base de données : Postgres en production, SQLite en dev si DATABASE_URL absent
    # (même convention que CIAM — voir backend/api/core/database.py)
    database_url: str = "sqlite:///./database/datalab.db"

    # Authentification — JWT HS256 (voir api/core/security.py)
    # ⚠️ OBLIGATOIRE à changer en production. Générer :
    #    python -c "import secrets; print(secrets.token_hex(64))"
    jwt_secret_key: str = "changez-cette-cle-en-production"

    # Journalisation
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",  # les variables Streamlit legacy (.env historique) ne doivent pas faire planter le chargement
    )


@lru_cache
def get_settings() -> Settings:
    """Instance unique des paramètres, mise en cache pour tout le process."""
    return Settings()
