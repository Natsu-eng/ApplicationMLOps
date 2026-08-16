"""Environnement Alembic — DataLab Pro (Lot 1.1, correctif C3).

L'URL de connexion n'est JAMAIS lue depuis `alembic.ini` : elle vient de
`get_settings().database_url` (même source que le reste de l'application,
`backend/.env`) pour ne jamais risquer une migration jouée contre la
mauvaise base par divergence de configuration.
"""
from __future__ import annotations

from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# Import du package `api` — nécessaire pour résoudre `api.core.*` sans
# dépendre du répertoire courant au moment de l'exécution (`alembic` peut
# être lancé depuis n'importe où si `backend/` est sur PYTHONPATH, mais on
# le garantit explicitement ici plutôt que de compter sur `prepend_sys_path`).
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.core.config import get_settings  # noqa: E402
from api.core.database import Base  # noqa: E402
from api.core import models  # noqa: E402,F401 — enregistre les 22 tables sur Base.metadata

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


_PLACEHOLDER_URL = "driver://user:pass@localhost/dbname"  # valeur par défaut d'alembic.ini


def _database_url() -> str:
    """Priorité à une URL explicitement fournie au `Config` Alembic (permet
    à `tests/test_alembic_migration.py` de piloter une base de test isolée
    sans toucher la config applicative globale) ; sinon, même source que le
    reste de l'application (`get_settings().database_url`) — jamais l'URL
    placeholder d'`alembic.ini`, pour ne jamais risquer une migration jouée
    contre la mauvaise base par divergence de configuration."""
    override = config.get_main_option("sqlalchemy.url")
    if override and override != _PLACEHOLDER_URL:
        return override
    url = get_settings().database_url
    # Railway/Heroku exposent parfois "postgres://" — SQLAlchemy 2.x exige "postgresql://"
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url


def run_migrations_offline() -> None:
    url = _database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = _database_url()
    connectable = engine_from_config(configuration, prefix="sqlalchemy.", poolclass=pool.NullPool)

    with connectable.connect() as connection:
        # render_as_batch : SQLite ne supporte pas ALTER COLUMN/DROP COLUMN
        # nativement — Alembic recrée la table via une table temporaire dans
        # ce mode. Sans effet sur Postgres (ALTER natif utilisé). Nécessaire
        # pour toute migration future au-delà du schéma initial (simple
        # CREATE TABLE, qui n'en a pas besoin) — activé dès maintenant pour
        # ne pas l'oublier le jour où une migration ALTER sera écrite.
        is_sqlite = connection.engine.dialect.name == "sqlite"
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=is_sqlite,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
