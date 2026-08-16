"""Connexion base de données (SQLAlchemy 2.x) — PostgreSQL en production, SQLite en
développement si `DATABASE_URL` n'est pas défini.

Aucun modèle ORM n'est encore déclaré au Lot 0 : `Base` est le point d'ancrage sur
lequel les entités métier (User, Dataset, TrainingJob...) viendront s'enregistrer
à partir du Lot 1, sans que ce fichier ait besoin d'être modifié.
"""
from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from api.core.config import get_settings

logger = logging.getLogger("datalab.db")

_settings = get_settings()
_db_url = _settings.database_url

# Railway/Heroku exposent parfois "postgres://" — SQLAlchemy 2.x exige "postgresql://"
if _db_url.startswith("postgres://"):
    _db_url = _db_url.replace("postgres://", "postgresql://", 1)

_is_sqlite = _db_url.startswith("sqlite")

# En SQLite, le dossier cible doit exister avant la première connexion
if _is_sqlite:
    _sqlite_path = _db_url.replace("sqlite:///", "", 1)
    if _sqlite_path not in ("", ":memory:"):
        Path(_sqlite_path).resolve().parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    _db_url,
    connect_args={"check_same_thread": False} if _is_sqlite else {},
    pool_pre_ping=not _is_sqlite,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Classe de base des modèles ORM (héritée par les entités métier à partir du Lot 1)."""


def get_db():
    """Dépendance FastAPI : ouvre une session par requête, la ferme systématiquement."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _add_column_if_missing(table: str, column: str, column_sql_type: str) -> None:
    """GELÉ depuis le Lot 1.1 (correctif C3, AUDIT_DATALAB_2026-08-16.md) —
    remplacé par Alembic (`run_migrations()` ci-dessous). Ne plus y ajouter
    d'appel : toute évolution de schéma passe désormais par
    `alembic revision --autogenerate`, voir `backend/alembic/versions/`.
    Conservée uniquement pour référence historique (les colonnes qu'elle
    ajoutait sont maintenant des colonnes normales de `api/core/models.py`,
    créées par la révision initiale `594bce594adf`)."""
    inspector = inspect(engine)
    if table not in inspector.get_table_names():
        return  # la table sera créée avec la bonne colonne par create_all()
    existing_columns = {col["name"] for col in inspector.get_columns(table)}
    if column in existing_columns:
        return
    with engine.begin() as conn:
        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {column_sql_type}"))
    logger.info("[DB] Migration : colonne %s.%s ajoutée", table, column)


def _alembic_config(db_url: str | None = None):
    """Config Alembic pointée sur `backend/alembic.ini` — utilisée aussi
    bien au démarrage de l'API (`run_migrations`) que par les scripts/tests
    qui doivent piloter Alembic par programme plutôt qu'en ligne de commande.
    `db_url` explicite (voir `alembic/env.py::_database_url`) : permet aux
    tests de pointer une base isolée sans toucher `DATABASE_URL`/la config
    applicative globale."""
    from alembic.config import Config

    backend_dir = Path(__file__).resolve().parent.parent.parent
    cfg = Config(str(backend_dir / "alembic.ini"))
    cfg.set_main_option("script_location", str(backend_dir / "alembic"))
    if db_url:
        cfg.set_main_option("sqlalchemy.url", db_url)
    return cfg


def run_migrations(db_url: str | None = None) -> None:
    """Amène le schéma à la révision `head` — remplace
    `Base.metadata.create_all()` + `_add_column_if_missing` (Lot 1.1,
    correctif C3). `db_url` optionnel : par défaut la base applicative
    (`_db_url`/`engine` de ce module) ; permet aux tests de cibler une base
    isolée (`tests/test_alembic_migration.py`).

    Deux cas distincts, jamais confondus :
    - **Base neuve** (aucune des 22 tables n'existe) : `alembic upgrade
      head` crée tout le schéma depuis la révision initiale.
    - **Base déjà en service** (tables déjà présentes, pas de table
      `alembic_version` — c'est-à-dire toute base créée avant ce lot par
      l'ancien `Base.metadata.create_all()`) : rejouer la révision initiale
      échouerait (`CREATE TABLE` sur une table déjà existante). On marque
      donc cette révision comme déjà appliquée via `alembic stamp head`,
      SANS exécuter son SQL — les données existantes ne sont jamais
      touchées. Testé avec de vraies données dans
      `tests/test_alembic_migration.py::test_existing_pre_alembic_database_is_stamped_not_recreated`.
    """
    from alembic import command

    target_engine = create_engine(db_url) if db_url else engine
    cfg = _alembic_config(db_url)
    inspector = inspect(target_engine)
    existing_tables = set(inspector.get_table_names())
    is_pre_alembic_database = "alembic_version" not in existing_tables and "organizations" in existing_tables

    if is_pre_alembic_database:
        command.stamp(cfg, "head")
        logger.info(
            "[DB] Base pré-Alembic détectée (tables déjà présentes, jamais migrée) — "
            "révision initiale marquée appliquée sans rejeu"
        )
    else:
        command.upgrade(cfg, "head")

    if db_url:
        target_engine.dispose()


def init_db() -> None:
    """Crée/met à jour le schéma via Alembic (voir `run_migrations()`)."""
    # Import local (et non en tête de module) pour éviter l'import circulaire :
    # api.core.models importe déjà `Base` depuis ce fichier. Import conservé
    # ici (même si `run_migrations` ne s'appuie plus sur `Base.metadata`
    # directement) car `env.py` d'Alembic importe lui aussi `api.core.models`
    # — le garder ici documente que TOUTES les tables doivent être
    # enregistrées avant toute opération de schéma.
    from api.core import models  # noqa: F401

    run_migrations()
    logger.info("[DB] Prête (%s)", "SQLite" if _is_sqlite else "PostgreSQL")


def check_connection() -> bool:
    """Vérifie que la base de données répond — consommé par GET /api/health."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        logger.exception("[DB] Connexion impossible")
        return False
