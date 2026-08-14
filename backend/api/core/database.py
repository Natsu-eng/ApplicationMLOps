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
    """Ajoute une colonne à une table déjà existante si elle est absente —
    migration idempotente maison plutôt qu'Alembic (voir ARCHITECTURE.md).
    `create_all()` ne modifie jamais une table existante, seulement les
    tables manquantes : sans ça, ajouter un champ à un modèle ORM casserait
    silencieusement toute base créée avant l'ajout."""
    inspector = inspect(engine)
    if table not in inspector.get_table_names():
        return  # la table sera créée avec la bonne colonne par create_all()
    existing_columns = {col["name"] for col in inspector.get_columns(table)}
    if column in existing_columns:
        return
    with engine.begin() as conn:
        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {column_sql_type}"))
    logger.info("[DB] Migration : colonne %s.%s ajoutée", table, column)


def init_db() -> None:
    """Crée les tables déclarées par les modèles ORM enregistrés sur `Base`,
    puis applique les migrations additives connues (voir `_add_column_if_missing`)."""
    # Import local (et non en tête de module) pour éviter l'import circulaire :
    # api.core.models importe déjà `Base` depuis ce fichier.
    from api.core.models import (  # noqa: F401
        AuditLog,
        ClusterCandidateRecord,
        ClusterModel,
        ClusteringJob,
        Dataset,
        MLModel,
        ModelCandidate,
        Organization,
        TrainingJob,
        User,
    )

    Base.metadata.create_all(bind=engine)
    _add_column_if_missing("ml_models", "feature_schema_json", "TEXT")
    _add_column_if_missing("ml_models", "evaluation_json", "TEXT")
    _add_column_if_missing("ml_models", "feature_engineering_json", "TEXT")
    _add_column_if_missing("training_jobs", "feature_engineering_json", "TEXT")
    # Lot Explicabilité globale — beeswarm SHAP, importance par permutation,
    # calibration, courbe d'apprentissage. NULL sur les modèles déjà
    # entraînés (rétrocompat) : le frontend affiche "réentraînez pour
    # l'obtenir" plutôt que de planter.
    _add_column_if_missing("ml_models", "shap_beeswarm_json", "TEXT")
    _add_column_if_missing("ml_models", "permutation_importance_json", "TEXT")
    _add_column_if_missing("ml_models", "calibration_json", "TEXT")
    _add_column_if_missing("ml_models", "learning_curve_json", "TEXT")
    # Lot 9 — registre de modèles versionné (stage/promoted_at, NULL = jamais promu).
    _add_column_if_missing("ml_models", "stage", "VARCHAR(20)")
    _add_column_if_missing("ml_models", "promoted_at", "TIMESTAMP")
    # Durcissement SaaS (AUDIT_ROADMAP.md, H2) — horodatage du dernier
    # signal de vie écrit par le worker (job.status="running" ou chaque
    # étape de progression). Permet de repérer un job "running" dont le
    # worker a crashé sans jamais le marquer "failed" (services/job_watchdog.py).
    _add_column_if_missing("training_jobs", "progress_updated_at", "TIMESTAMP")
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
