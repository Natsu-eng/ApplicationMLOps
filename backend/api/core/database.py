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


class SchemaMismatchError(RuntimeError):
    """La base est pré-Alembic (tables déjà présentes, pas de table
    `alembic_version`) mais son schéma ne correspond pas exactement à ce
    que la révision initiale est censée avoir créé — voir
    `_schema_matches_metadata`. Ne JAMAIS estampiller `head` dans ce cas :
    ce serait mentir sur l'état réel de la base et fermer le seul chemin
    de réparation (un `alembic upgrade head` sur une base honnêtement non
    stampée aurait au moins tenté de créer ce qui manque)."""


def _schema_matches_metadata(target_engine) -> list[str]:
    """Compare le schéma RÉEL de `target_engine` à `Base.metadata` (les 22
    tables/colonnes attendues, exactement ce que la révision initiale crée
    — même source, donc comparaison fiable sans dupliquer la définition du
    schéma attendu). Retourne la liste des écarts (vide = conforme).

    Correctif (retour utilisateur) : le critère précédent ("alembic_version
    absente + table organizations présente") ne prouvait rien sur les 21
    AUTRES tables — une base restaurée depuis une sauvegarde antérieure à
    l'ajout du pilier Vision, par exemple, aurait été estampillée "à jour"
    avec des tables manquantes, sans qu'aucune migration ne les crée
    jamais. Vérifie maintenant table par table ET colonne par colonne."""
    inspector = inspect(target_engine)
    actual_tables = set(inspector.get_table_names())
    problems: list[str] = []
    for table in Base.metadata.sorted_tables:
        if table.name not in actual_tables:
            problems.append(f"table manquante : {table.name}")
            continue
        actual_columns = {col["name"] for col in inspector.get_columns(table.name)}
        missing_columns = [c.name for c in table.columns if c.name not in actual_columns]
        for column_name in missing_columns:
            problems.append(f"colonne manquante : {table.name}.{column_name}")
    return problems


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
      échouerait (`CREATE TABLE` sur une table déjà existante). SI son
      schéma correspond exactement à ce qu'attend la révision initiale
      (`_schema_matches_metadata`), on la marque comme déjà appliquée via
      `alembic stamp head`, SANS exécuter son SQL — les données existantes
      ne sont jamais touchées. SINON (schéma partiel — ex. une base
      restaurée depuis une sauvegarde antérieure à un lot qui a ajouté des
      tables) : `SchemaMismatchError`, jamais un stamp à l'aveugle qui
      fermerait le seul chemin de réparation. Testé avec de vraies données
      dans `tests/test_alembic_migration.py`.

    **Vérification finale, après les deux chemins ci-dessus** (correctif —
    incident réel) : un stamp erroné effectué par une version ANTÉRIEURE,
    moins stricte, de cette fonction (avant l'ajout de
    `_schema_matches_metadata`) laisse une base marquée `head` alors que
    son schéma réel diverge — `vision_anomaly_models` a ainsi été stampée
    `head` sans les colonnes `n_calibration`/`n_evaluation`, ajoutées au
    modèle après coup. Une fois `alembic_version` posée à `head`, chaque
    redémarrage suivant passait par la branche `else` (`upgrade head`,
    no-op puisque déjà à `head`) SANS jamais revérifier le schéma réel :
    l'écart restait invisible jusqu'au premier appel API en 500. Le
    contrôle est donc désormais répété après `stamp` ET après `upgrade`,
    pas seulement avant un stamp — seule façon de détecter un schéma qui a
    dérivé APRÈS avoir atteint `head` (stamp historique erroné, migration
    partiellement appliquée, intervention manuelle sur la base).
    """
    from alembic import command

    target_engine = create_engine(db_url) if db_url else engine
    cfg = _alembic_config(db_url)
    inspector = inspect(target_engine)
    existing_tables = set(inspector.get_table_names())
    is_pre_alembic_database = "alembic_version" not in existing_tables and "organizations" in existing_tables

    def _fail(message: str, problems: list[str]) -> None:
        if db_url:
            target_engine.dispose()
        raise SchemaMismatchError(message + " Écarts détectés : " + "; ".join(problems))

    if is_pre_alembic_database:
        problems = _schema_matches_metadata(target_engine)
        if problems:
            _fail(
                "Base pré-Alembic détectée, mais son schéma ne correspond pas exactement à la révision "
                "initiale — refus de démarrer plutôt que d'estampiller à l'aveugle (ce qui fermerait "
                "définitivement le chemin de réparation).",
                problems,
            )
        command.stamp(cfg, "head")
        logger.info(
            "[DB] Base pré-Alembic détectée (tables déjà présentes, jamais migrée, schéma conforme) — "
            "révision initiale marquée appliquée sans rejeu"
        )
    else:
        command.upgrade(cfg, "head")

    # Filet de sécurité général (voir docstring) : reconfirme la conformité
    # une fois `head` atteint, quel que soit le chemin emprunté ci-dessus.
    post_migration_problems = _schema_matches_metadata(target_engine)
    if post_migration_problems:
        _fail(
            "La base est marquée `head` mais son schéma réel ne correspond pas à celui attendu — refus de "
            "démarrer plutôt que de servir du trafic contre un schéma incomplet. Génère et applique une "
            "migration de rattrapage (`alembic revision --autogenerate`) pour combler l'écart.",
            post_migration_problems,
        )

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
