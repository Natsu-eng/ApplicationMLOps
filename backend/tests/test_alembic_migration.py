"""Tests du chemin de migration Alembic (Lot 1.1, correctif C3,
AUDIT_DATALAB_2026-08-16.md) — base neuve, downgrade réel, et surtout le
cas qui compte le plus : une base DÉJÀ EN SERVICE (créée par l'ancien
`Base.metadata.create_all()`, jamais migrée) passe à Alembic sans perte de
données. Chaque test utilise sa propre base SQLite isolée (`tmp_path`),
jamais la base de test partagée de `conftest.py`."""
from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from api.core.database import Base, SchemaMismatchError, _alembic_config, run_migrations

# Enregistre les 22 tables sur Base.metadata (même import que env.py) —
# sans ça, Base.metadata.create_all() ne créerait rien.
from api.core import models  # noqa: F401

EXPECTED_TABLES = {
    "organizations", "users", "audit_logs", "datasets", "vision_datasets",
    "anomaly_jobs", "clustering_jobs", "dimensionality_jobs", "training_jobs",
    "vision_anomaly_jobs", "vision_classification_jobs", "anomaly_models",
    "anomaly_observations", "cluster_candidates", "cluster_models",
    "dimensionality_models", "dimensionality_points", "ml_models",
    "model_candidates", "vision_anomaly_models", "vision_classification_models",
    "vision_anomaly_examples",
}


def _sqlite_url(tmp_path: Path, name: str) -> str:
    return f"sqlite:///{(tmp_path / name).as_posix()}"


def test_fresh_database_upgrade_creates_all_tables(tmp_path):
    """Base neuve (aucune table) : `alembic upgrade head` doit créer les 22
    tables ET la table `alembic_version` (preuve que la révision a bien été
    appliquée, pas juste tentée)."""
    url = _sqlite_url(tmp_path, "fresh.db")

    run_migrations(db_url=url)

    engine = create_engine(url)
    tables = set(inspect(engine).get_table_names())
    assert EXPECTED_TABLES <= tables
    assert "alembic_version" in tables
    engine.dispose()


def test_downgrade_actually_drops_all_tables(tmp_path):
    """Exigence explicite : aucune migration sans `downgrade` réellement
    testé. Après upgrade puis downgrade vers `base`, les 22 tables doivent
    avoir disparu — pas seulement un `downgrade()` qui ne fait rien."""
    url = _sqlite_url(tmp_path, "downgrade.db")
    run_migrations(db_url=url)

    cfg = _alembic_config(url)
    command.downgrade(cfg, "base")

    engine = create_engine(url)
    tables = set(inspect(engine).get_table_names())
    assert not (EXPECTED_TABLES & tables), f"Tables non supprimées par le downgrade : {EXPECTED_TABLES & tables}"
    engine.dispose()


def test_existing_pre_alembic_database_is_stamped_not_recreated(tmp_path):
    """LE test central du correctif C3 : une base déjà en service (tables
    créées par l'ancien `Base.metadata.create_all()`, jamais migrée) ne doit
    JAMAIS voir sa révision initiale rejouée (elle échouerait — CREATE TABLE
    sur une table déjà existante) ni perdre de données. `run_migrations()`
    doit détecter ce cas et utiliser `stamp`, pas `upgrade`."""
    url = _sqlite_url(tmp_path, "existing.db")
    engine = create_engine(url)

    # Simule l'état "avant Alembic" : tables créées directement, comme le
    # faisait l'ancien init_db() avant ce lot.
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        session.execute(text("INSERT INTO organizations (name) VALUES ('Bureau existant')"))
        session.commit()

    tables_before = set(inspect(engine).get_table_names())
    assert "alembic_version" not in tables_before  # confirme l'état pré-Alembic
    engine.dispose()

    # Ne doit lever aucune exception (contrairement à un `upgrade head` naïf
    # sur cette même base — voir test suivant).
    run_migrations(db_url=url)

    engine = create_engine(url)
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT name FROM organizations")).fetchall()
    assert [r[0] for r in rows] == ["Bureau existant"]  # donnée préexistante intacte

    inspector = inspect(engine)
    assert "alembic_version" in inspector.get_table_names()
    with engine.connect() as conn:
        version = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
    assert version is not None  # marquée à head, pas laissée vide
    engine.dispose()


def test_pre_alembic_database_with_missing_table_is_refused_not_stamped(tmp_path):
    """Correctif (retour utilisateur) : l'ancien critère de détection
    ("alembic_version absente + organizations présente") ne prouvait rien
    sur les 21 AUTRES tables — une base restaurée depuis une sauvegarde
    antérieure à l'ajout du pilier Vision, par exemple, aurait été
    estampillée "à jour" avec des tables manquantes, sans qu'aucune
    migration ne les crée jamais. `run_migrations()` doit maintenant
    comparer le schéma réel à `Base.metadata` table par table et refuser
    de démarrer si ça diverge, avec un diagnostic précis — jamais un
    stamp à l'aveugle."""
    url = _sqlite_url(tmp_path, "missing_table.db")
    engine = create_engine(url)

    # Simule une base pré-Alembic à laquelle il manque une table entière
    # (ex. restaurée depuis une sauvegarde antérieure au pilier Vision) :
    # crée tout le schéma attendu PUIS supprime une table.
    Base.metadata.create_all(bind=engine)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE vision_anomaly_examples"))

    tables_before = set(inspect(engine).get_table_names())
    assert "alembic_version" not in tables_before
    assert "vision_anomaly_examples" not in tables_before
    engine.dispose()

    with pytest.raises(SchemaMismatchError, match="vision_anomaly_examples"):
        run_migrations(db_url=url)

    # Confirme que rien n'a été estampillé (le diagnostic reste valable si
    # on relance après avoir réparé la base à la main).
    engine = create_engine(url)
    tables_after = set(inspect(engine).get_table_names())
    assert "alembic_version" not in tables_after
    engine.dispose()


def test_pre_alembic_database_with_missing_column_is_refused_not_stamped(tmp_path):
    """Variante colonne (demandée explicitement) du test précédent : une
    base pré-Alembic à laquelle il manque une SEULE colonne (pas une table
    entière) doit être refusée exactement de la même façon."""
    url = _sqlite_url(tmp_path, "missing_column.db")
    engine = create_engine(url)

    Base.metadata.create_all(bind=engine)
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE vision_anomaly_models DROP COLUMN n_calibration"))

    tables_before = set(inspect(engine).get_table_names())
    assert "alembic_version" not in tables_before
    engine.dispose()

    with pytest.raises(SchemaMismatchError, match="vision_anomaly_models.n_calibration"):
        run_migrations(db_url=url)

    engine = create_engine(url)
    assert "alembic_version" not in set(inspect(engine).get_table_names())
    engine.dispose()


def test_already_stamped_database_with_drifted_schema_is_refused_at_every_startup(tmp_path):
    """Reproduit l'INCIDENT RÉEL : une base déjà marquée `alembic_version =
    head` (par une version antérieure, moins stricte, de `run_migrations`,
    ou par toute autre voie) mais dont le schéma réel a dérivé — ici,
    `vision_anomaly_models` sans `n_calibration`/`n_evaluation`. Comme
    `alembic_version` est déjà présente, l'ancienne logique tombait dans la
    branche `upgrade head`, un no-op silencieux (déjà à `head`), et
    l'écart restait invisible jusqu'au premier appel API en 500. La
    vérification post-migration doit maintenant le détecter à CHAQUE
    démarrage, pas seulement au moment d'un stamp."""
    url = _sqlite_url(tmp_path, "drifted_head.db")
    engine = create_engine(url)

    # Reproduit l'état incriminé : schéma complet, migré normalement une
    # première fois (alembic_version posée à head par un run_migrations
    # antérieur), PUIS une colonne disparaît (dérive post-head — stamp
    # historique erroné, migration partiellement appliquée, etc.).
    run_migrations(db_url=url)
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE vision_anomaly_models DROP COLUMN n_evaluation"))

    tables_before = set(inspect(engine).get_table_names())
    assert "alembic_version" in tables_before  # bien déjà stampée/migrée, contrairement aux tests précédents
    engine.dispose()

    with pytest.raises(SchemaMismatchError, match="vision_anomaly_models.n_evaluation"):
        run_migrations(db_url=url)


def test_naive_upgrade_on_existing_schema_would_fail_without_stamp(tmp_path):
    """Documente POURQUOI la détection stamp/upgrade est nécessaire : sans
    elle, rejouer la révision initiale sur une base où les tables existent
    déjà lève une erreur SQL (CREATE TABLE sur une table déjà existante).
    Si ce test se met à échouer (l'erreur disparaît), la détection dans
    `run_migrations()` n'est peut-être plus nécessaire — à vérifier avant de
    la retirer."""
    url = _sqlite_url(tmp_path, "naive.db")
    engine = create_engine(url)
    Base.metadata.create_all(bind=engine)
    engine.dispose()

    cfg = _alembic_config(url)
    with pytest.raises(Exception):
        command.upgrade(cfg, "head")
