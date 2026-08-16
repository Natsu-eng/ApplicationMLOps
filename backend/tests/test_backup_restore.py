"""Preuve que la sauvegarde se restaure réellement — Lot 1.2 (correctif C6,
AUDIT_DATALAB_2026-08-16.md) : « un script de sauvegarde jamais restauré ne
compte pas ». Cycle complet sur un vrai Postgres, dans un schéma jetable
dédié de la base configurée (jamais `public`, jamais les données réelles) :
peuple le schéma, le sauvegarde, SUPPRIME le schéma (simule une perte de
données réelle), restaure depuis le dump, vérifie que les données sont
revenues. Le schéma jetable est supprimé même si le test échoue.

Isolation par SCHÉMA plutôt que par base séparée : le rôle applicatif
(`datalab`) n'a délibérément PAS le privilège CREATEDB (moindre privilège,
vérifié dans cet environnement) — créer/supprimer un schéma dans une base
où l'on a déjà un accès complet ne demande, elle, aucun privilège
supplémentaire.

Ignoré si aucun Postgres réel n'est joignable ou si `pg_dump`/`pg_restore`
sont absents du PATH (ex. CI sans service Postgres) — un mock n'aurait
rien prouvé ici, seul un vrai moteur compte pour cette preuve."""
from __future__ import annotations

import os
import shutil

import psycopg2
import pytest
from sqlalchemy import create_engine, text

from api.core.database import Base
from api.core import models  # noqa: F401 — enregistre les tables sur Base.metadata
from scripts.backup_db import backup_database, backup_storage
from scripts.restore_db import restore_database, restore_storage

# Délibérément PAS `get_settings().database_url` : `tests/conftest.py`
# réécrit inconditionnellement `DATABASE_URL` vers un SQLite temporaire dès
# son propre import (isolation de la suite), donc `get_settings()` ne verra
# jamais un Postgres réel dans CE process pytest, quel que soit
# l'environnement. Variable dédiée, indépendante de la config applicative —
# vaut par défaut l'URL du Postgres de développement local, joignable dans
# cet environnement de dev, et surchargeable en CI pour pointer le service
# Postgres de la CI (voir .github/workflows/ci.yml).
_DATABASE_URL = os.environ.get(
    "BACKUP_TEST_DATABASE_URL",
    "postgresql://datalab:datalab_dev_password@localhost:5432/datalab",
)
_SCHEMA = "_datalab_backup_test"


def _postgres_reachable() -> bool:
    try:
        conn = psycopg2.connect(_DATABASE_URL, connect_timeout=3)
        conn.close()
        return True
    except Exception:
        return False


db_test = pytest.mark.skipif(
    shutil.which("pg_dump") is None or shutil.which("pg_restore") is None or not _postgres_reachable(),
    reason="Postgres réel (ou pg_dump/pg_restore) non joignable — cycle de sauvegarde non testable ici",
)


def _drop_test_schema(engine) -> None:
    with engine.begin() as conn:
        conn.execute(text(f'DROP SCHEMA IF EXISTS "{_SCHEMA}" CASCADE'))


@db_test
def test_database_backup_restore_cycle_preserves_data(tmp_path):
    engine = create_engine(_DATABASE_URL)
    _drop_test_schema(engine)  # nettoyage défensif d'un run précédent interrompu
    try:
        with engine.begin() as conn:
            conn.execute(text(f'CREATE SCHEMA "{_SCHEMA}"'))
        schema_engine = create_engine(_DATABASE_URL, connect_args={"options": f"-csearch_path={_SCHEMA}"})
        Base.metadata.create_all(bind=schema_engine)
        with schema_engine.begin() as conn:
            conn.execute(text("INSERT INTO organizations (name) VALUES ('Bureau de test sauvegarde')"))
            org_id = conn.execute(text("SELECT id FROM organizations")).scalar()
            conn.execute(
                text(
                    "INSERT INTO users (email, nom, hashed_password, role, organization_id, actif) "
                    "VALUES ('test@bureau.fr', 'Testeur', 'hash', 'owner', :org_id, true)"
                ),
                {"org_id": org_id},
            )
        schema_engine.dispose()

        dump_path = tmp_path / "test.dump"
        backup_database(_DATABASE_URL, dump_path, schema=_SCHEMA)
        assert dump_path.exists() and dump_path.stat().st_size > 0

        # Simule une perte de données réelle — le schéma disparaît complètement.
        _drop_test_schema(engine)
        with engine.connect() as conn:
            schemas = conn.execute(
                text("SELECT schema_name FROM information_schema.schemata WHERE schema_name = :s"),
                {"s": _SCHEMA},
            ).fetchall()
        assert schemas == []  # confirme la "perte" avant de prouver la restauration

        restore_database(dump_path, _DATABASE_URL)

        with engine.connect() as conn:
            orgs = conn.execute(text(f'SELECT name FROM "{_SCHEMA}".organizations')).fetchall()
            users = conn.execute(
                text(f'SELECT email, organization_id FROM "{_SCHEMA}".users')
            ).fetchall()

        assert [r[0] for r in orgs] == ["Bureau de test sauvegarde"]
        assert len(users) == 1
        assert users[0][0] == "test@bureau.fr"
    finally:
        _drop_test_schema(engine)
        engine.dispose()


def test_storage_backup_restore_cycle_preserves_files(tmp_path):
    """Aucune dépendance Postgres — toujours exécuté."""
    source_storage = tmp_path / "storage_source"
    (source_storage / "datasets" / "1").mkdir(parents=True)
    (source_storage / "datasets" / "1" / "data.csv").write_text("a,b\n1,2\n")
    (source_storage / "models" / "1").mkdir(parents=True)
    (source_storage / "models" / "1" / "model.joblib").write_bytes(b"\x00\x01\x02")

    archive_path = tmp_path / "storage_backup.tar.gz"
    backup_storage(source_storage, archive_path)
    assert archive_path.exists()

    restored_dir = tmp_path / "storage_restored"
    restore_storage(archive_path, restored_dir)

    assert (restored_dir / "datasets" / "1" / "data.csv").read_text() == "a,b\n1,2\n"
    assert (restored_dir / "models" / "1" / "model.joblib").read_bytes() == b"\x00\x01\x02"
