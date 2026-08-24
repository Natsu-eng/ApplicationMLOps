"""Bornes du pool de connexions + délai maximal d'exécution SQL (Phase 2,
AUDIT_BACKEND_2026-08-23.md, Axe F.7/F.8) — vérifié contre un VRAI
PostgreSQL (le comportement dépend du driver réel, pas simulable
fidèlement avec SQLite, qui n'a ni pool véritable ni `statement_timeout`).

Sous-process, même technique que `test_database_startup.py` :
`api.core.database` construit son moteur une seule fois à l'import,
`tests/conftest.py` a déjà forcé `DATABASE_URL` vers SQLite pour le reste
de la suite — impossible de réimporter le module avec une autre config
dans CE process."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import psycopg2
import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_DATABASE_URL = os.environ.get(
    "BACKUP_TEST_DATABASE_URL",
    "postgresql://datalab:datalab_dev_password@localhost:5432/datalab",
)


def _postgres_reachable() -> bool:
    try:
        conn = psycopg2.connect(_DATABASE_URL, connect_timeout=3)
        conn.close()
        return True
    except Exception:
        return False


pg_test = pytest.mark.skipif(
    not _postgres_reachable(), reason="Postgres réel non joignable — pool/statement_timeout non testables ici"
)


def _run_in_subprocess(code: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["DATABASE_URL"] = _DATABASE_URL
    env["ENVIRONMENT"] = "development"  # évite le hard-fail prod (Phase 1) sur cette URL de test
    return subprocess.run(
        [sys.executable, "-c", code], cwd=str(_BACKEND_DIR), env=env, capture_output=True, text=True, timeout=30
    )


@pg_test
def test_statement_timeout_is_applied_on_postgres_connections():
    result = _run_in_subprocess(
        "from api.core.database import engine\n"
        "with engine.connect() as conn:\n"
        "    print(conn.exec_driver_sql('SHOW statement_timeout').scalar())\n"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "30s"


@pg_test
def test_statement_timeout_actually_cancels_a_slow_query():
    """Ne se contente pas de lire le réglage — prouve qu'il agit réellement :
    une requête plus longue que le timeout doit être annulée par PostgreSQL
    lui-même, pas seulement configurée en théorie."""
    result = _run_in_subprocess(
        "from psycopg2.errors import QueryCanceled\n"
        "from sqlalchemy.exc import OperationalError\n"
        "from api.core.database import engine\n"
        "with engine.connect() as conn:\n"
        "    conn.exec_driver_sql('SET statement_timeout = 200')\n"  # 200ms — court pour un test rapide
        "    try:\n"
        "        conn.exec_driver_sql('SELECT pg_sleep(2)')\n"
        "        print('PAS_ANNULEE')\n"
        "    except OperationalError as exc:\n"
        # Le type d'exception psycopg2 fait foi, jamais le texte du message
        # (localisé par le serveur PostgreSQL — "query cancelled" en
        # anglais, "annulation de la requête..." si le serveur est en
        # français, vérifié en direct sur ce poste).
        "        print('ANNULEE' if isinstance(exc.orig, QueryCanceled) else f'AUTRE_ERREUR: {exc}')\n"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ANNULEE"


@pg_test
def test_pool_size_and_max_overflow_are_explicitly_configured():
    """Correctif : avant cette phase, `create_engine()` ne passait ni
    `pool_size` ni `max_overflow` — les défauts SQLAlchemy (5+10=15) ne
    reflétaient aucun dimensionnement réfléchi face à la topologie réelle
    (2 workers gunicorn + jusqu'à 3 process RQ, chacun avec son propre
    pool)."""
    result = _run_in_subprocess(
        "from api.core.database import engine\nprint(engine.pool.size(), engine.pool._max_overflow)\n"
    )
    assert result.returncode == 0, result.stderr
    size, max_overflow = result.stdout.strip().split()
    assert int(size) == 10
    assert int(max_overflow) == 5


def test_sqlite_engine_has_no_statement_timeout_option(client):
    """Non-régression : SQLite (dev/test, `tests/conftest.py`) n'a pas de
    `statement_timeout` — `connect_args` ne doit jamais lui passer l'option
    Postgres `-c statement_timeout=...` (SQLite planterait à la connexion,
    `check_same_thread` est la seule option attendue ici). `client`
    (fixture SQLite) suffit à prouver que l'API démarre et répond
    normalement avec ce moteur — la branche `_is_sqlite` de
    `create_engine()` n'a pas régressé."""
    from api.core.database import _is_sqlite

    assert _is_sqlite is True
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["database"] == "up"
