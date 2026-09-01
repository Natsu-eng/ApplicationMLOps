"""Régression du correctif de dérive de fuseau horaire (migration
`2c88481be342_fix_timezone_and_notnull_drift.py`) — vérifié contre un VRAI
PostgreSQL, jamais simulable avec SQLite (qui ne distingue pas
`TIMESTAMP`/`TIMESTAMPTZ`, donc n'aurait jamais pu révéler ce bug).

Contexte : `ml_models.promoted_at` et `training_jobs.progress_updated_at`
étaient `TIMESTAMP WITHOUT TIME ZONE` alors que le modèle déclare
`DateTime(timezone=True)` (comme toutes les autres colonnes
`progress_updated_at` du dépôt). Avec un fuseau de session Postgres
différent d'UTC (`Africa/Casablanca`, UTC+1, sur ce poste), une valeur
UTC-aware insérée dans une colonne `TIMESTAMP` était convertie au fuseau
de session PUIS dépouillée de son fuseau — relue ensuite comme naïve et
réétiquetée "UTC" par `domains/shared/job_watchdog.py::_as_aware_utc`
(qui suppose explicitement, par son propre docstring, que Postgres
conserve toujours le fuseau) : la valeur relue était donc décalée d'1h,
faisant sous-estimer d'1h l'ancienneté réelle d'un job détecté bloqué.

Même technique de sous-process que `test_database_pool_and_timeout.py` :
utilise directement `psycopg2`, jamais le moteur applicatif (déjà fixé sur
SQLite pour le reste de la suite par `tests/conftest.py`)."""

from __future__ import annotations

import os
from datetime import datetime, timezone

import psycopg2
import pytest

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
    not _postgres_reachable(), reason="Postgres réel non joignable — dérive de fuseau non testable ici"
)


@pg_test
def test_previously_drifted_columns_are_now_timezone_aware_and_not_null():
    """Vérifie le SCHÉMA réel — garde-fou si une future migration
    (autogenerate ou manuelle) réintroduisait la dérive sans qu'on s'en
    rende compte (comme la Phase 3 l'avait fait initialement)."""
    conn = psycopg2.connect(_DATABASE_URL, connect_timeout=3)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            select table_name, column_name, data_type, is_nullable
            from information_schema.columns
            where (table_name, column_name) in (
                ('ml_models', 'promoted_at'),
                ('training_jobs', 'progress_updated_at'),
                ('password_reset_tokens', 'created_at')
            )
            """
        )
        rows = {(t, c): (dt, n) for t, c, dt, n in cur.fetchall()}
        assert rows[("ml_models", "promoted_at")][0] == "timestamp with time zone"
        assert rows[("training_jobs", "progress_updated_at")][0] == "timestamp with time zone"
        assert rows[("password_reset_tokens", "created_at")] == ("timestamp with time zone", "NO")
    finally:
        conn.close()


@pg_test
def test_progress_updated_at_roundtrip_preserves_the_exact_utc_instant():
    """Le bug réel : avant le correctif, cet aller-retour aurait renvoyé un
    datetime naïf décalé du fuseau de session (+1h sur ce poste), jamais
    l'instant exact écrit. Transaction jamais commitée (`rollback()` dans
    le `finally`) — ne laisse rien dans la base de développement, même
    principe que le reste de la suite Postgres-only de ce dépôt."""
    conn = psycopg2.connect(_DATABASE_URL, connect_timeout=3)
    try:
        cur = conn.cursor()
        cur.execute("select id from training_jobs limit 1")
        row = cur.fetchone()
        if row is None:
            pytest.skip("Aucun training_job existant pour tester l'aller-retour")
        job_id = row[0]

        known_instant = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        cur.execute(
            "update training_jobs set progress_updated_at = %s where id = %s",
            (known_instant, job_id),
        )
        cur.execute("select progress_updated_at from training_jobs where id = %s", (job_id,))
        read_back = cur.fetchone()[0]

        assert read_back.tzinfo is not None, "doit revenir timezone-aware, plus jamais naïf"
        assert read_back.astimezone(timezone.utc) == known_instant
    finally:
        conn.rollback()
        conn.close()
