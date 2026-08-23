"""Phase 1 (AUDIT_BACKEND_2026-08-23.md, Axe D) — le démarrage doit être
bloqué en production si DATABASE_URL pointe vers SQLite, ou contient encore
le mot de passe placeholder `CHANGE_ME` (voir `.env.example`). Même
principe que `test_security.py` pour JWT_SECRET_KEY : le contrôle a lieu à
l'import (module-level), testé en sous-process pour reproduire fidèlement
un démarrage à froid.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent


def _run_import_in_subprocess(env_overrides: dict[str, str]) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-c", "import api.core.database"],
        cwd=str(_BACKEND_DIR),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_sqlite_blocks_startup_in_production():
    result = _run_import_in_subprocess({"ENVIRONMENT": "production", "DATABASE_URL": "sqlite:///./database/datalab.db"})
    assert result.returncode != 0
    assert "DATABASE_URL" in result.stderr


def test_sqlite_allowed_in_development():
    result = _run_import_in_subprocess({"ENVIRONMENT": "development", "DATABASE_URL": "sqlite:///./database/datalab.db"})
    assert result.returncode == 0


def test_placeholder_password_blocks_startup_in_production():
    result = _run_import_in_subprocess(
        {"ENVIRONMENT": "production", "DATABASE_URL": "postgresql://datalab:CHANGE_ME@db:5432/datalab"}
    )
    assert result.returncode != 0
    assert "CHANGE_ME" in result.stderr


def test_real_postgres_url_never_blocks_startup_in_production():
    result = _run_import_in_subprocess(
        {"ENVIRONMENT": "production", "DATABASE_URL": "postgresql://datalab:un-vrai-mot-de-passe@db:5432/datalab"}
    )
    assert result.returncode == 0
