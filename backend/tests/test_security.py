"""Lot durcissement (H3, AUDIT_ROADMAP.md) — le démarrage doit être bloqué
en production si JWT_SECRET_KEY est resté à sa valeur par défaut.

Le contrôle a lieu à l'IMPORT de api.core.security (module-level, capture
_settings une seule fois) — donc testé en sous-process avec un environnement
dédié, plutôt qu'en réimportant le module dans le process de test (déjà
importé avec la clé de test de conftest.py, un simple reload ne reproduirait
pas fidèlement un vrai démarrage à froid).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent


def _run_import_in_subprocess(env_overrides: dict[str, str]) -> subprocess.CompletedProcess:
    import os

    env = os.environ.copy()
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-c", "import api.core.security"],
        cwd=str(_BACKEND_DIR),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_default_jwt_key_blocks_startup_in_production():
    result = _run_import_in_subprocess(
        {"ENVIRONMENT": "production", "JWT_SECRET_KEY": "changez-cette-cle-en-production"}
    )
    assert result.returncode != 0
    assert "JWT_SECRET_KEY" in result.stderr


def test_default_jwt_key_only_warns_in_development():
    result = _run_import_in_subprocess(
        {"ENVIRONMENT": "development", "JWT_SECRET_KEY": "changez-cette-cle-en-production"}
    )
    assert result.returncode == 0


def test_custom_jwt_key_never_blocks_startup_in_production():
    result = _run_import_in_subprocess(
        {"ENVIRONMENT": "production", "JWT_SECRET_KEY": "une-vraie-cle-generee-aleatoirement-64-hex"}
    )
    assert result.returncode == 0
