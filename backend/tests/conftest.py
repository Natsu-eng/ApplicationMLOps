"""Fixtures pytest partagées.

Base de données de test isolée (SQLite fichier temporaire, jamais la base de
développement/production) recréée avant chaque test, et client FastAPI de
test. `DATABASE_URL`/`JWT_SECRET_KEY` sont surchargées AVANT tout import de
`api.core.*` : `get_settings()` est mis en cache (`lru_cache`) dès le premier
import, donc l'ordre des imports ci-dessous est important.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

_TEST_DB_PATH = Path(tempfile.gettempdir()) / "datalab_test.db"
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB_PATH}"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-not-for-production"

from api.core import models  # noqa: E402,F401 — enregistre les tables sur Base.metadata
from api.core.database import Base, get_db  # noqa: E402
from api.main import app  # noqa: E402

_test_engine = create_engine(f"sqlite:///{_TEST_DB_PATH}", connect_args={"check_same_thread": False})
_TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_test_engine)


def _override_get_db() -> Iterator:
    db = _TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def _fresh_database() -> Iterator[None]:
    """Schéma vierge avant CHAQUE test — isolation totale, aucun état ne fuit
    d'un test à l'autre (contrairement à une base partagée réutilisée)."""
    Base.metadata.drop_all(bind=_test_engine)
    Base.metadata.create_all(bind=_test_engine)
    app.dependency_overrides[get_db] = _override_get_db
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def db_session() -> Iterator:
    """Session DB directe — pour les tests qui doivent insérer des données
    comme le ferait un worker (hors requête HTTP), ex : test_inference.py."""
    db = _TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
