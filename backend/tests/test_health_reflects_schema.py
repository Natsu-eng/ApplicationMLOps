"""Le healthcheck doit refléter l'état du SCHÉMA, pas seulement la
joignabilité de la base — revue post-Phase 5.

Régression visée : le bug #2 (Dockerfile ne copiait pas `alembic/`) est resté
invisible parce que `/api/health` répondait 200 `{"status": "ok",
"database": "up"}` alors qu'aucune migration ne s'était appliquée. Docker
marquait le conteneur sain et l'orchestrateur y envoyait du trafic."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.core import database


@pytest.fixture
def client():
    from api.main import app

    return TestClient(app)


def test_health_is_200_and_ok_when_schema_applied(client, monkeypatch):
    monkeypatch.setitem(database._MIGRATION_STATE, "ok", True)
    monkeypatch.setitem(database._MIGRATION_STATE, "error", None)
    monkeypatch.setattr("api.main.check_connection", lambda: True)
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["schema"] == "ready"


def test_health_is_503_when_migrations_never_applied(client, monkeypatch):
    """Le cas du bug #2 : base joignable, schéma périmé. Sans ce test, un
    healthcheck vert masque à nouveau une production cassée."""
    monkeypatch.setitem(database._MIGRATION_STATE, "ok", False)
    monkeypatch.setitem(database._MIGRATION_STATE, "error", "CommandError: Path doesn't exist")
    monkeypatch.setattr("api.main.check_connection", lambda: True)
    r = client.get("/api/health")
    assert r.status_code == 503, "un schéma périmé doit faire échouer le healthcheck"
    body = r.json()
    assert body["status"] == "degraded"
    assert body["database"] == "up"      # la base répond...
    assert body["schema"] == "stale"     # ...mais le schéma est périmé


def test_health_is_503_when_database_unreachable(client, monkeypatch):
    monkeypatch.setitem(database._MIGRATION_STATE, "ok", True)
    monkeypatch.setattr("api.main.check_connection", lambda: False)
    r = client.get("/api/health")
    assert r.status_code == 503
    assert r.json()["database"] == "down"


def test_schema_error_never_disclosed_in_production(client, monkeypatch):
    """Le détail de l'erreur aide en local ; sur un endpoint non authentifié
    en production, c'est une divulgation gratuite."""
    monkeypatch.setitem(database._MIGRATION_STATE, "ok", False)
    monkeypatch.setitem(database._MIGRATION_STATE, "error", "OperationalError: permission denied")
    monkeypatch.setattr("api.main.check_connection", lambda: True)
    monkeypatch.setattr("api.main._IS_PROD", True)
    assert "schema_error" not in client.get("/api/health").json()
