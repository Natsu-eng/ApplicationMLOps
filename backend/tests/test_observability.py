"""Tests de api/core/observability.py (Lot 4, correctif I7,
AUDIT_DATALAB_2026-08-16.md §I7) : logs JSON, corrélation par request_id,
métriques Prometheus. Sentry non testé ici (un simple appel conditionnel
à `sentry_sdk.init`, voir api/main.py) — seul son défaut désactivé
(`sentry_dsn=None`) est vérifié."""
from __future__ import annotations

import json
import logging

from starlette.requests import Request

from api.core.config import get_settings
from api.core.observability import JsonFormatter, _route_template


def _make_record(message: str, request_id: str | None = None) -> logging.LogRecord:
    record = logging.LogRecord("datalab.test", logging.INFO, __file__, 1, message, (), None)
    if request_id is not None:
        record.request_id = request_id
    return record


# ── JsonFormatter ────────────────────────────────────────────────────────

def test_json_formatter_produces_parseable_json_with_expected_fields():
    record = _make_record("un message", request_id="abc-123")
    payload = json.loads(JsonFormatter().format(record))
    assert payload["level"] == "INFO"
    assert payload["logger"] == "datalab.test"
    assert payload["message"] == "un message"
    assert payload["request_id"] == "abc-123"


def test_json_formatter_defaults_request_id_to_dash_when_absent():
    """Cas d'un log émis hors requête HTTP (démarrage, worker RQ) — le
    filtre qui pose `request_id` en temps normal n'est pas dans le chemin."""
    record = _make_record("sans contexte HTTP")
    payload = json.loads(JsonFormatter().format(record))
    assert payload["request_id"] == "-"


def test_json_formatter_includes_exception_when_present():
    try:
        raise ValueError("échec de test")
    except ValueError:
        import sys

        record = logging.LogRecord(
            "datalab.test", logging.ERROR, __file__, 1, "erreur", (), sys.exc_info()
        )
    payload = json.loads(JsonFormatter().format(record))
    assert "ValueError" in payload["exception"]
    assert "échec de test" in payload["exception"]


# ── request_id (middleware, bout-en-bout via TestClient) ────────────────────

def test_missing_request_id_header_generates_one(client):
    resp = client.get("/api/health")
    assert resp.headers.get("x-request-id")


def test_provided_valid_uuid_request_id_header_is_preserved(client):
    """Phase 1 (AUDIT_BACKEND_2026-08-23.md, Axe E) — un UUID valide fourni
    par le client reste préservé, utile pour du traçage distribué où le
    client génère déjà son propre identifiant de corrélation."""
    resp = client.get("/api/health", headers={"X-Request-ID": "d290f1ee-6c54-4b01-90e6-d701748f0851"})
    assert resp.headers["x-request-id"] == "d290f1ee-6c54-4b01-90e6-d701748f0851"


def test_provided_non_uuid_request_id_header_is_replaced(client):
    """Correctif Phase 1 (AUDIT_BACKEND_2026-08-23.md, Axe E) — avant ce
    correctif, un `X-Request-ID` arbitraire fourni par le client était
    accepté tel quel et injecté dans chaque ligne de log JSON, sans
    validation (confusion de corrélation, contenu arbitraire pour un outil
    de logs en aval). Seul un UUID valide est désormais accepté ; une
    chaîne arbitraire ("mon-id-fixe", vérifié en direct qu'elle était
    reflétée avant ce correctif) est ignorée, comme si rien n'avait été
    fourni."""
    resp = client.get("/api/health", headers={"X-Request-ID": "mon-id-fixe"})
    assert resp.headers["x-request-id"] != "mon-id-fixe"
    assert resp.headers["x-request-id"]  # un UUID généré est bien présent


def test_two_requests_get_different_generated_ids(client):
    first = client.get("/api/health").headers["x-request-id"]
    second = client.get("/api/health").headers["x-request-id"]
    assert first != second


# ── _route_template (gabarit de route, jamais le chemin brut) ───────────────

def test_route_template_uses_the_matched_route_path_not_the_raw_path():
    class _FakeRoute:
        path = "/datasets/{dataset_id}"

    scope = {"type": "http", "method": "GET", "path": "/datasets/42", "route": _FakeRoute(), "headers": []}
    assert _route_template(Request(scope)) == "/datasets/{dataset_id}"


def test_route_template_falls_back_to_raw_path_when_no_route_matched():
    """404 : aucune route ne matche, `scope["route"]` n'est jamais posé."""
    scope = {"type": "http", "method": "GET", "path": "/n-existe-pas", "headers": []}
    assert _route_template(Request(scope)) == "/n-existe-pas"


# ── /metrics ─────────────────────────────────────────────────────────────

def test_metrics_endpoint_exposes_prometheus_format(client):
    client.get("/api/health")
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert "http_requests_total" in resp.text
    assert "http_request_duration_seconds" in resp.text
    assert "http_requests_in_progress" in resp.text


def test_metrics_labels_use_the_route_template_for_a_known_static_route(client):
    client.get("/api/health")
    resp = client.get("/metrics")
    lines = [l for l in resp.text.splitlines() if l.startswith("http_requests_total") and "/api/health" in l]
    assert lines, "aucune série http_requests_total pour /api/health"


# ── Sentry (config seulement — voir api/main.py pour l'initialisation) ──────

def test_sentry_disabled_by_default():
    assert get_settings().sentry_dsn is None
