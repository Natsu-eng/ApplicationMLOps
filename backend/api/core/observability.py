"""Observabilité — Lot 4, correctif I7 (AUDIT_DATALAB_2026-08-16.md §I7).

Avant ce correctif : logs texte libre sans corrélation possible entre les
lignes d'une même requête, aucune métrique exposée, aucun rapport
d'exception centralisé — un incident en production ne laissait que des
logs bruts à parcourir à la main, sans moyen de les relier entre eux ni de
mesurer la charge réelle.

Trois briques, indépendantes les unes des autres :
- `configure_logging` : un log JSON par ligne (`%(message)s` structuré),
  `request_id` injecté sur CHAQUE ligne émise pendant une requête HTTP
  (corrélation), `"-"` en dehors d'une requête (démarrage, worker RQ).
- `RequestIdMiddleware` : lit `X-Request-ID` du client si fourni (traçage
  bout-en-bout derrière un reverse proxy qui le fixe déjà), sinon en
  génère un ; le renvoie toujours dans la réponse.
- `PrometheusMiddleware` + `/metrics` (câblé dans `api/main.py`) :
  compteur de requêtes, histogramme de latence, jauge de requêtes en
  cours — labellisés par le GABARIT de route (`/datasets/{dataset_id}`),
  jamais le chemin brut (`/datasets/42`), pour ne pas faire exploser la
  cardinalité des métriques avec un label par id.

Sentry est initialisé séparément dans `api/main.py` (pas ici) : simple
appel conditionnel à `sentry_sdk.init`, rien à mutualiser avec le reste de
ce module.
"""
from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Optional

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_RESERVED_LOG_RECORD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message"}


class _RequestIdFilter(logging.Filter):
    """Injecte `request_id` (contextvar) sur chaque enregistrement de log —
    sans ce filtre, `%(request_id)s` planterait sur les logs émis hors
    d'une requête HTTP (démarrage de l'app, worker RQ)."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    """Une ligne JSON par log — exploitable par un collecteur de logs
    (ELK, Loki, CloudWatch...) sans parsing regex fragile sur du texte
    libre."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        # Champs supplémentaires passés via `logger.info(..., extra={...})`
        # — jamais les attributs internes du LogRecord lui-même.
        for key, value in record.__dict__.items():
            if key not in _RESERVED_LOG_RECORD_ATTRS and key != "request_id":
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    """Remplace `logging.basicConfig` — un seul handler stdout, format
    JSON, `request_id` corrélé. Appelé une fois au démarrage de l'API
    (`api/main.py`)."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(_RequestIdFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Ajoutée EN DERNIER dans `api/main.py` (donc la plus externe — voir
    la doc Starlette sur l'ordre d'empilement des middlewares) : sans ça,
    les logs émis par `CORSMiddleware`/`MaxJsonBodySizeMiddleware` pour
    une requête donnée n'auraient pas encore de `request_id`."""

    async def dispatch(self, request: Request, call_next):
        incoming = request.headers.get("x-request-id")
        request_id = incoming if incoming else str(uuid.uuid4())
        token = request_id_var.set(request_id)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = request_id
        return response


# ── Métriques Prometheus ────────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Nombre de requêtes HTTP traitées",
    ["method", "path", "status"],
)
REQUEST_LATENCY_SECONDS = Histogram(
    "http_request_duration_seconds",
    "Durée des requêtes HTTP en secondes",
    ["method", "path"],
)
REQUESTS_IN_PROGRESS = Gauge(
    "http_requests_in_progress",
    "Nombre de requêtes HTTP actuellement en cours de traitement",
)


def _route_template(request: Request) -> str:
    """Gabarit de route (`/datasets/{dataset_id}`) plutôt que le chemin
    brut (`/datasets/42`) — un label Prometheus par id ferait exploser la
    cardinalité de la métrique en production. `scope["route"]` n'est posé
    QU'APRÈS résolution du routing par Starlette (donc seulement lisible
    après `call_next`, jamais avant) ; absent sur un 404 (aucune route ne
    matche) — on retombe alors sur le chemin brut, seul cas où le risque
    de cardinalité est de toute façon borné par les tentatives d'un
    attaquant, pas par un usage légitime répété."""
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return path if path is not None else request.url.path


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        REQUESTS_IN_PROGRESS.inc()
        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            REQUESTS_IN_PROGRESS.dec()
        duration = time.perf_counter() - start
        path = _route_template(request)
        REQUEST_COUNT.labels(method=request.method, path=path, status=response.status_code).inc()
        REQUEST_LATENCY_SECONDS.labels(method=request.method, path=path).observe(duration)
        return response


def metrics_response() -> Response:
    """Corps de `GET /metrics` (câblé dans `api/main.py`) — format texte
    Prometheus standard, scrappable par n'importe quel collecteur
    compatible (Prometheus lui-même, Grafana Agent, VictoriaMetrics...)."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
