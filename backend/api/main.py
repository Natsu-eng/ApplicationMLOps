"""
DataLab Pro — API FastAPI
==========================
Squelette du Lot 0 : démarrage de l'application, CORS, healthcheck. Les
endpoints métier (auth, datasets, entraînement, évaluation...) arrivent lot
par lot — voir backend/workflow.md pour l'état d'avancement.

Lancer en local :
    cd backend
    uvicorn api.main:app --reload --port 8000
    → doc interactive : http://localhost:8000/docs
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from api.core.config import get_settings
from api.core.database import check_connection, init_db
from api.core.observability import (
    PrometheusMiddleware,
    RequestIdMiddleware,
    configure_logging,
    metrics_response,
    request_id_var,
)
from domains.anomalies.router import router as anomalies_router
from domains.auth.router import router as auth_router
from domains.auth.router import feedback_router, users_router
from domains.clustering.router import router as clustering_router
from domains.dashboard.router import router as dashboard_router
from domains.datasets.router import router as datasets_router
from domains.dimensionality.router import router as dimensionality_router
from domains.training.router import router as training_router
from domains.vision.anomalies.router import router as vision_anomalies_router
from domains.vision.classification.router import router as vision_classification_router
from domains.vision.datasets.router import router as vision_datasets_router

settings = get_settings()

configure_logging(settings.log_level)
logger = logging.getLogger("datalab.api")

if settings.sentry_dsn:
    # Lot 4, correctif I7 — actif SEULEMENT si SENTRY_DSN est défini
    # (backend/.env) : aucun comportement différent en dev/CI sans DSN.
    import sentry_sdk

    sentry_sdk.init(dsn=settings.sentry_dsn, environment=settings.environment, traces_sample_rate=0.1)
    logger.info("[STARTUP] Sentry activé (environment=%s)", settings.environment)
else:
    logger.info("[STARTUP] Sentry désactivé (SENTRY_DSN absent)")


class MaxJsonBodySizeMiddleware(BaseHTTPMiddleware):
    """Rejette un corps JSON trop volumineux avant de le laisser atteindre
    l'endpoint (Lot 1.4, §C.2.7, AUDIT_DATALAB_2026-08-16.md) —
    `POST /training/jobs/{id}/predict` accepte un dictionnaire arbitraire,
    sans aucune borne jusqu'ici.

    Scopé à `Content-Type: application/json` uniquement : les uploads
    (`multipart/form-data`) ont déjà leurs propres limites dédiées, plus
    élevées (`max_upload_size_mb`, `max_vision_upload_size_mb`), vérifiées
    plus loin dans leurs endpoints respectifs — cette middleware ne doit
    jamais les bloquer.

    Vérifie `Content-Length` — fourni par tout client HTTP standard pour un
    corps JSON. Un client contournant délibérément cet en-tête (transfer
    chunked) échappe à cette vérification : hors du modèle de menace visé
    ici (requête accidentellement/naïvement trop grande), une défense plus
    stricte relèverait d'un reverse proxy (`client_max_body_size` nginx)."""

    def __init__(self, app, max_bytes: int):
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next):
        content_type = request.headers.get("content-type", "")
        if content_type.startswith("application/json"):
            content_length = request.headers.get("content-length")
            if content_length is not None and int(content_length) > self.max_bytes:
                return JSONResponse(
                    status_code=413,
                    content={
                        "detail": {
                            "code": "CORPS_TROP_VOLUMINEUX",
                            "message": f"Corps de requête trop volumineux (max {self.max_bytes // (1024 * 1024)} Mo)",
                        }
                    },
                )
        return await call_next(request)


# Correctif Phase 1 (AUDIT_BACKEND_2026-08-23.md, Axe E) — aucun en-tête de
# sécurité, vérifié en direct (requête réelle contre le backend démarré
# localement) : ni HSTS, ni X-Content-Type-Options, ni X-Frame-Options/CSP,
# ni Referrer-Policy, ni Permissions-Policy. Exempté sur /docs, /redoc,
# /openapi.json (Swagger UI charge ses assets depuis un CDN — CSP stricte
# incompatible ; ces routes sont de toute façon désactivées en production,
# voir plus haut) : la CSP ne s'applique qu'au SPA et à l'API JSON.
_CSP = (
    "default-src 'self'; "
    "script-src 'self'; "
    # 'unsafe-inline' réservé à style-src : les graphiques (Recharts, SVG)
    # posent des styles inline sur les éléments — script-src reste, lui,
    # strict (aucune exécution de script arbitraire, c'est le risque réel).
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob:; "
    "font-src 'self'; "
    "connect-src 'self'; "
    "frame-ancestors 'none'; "
    "base-uri 'self'; "
    "form-action 'self'"
)
_DOCS_PATHS = frozenset({"/docs", "/redoc", "/openapi.json"})


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        # N'a d'effet que servi sur HTTPS (le navigateur ignore ce header
        # sur une réponse HTTP) — envoyé inconditionnellement : ce backend
        # ne sait pas s'il est atteint via un terminaison TLS en amont
        # (load balancer, nginx avec certificat) ou en clair.
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
        if request.url.path not in _DOCS_PATHS:
            response.headers["Content-Security-Policy"] = _CSP
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialisation au démarrage.

    Un échec d'initialisation DB ne doit jamais empêcher l'API de répondre :
    on le journalise et l'API démarre en mode dégradé (le healthcheck le
    reflète). Voir ARCHITECTURE.md, même choix que CIAM pour les modules
    optionnels.
    """
    try:
        init_db()
    except Exception:
        logger.exception("[STARTUP] Initialisation DB échouée — API démarrée en mode dégradé")
    yield
    logger.info("[SHUTDOWN] Arrêt de l'API")


_IS_PROD = settings.environment == "production"

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
    # Correctif Phase 1 (AUDIT_BACKEND_2026-08-23.md, Axe E) — /docs et
    # /openapi.json étaient exposés sans authentification quel que soit
    # l'environnement (vérifié en direct : 200 sur les deux). La
    # documentation interactive complète (schémas, tous les endpoints)
    # n'a rien à faire en production.
    docs_url=None if _IS_PROD else "/docs",
    redoc_url=None if _IS_PROD else "/redoc",
    openapi_url=None if _IS_PROD else "/openapi.json",
)

# CORS — le frontend Vite (dev) et l'URL de production déclarée en config.
# Correctif Phase 1 (AUDIT_BACKEND_2026-08-23.md, Axe E) — les origines de
# dev (localhost:5173) étaient whitelistées INCONDITIONNELLEMENT, y compris
# en production, avec allow_credentials=True : vérifié en direct (preflight
# accepté avec Origin: http://localhost:5173 même hors dev). Un serveur qui
# tournerait sur ce port sur la machine d'une victime aurait pu émettre des
# requêtes cross-origin authentifiées vers l'API de production.
_allowed_origins = (
    [settings.frontend_url]
    if _IS_PROD
    else list({"http://localhost:5173", "http://127.0.0.1:5173", settings.frontend_url})
)
# Lot 1.4 (§C.2.7/§D.4, AUDIT_DATALAB_2026-08-16.md) — méthodes/en-têtes
# resserrés à ce que l'API utilise réellement (GET/POST/PATCH/DELETE,
# Authorization + Content-Type) plutôt que "*". Les origines étaient déjà
# explicites (pas de risque critique), mais un "*" sur méthodes/en-têtes
# n'a aucune justification ici. `expose_headers` : `Content-Disposition`
# doit être lisible côté JS pour `api.training.exportModel()`
# (frontend/src/api/client.ts), qui lit le nom de fichier suggéré par le
# serveur — invisible en JS cross-origin sans cette exposition explicite.
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
    expose_headers=["Content-Disposition"],
)
app.add_middleware(MaxJsonBodySizeMiddleware, max_bytes=settings.max_json_body_size_mb * 1024 * 1024)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(PrometheusMiddleware)
# RequestIdMiddleware EN DERNIER : le dernier `add_middleware` devient le
# plus externe de la pile Starlette — sans ça, les logs émis par les
# middlewares ci-dessus n'auraient pas encore de `request_id` (Lot 4,
# correctif I7).
app.add_middleware(RequestIdMiddleware)

# Préfixe /api sur TOUS les routers métier (correctif — incident réel de
# déploiement) : nginx (nginx/nginx.conf) ne proxifie que `location /api/`
# vers le backend, tout le reste tombe dans `try_files ... /index.html` et
# renvoie le HTML de la SPA en 200 — un POST /auth/login sans préfixe
# recevait donc du HTML, jamais le backend. `include_router(router,
# prefix="/api")` compose avec le préfixe propre du router (ex. "/auth"),
# résultat "/api/auth" — les routers eux-mêmes restent inchangés.
app.include_router(auth_router, prefix="/api")
app.include_router(users_router, prefix="/api")
app.include_router(feedback_router, prefix="/api")
app.include_router(datasets_router, prefix="/api")
app.include_router(training_router, prefix="/api")
app.include_router(clustering_router, prefix="/api")
app.include_router(dimensionality_router, prefix="/api")
app.include_router(anomalies_router, prefix="/api")
app.include_router(vision_datasets_router, prefix="/api")
app.include_router(vision_classification_router, prefix="/api")
app.include_router(vision_anomalies_router, prefix="/api")
app.include_router(dashboard_router, prefix="/api")


@app.get("/api/health", tags=["système"])
def health() -> dict:
    """Healthcheck — utilisé par Docker/Railway et par le frontend au démarrage."""
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "database": "up" if check_connection() else "down",
    }


@app.get("/metrics", tags=["système"])
def metrics():
    """Métriques Prometheus (Lot 4, correctif I7) — pas de préfixe `/api`
    ni d'authentification, même convention que `/api/health` : un
    collecteur de métriques n'a pas de session utilisateur."""
    return metrics_response()


# ── Gestionnaires d'erreur globaux (Phase 1, AUDIT_BACKEND_2026-08-23.md,
# Axe E) ──────────────────────────────────────────────────────────────────
# Avant ce correctif, seules les `HTTPException` levées explicitement par le
# code métier portaient l'enveloppe `{"detail": {"code", "message"}}` — les
# erreurs produites par FastAPI/Starlette lui-même (404 sans route, 422 de
# validation Pydantic, 401 "Not authenticated" d'OAuth2PasswordBearer sans
# en-tête, 500 non gérée) y échappaient, vérifié en direct :
# `GET /api/does-not-exist` renvoyait `{"detail":"Not Found"}`, sans code
# stable, sans `request_id` dans le corps. Les trois gestionnaires
# ci-dessous unifient l'enveloppe partout, sans changer les codes HTTP.

def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "-")


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    detail = exc.detail
    if isinstance(detail, dict) and "code" in detail and "message" in detail:
        body = {**detail, "request_id": _request_id(request)}
    else:
        # `exc.detail` est une chaîne brute (ex. "Not authenticated" posé
        # par OAuth2PasswordBearer, "Not Found" par le routeur Starlette
        # lui-même) — jamais renvoyée telle quelle, toujours reformulée en
        # français avec un code stable dérivé du statut.
        code = {401: "AUTH_NON_AUTHENTIFIE", 404: "NON_TROUVE", 405: "METHODE_NON_AUTORISEE"}.get(
            exc.status_code, "ERREUR_HTTP"
        )
        message = {
            401: "Authentification requise.",
            404: "Ressource introuvable.",
            405: "Méthode non autorisée pour cette ressource.",
        }.get(exc.status_code, str(detail) if detail else "Une erreur est survenue.")
        body = {"code": code, "message": message, "request_id": _request_id(request)}
    return JSONResponse(status_code=exc.status_code, content={"detail": body}, headers=dict(exc.headers or {}))


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """422 Pydantic — le message par défaut ("Field required", en anglais
    technique) n'est ni en français ni actionnable. Reformule un résumé
    lisible ; le détail champ par champ reste disponible sous `errors` pour
    le débogage frontend, sans être LE message affiché à l'utilisateur."""
    fields = ", ".join(".".join(str(p) for p in err["loc"][1:]) for err in exc.errors()) or "champ(s) invalide(s)"
    body = {
        "code": "VALIDATION_ECHOUEE",
        "message": f"Requête invalide : vérifiez {fields}.",
        "request_id": _request_id(request),
        "errors": exc.errors(),
    }
    return JSONResponse(status_code=422, content={"detail": body})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Filet de sécurité final — ne doit normalement jamais se déclencher
    sur un chemin métier (chaque domaine traduit déjà ses erreurs
    attendues), mais si un bug échappe à tout le reste, le client ne doit
    JAMAIS recevoir `str(exc)` brut (fuite de détails internes) : le
    `request_id` est la seule chose qu'on lui donne pour permettre le
    support, l'exception complète part dans les logs structurés (déjà
    corrélés par `request_id`, voir observability.py)."""
    logger.exception("[UNHANDLED] %s %s", request.method, request.url.path)
    body = {
        "code": "ERREUR_INTERNE",
        "message": "Une erreur inattendue est survenue. Réessayez ; si le problème persiste, "
        f"contactez le support avec la référence {_request_id(request)}.",
        "request_id": _request_id(request),
    }
    return JSONResponse(status_code=500, content={"detail": body})
