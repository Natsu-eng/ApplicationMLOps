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
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from api.core.config import get_settings
from api.core.database import check_connection, init_db
from api.routers.anomalies import router as anomalies_router
from api.routers.auth import router as auth_router
from api.routers.clustering import router as clustering_router
from api.routers.datasets import router as datasets_router
from api.routers.dimensionality import router as dimensionality_router
from api.routers.training import router as training_router
from api.routers.vision_anomalies import router as vision_anomalies_router
from api.routers.vision_classification import router as vision_classification_router
from api.routers.vision_datasets import router as vision_datasets_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("datalab.api")

settings = get_settings()


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


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

# CORS — le frontend Vite (dev) et l'URL de production déclarée en config
_allowed_origins = list({
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    settings.frontend_url,
})
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

# Préfixe /api sur TOUS les routers métier (correctif — incident réel de
# déploiement) : nginx (nginx/nginx.conf) ne proxifie que `location /api/`
# vers le backend, tout le reste tombe dans `try_files ... /index.html` et
# renvoie le HTML de la SPA en 200 — un POST /auth/login sans préfixe
# recevait donc du HTML, jamais le backend. `include_router(router,
# prefix="/api")` compose avec le préfixe propre du router (ex. "/auth"),
# résultat "/api/auth" — les routers eux-mêmes restent inchangés.
app.include_router(auth_router, prefix="/api")
app.include_router(datasets_router, prefix="/api")
app.include_router(training_router, prefix="/api")
app.include_router(clustering_router, prefix="/api")
app.include_router(dimensionality_router, prefix="/api")
app.include_router(anomalies_router, prefix="/api")
app.include_router(vision_datasets_router, prefix="/api")
app.include_router(vision_classification_router, prefix="/api")
app.include_router(vision_anomalies_router, prefix="/api")


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
