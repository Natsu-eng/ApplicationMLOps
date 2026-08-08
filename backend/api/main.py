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

from api.core.config import get_settings
from api.core.database import check_connection, init_db
from api.routers.auth import router as auth_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("datalab.api")

settings = get_settings()


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
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)


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
