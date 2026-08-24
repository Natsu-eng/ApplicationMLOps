# ═══════════════════════════════════════════════════════════════════════════
# DOCKERFILE — Backend FastAPI DataLab Pro
# ═══════════════════════════════════════════════════════════════════════════
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libgomp1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY domains/ ./domains/
COPY workers/ ./workers/
# Phase 5 (AUDIT_BACKEND_2026-08-23.md, Axe J) -- jamais copies avant ce
# correctif : api/core/database.py::init_db() (appele au demarrage par
# api/main.py::lifespan) appelle run_migrations(), qui a besoin
# d'alembic.ini + alembic/ (env.py, versions/*.py) pour fonctionner.
# Masque jusqu'ici par le bug plus severe de la Decision 1 (domains/ non
# copie -- l'import de l'app echouait avant meme d'atteindre ce code) --
# une fois ce premier bug corrige en Phase 1, ce second bug devenait
# atteignable : en production, CHAQUE demarrage aurait execute init_db()
# dans son bloc try/except (voir lifespan()), qui journalise l'echec et
# demarre quand meme en mode degrade -- silencieusement, aucune migration
# n'aurait jamais ete appliquee sur une vraie base de production.
COPY alembic.ini .
COPY alembic/ ./alembic/

RUN mkdir -p database logs storage/datasets storage/models && \
    useradd -m -u 1000 -s /bin/bash appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["gunicorn", "api.main:app", \
     "--workers", "2", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
