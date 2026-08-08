# ARCHITECTURE.md — Backend DataLab Pro

> **Lot 0 — squelette.** Ce document est vivant : il grandit à chaque lot livré.
> Historique des décisions : voir [`workflow.md`](workflow.md).

## 1. Vue d'ensemble

| Couche | Technologie | Version |
|---|---|---|
| API | FastAPI | 0.136.1 |
| ORM | SQLAlchemy | 2.0.49 |
| Base de données production | PostgreSQL | via `psycopg2-binary` 2.9.11 |
| Base de données développement | SQLite | fallback automatique si `DATABASE_URL` absent |
| Configuration | pydantic-settings | 2.10.1 |

Stack choisie par parité avec **CIAM** (`concrete-ai-platform`), déjà
éprouvée en production. Le détail des choix retenus pour DataLab Pro
(organisation/équipe multi-utilisateurs, RQ+Redis pour les tâches longues,
ordre des lots) est documenté dans le diagnostic de migration présenté et
validé avant le début du code.

## 2. Schéma des couches (état Lot 0)

```
┌───────────────────────────────────────────────┐
│  FRONTEND  (React — voir ../frontend/)          │
└────────────────────┬────────────────────────────┘
                      │ HTTP JSON
                      ▼
┌───────────────────────────────────────────────┐
│  api/main.py — FastAPI                           │
│    lifespan()  : init_db() (non bloquant)         │
│    CORS        : origines frontend autorisées       │
│    GET /api/health                                    │
└────────────────────┬────────────────────────────┘
                      ▼
┌───────────────────────────────────────────────┐
│  api/core/database.py                            │
│    engine, SessionLocal, Base (ORM)                │
│    PostgreSQL (DATABASE_URL) ou SQLite (dev)         │
└───────────────────────────────────────────────┘
```

Ce schéma se complète lot par lot :

| Lot | Ajout prévu |
|---|---|
| 1 | `api/core/security.py` (JWT + bcrypt), `api/core/models.py` (User, Organization), `api/routers/auth.py` |
| 2 | `api/routers/datasets.py`, `api/core/storage.py` |
| 3 | `api/core/job_queue.py` (RQ + Redis), `workers/`, `api/routers/ws_progress.py` |
| ≥4 | `services/` — portage de la logique ML pure identifiée dans le diagnostic (catalogue de modèles, preprocessing, orchestrateurs, métriques, visualisations) |

## 3. Points d'entrée

- **Fichier d'entrée** : `api/main.py`
- **Commande dev** : `uvicorn api.main:app --reload --port 8000`
- **Commande prod** : `gunicorn api.main:app --workers 2 --worker-class uvicorn.workers.UvicornWorker` (voir `Dockerfile`)
- **Port** : 8000

## 4. Variables d'environnement

| Variable | Lue dans | Défaut développement |
|---|---|---|
| `DATABASE_URL` | `api/core/database.py` | SQLite `backend/database/datalab.db` |
| `FRONTEND_URL` | `api/main.py` (CORS) | `http://localhost:5173` |
| `ENVIRONMENT` | `api/core/config.py` | `development` |
| `LOG_LEVEL` | `api/core/config.py` | `INFO` |

## 5. Conventions reprises de CIAM

- Un échec d'initialisation non critique (base de données indisponible au
  démarrage) ne bloque jamais le démarrage de l'API : il est journalisé et
  `GET /api/health` reflète l'état réel (`"database": "down"`) plutôt que de
  faire planter tout le service.
- Configuration centralisée dans un seul module (`api/core/config.py`),
  jamais de valeurs en dur dispersées dans le code métier.
- Un seul `.env` par dossier (`backend/.env`, `frontend/.env.local`), jamais
  suivi par git.
- Migrations de schéma idempotentes plutôt qu'Alembic (à réévaluer si le
  schéma grandit significativement en cours de route).
