# ARCHITECTURE.md — Backend DataLab Pro

> **Lot 1 — authentification + organisations.** Ce document est vivant : il
> grandit à chaque lot livré. Historique des décisions : voir
> [`workflow.md`](workflow.md).

## 1. Vue d'ensemble

| Couche | Technologie | Version |
| --- | --- | --- |
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

## 2. Schéma des couches (état Lot 1)

```text
┌────────────────────────────────────────────────────┐
│  FRONTEND  (React — voir ../frontend/)               │
│    AuthContext (token localStorage) + ProtectedRoute   │
└─────────────────────┬────────────────────────────────┘
                       │ HTTP JSON (Bearer JWT sur les routes protégées)
                       ▼
┌────────────────────────────────────────────────────┐
│  api/main.py — FastAPI                                │
│    lifespan()  : init_db() (non bloquant)               │
│    CORS        : origines frontend autorisées              │
│    GET /api/health                                            │
│    router auth (prefix /auth)                                   │
└─────────────────────┬────────────────────────────────┘
                       ▼
┌────────────────────────────────────────────────────┐
│  api/routers/auth.py                                  │
│    register/login/me/team ; get_current_user, require_owner │
└─────────────────────┬────────────────────────────────┘
                       ▼
┌────────────────────────────────────────────────────┐
│  api/core/{security,database,models}.py                │
│    JWT HS256 + bcrypt · engine/SessionLocal/Base ORM      │
│    Organization, User (organization_id sur tout le reste)   │
│    PostgreSQL (DATABASE_URL) ou SQLite (dev)                  │
└────────────────────────────────────────────────────┘
```

Ce schéma se complète lot par lot :

| Lot | Ajout |
| --- | --- |
| 1 (livré) | `api/core/security.py` (JWT + bcrypt), `api/core/models.py` (User, Organization), `api/routers/auth.py` |
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
| --- | --- | --- |
| `DATABASE_URL` | `api/core/database.py` | SQLite `backend/database/datalab.db` |
| `FRONTEND_URL` | `api/main.py` (CORS) | `http://localhost:5173` |
| `ENVIRONMENT` | `api/core/config.py` | `development` |
| `LOG_LEVEL` | `api/core/config.py` | `INFO` |
| `JWT_SECRET_KEY` | `api/core/security.py` | clé de développement codée en dur (avertissement journalisé) |

## 5. Authentification et multi-tenant (Lot 1)

- **JWT HS256** (`python-jose`), TTL 24h, `sub` = id utilisateur (toujours
  sérialisé en string). Hash de mot de passe : bcrypt, 12 rounds.
- **Dépendances FastAPI empilables** (`api/routers/auth.py`), même pattern
  que CIAM : `get_current_user` (valide le token, charge l'utilisateur actif)
  → `require_owner` (réservé au propriétaire de l'organisation).
- **Modèle multi-tenant retenu** : `Organization` (bureau d'études) ⟶
  plusieurs `User` (`role` = `owner` ou `member`). `POST /auth/register` crée
  l'organisation et son premier utilisateur (`owner`) en une seule opération.
  Seul l'`owner` peut ajouter des membres (`POST /auth/team/members`).
- **Isolation** : toute requête qui liste des utilisateurs filtre par
  `User.organization_id == current_user.organization_id` — jamais par un
  identifiant fourni par le client. Les lots suivants (datasets, jobs,
  modèles) suivent la même règle : chaque table métier porte un
  `organization_id`, jamais visible en dehors de son organisation.
- **Codes d'erreur structurés**, repris de CIAM :
  `{"detail": {"code": "AUTH_...", "message": "..."}}` — le frontend peut
  distinguer les cas (`AUTH_EMAIL_DEJA_UTILISE`, `AUTH_OWNER_REQUIS`...)
  sans parser le message textuel.

## 6. Conventions reprises de CIAM

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
