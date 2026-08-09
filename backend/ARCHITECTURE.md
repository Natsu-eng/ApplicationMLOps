# ARCHITECTURE.md — Backend DataLab Pro

> **Lot 2 — datasets tabulaires.** Ce document est vivant : il grandit à
> chaque lot livré. Historique des décisions : voir
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

## 2. Schéma des couches (état Lot 2)

```text
┌────────────────────────────────────────────────────┐
│  FRONTEND  (React — voir ../frontend/)               │
│    AuthContext (token localStorage) + ProtectedRoute   │
│    AppShell (navigation) + components/ui/ (design system)│
└─────────────────────┬────────────────────────────────┘
                       │ HTTP JSON (Bearer JWT sur les routes protégées)
                       ▼
┌────────────────────────────────────────────────────┐
│  api/main.py — FastAPI                                │
│    lifespan()  : init_db() (non bloquant)               │
│    CORS        : origines frontend autorisées              │
│    GET /api/health                                            │
│    routers auth (/auth) et datasets (/datasets)                 │
└─────────────────────┬────────────────────────────────┘
                       ▼
┌────────────────────────────────────────────────────┐
│  api/routers/{auth,datasets}.py                       │
│    get_current_user, require_owner (dépendances communes)│
└─────────────────────┬────────────────────────────────┘
              ┌────────┴────────┐
              ▼                 ▼
┌───────────────────────┐ ┌───────────────────────────┐
│ api/core/{security,      │ │ services/datasets.py         │
│ database,models,storage}  │ │  lecture/validation pure       │
│  JWT+bcrypt · SQLAlchemy    │ │  (csv/parquet/xlsx/xls/json)    │
│  Organization,User,Dataset   │ │ api/core/storage.py               │
│  Postgres (prod)/SQLite (dev)  │ │  disque local, storage/datasets/    │
└───────────────────────┘ └───────────────────────────┘
```

Ce schéma se complète lot par lot :

| Lot | Ajout |
| --- | --- |
| 1 (livré) | `api/core/security.py` (JWT + bcrypt), `api/core/models.py` (User, Organization), `api/routers/auth.py` |
| 2 (livré) | `api/routers/datasets.py`, `api/core/storage.py`, `services/datasets.py` (première brique de la couche `services/`) |
| 3 | `api/core/job_queue.py` (RQ + Redis), `workers/`, `api/routers/ws_progress.py` |
| ≥4 | `services/` continue de grandir — portage de la logique ML pure identifiée dans le diagnostic (catalogue de modèles, preprocessing, orchestrateurs, métriques, visualisations) |

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
| `MAX_UPLOAD_SIZE_MB` | `api/routers/datasets.py` | `200` |

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

## 6. Datasets tabulaires (Lot 2)

- **Stockage** : disque local, `backend/storage/datasets/{organization_id}/{dataset_id}{extension}`
  — monté en volume Docker pour persister entre redémarrages
  (`docker-compose.yml`). Migration vers un stockage objet compatible S3
  prévue quand le volume de clients l'impose (voir le diagnostic de
  migration, section D2) — non nécessaire tant qu'on reste sur un seul hôte.
- **Modèle de données** : `Dataset` appartient à l'organisation entière
  (`organization_id`), pas seulement à qui l'a uploadé — cohérent avec le
  principe d'équipe partagée. `columns_json` stocke le schéma (nom + type)
  calculé une seule fois à l'upload, pour ne pas re-parser le fichier à
  chaque affichage de la liste.
- **Upload synchrone, sans tâche de fond** : le fichier est lu entièrement en
  mémoire (`await file.read()`) puis parsé avant de répondre — acceptable
  tant qu'il n'y a pas de file de tâches (Lot 3) et que la limite de taille
  (`MAX_UPLOAD_SIZE_MB`, 200 Mo par défaut) reste raisonnable pour une
  requête HTTP directe. Au-delà, ou pour une vraie barre de progression, ce
  sera le même mécanisme que l'entraînement (RQ + Redis).
- **Isolation en profondeur** : filtrage systématique par
  `Dataset.organization_id == current_user.organization_id` en base, **et**
  l'`organization_id` fait partie du chemin sur disque — deux niveaux
  indépendants, pas un seul point de défaillance.

## 7. Conventions reprises de CIAM

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
