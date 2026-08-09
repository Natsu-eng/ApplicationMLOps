# DataLab Pro — Backend (API FastAPI)

Backend de **DataLab Pro**, SaaS destiné aux bureaux d'études pour entraîner et
évaluer leurs propres modèles de machine learning et de deep learning (vision,
détection d'anomalies) sur leurs propres données. Architecture inspirée de
**CIAM** (`concrete-ai-platform`), déjà en production — voir
[`../README.md`](../README.md) pour la vue d'ensemble produit.

## État d'avancement

Ce backend est construit **lot par lot**, chaque lot livrant quelque chose qui
fonctionne (voir [`workflow.md`](workflow.md) pour le détail).

> **Lot 3 (entraînement ML supervisé) — état actuel.** Authentification,
> upload/catalogage de datasets, et entraînement de bout en bout
> (LightGBM/XGBoost/CatBoost + Optuna + SHAP + intervalles conformes CQR) en
> tâche de fond (RQ + Redis) avec suivi de progression en direct.

## Stack

| Couche | Technologie | Version |
| --- | --- | --- |
| API | FastAPI + Uvicorn (Gunicorn en production) | 0.136.1 |
| ORM | SQLAlchemy | 2.0.49 |
| Base de données | PostgreSQL (prod) / SQLite (dev, par défaut) | — |
| Configuration | pydantic-settings (lecture de `.env`) | 2.10.1 |
| Authentification | JWT HS256 (`python-jose`) + bcrypt | 3.5.0 / 5.0.0 |
| Datasets | pandas + openpyxl (Excel) + pyarrow (Parquet) | 2.2.2 |
| ML | scikit-learn + LightGBM + XGBoost + CatBoost | 1.3.2 / 4.3.0 / 2.0.3 / 1.2.5 |
| Recherche d'hyperparamètres | Optuna (TPE) | 3.6.1 |
| Explicabilité | SHAP (TreeExplainer) | 0.45.0 |
| File de tâches | RQ + Redis | 1.16.2 / 5.0.4 |

## Démarrage local

```bash
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1    # PowerShell — cmd.exe : .venv\Scripts\activate.bat — Linux/Mac : source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env        # cp sur Linux/Mac — les valeurs par défaut suffisent en dev
uvicorn api.main:app --reload --port 8000
# → http://localhost:8000/docs (documentation interactive Swagger)
```

L'entraînement de modèles (Lot 3) a besoin en plus de **Redis** et d'un
**worker** — sans ça, un job reste bloqué en `queued` :

```bash
# Redis (le plus simple : Docker, même hors docker-compose)
docker run -d --name datalab_redis -p 6379:6379 redis:7-alpine

# Worker — dans un second terminal, même venv activé
cd backend
python -m workers.run_worker
```

## Structure

```text
backend/
├── api/
│   ├── main.py             ← point d'entrée FastAPI : CORS, cycle de vie, healthcheck, routers
│   ├── core/
│   │   ├── config.py        ← paramètres applicatifs centralisés (pydantic-settings)
│   │   ├── database.py       ← connexion SQLAlchemy, session, Base ORM
│   │   ├── models.py          ← Organization, User, Dataset, TrainingJob, MLModel
│   │   ├── security.py         ← JWT (python-jose) + hashing bcrypt
│   │   ├── storage.py           ← chemins des fichiers (datasets + modèles) sur disque
│   │   └── job_queue.py           ← file RQ + connexion Redis
│   └── routers/
│       ├── auth.py               ← inscription, connexion, profil, gestion d'équipe
│       ├── datasets.py             ← upload, liste, aperçu, suppression de datasets
│       └── training.py               ← lancement, suivi et résultat des entraînements
├── services/
│   ├── datasets.py             ← lecture/validation pure des fichiers tabulaires
│   ├── ml_task.py                ← détection classification/régression
│   ├── ml_preprocessing.py         ← dédoublonnage, split anti-fuite, imputation/encodage
│   └── ml_training.py                ← Optuna, sélection sur CV, SHAP, CQR Mondrian
├── workers/
│   ├── run_worker.py             ← point d'entrée du worker (SimpleWorker, portable Windows/Linux)
│   └── training_worker.py          ← fonction exécutée par le worker pour chaque job
├── storage/{datasets,models}/      ← fichiers uploadés + artefacts entraînés (gitignorés, volume Docker)
├── database/                 ← base SQLite de développement (générée au démarrage, gitignorée)
├── tests/                     ← pytest — voir section Tests ci-dessous
├── requirements.txt
├── .env.example                ← variables documentées, aucune valeur réelle
├── Dockerfile
├── README.md                     ← ce fichier
├── ARCHITECTURE.md                ← schéma des couches et conventions techniques
└── workflow.md                     ← avancement lot par lot, décisions prises
```

## Tests

Suite pytest, base de données isolée (SQLite temporaire recréée à chaque
test, jamais la base de dev), pas besoin de backend/frontend démarrés.
L'entraînement (Lot 3) y est testé sans dépendre de Redis : les fonctions
pures (`services/ml_training.py`, `services/ml_preprocessing.py`) sont
appelées directement, et le router `/training/jobs` est testé avec la file
RQ mockée (voir `tests/test_training_api.py`).

```bash
cd backend
pytest                 # tous les tests
pytest -v tests/test_ml_training.py   # un fichier en particulier
```

## Docker

```bash
# depuis backend/
cp .env.example .env   # ajuster POSTGRES_PASSWORD si besoin

# depuis la racine du dépôt
docker compose up -d --build
docker compose logs -f backend
```

## Variables d'environnement

Voir [`.env.example`](.env.example) — entièrement commenté, aucune valeur
réelle n'y figure. Le vrai `.env` n'est jamais suivi par git (`.gitignore`).

---

Documentation complète : [`../README.md`](../README.md) (vue d'ensemble
produit) · [`ARCHITECTURE.md`](ARCHITECTURE.md) (schéma technique) ·
[`workflow.md`](workflow.md) (avancement lot par lot)
