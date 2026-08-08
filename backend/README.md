# DataLab Pro — Backend (API FastAPI)

Backend de **DataLab Pro**, SaaS destiné aux bureaux d'études pour entraîner et
évaluer leurs propres modèles de machine learning et de deep learning (vision,
détection d'anomalies) sur leurs propres données. Architecture inspirée de
**CIAM** (`concrete-ai-platform`), déjà en production — voir
[`../README.md`](../README.md) pour la vue d'ensemble produit.

## État d'avancement

Ce backend est construit **lot par lot**, chaque lot livrant quelque chose qui
fonctionne (voir [`workflow.md`](workflow.md) pour le détail).

> **Lot 0 (squelette) — état actuel.** L'API démarre, expose `GET /api/health`,
> se connecte à la base de données. Aucune route métier (auth, datasets,
> entraînement...) n'existe encore — c'est volontaire.

## Stack

| Couche | Technologie | Version |
|---|---|---|
| API | FastAPI + Uvicorn (Gunicorn en production) | 0.136.1 |
| ORM | SQLAlchemy | 2.0.49 |
| Base de données | PostgreSQL (prod) / SQLite (dev, par défaut) | — |
| Configuration | pydantic-settings (lecture de `.env`) | 2.10.1 |

## Démarrage local

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows — sur Linux/Mac : source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env        # cp sur Linux/Mac — les valeurs par défaut suffisent en dev
uvicorn api.main:app --reload --port 8000
# → http://localhost:8000/docs (documentation interactive Swagger)
```

## Structure

```
backend/
├── api/
│   ├── main.py            ← point d'entrée FastAPI : CORS, cycle de vie, healthcheck
│   └── core/
│       ├── config.py       ← paramètres applicatifs centralisés (pydantic-settings)
│       └── database.py      ← connexion SQLAlchemy, session, Base ORM
├── database/                 ← base SQLite de développement (générée au démarrage, gitignorée)
├── requirements.txt
├── .env.example                ← variables documentées, aucune valeur réelle
├── Dockerfile
├── README.md                     ← ce fichier
├── ARCHITECTURE.md                ← schéma des couches et conventions techniques
└── workflow.md                     ← avancement lot par lot, décisions prises
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
