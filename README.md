# DataLab Pro

Outil d'entraînement de modèles ML / deep learning / vision pour bureaux
d'études — en cours de migration d'un outil académique Streamlit vers un
**SaaS FastAPI + React** multi-utilisateurs. Architecture inspirée de
**CIAM** (`concrete-ai-platform`), déjà en production.

---

## État du dépôt

Le dépôt contient deux choses en parallèle, le temps de la migration :

| | Rôle |
| --- | --- |
| **`backend/` + `frontend/`** | Le nouveau SaaS, construit **lot par lot**. État actuel : **Lot 2** — auth + organisations, upload et catalogage de datasets tabulaires. Voir [`backend/workflow.md`](backend/workflow.md). |
| **`src/`, `ui/`, `helpers/`, `monitoring/`, `orchestrators/`...** | L'application **Streamlit historique**, conservée intacte comme référence pendant le portage progressif de sa logique ML vers `backend/`. Documentation : [`docs/legacy/README.md`](docs/legacy/README.md). |

Rien n'est supprimé de l'application historique tant que sa logique n'a pas
été portée et validée dans le nouveau backend — voir le diagnostic de
migration pour le détail de ce qui est réutilisable tel quel et ce qui est
du code mort identifié (à trancher au fil des lots).

---

## Démarrage rapide — nouveau SaaS

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1    # PowerShell — cmd.exe : .venv\Scripts\activate.bat — Linux/Mac : source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env        # cp sur Linux/Mac — les valeurs par défaut suffisent en dev
uvicorn api.main:app --reload --port 8000
# → http://localhost:8000/docs
```

### Frontend

```bash
cd frontend
npm install
copy .env.example .env.local   # cp sur Linux/Mac — laisser VITE_API_URL vide en dev
npm run dev
# → http://localhost:5173
```

### Docker (backend + frontend + PostgreSQL)

```bash
cp backend/.env.example backend/.env
docker compose up -d --build
docker compose logs -f backend
```

Détails complets (structure, variables d'environnement, conventions) : voir
les READMEs de chaque dossier ci-dessous.

---

## Documentation

| Document | Contenu |
| --- | --- |
| [`backend/README.md`](backend/README.md) | Backend : stack, démarrage, structure des fichiers |
| [`backend/ARCHITECTURE.md`](backend/ARCHITECTURE.md) | Backend : schéma des couches, conventions techniques |
| [`backend/workflow.md`](backend/workflow.md) | Avancement lot par lot, décisions prises |
| [`frontend/README.md`](frontend/README.md) | Frontend : stack, démarrage, structure des fichiers |
| [`docs/legacy/README.md`](docs/legacy/README.md) | Application Streamlit historique (référence pendant la migration) |

## Application Streamlit historique — toujours démarrable

```bash
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
streamlit run src/app/main.py
```

---

**Statut** : migration en cours — Lot 0 (squelette), Lot 1 (authentification et
organisations) et Lot 2 (datasets tabulaires) livrés. Voir
[`backend/workflow.md`](backend/workflow.md) pour la suite.
