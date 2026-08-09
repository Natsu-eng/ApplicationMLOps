# ARCHITECTURE.md — Backend DataLab Pro

> **Lot 4a — prédiction sur modèle entraîné.** Ce document est vivant : il
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

## 2. Schéma des couches (état Lot 3)

```text
┌────────────────────────────────────────────────────┐
│  FRONTEND  (React — voir ../frontend/)               │
│    AppShell + components/ui/ (design system)           │
│    Training : formulaire + polling progression + résultat│
└─────────────────────┬────────────────────────────────┘
                       │ HTTP JSON (Bearer JWT)
                       ▼
┌────────────────────────────────────────────────────┐
│  api/main.py — FastAPI                                │
│    routers auth (/auth), datasets (/datasets),          │
│    training (/training)                                   │
└─────────────────────┬────────────────────────────────┘
              ┌────────┼──────────────────┐
              ▼        ▼                  ▼
┌───────────────┐ ┌───────────────┐ ┌─────────────────────┐
│ api/core/         │ services/datasets   │ api/routers/training.py │
│ {security,database,│ (lecture pure)      │  crée TrainingJob,          │
│ models,storage}.py │                     │  enfile sur RQ (non bloquant)│
└───────────────┘ └───────────────┘ └───────────┬─────────┘
                                                  │ redis (queue "training")
                                                  ▼
                                     ┌─────────────────────────┐
                                     │  WORKER — process séparé    │
                                     │  workers/run_worker.py         │
                                     │  (SimpleWorker+TimerDeathPenalty,│
                                     │   portable Windows/Linux)         │
                                     │    → workers/training_worker.py     │
                                     │    → services/ml_preprocessing.py     │
                                     │      (dédoublonnage, split anti-fuite)│
                                     │    → services/ml_training.py           │
                                     │      (Optuna, SHAP, CQR Mondrian)        │
                                     │    → persiste MLModel + artefact joblib   │
                                     │      + progression DB à chaque étape        │
                                     └─────────────────────────────────────────────┘
```

Le frontend ne parle jamais au worker directement : il interroge
`GET /training/jobs/{id}` par polling, qui lit la même table `TrainingJob`
que le worker met à jour — une seule source de vérité pour la progression.

Ce schéma se complète lot par lot :

| Lot | Ajout |
| --- | --- |
| 1 (livré) | `api/core/security.py` (JWT + bcrypt), `api/core/models.py` (User, Organization), `api/routers/auth.py` |
| 2 (livré) | `api/routers/datasets.py`, `api/core/storage.py`, `services/datasets.py` (première brique de la couche `services/`) |
| 3 (livré) | `api/core/job_queue.py` (RQ + Redis), `workers/` (worker portable), `services/ml_*.py`, `api/routers/training.py` |
| 4-5 | `services/` continue de grandir — visualisations Plotly, catalogue ML complet (sklearn, SMOTE, clustering) |

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
| `REDIS_URL` | `api/core/job_queue.py` | `redis://localhost:6379/0` |
| `OPTUNA_TRIALS_DEFAULT` | `api/routers/training.py` | `20` |
| `CV_FOLDS_DEFAULT` | `api/routers/training.py` | `4` |
| `SHAP_SAMPLE_SIZE` | `services/ml_training.py` | `500` |
| `CQR_ALPHA` | `services/ml_training.py` | `0.20` (intervalle à 80 %) |
| `CQR_N_STRATA` | `services/ml_training.py` | `5` |

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

## 7. Entraînement ML supervisé (Lot 3)

- **Méthodologie source** : un notebook de référence partagé par l'équipe
  (`Notebook_Pipeline_Resistance.ipynb`, lu intégralement puis supprimé du
  dépôt — sa méthodologie est reprise ici, pas son sujet métier). Points
  repris tels quels :
  - **sélection du meilleur modèle sur le score de validation croisée,
    jamais sur le score test** — le score test n'est rapporté qu'à titre
    d'estimation finale (`services/ml_training.py::train_and_evaluate`).
  - **split et CV groupés anti-fuite** : si l'utilisateur fournit une
    colonne de groupe (ex : plusieurs mesures d'un même échantillon),
    `GroupShuffleSplit`/`GroupKFold` remplacent le split/CV classiques, avec
    une vérification explicite (assertion) qu'aucun groupe ne se retrouve
    des deux côtés (`services/ml_preprocessing.py::DataLeakageError`).
  - **CQR Mondrian** : intervalles de confiance conformes calibrés par
    strate de prédiction (5 strates par défaut), pas un quantile unique —
    corrige la sous-couverture aux valeurs extrêmes d'un split conformal
    simple.
  - **SHAP TreeExplainer** pour l'explicabilité globale (importance moyenne
    par feature), cohérent avec des modèles d'arbres.
- **Pourquoi seulement 3 algorithmes (LightGBM/XGBoost/CatBoost)** : tous
  compatibles nativement avec SHAP `TreeExplainer` et la régression
  quantile (nécessaire pour CQR) — permet une profondeur d'explicabilité et
  de calibration uniforme sur les trois. Le catalogue sklearn plus large
  (linéaire, SVM, KNN, + SMOTE) arrive au Lot 5 sans cette même profondeur.
- **Tâche de fond obligatoire** : un entraînement (3 algos × recherche
  Optuna × validation croisée) prend de quelques dizaines de secondes à
  plusieurs minutes — jamais dans le cycle requête/réponse HTTP. La requête
  `POST /training/jobs` ne fait que créer la ligne `TrainingJob` et enfiler
  le job RQ ; tout le calcul a lieu dans `workers/training_worker.py`.
- **RQ sur Windows** : la configuration RQ par défaut suppose Unix
  (`os.fork()` pour isoler chaque job, `signal.SIGALRM` pour les timeouts).
  Les deux sont absents sur Windows. `workers/run_worker.py` utilise
  `SimpleWorker` (exécution dans le process du worker, pas de fork) et
  `TimerDeathPenalty` (timeout par thread) — un seul point d'entrée qui
  fonctionne identiquement en dev Windows et en conteneur Linux
  (`docker-compose.yml`, service `worker`), pas un cas spécial cousu pour
  une seule plateforme.
- **Progression sans WebSocket** : le worker écrit `progress_step`/
  `progress_percent` dans `TrainingJob` à chaque étape (par exemple à
  chaque essai Optuna) ; le frontend fait du polling REST
  (`GET /training/jobs/{id}` toutes les 3 secondes tant qu'un job est actif).
  Suffisant et plus simple à fiabiliser qu'un WebSocket pour ce volume
  d'événements — à réévaluer si le besoin de vrai temps réel se confirme.
- **Persistance du résultat** : `MLModel` stocke les métriques, le résumé
  SHAP et les paramètres CQR en JSON (interrogeables sans désérialiser
  l'artefact), et référence un bundle `joblib` (modèle + preprocessor +
  régresseurs de quantile CQR) sur disque
  (`storage/models/{organization_id}/{training_job_id}.joblib`) — base du
  registre de modèles versionné prévu au Lot 9 (pas encore de versioning ni
  d'endpoint de téléchargement de l'artefact brut, mais l'inférence
  elle-même existe depuis le Lot 4a — voir section 8).

## 8. Prédiction / inférence (Lot 4a)

- **Referme la boucle ouverte au Lot 3** : un modèle entraîné sans pouvoir
  être réutilisé n'a pas de valeur pour un bureau d'études — signalé
  explicitement par un retour utilisateur avant le début du Lot 4b.
- `services/ml_inference.py::predict_one` charge le bundle joblib
  (`preprocessor` + `model` + éventuels régresseurs de quantile `cqr`),
  construit une ligne à partir d'un dict `{colonne: valeur}` fourni par le
  frontend, applique le même `preprocessor.transform()` qu'à l'entraînement
  (jamais un `fit` — les statistiques d'imputation/normalisation restent
  celles apprises sur le train), puis prédit.
- **Régression** : recalcule l'intervalle conforme (CQR) sur la nouvelle
  observation avec les mêmes régresseurs de quantile et les mêmes strates
  que l'entraînement (`strata_bounds`/`qhat_per_stratum` persistés dans le
  bundle) — pas un intervalle générique, celui calibré pour ce modèle
  précis. Le `clip_negative` décidé à l'entraînement (cible historiquement
  positive ou non) est réappliqué à l'identique.
- **Classification** : renvoie la classe prédite (via `class_names` persisté
  dans le bundle, pour ne jamais exposer un indice numérique brut à
  l'utilisateur) et les probabilités par classe si le modèle les expose.
- `MLModel.feature_schema_json` (nom + type de chaque variable d'entrée,
  dérivé du schéma du dataset au moment de l'entraînement) permet au
  frontend de générer un formulaire de saisie adapté sans redemander le
  dataset d'origine — première fois qu'un champ est ajouté à une table
  déjà existante, d'où l'introduction de `_add_column_if_missing` (migration
  additive idempotente, voir `api/core/database.py`) plutôt que d'exiger
  une base vierge à chaque évolution de schéma.

## 9. Conventions reprises de CIAM

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
