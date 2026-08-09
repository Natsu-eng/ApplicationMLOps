# workflow.md — Backend DataLab Pro : avancement lot par lot

> Mis à jour à chaque lot livré. Décisions produit issues du diagnostic de
> migration présenté et validé avant tout code : modèle **Organisation/équipe**
> pour le multi-utilisateurs, **ML classique tabulaire porté en premier**
> (avant la vision), **RQ + Redis** pour les tâches d'entraînement de fond,
> positionnement **généraliste multi-secteurs**.

## Lot 0 — Squelette (livré)

- [x] `api/main.py` : l'application FastAPI démarre, CORS configuré, `GET /api/health`
- [x] `api/core/config.py` : configuration centralisée (pydantic-settings, lecture de `backend/.env`)
- [x] `api/core/database.py` : connexion SQLAlchemy (PostgreSQL en prod, SQLite en dev), `Base` ORM prête à recevoir des modèles au Lot 1
- [x] `Dockerfile` (backend) + `docker-compose.yml` (racine) : services `db` (PostgreSQL) + `backend`
- [x] Frontend Vite + React + TypeScript + Tailwind : appelle `GET /api/health` au chargement, affiche le statut

**Vérifié** : `uvicorn api.main:app --reload` démarre et répond sur `/api/health` ; `npm run build` du frontend passe sans erreur.

## Lot 1 — Authentification + Organisation/équipe (livré)

- [x] `api/core/models.py` : `Organization` (bureau d'études) et `User` (`role` = `owner`/`member`, FK `organization_id`) — premier modèle multi-tenant, toute donnée métier future portera `organization_id`
- [x] `api/core/security.py` : JWT HS256 (`python-jose`) + hashing bcrypt (12 rounds) — pattern identique à `concrete-ai-platform/backend/api/core/security.py`
- [x] `api/routers/auth.py` :
  - `POST /auth/register` — crée une Organisation **et** son premier utilisateur (`owner`) en une seule opération
  - `POST /auth/login` — `OAuth2PasswordRequestForm`, retourne un JWT Bearer (24h)
  - `GET /auth/me`, `PATCH /auth/me`, `PATCH /auth/me/password`, `POST /auth/logout`
  - `GET /auth/team/members` — liste les membres de **son** organisation uniquement
  - `POST /auth/team/members` — réservé au `owner` (`require_owner`), ajoute un membre à son organisation
- [x] `api/main.py` : router `auth` monté ; `init_db()` enregistre désormais les tables `organizations`/`users`
- [x] Frontend : `AuthContext` (token en `localStorage`, vérifie la session au chargement via `/auth/me`), `ProtectedRoute`, pages `Login`, `Register`, `Dashboard` (profil + équipe + formulaire d'ajout de membre pour le owner), routing `react-router-dom`

**Vérifié** :

- Backend, via curl, sur SQLite **et** PostgreSQL réel (base `datalab` créée manuellement dans pgAdmin) : inscription de deux organisations distinctes, connexion, ajout de membre par le owner, **refus 403** d'un membre qui tente d'ajouter un membre, `GET /auth/team/members` de l'organisation B ne renvoie **jamais** les membres de l'organisation A (isolation confirmée), erreurs (mauvais mot de passe, email déjà utilisé, token absent) toutes correctement codées.
- Frontend : `npm run build` sans erreur TypeScript ; flux register → `/auth/me` → health vérifié en conditions réelles à travers le proxy Vite (`http://localhost:5173/auth/register`, `/auth/me`, `/api/health`).
- Non vérifié : le rendu visuel réel dans un navigateur (pas d'outil d'interaction navigateur disponible dans cette session) — seule la mécanique réseau/état a été testée.

**Ce qui n'est volontairement pas dans ce lot** (à ajouter plus tard si besoin) : réinitialisation de mot de passe par e-mail (nécessite l'infrastructure SMTP, pas encore montée), invitation par e-mail (l'ajout de membre se fait aujourd'hui par mot de passe temporaire direct, décidé par le owner), panneau d'administration multi-organisations.

## Lot 2 — Upload et gestion de datasets tabulaires (livré)

- [x] `api/core/models.py::Dataset` — appartient à l'organisation entière (pas seulement à qui l'a uploadé), cohérent avec le principe d'équipe partagée
- [x] `api/core/storage.py` — stockage disque local, `storage/datasets/{organization_id}/{dataset_id}{ext}` (isolation défense-en-profondeur en plus du filtrage DB)
- [x] `services/datasets.py` — logique pure (lecture csv/parquet/xlsx/xls/json, extraction de schéma, échantillonnage) : premier module du dossier `services/` planifié dans `ARCHITECTURE.md`, portage simplifié de `src/data/data_loader.py` de l'app historique
- [x] `api/routers/datasets.py` : `POST /datasets` (upload), `GET /datasets` (liste), `GET /datasets/{id}` (détail + schéma), `GET /datasets/{id}/preview` (échantillon), `DELETE /datasets/{id}` — tous filtrés par organisation, accessibles à tout membre (pas réservé au owner)
- [x] `docker-compose.yml` : volume `backend/storage` monté pour persister les datasets entre redémarrages
- [x] Frontend : nouveau système de composants (`components/ui/` : Card, Button, Badge, Avatar, Input, Modal), `AppShell` (navigation commune), page `Datasets` (zone de dépôt drag & drop, grille de cartes, aperçu en modale), Dashboard et pages d'authentification alignés sur le même système visuel

**Vérifié** :

- Backend, via curl (upload multipart réel, pas seulement JSON) : upload csv → schéma et comptage de lignes/colonnes corrects, aperçu, suppression (fichier physique supprimé du disque, vérifié), isolation confirmée entre deux organisations (liste vide + 404 sur accès croisé à l'id d'un autre org).
- **Bug réel trouvé et corrigé pendant les tests** : `DatasetSummary.model_validate(dataset, from_attributes=True)` faisait planter la sérialisation (500) car le champ Pydantic `uploaded_by: str` entrait en collision de nom avec la relation SQLAlchemy `Dataset.uploaded_by` (objet `User`) — corrigé en construisant la réponse explicitement plutôt que via l'auto-mapping par attribut.
- Frontend : `npm run build` sans erreur TypeScript ; flux complet (login → upload → liste) vérifié à travers le proxy Vite réel.
- Non vérifié : rendu visuel réel en navigateur (pas d'outil d'interaction navigateur dans cette session) — les deux serveurs de dev sont laissés actifs pour vérification visuelle directe.

**Scope volontairement limité** : upload synchrone en mémoire (limite 200 Mo, configurable) — pas de tâche de fond ni de barre de progression pour l'instant (arrive avec la file de tâches du Lot 3) ; pas de nettoyage/preprocessing dans ce lot (juste upload + catalogage + aperçu) ; pas de support Dask pour les très gros fichiers (identifié comme portable plus tard si besoin réel).

## Lot 3 — Entraînement ML supervisé de bout en bout (livré)

Méthodologie alignée sur un notebook de référence partagé par l'équipe
(`Notebook_Pipeline_Resistance.ipynb`, lu intégralement puis supprimé du
dépôt comme convenu — sa méthodologie est documentée ici et dans
`ARCHITECTURE.md`, pas son contenu métier). Portée volontairement au
supervisé (classification/régression) ; le clustering (non supervisé
tabulaire) arrive au Lot 5 avec le catalogue ML complet.

- [x] `api/core/models.py::TrainingJob` (suivi/progression) et `MLModel`
  (résultat : métriques, SHAP, CQR, fiche modèle)
- [x] `api/core/job_queue.py` — file RQ + Redis (`training_queue`)
- [x] `workers/training_worker.py` — fonction exécutée par le worker (process
  séparé), persiste la progression directement en base à chaque étape
- [x] `workers/run_worker.py` — point d'entrée `SimpleWorker` +
  `TimerDeathPenalty` : **RQ utilise par défaut `os.fork()` et
  `signal.SIGALRM`, tous deux absents sur Windows** — remplacés par une
  exécution mono-process et un timeout par thread, qui fonctionnent
  identiquement sur Windows et Linux/Docker (pas un correctif Windows cachée,
  le même point d'entrée partout)
- [x] `services/ml_task.py` — détection classification/régression
- [x] `services/ml_preprocessing.py` — dédoublonnage exact, split anti-fuite
  (`GroupShuffleSplit`/`GroupKFold` si une colonne de groupe est fournie,
  avec vérification explicite par assertion), imputation + normalisation/one-hot
- [x] `services/ml_training.py` — cœur du lot :
  - 3 algorithmes comparés (LightGBM, XGBoost, CatBoost), recherche
    d'hyperparamètres **Optuna** (TPE) par algorithme, validation croisée
    groupée si applicable
  - **sélection du meilleur modèle sur le score de CV, jamais sur le score
    test** (le score test n'est qu'une estimation finale rapportée)
  - métriques + intervalles de confiance par bootstrap (R²/RMSE/MAE en
    régression, accuracy/F1/precision/recall/AUC en classification), plus
    ΔR² train-test comme indicateur de surapprentissage
  - **explicabilité SHAP** (`TreeExplainer`, importance globale par feature)
  - **CQR (Conformal Quantile Regression), variante Mondrian** en régression :
    deux régresseurs de quantile, calibration par strate de prédiction (corrige
    la sous-couverture aux valeurs extrêmes d'un split conformal simple)
  - fiche modèle (model card) et artefact joblib (modèle + preprocessor + CQR)
    persistés — base du futur registre de modèles (Lot 9)
- [x] `api/routers/training.py` : `POST /training/jobs` (lance, enfile sur
  RQ), `GET /training/jobs` (liste, isolée par organisation), `GET
  /training/jobs/{id}` (statut/progression, pour le polling), `GET
  /training/jobs/{id}/model` (résultat complet)
- [x] `docker-compose.yml` : service `redis` + service `worker` (même image
  que le backend, `command: python -m workers.run_worker`), volume
  `backend/storage` partagé entre `backend` et `worker`
- [x] Frontend : page `Training` (choix dataset/cible/colonne de groupe,
  curseurs essais Optuna/taille de test, historique avec **progression en
  temps réel par polling**), `ModelResultModal` (métriques, barres
  d'importance SHAP, couverture CQR, fiche modèle)

**Vérifié** :

- Service d'entraînement testé en direct (hors file, appel Python direct)
  sur données synthétiques : régression avec colonne de groupe (doublons
  volontaires correctement retirés, fuite vérifiée nulle), classification
  binaire — SHAP retrouve correctement les variables les plus influentes
  (cohérent avec les coefficients utilisés pour générer les données), CQR
  atteint une couverture empirique proche de la cible (85 %/80 % puis
  82,5 %/80 % sur deux runs).
- **Pipeline asynchrone complet testé via l'API HTTP réelle + un vrai worker
  RQ séparé** (pas un appel direct) : upload dataset → `POST
  /training/jobs` → progression visible par polling (`queued` → `running`
  avec étapes détaillées → `completed`) → résultat complet via `GET
  .../model`. Isolation confirmée : une organisation B ne voit aucun job ni
  dataset d'une organisation A (liste vide + 404 sur accès direct par id).
- Testé sur PostgreSQL réel, avec Redis lancé via Docker (Docker Desktop
  n'était pas démarré au début du lot — relancé en cours de route).
- **Deux bugs d'incompatibilité Windows trouvés et corrigés pendant les
  tests** (RQ est conçu autour de primitives Unix) : `os.fork()` absent →
  `SimpleWorker` ; `signal.SIGALRM` absent → `TimerDeathPenalty`. Voir
  `workers/run_worker.py`.
- Frontend : `npm run build` sans erreur TypeScript ; flux complet (upload →
  lancement → suivi → résultat) vérifié à travers le proxy Vite réel avec
  un vrai worker actif en arrière-plan.
- Non vérifié : rendu visuel réel en navigateur (pas d'outil d'interaction
  navigateur dans cette session) — les trois processus (API, worker,
  frontend) sont laissés actifs pour vérification directe.

**Scope volontairement limité** : catalogue restreint à 3 algorithmes de
gradient boosting (permet SHAP + CQR uniformes et de haute qualité plutôt
qu'un catalogue large sans profondeur) — le catalogue sklearn complet
(linéaire, SVM, KNN, SMOTE...) arrive au Lot 5 ; pas d'endpoint de
prédiction/inférence sur un modèle entraîné (l'artefact est persisté mais
pas encore servi) ; pas de suppression de job/modèle depuis l'API ; pas de
WebSocket (polling REST uniquement pour l'instant, suffisant et plus simple
à fiabiliser) ; pas de clustering (non supervisé tabulaire, Lot 5) ; SHAP
limité à l'importance globale (dependence/waterfall plots plus riches :
Lot 4).

### Correctifs post-livraison (trouvés en usage réel par l'utilisateur)

- **Bug SHAP multiclasse** : `explainer.shap_values(X)` renvoie soit une
  liste d'une matrice par classe (API historique), soit un seul tableau
  3D `(n_échantillons, n_features, n_classes)` (API unifiée récente) selon
  la version de SHAP/le backend d'arbre. `_compute_shap_summary` ne gérait
  que le premier cas — sur un dataset réel à 3 classes (Iris), le second
  cas produisait `IndexError: only integer scalar arrays can be converted
  to a scalar index`. Corrigé dans `services/ml_training.py`, testé sur le
  dataset réel de l'utilisateur (job relancé avec succès) et couvert par
  `tests/test_ml_training.py::test_multiclass_classification_shap_does_not_crash`.
- **Suite de tests pytest ajoutée rétroactivement** (`backend/tests/`,
  22 tests) : jusqu'ici les Lots 1-3 n'étaient vérifiés qu'à la main
  (curl/scripts jetables) — désormais couverts par des tests qui restent
  dans le dépôt. Voir la section *Tests* de `backend/README.md`.

## Lot 4a — Prédiction sur un modèle entraîné + guidage (livré)

Rétrospective déclenchée par un retour utilisateur explicite : après le
Lot 3, un modèle s'entraînait mais ne servait à rien — impossible de
l'utiliser sur une nouvelle donnée. C'est désormais corrigé, avant tout
travail de visualisation (Lot 4b).

- [x] `services/ml_inference.py` — charge le bundle joblib (modèle +
  preprocessor + régresseurs CQR), construit une ligne à partir de la saisie
  utilisateur, prédit (+ intervalle de confiance en régression, +
  probabilités par classe en classification)
- [x] `api/core/models.py::MLModel.feature_schema_json` — schéma (nom +
  type) des variables d'entrée, dérivé du schéma du dataset au moment de
  l'entraînement, pour que le frontend génère un formulaire adapté sans
  redemander le dataset d'origine
- [x] **Première migration de schéma additive** (`api/core/database.py::_add_column_if_missing`)
  — pattern idempotent façon CIAM, introduit à ce lot car c'est la première
  fois qu'un champ est ajouté à une table déjà existante (`create_all()` ne
  modifie jamais les tables déjà créées)
- [x] `POST /training/jobs/{id}/predict` — isolé par organisation comme le reste
- [x] Frontend : `PredictionForm` (formulaire généré dynamiquement depuis
  `feature_schema`, intégré à `ModelResultModal`), `ui/Tooltip.tsx` +
  info-bulles en langage clair sur les métriques (R², RMSE, F1, AUC-ROC,
  score de CV, SHAP, CQR) — répond au besoin de guidage pour des
  utilisateurs qui ne sont pas data scientists de métier
- [x] **Sélection manuelle des variables d'entraînement** exposée dans le
  formulaire `Training` (le backend l'acceptait déjà depuis le Lot 3 côté
  API, seule l'UI manquait) — décocher une colonne sans valeur prédictive
  (ex. un identifiant) plutôt que de tout laisser par défaut

**Vérifié** :

- Tests pytest (`tests/test_inference.py`) : entraînement réel (pas mocké)
  exécuté en process de test, bundle persisté, prédiction avec intervalle
  CQR plausible, rejet propre d'une variable manquante.
- **Bout en bout en conditions réelles**, API + worker RQ réel + Redis via
  Docker : entraînement sur le dataset Iris de l'utilisateur avec sélection
  manuelle de 4 variables, puis prédiction sur deux fleurs aux
  caractéristiques opposées — `Iris-setosa` (91,9 % de confiance) pour de
  petites pétales, `Iris-virginica` (90,3 %) pour de grandes pétales :
  cohérent botaniquement, pas seulement "ça ne plante pas".
- **Incident Redis trouvé et corrigé pendant ce test** : Docker Desktop
  s'était arrêté entre deux sessions de travail, le conteneur Redis
  autonome utilisé en dev (hors `docker-compose`) ne redémarrait pas tout
  seul avec lui — reconfiguré avec `--restart unless-stopped`.
- 24/24 tests pytest toujours au vert après ce lot.

## Prochains lots (résumé — détail complet dans le diagnostic de migration et les échanges de cadrage)

| Lot | Contenu | Livrable testable |
| --- | --- | --- |
| 4b | EDA dataset (stats, histogrammes, corrélations, valeurs manquantes) + visualisations d'évaluation (matrice de confusion, ROC/PR, résidus) — **Recharts**, pas Plotly (plus léger, thémable à notre design system, déjà éprouvé par CIAM) | Explorer un dataset avant d'entraîner ; voir les courbes d'un modèle, pas seulement ses métriques brutes |
| 4c | Ingénierie de variables : créer des variables dérivées (ratios, transformations, extraction de dates) avant l'entraînement | Un utilisateur peut créer une nouvelle colonne calculée et l'utiliser comme feature |
| 5 | Catalogue ML complet comparé automatiquement (RandomForest, régression linéaire/logistique, SVM, KNN, Naive Bayes, + SMOTE, + clustering) | Un non-expert bénéficie d'un pool de candidats large sans avoir à choisir un algorithme |
| 6-8 | Upload / entraînement / évaluation vision (détection d'anomalies) | Parité fonctionnelle côté vision |
| 9 | Registre de modèles unifié (versioning, export) | Remplace les 3 mécanismes de persistance de l'app historique |
| 10 | Durcissement SaaS (erreurs, audit, quotas) | Prêt pour un client pilote |

Ce fichier sera complété à chaque lot livré avec le détail réel (fichiers
créés, endpoints exposés, décisions techniques prises en cours de route) —
même format que le `workflow.md` de CIAM, sourcé fichier par fichier. Voir
aussi [`../recap.md`](../recap.md) pour une synthèse lisible de l'ensemble,
mise à jour au même rythme.
