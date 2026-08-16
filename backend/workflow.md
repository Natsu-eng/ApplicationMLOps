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

### Correctif — suppression d'un entraînement

Signalé en usage réel : l'historique d'entraînements grandit sans limite,
sans aucun moyen de le nettoyer (ni un job resté bloqué en `queued` faute de
worker, ni un test qu'on ne veut plus garder).

- [x] `DELETE /training/jobs/{id}` — supprime le job et le modèle associé
  (cascade DB), efface l'artefact `joblib` du disque, et tente une
  annulation best-effort du job RQ sous-jacent s'il est encore `queued`/
  `running` (sans danger si ça échoue : `training_worker.py` gère déjà
  l'absence du job en base sans planter)
- [x] Frontend : bouton supprimer sur chaque carte d'entraînement, double
  clic de confirmation (pas de popup modale intrusive)
- [x] 2 tests pytest dédiés (suppression + isolation — une organisation ne
  peut pas supprimer l'entraînement d'une autre), 26/26 sur l'ensemble de
  la suite

## Lot 4b — Exploration de données (EDA) et graphiques d'évaluation (livré)

Deux manques signalés explicitement par l'utilisateur après le Lot 3 : pas
moyen d'explorer un dataset avant de lancer un entraînement, et le résultat
d'un modèle ne montrait que des métriques chiffrées, sans graphique.
Choix de bibliothèque également tranché à ce lot : **Recharts plutôt que
Plotly** (plus léger, thémable au système de design existant — dégradé
teal/verre dépoli —, déjà éprouvé par CIAM).

- [x] `services/dataset_eda.py` — statistiques par colonne (numérique :
  moyenne/écart-type/min/max/médiane ; catégorielle : cardinalité/valeurs
  les plus fréquentes), matrice de corrélation de Pearson (colonnes
  numériques uniquement), résumé des valeurs manquantes, histogramme à la
  demande (bins pour le numérique, top-N + "Autres" pour le catégoriel) —
  `_clean_float` neutralise les `NaN`/`inf` en `None` avant sérialisation
  JSON (sinon `json.dumps` échoue silencieusement côté client)
- [x] `api/routers/datasets.py` : `GET /datasets/{id}/eda` (statistiques +
  corrélations + valeurs manquantes en un seul appel), `GET
  /datasets/{id}/histogram?column=X&bins=N` (à la demande, pour ne pas
  calculer tous les histogrammes d'un dataset large à chaque ouverture) —
  isolés par organisation comme le reste
- [x] `services/ml_training.py` — évaluation persistée au moment de
  l'entraînement (une seule fois, pas recalculée à chaque consultation) :
  `_compute_classification_evaluation` (matrice de confusion, courbes
  ROC/PR par classe en un-contre-tous pour le multiclasse),
  `_compute_regression_evaluation` (valeurs réelles/prédites et résidus sur
  le jeu de test, sous-échantillonnés à 300 points via `_downsample_curve`
  pour ne pas alourdir la réponse JSON sur un gros dataset)
- [x] `api/core/models.py::MLModel.evaluation_json` — deuxième migration
  additive (même pattern `_add_column_if_missing` qu'au Lot 4a), confirmée
  appliquée sur la base PostgreSQL réelle au redémarrage du backend
  (`[DB] Migration : colonne ml_models.evaluation_json ajoutée`)
- [x] `api/routers/training.py::MLModelDetail.evaluation` — désérialise
  `evaluation_json`, `{}` si absent (modèles entraînés avant ce lot,
  rétrocompatible sans script de backfill)
- [x] Frontend : `components/ui/Heatmap.tsx` — grille CSS maison (pas de
  dépendance graphique dédiée, Recharts n'a pas de heatmap native), deux
  variantes (`diverging` centrée sur 0 pour les corrélations, `sequential`
  pour les comptages/matrice de confusion) ; `components/datasets/EdaModal.tsx`
  (valeurs manquantes en barres, corrélations en heatmap, histogramme avec
  sélecteur de colonne, tableau récapitulatif) ; `components/training/EvaluationCharts.tsx`
  (`ClassificationCharts` : matrice de confusion + ROC + PR ;
  `RegressionCharts` : prédit-vs-réel + résidus, avec ligne de référence
  diagonale/zéro) ; bouton "Explorer" sur chaque carte dataset
  (`pages/Datasets.tsx`) ; graphiques intégrés dans `ModelResultModal` juste
  après les métriques de performance
- [x] `package.json` : `recharts` ajouté, upgrade vers la v3 en cours de
  route (la v2 était dépréciée) — a changé la signature du prop `formatter`
  du `Tooltip` (voir correctif ci-dessous)
- [x] Tests pytest : `tests/test_dataset_eda.py` (statistiques
  numériques/catégorielles, corrélations, valeurs manquantes, histogramme
  numérique et catégoriel, isolation par organisation) + extension de
  `tests/test_ml_training.py` (présence de `evaluation` en classification et
  régression, cohérence des dimensions matrice de confusion / classes)

**Vérifié** :

- Suite pytest complète : **35/35** tests au vert (contre 26 avant ce lot).
- `npm run build` (frontend) sans erreur TypeScript.
- **Bout en bout en conditions réelles**, API + worker RQ réel + PostgreSQL,
  sur les vrais datasets de l'utilisateur :
  - `GET /datasets/7/eda` (Iris, 150 lignes) : statistiques numériques et
    catégorielles correctes, `Species` bien détectée catégorielle avec ses
    3 valeurs à 50 occurrences chacune ; `GET /datasets/7/histogram` sur
    `SepalLengthCm` : 8 bins cohérents avec la distribution connue du
    dataset.
  - Entraînement classification réel sur Iris (job id 11, CatBoost,
    accuracy 0,933) : `evaluation` contient une matrice de confusion 3×3
    plausible (`[[10,0,0],[0,9,1],[0,1,9]]`), les 3 classes, des courbes
    ROC/PR par classe.
  - Entraînement régression réel sur Concrete Compressive Strength (1030
    lignes, job id 12, CatBoost, R² test 0,933) : `evaluation` contient 201
    triplets réel/prédit/résidu (échantillonnage à 300 points non
    déclenché ici, jeu de test plus petit), CQR cohérent (couverture
    empirique 83,6 % pour une cible de 80 %).
- **Processus périmés retrouvés en cours de vérification** (déjà rencontré
  au Lot 3) : un ancien `uvicorn`/worker lancé depuis l'interpréteur Python
  global (pas le `.venv`) occupait encore le port 8000 et servait du code
  d'avant ce lot — identifié via `Get-NetTCPConnection`, arrêté, backend et
  worker relancés depuis `.venv`, migration `evaluation_json` confirmée
  appliquée au redémarrage propre.
- Non vérifié : rendu visuel réel en navigateur (pas d'outil d'interaction
  navigateur dans cette session) — build TypeScript propre et payloads API
  corrects sont les seules garanties disponibles ; les trois processus
  (API, worker, frontend) sont laissés actifs pour vérification visuelle
  directe.

### Correctif — Recharts v3 et le prop `formatter` du `Tooltip`

`Tooltip.formatter` en Recharts v3 attend `Formatter<ValueType, NameType>`
où `ValueType` peut être `undefined` — un callback typé explicitement
`(v: number) => string` ne compile plus (`npm run build` échouait sur 3
occurrences : `EdaModal.tsx`, `EvaluationCharts.tsx` ×2). Corrigé en
laissant le paramètre non typé et en forçant la conversion à l'intérieur :
`formatter={(v) => Number(v).toFixed(3)}`.

### Correctif — suppression d'un entraînement terminé impossible sur PostgreSQL

Signalé en usage réel : supprimer un entraînement **terminé** (donc avec un
`MLModel` associé) échouait systématiquement en 500 sur PostgreSQL, alors
que la suppression d'un job encore `queued` (sans modèle) fonctionnait —
d'où une confusion initiale côté utilisateur (« ça a marché une fois,
maintenant plus »), le symptôme visible dépendant en fait de si le job
supprimé avait ou non produit un modèle.

- **Cause** : dans `delete_training_job` (`api/routers/training.py`), la
  route accède à `job.model` (pour effacer l'artefact `joblib`) **avant**
  `db.delete(job)`. Cet accès charge le `MLModel` dans la session
  SQLAlchemy ; au flush, l'ORM tente alors de mettre `NULL` sur
  `ml_models.training_job_id` pour "dissocier" l'objet déjà chargé — hors
  cette colonne est `NOT NULL`, ce qui lève une `IntegrityError`. Le
  `ON DELETE CASCADE` déclaré sur la contrainte FK (voir
  `MLModel.training_job_id`) n'est jamais atteint : l'ORM échoue avant.
  `passive_deletes=True` seul (ajouté sur `TrainingJob.model`, bonne
  pratique conservée) ne suffit pas ici car il ne protège que les relations
  *non chargées* au moment du flush — la nôtre l'est, explicitement, par la
  route elle-même.
- **Correctif** : suppression explicite de `job.model` en Python
  (`db.delete(job.model)`) avant `db.delete(job)`, plutôt que de compter
  sur le cascade DB dans un cas où l'objet est de toute façon déjà chargé —
  robuste indépendamment du moteur de base (Postgres/SQLite) et de l'état
  de chargement de la relation.
- **Pourquoi les tests ne l'avaient pas détecté** : `test_delete_removes_job_from_history`
  supprime un job juste après sa création (`queued`, jamais de `MLModel`
  associé) — le chemin qui plante n'était simplement jamais exercé. Nouveau
  test `test_delete_completed_job_with_model` (`tests/test_training_api.py`)
  qui insère un `MLModel` réel avant suppression, comme le ferait le
  worker.
- **Correctif frontend associé** : une suppression en échec ne montrait
  strictement rien à l'utilisateur (pas de bannière d'erreur, carte
  inchangée) — `onDelete` (`pages/Training.tsx`) affiche désormais l'erreur
  via la bannière déjà utilisée pour le chargement de la liste, et
  rafraîchit systématiquement l'historique (y compris en cas d'échec, pour
  faire disparaître une carte déjà supprimée ailleurs).
- **Vérifié** : reproduit puis corrigé en conditions réelles sur la base
  PostgreSQL de développement (`DELETE /training/jobs/7`, job terminé avec
  modèle CatBoost : 500 avant correctif, 204 après) ; suite pytest complète
  au vert après correctif (94/94, dont le nouveau test de régression).

## Lot 5 — Catalogue supervisé élargi, architecture modulable par registre (livré)

Portée stricte : ML supervisé uniquement (classification/régression). Le
non-supervisé (clustering) et SMOTE restent hors périmètre — pas amorcés
dans ce lot. Phase 1 (lecture seule, audit de l'existant + plan
d'architecture) validée avant toute implémentation ; Phase 2 livrée en 6
commits isolés sur la branche `lot5-catalogue-supervise-elargi`.

- [x] `services/ml_registry.py` (nouveau) — registre de modèles :
  `ModelSpec` déclare, pour chaque entrée, son identifiant, sa famille
  (arbre_ensemble/lineaire/distance_noyau), les tâches supportées, son
  constructeur d'estimateur (`build_estimator(task_type, seed, params,
  final_fit)` — absorbe les particularités par estimateur : `random_state`
  absent pour KNN/SVR, `probability=True` différé au candidat final pour
  SVC), son espace Optuna, son type d'explainer SHAP (`tree`/`linear`/
  `kernel`), et son appartenance au sous-ensemble par défaut (`is_default`).
  9 modèles enregistrés : LightGBM, XGBoost, CatBoost, RandomForest,
  ExtraTrees (arbres/ensembles) ; régression Ridge/LogisticRegression, un
  seul spec `linear_reg` (linéaire) ; SVM (SVR/SVC), KNN, Naive Bayes
  (distance/noyau — Naive Bayes y est rattaché par convention de routage
  vers KernelExplainer, pas par nature mathématique, commenté dans le code).
- [x] `services/ml_training.py` refactoré pour consommer le registre
  (`models_for_task()`) — **aucun nom d'algorithme en dur** dans le moteur ;
  ajouter un modèle = ajouter une entrée au registre (critère d'acceptation
  n°1 du cadrage, démontré par le commit 4 : 6 nouveaux modèles ajoutés sans
  toucher `_optimize_one_model`/`train_and_evaluate` au-delà du câblage
  registre déjà en place).
- [x] **Score de sélection robuste** (`_classification_selection_score`) —
  remplace le scorer texte `"roc_auc_ovr_weighted"` (exigeait
  `predict_proba`, aurait imposé `SVC(probability=True)` partout : mesuré
  **~9,3× plus lent par fit** — 0,786s vs 0,084s sur un dataset réel
  1025×13, 5 répétitions). Nouveau scorer : `predict_proba` si disponible,
  sinon `decision_function`, sinon repli sur l'accuracy — tous les candidats
  de classification restent comparés sur la même échelle AUC.
- [x] **SHAP routé par famille** (`_build_explainer`) — `TreeExplainer`
  (arbres, inchangé), `LinearExplainer` (linéaire), `KernelExplainer`
  (distance/noyau, fond résumé par k-means ≤ 50 points, échantillon expliqué
  ≤ 50, désactivé au-delà de 50 variables). Les deux nouveaux explainers
  reçoivent explicitement les données dans l'espace **préprocessé**
  (`X_train_proc`/`X_test_proc`), jamais brutes — un explainer nourri du
  mauvais espace produit des valeurs SHAP fausses sans lever d'erreur.
  Sanity check numérique dédié : les valeurs SHAP d'un modèle linéaire
  connu égalent `coef_ * (x - moyenne du fond)` dans l'espace préprocessé.
  `_compute_explainability` dégrade proprement (jamais de plantage du job) :
  `model_card.explainability = {"status": "ok"|"degraded", "message": ...}`,
  affiché côté frontend (`ModelResultModal`) au lieu de laisser la section
  SHAP disparaître silencieusement.
- [x] **CQR confirmé indépendant du modèle gagnant** (déjà vrai depuis le
  Lot 3 — les régresseurs de quantile sont toujours des LightGBM dédiés) :
  aucune adaptation nécessaire pour que Ridge/SVR/RandomForest/ExtraTrees/
  KNN gagnants aient un CQR fonctionnel.
- [x] **Préprocessing confirmé déjà indifférencié par famille** (Phase 1) :
  `build_preprocessor` scale déjà tout le numérique inconditionnellement —
  aucune modification nécessaire pour satisfaire le besoin de scaling de
  SVM/KNN/linéaire. `ModelSpec.requires_scaling` reste déclaratif.
- [x] **Sélection par défaut = stratégie produit "B"** — par défaut, seuls
  les 4 modèles robustes/rapides tournent (boosters + RandomForest,
  `subset="default"`) ; les 5 autres restent dans le registre, disponibles
  mais pas lancés (mécanique d'activation prête pour le Lot E, pas encore
  exposée en API/UI). Benchmark réel (dataset 1026×13, 15 essais Optuna,
  4 folds) : catalogue par défaut 235,5s, catalogue complet 251,2s (+7 % —
  nettement moins que l'estimation prudente de la Phase 1, les modèles
  ajoutés étant bon marché à fit face au coût dominant de la recherche
  Optuna des boosters, commun aux deux configurations).
- [x] Frontend `ModelResultModal.tsx` — affiche le message clair de
  `explainability_status` quand SHAP dégrade (seule modification UI de ce
  lot, cadrage explicite : pas de refonte de cartes).

**Vérifié** :

- Suite pytest complète verte à chaque commit (gate systématique, jamais un
  commit sans validation) : 134 → 138 → 143 → 145 → 146 tests au fil des 5
  commits backend (partis de 134 tests Lot 4c ; le 6ᵉ commit est
  frontend-only, pas de nouveau test pytest).
- Fold-safety (Lot A) prouvée structurellement pour les modèles à scaling
  requis (SVM/KNN/linéaire), pas seulement les arbres : le préprocesseur
  reste cloné/refit à l'intérieur de chaque fold.
- Chaque modèle du registre s'entraîne et prédit sans erreur, sur les deux
  tâches qu'il supporte, sur un dataset réel.
- Non-régression confirmée : l'assertion figée sur les 3 boosters historiques
  (`test_regression_pipeline_end_to_end`) a été rendue dynamique — un signal
  fortement linéaire peut désormais légitimement faire gagner un autre
  modèle du catalogue sans casser le test.

**Scope volontairement limité** (périmètre acté au cadrage, à ne pas
rouvrir) : clustering et SMOTE hors périmètre ; sélection expert des modèles
(UI pour activer ExtraTrees/linéaire/SVM/KNN/Naive Bayes) prévue au Lot E,
la mécanique (`subset`) est prête côté moteur mais pas exposée en API.

## Lot déséquilibre — rééquilibrage des classes par pondération (livré)

Portée stricte : `class_weight`/`sample_weight` (pondération native aux
modèles), PROPOSÉ à l'utilisateur, jamais appliqué d'office. SMOTE et tout
rééchantillonnage synthétique restent hors périmètre (fuite-sensible,
réservés à un lot expert ultérieur). Phase 1 (lecture seule, audit + plan)
validée avant implémentation ; Phase 2 livrée en 4 commits isolés sur la
branche `lot-desequilibre-class-weight`.

- [x] **Découverte clé qui a simplifié le lot** : `class_weight="balanced"`
  (sklearn) est l'équivalent exact de `sample_weight =
  compute_sample_weight("balanced", y)` passé à `.fit()` — c'est son
  implémentation interne. Ce mécanisme unique est supporté nativement par
  sklearn, LightGBM, XGBoost et CatBoost (vérifié par introspection des
  signatures réellement installées), binaire et multiclasse, sans branche
  par librairie. Seul **KNN** n'a aucune notion de pondération d'échantillon
  (vote par plus proches voisins) — GaussianNB, contrairement à l'hypothèse
  initiale du cadrage, le supporte bien.
- [x] `services/ml_registry.py` — `ModelSpec.supports_rebalancing: bool`,
  déclaratif, `True` partout sauf `knn`.
- [x] `services/ml_training.py` — `TrainingConfig.class_rebalancing`
  (défaut `False`). Actif en classification : un poids par échantillon du
  train calculé une seule fois, routé vers `model__sample_weight` pendant
  la recherche Optuna (`cross_val_score` le découpe lui-même par fold —
  vérifié empiriquement, aucune fuite) et pour le refit final du modèle
  retenu, seulement si `supports_rebalancing`. Ignoré en régression.
  `model_card` expose `class_rebalancing_requested`/`applied` pour la
  transparence frontend (même pattern que `explainability_status`, Lot 5).
- [x] API — `TrainingJobCreate.class_rebalancing`, volontairement **sibling**
  de `feature_engineering` (pas imbriqué dedans) : contrairement à
  `feature_engineering_json`, ce choix n'est jamais rejoué à l'inférence
  (il ne modifie que la pondération vue pendant l'entraînement, pas la
  forme du pipeline) — il transite par `config_json`, aucune migration DB.
- [x] Frontend — `ClassRebalancingSuggestion.tsx`, même pattern d'approbation
  que `FeatureEngineeringSuggestions` (Lot 4c : case à cocher, badge
  "Garde-fou", explication dépliable), branché sur le garde-fou Lot B
  existant (`desequilibre_classes`, `GET .../quality-check`) — pas de
  nouvelle détection, message d'arbitrage en langage clair utilisant le
  ratio réel du dataset.

**Vérifié** :

- Suite pytest complète verte (146 → 152 tests).
- Routage du `sample_weight` prouvé structurellement (mock de
  `cross_val_score`) pour un modèle qui le supporte, et absence de
  transmission pour KNN (pas de crash).
- Anti-fuite (Lot A) confirmée : split train/test/CV strictement identique
  avec ou sans le flag.
- **Preuve empirique que le rééquilibrage agit réellement** (pas seulement
  câblé) : sur un dataset synthétique déséquilibré (~92/8), activer le
  rééquilibrage fait passer le rappel de la classe minoritaire de 0,125 à
  0,625 — au prix attendu d'un rappel global plus faible, l'arbitrage même
  que le message affiché à l'utilisateur décrit.

**Scope volontairement limité** : SMOTE/rééchantillonnage synthétique hors
périmètre (lot expert futur) ; réglage de seuil de décision hors périmètre
(autre lot) ; pas de mode guidé/expert dédié (Lot E).

## Lot D — Leaderboard : rendre visible le travail de comparaison (livré)

Avant ce lot, un entraînement comparait plusieurs modèles mais seul le
gagnant était persisté (`MLModel`) — les autres candidats, leurs scores, et
la raison du choix restaient invisibles (un simple `logger.info`, jamais
lu). Phase 1 (lecture seule, audit + plan) validée avant implémentation ;
Phase 2 livrée en 4 commits isolés sur la branche `lot-d-leaderboard`.
Périmètre Niveau 1 uniquement (leaderboard intra-job) — la comparaison
inter-jobs (Niveau 2) a été volontairement reportée à un lot D-bis dédié
pour ne pas bâcler ni l'un ni l'autre.

- [x] **Bug corrigé en premier, dans son propre commit** :
  `_headline_metric` (`api/routers/training.py`) affichait `accuracy` sur
  la carte d'historique pour tout job de classification — trompeur sur un
  dataset déséquilibré (un modèle qui ignore la classe rare peut afficher
  95 % d'exactitude). Corrigé pour afficher `cv_score`, la métrique qui a
  réellement départagé les candidats. Régression inchangée (`r2_test`,
  déjà correct).
- [x] Nouvelle table `ModelCandidate` (`api/core/models.py`) — un candidat
  = algorithme, famille, `selection_score` (LA métrique qui a choisi le
  gagnant : ROC-AUC pondérée en classification, R² en régression, jamais
  l'accuracy), rang, variance inter-folds, erreur en unité réelle
  (régression). Table dédiée plutôt qu'un JSON sur `TrainingJob` — le
  tri/filtre inter-jobs prévu au lot D-bis a besoin de colonnes
  requêtables. Migration triviale (`create_all` gère les tables
  manquantes, aucune colonne à ajouter ailleurs) — rétrocompatible **par
  absence de lignes**, jamais par backfill : les jobs antérieurs à ce lot
  n'ont simplement aucun candidat persisté.
- [x] `services/ml_training.py` — `_optimize_one_model` bascule de
  `cross_val_score` à `cross_validate` : même calcul (un fit par fold,
  aucun ré-entraînement supplémentaire), mais expose le détail par fold de
  chaque scorer. La variance inter-folds (score par fold de l'essai
  Optuna gagnant) est capturée via `trial.set_user_attr` — décision prise
  après avoir confirmé en Phase 1 qu'elle n'était pas récupérable sans
  toucher la boucle d'entraînement, mais qu'elle ne coûtait rien de plus à
  calculer puisque `cross_validate` la produit déjà. En régression, un
  second scorer (RMSE, `neg_root_mean_squared_error`) est évalué sur les
  mêmes prédictions déjà produites par chaque fold — le R² seul n'étant
  pas lisible pour un bureau d'études, l'erreur en unité réelle est
  affichée à côté, jamais à la place du score de sélection.
  `TrainedModelResult.all_candidates` porte désormais tous les candidats
  triés par score de sélection décroissant.
- [x] `workers/training_worker.py` — persiste une ligne `ModelCandidate`
  par candidat du catalogue par défaut, dans la **même transaction** que
  `MLModel`/`job.status` : garantit que le gagnant et la ligne
  `is_winner=True` désignent toujours le même modèle par construction
  (mêmes variables sources dans `ml_training.py`, jamais recalculées
  séparément côté worker) — vérifié par un test dédié plutôt que supposé.
- [x] `GET /training/jobs/{id}/candidates` — leaderboard du job, même
  pattern d'isolation `_get_org_job` que le reste du router.
  Rétrocompatible : `candidates: []` (jamais une erreur) pour un job sans
  ligne persistée, le frontend se rabat alors sur le seul gagnant déjà
  disponible via `GET .../model`.
- [x] Frontend `ModelResultModal.tsx` — section "Modèles comparés" sous la
  Performance du gagnant : classement sur `selection_metric_label`, phrase
  en langage clair ("X retenu : meilleur ROC-AUC en validation croisée,
  devant Y de N points"), `BoxPlotChart` (Lot B) réutilisé tel quel pour la
  variance inter-folds quand disponible — aucun nouveau composant de
  graphe créé.

**Vérifié** :

- Suite pytest complète verte (152 → 165 tests).
- Cohérence gagnant garantie et testée explicitement : le modèle de
  `MLModel` et la ligne `ModelCandidate.is_winner=True` portent toujours
  le même algorithme et le même score (`test_worker_winner_consistent_...`).
- Classement piloté par le score de sélection, pas l'accuracy, prouvé sur
  un scénario construit où les deux métriques divergent (pas seulement une
  assertion structurelle sur le code).
- Rétrocompatibilité vérifiée par requête HTTP réelle sur un job sans
  candidat persisté (pas seulement en théorie).
- Isolation multi-tenant vérifiée sur le nouvel endpoint.
- `tsc -b` (typecheck strict, `noUnusedLocals`/`noUnusedParameters` actifs)
  vert côté frontend — `eslint` indisponible dans cet environnement
  (dépendance non installée localement), non vérifié par ce lot.

**Scope volontairement limité** (acté en Phase 1, à ne pas rouvrir) :
comparaison inter-jobs (tri, diff de config) reportée à un lot D-bis ; pas
de mode guidé/expert, pas de refonte UX globale, pas de nouveaux graphes
d'évaluation (tout ça = Lot E) — ce lot montre les modèles déjà comparés,
il ne refond pas l'écran.

## Fix — Échec d'entraînement affiché en stack trace brute ("bad allocation") (livré)

Un entraînement sur `predictive_maintenance.csv` (10 000 lignes, colonne
`Product ID` quasi-identifiant — 10 000 valeurs uniques) a échoué et affiché
à l'utilisateur une stack trace Python brute (chemins de fichiers internes
compris), se terminant par `_catboost.CatBoostError: bad allocation`. Phase
1 (lecture seule) a d'abord confirmé la cause apparente sur CE dataset
(one-hot sans garde-fou sur un identifiant), puis une reformulation de
l'utilisateur a demandé une correction GÉNÉRALISÉE, pas un contournement
pour un seul fichier — l'investigation a alors trouvé la vraie cause
structurelle.

- [x] **Cause racine, généralisée (pas spécifique à ce dataset)** :
  `services/ml_training.py` convertissait explicitement en tableau **dense**
  (`np.asarray(X.todense())`) le résultat du préprocesseur à 5 endroits (fit
  final, CQR), alors que les 4 modèles du catalogue par défaut
  (LightGBM/XGBoost/CatBoost/RandomForest) et `shap.TreeExplainer`
  acceptent **nativement le sparse** — vérifié empiriquement, pas supposé.
  Une colonne quasi-identifiant one-hotée reste minuscule en sparse (un seul
  `1.0` par ligne) mais explose en dense (des centaines de Mo pour un
  dataset par ailleurs modeste). Densifier était inutile et est la cause
  structurelle, quel que soit le dataset — pas un bug introduit par un lot
  récent (le comportement existe depuis le Lot 3, jamais réévalué).
- [x] `_compute_explainability` densifie désormais UNIQUEMENT quand
  `explainer_kind` l'exige (`LinearExplainer`/`KernelExplainer`, familles
  hors catalogue par défaut, réservées au mode expert futur Lot E), et
  seulement un échantillon BORNÉ (`_KERNEL_SHAP_BACKGROUND_SIZE`), jamais le
  train complet — sûr même sur un futur modèle de cette famille sur un gros
  dataset.
- [x] **Volet A — échouer proprement** : `workers/training_worker.py`,
  nouvelle fonction `_user_safe_error_message` — whitelist restreinte
  (mémoire insuffisante, détectée par type `MemoryError` ou motif textuel
  `"bad allocation"`/`"unable to allocate"`/`"out of memory"`, insensible à
  la casse) traduite en langage clair et actionnable ; message générique sûr
  par défaut pour toute autre cause. Le détail technique complet (type,
  message d'origine, traceback) continue d'aller dans les logs serveur
  (`logger.error`, inchangé) — jamais affiché à l'utilisateur.
  `DataLeakageError` (déjà un message français auto-rédigé, sûr) n'est pas
  affecté par cette traduction.
- [x] **Volet B — transparence amont, recalibrée** : le garde-fou Lot B
  `cardinalite_excessive` (`services/data_quality.py::_detect_high_cardinality`)
  existait déjà et aurait signalé `Product ID` (identifiant, pas de valeur
  prédictive) — enrichi d'un `n_estimated_onehot_columns` informatif dans
  `details`. Un détecteur de "risque mémoire" dédié, envisagé dans le
  cadrage initial, n'a **pas** été ajouté : le fix racine (sparse préservé)
  neutralise déjà le cas dominant (one-hot sur identifiant) pour tout le
  catalogue par défaut — en construire un aurait sur-corrigé un risque déjà
  largement fermé. Le risque résiduel (ex. `GaussianNB`, qui exige du dense,
  en mode expert futur) est noté mais pas actionnable avant que le Lot E
  n'expose ces modèles.

**Vérifié** :

- Suite pytest complète verte (165 → 174 tests).
- **Preuve structurelle, pas seulement fonctionnelle** : un test instrumente
  `scipy.sparse.csr_matrix.todense` et prouve qu'il n'est JAMAIS appelé
  pendant un entraînement complet (classification + régression avec CQR) sur
  un dataset synthétique à colonne quasi-identifiant — la preuve porte sur
  le mécanisme (aucune densification), pas sur un seuil de taille qui
  serait, par construction, spécifique à un dataset.
  Généralisé à N'IMPORTE QUEL dataset avec une colonne à cardinalité
  élevée.
- `LinearExplainer` (mode expert futur) reste correct avec le nouveau fond
  borné, sur une entrée sparse en amont.
- Message d'échec traduit vérifié bout en bout via le worker réel (pas
  seulement la fonction de traduction isolée) : jamais de "Traceback",
  jamais de chemin de fichier (`E:\`), jamais de `.py` dans
  `job.error_message`, quelle que soit l'exception d'origine.
- Non-régression : les 174 tests couvrent aussi la suite déséquilibre/Lot D
  déjà en place, dont plusieurs s'appuyaient sur `cross_val_score`/
  `cross_validate` mockés — inchangés par ce fix (aucun changement à la
  boucle de recherche Optuna elle-même, seulement au fit final/CQR/SHAP).

## Lot E1-ter — Refonte structurelle et design des pages dashboard/données/entraînement/résultats (livré)

Frontend uniquement. Après E1/E1-bis (socle visuel, thème clair, navigation
par piliers), la STRUCTURE des 4 pages métier restait en dessous du niveau
attendu — hiérarchie inversée sur Données, formulaire d'entraînement sans
étapes, et un bug de crédibilité sur le graphe de variance CV. Corrigé en 5
commits isolés sur `fix-entrainement-memoire-echec-propre`.

- [x] **Bug corrigé en premier, commit séparé** — graphe "Variance entre
  les découpages de validation croisée" (`ModelResultModal.tsx`) affichant
  des valeurs à 9 chiffres sur l'axe Y. Diagnostic (reproduit
  empiriquement via `cross_validate` sur un target quasi-constant) : ce
  n'est PAS un bug de lecture/parsing — le R² par fold est mathématiquement
  non borné en dessous et s'effondre légitimement quand la cible d'un fold
  a une variance quasi nulle (petit `GroupKFold` sur un dataset modeste).
  Corrigé côté données de graphe uniquement (frontend), sans toucher au
  moteur ML : `frontend/src/utils/cvScore.ts` (nouveau) borne les scores à
  [0, 1] pour l'affichage (`clampUnitScore`), appliqué au boxplot et au
  score affiché par candidat dans le leaderboard — la sélection du modèle
  côté backend reste sur la valeur brute, inchangée. Vitest ajouté au
  frontend (absent jusqu'ici) + test reproduisant le fold dégénéré.
- [x] **Page Dashboard** — vue d'ensemble de l'activité (4 tuiles
  statistiques : datasets/entraînements/en cours/membres, badges d'icône
  colorés, compteur animé) au-dessus de la gestion d'équipe (inchangée dans
  le fond) ; salutation contextuelle à l'heure réelle du navigateur ; deux
  listes "Derniers entraînements"/"Derniers datasets" (5 plus récents,
  clic → `ModelResultModal`).
- [x] **Page Données** — hiérarchie inversée : bande d'upload compacte en
  tête (glisser-déposer toujours actif), grille de datasets dense en
  dessous (jusqu'à 4 colonnes), badge de compte à côté du titre.
- [x] **Page Entraînement** — restructurée en 5 étapes numérotées guidées
  (sélection des données → contrôle qualité en panneau dédié →
  améliorations automatiques → réglages avancés repliés par défaut,
  emplacement réservé au futur Mode Expert E2, non implémenté → lancer).
  Devenue une **page dédiée sans historique à côté** (sur demande
  explicite en cours de lot) : configurer, lancer, puis voir la
  progression (poll automatique) et le résultat EN PLACE sur la même page.
  `ModelResultView` extrait de `ModelResultModal.tsx` (contenu pur, sans
  chrome de modale) pour être réutilisé identique dans les deux contextes
  (modale depuis le dashboard, page pleine largeur ici). L'historique
  complet vit désormais sur le Dashboard ; suppression d'un entraînement
  toujours possible depuis la vue résultat/échec (pas de régression
  fonctionnelle).
- [x] **Page Résultats** (`ModelResultView`) — nouveau bloc "Interprétation
  du modèle" en langage clair (pourquoi ce modèle a gagné + variables SHAP
  dominantes), réutilise uniquement des données déjà calculées (leaderboard,
  `shap_summary`), aucun nouveau calcul. Courbes ROC/PR multiclasses (6
  classes emmêlées) : légende cliquable pour isoler une classe, survol pour
  mettre les autres en retrait (`useSeriesIsolation`/`IsolatableLegend`
  dans `EvaluationCharts.tsx`). Matrice de confusion (déjà en dégradé teal
  sur fond clair depuis E1-bis) : vérifiée lisible, pas de changement
  nécessaire.
- [x] **Infra partagée** (petit commit dédié, en tête) —
  `components/ui/ColorIconBadge.tsx` (nouveau) : palette d'accent partagée
  (bleu/teal/amber/violet, teinte déterministe par id) pour des cartes/
  listes moins monochromes, sur retour explicite en cours de lot ; logo
  complet (`public/logo.png`, recadré sur l'icône) remplace le badge "D"
  texte dans `AppShell` ; fond de page légèrement bleuté
  (`--color-canvas`), sur demande explicite.

**Vérifié** :

- `tsc -b` et `vite build` verts.
- Vitest : 5/5 (nouveau, `cvScore.test.ts`).
- Suite pytest backend intégralement verte (174 tests) — non-régression
  attendue et confirmée : aucun fichier backend touché par ce lot.
- Rendu visuel réel **non vérifié** par ce lot — aucun outil d'interaction
  navigateur disponible dans cet environnement ; revue visuelle page par
  page à faire par l'utilisateur.

**Scope volontairement limité** (acté en cadrage, à ne pas rouvrir) : mode
guidé/expert réel (E2, seul l'emplacement est prévu ici), explicabilité
SHAP locale enrichie, nouveaux graphes d'évaluation avancés — ce lot
restructure et style l'existant, il ne réinvente pas les fonctionnalités ML.

## Lot E2 — Mode guidé / mode expert (livré)

Backend + frontend. Le moteur supervisé (Lot 5) exposait déjà `subset`
(`services/ml_registry.models_for_task`) et plusieurs paramètres de
`TrainingConfig` (`cv_folds`, `cqr_alpha`), mais deux problèmes distincts :
`subset` était câblé en dur sur `"default"` dans `train_and_evaluate`
(jamais piloté par l'appelant), et `cv_folds`/`seed`/`cqr_alpha` n'étaient
soit jamais transmis par le frontend (typés côté client, ignorés), soit
jamais lisibles depuis le corps de la requête côté API (forcés depuis
`Settings`). Ce lot corrige les deux, puis expose le tout dans un panneau
"Mode expert" replié par défaut — le mode guidé (défaut) reste strictement
inchangé.

- [x] **`services/ml_training.py`** — `TrainingConfig.model_ids:
  Optional[list[str]]` (nouveau, défaut `None`). Dans `train_and_evaluate`,
  le catalogue comparé devient : si `model_ids` fourni, intersection avec
  `models_for_task(task_type, subset="all")` (filtré par id) ; sinon,
  comportement strictement inchangé (`subset="default"`). Garde-fou
  défensif si l'intersection est vide (ne devrait jamais arriver, l'API
  valide déjà en amont) : repli sur le sous-ensemble par défaut plutôt
  qu'un catalogue vide.
- [x] **`api/routers/training.py`** — `TrainingJobCreate` gagne
  `model_ids`, `seed`, `cqr_alpha` (tous optionnels, `None` = comportement
  d'avant ce lot). Validation en deux temps : ids inconnus du registre →
  400 `MODELES_INCONNUS` (avant lecture du dataset, fail-fast) ; puis,
  une fois la tâche détectée, intersection avec les modèles compatibles
  avec cette tâche (ex. Naive Bayes = classification uniquement) → 400
  `AUCUN_MODELE_COMPATIBLE` si l'intersection est vide. Le serveur ne fait
  jamais confiance au filtrage déjà fait côté UI. `seed`/`cqr_alpha` du
  corps de requête priment sur les défauts `Settings` quand fournis (avant
  ce lot : toujours forcés depuis `Settings`, champ absent du schéma).
  Nouvel endpoint `GET /training/models-catalog` (lecture pure du
  registre, aucun accès dataset) : les 9 modèles avec libellé lisible
  (régularise Ridge/LogisticRegression et SVR/SVC en un seul `label`),
  famille, `is_default`, `supported_tasks`, et un indicateur `slow` (SVM,
  KNN — surcoût mesuré au Lot 5) porté par une constante du router, pas du
  registre (question d'UX, pas de capacité du modèle).
- [x] **Frontend, `components/training/ExpertModePanel.tsx`** (nouveau) —
  interrupteur "Mode expert" (défaut OFF) + manettes, chacune avec un
  libellé clair et une aide en langage courant (`LabelWithHelp`, motif déjà
  utilisé ailleurs dans l'app) : essais Optuna (déplacé hors du formulaire
  guidé), blocs de validation croisée, graine aléatoire, confiance des
  intervalles CQR, rééquilibrage des classes (force/annule manuellement la
  suggestion automatique existante), et sélecteur de modèles (cases à
  cocher groupées par famille, catalogue chargé à la demande — un
  utilisateur qui n'ouvre jamais le mode expert ne déclenche aucun appel
  réseau supplémentaire). Modèles lents signalés par un badge
  d'avertissement.
- [x] **Frontend, `utils/trainingPayload.ts`** (nouveau) — construction du
  payload extraite de `TrainingForm` en fonction pure testable (pas
  d'infra de test de composants React dans ce dépôt) : `model_ids` n'est
  envoyé que si le mode expert est actif ET qu'une sélection existe —
  sinon toujours omis, comme avant ce lot.
- [x] **Rétrocompatibilité vérifiée par construction** : chaque manette
  experte démarre à la même valeur que le mode guidé (`DEFAULT_CV_FOLDS`,
  `DEFAULT_SEED`, `DEFAULT_CQR_ALPHA` dans `ExpertModePanel.tsx`, alignées
  sur `Settings.cv_folds_default`/`model_seed`/`cqr_alpha`) — activer le
  mode expert sans rien changer produit exactement le même payload que le
  mode guidé, testé explicitement.

**Vérifié** :

- Backend : suite pytest intégralement verte (182 tests, dont 8 nouveaux
  pour ce lot — 2 dans `test_ml_training.py::test_model_ids_*`, 5 dans
  `test_training_api.py::test_models_catalog_*`/`test_create_job_*`, 1 dans
  `test_training_worker.py::test_worker_respects_model_ids_from_config_json`).
- Frontend : `tsc -b`, `vite build` et `vitest run` verts (10 tests, dont 5
  nouveaux — `utils/trainingPayload.test.ts`, incluant le test "expert ON
  sans modification == guidé").
- Rendu visuel réel **non vérifié** par ce lot — aucun outil d'interaction
  navigateur disponible dans cet environnement ; revue visuelle à faire.

**Scope volontairement limité** (acté en cadrage) : `shap_sample_size` non
exposé (laissé au défaut serveur, jugé non nécessaire) ; pas de
détection du type de tâche avant soumission côté UI (le catalogue affiche
`supported_tasks` par modèle à titre informatif, mais le filtrage réel
tâche/modèle reste fait côté serveur à la création du job) ; explicabilité
SHAP locale enrichie et refonte visuelle fine hors périmètre.

## Lot Explicabilité globale — au-delà de l'importance moyenne (livré)

Livré par une session parallèle (voir mémoire `project_parallel_sessions`),
documenté ici a posteriori à partir du contenu réel des 5 commits (validé
par l'utilisateur). Avant ce lot, SHAP ne donnait que l'importance moyenne
par variable (barres, Lot 5) : on savait qu'une variable comptait, jamais si
elle poussait la prédiction vers le haut ou le bas pour un cas donné, ni
comment le modèle se comportait au-delà des seules métriques ponctuelles
(train/test).

- [x] **`services/ml_training.py`** — 4 nouveaux diagnostics sur le modèle
  gagnant : **beeswarm SHAP** (réutilise l'explainer et les `shap_values`
  déjà calculés par `_compute_shap_summary`, aucun second appel — distribution
  signée + valeur de la feature en couleur, bornée en variables/points pour
  un payload JSON raisonnable) ; **importance par permutation** (mesure
  indépendante du type de modèle, pour recouper le SHAP) ; **courbe de
  calibration** (classification uniquement, réutilise `proba_test`/`y_test`
  déjà calculés — aucun risque de fuite propre à ce calcul) ; **courbe
  d'apprentissage** (seul calcul réellement coûteux du lot : refit du modèle
  gagnant, hyperparamètres déjà figés par Optuna, sur des tailles de train
  croissantes, avec la MÊME validation croisée que la sélection du modèle —
  jamais sur le train complet vu par le modèle final). Chaque diagnostic
  dégrade proprement (statut `"ok"`/`"degraded"` + message FR, même motif que
  l'explicabilité SHAP du Lot 5) plutôt que de faire échouer l'entraînement.
- [x] **Bug réel trouvé et corrigé pendant ce lot** : `CatBoost` marque en
  lecture seule, comme effet de bord de son `Pool` interne, le tableau numpy
  qu'on lui passe à `predict()` — `permutation_importance` réutilise ce même
  tableau sur plusieurs répétitions, donc la 2ᵉ répétition échouait
  systématiquement quand CatBoost était le modèle retenu. Corrigé en passant
  un `DataFrame` (réaffectation de colonne côté pandas, jamais d'écriture
  in-place dans le buffer verrouillé par CatBoost).
- [x] **`api/core/models.py`/`database.py`** — 4 nouvelles colonnes JSON-as-
  Text sur `MLModel` (`shap_beeswarm_json`, `permutation_importance_json`,
  `calibration_json`, `learning_curve_json`), nullable, migration additive
  idempotente (même mécanisme que le reste du projet, pas d'Alembic).
- [x] **`workers/training_worker.py`** — les 4 champs sont écrits sur
  `MLModel` (calibration/courbe d'apprentissage peuvent être `None`/absents
  selon la tâche, jamais une erreur).
- [x] **`api/routers/training.py`** — `MLModelDetail` expose les 4 champs
  avec des défauts (`[]`/`{}`/`{}`/`None`) : un job entraîné avant ce lot
  répond avec ces défauts plutôt que de planter (même rétrocompatibilité par
  absence que le reste du projet).
- [x] **Frontend** — `GlobalExplainability.tsx` (`ShapBeeswarmChart` : jitter
  déterministe sans dépendance beeswarm dédiée, couleur par valeur de
  variable normalisée bleu→rouge ; `PermutationImportanceChart` : barres ±
  écart-type) ; `ReliabilityDiagnostics.tsx` (`CalibrationChart`/
  `LearningCurveChart`, réutilisent le motif d'isolation de série des
  courbes ROC/PR du Lot E1-ter, exporté depuis `EvaluationCharts.tsx` pour
  l'occasion) ; `ModelResultModal.tsx` — nouvelles sections "Explicabilité
  SHAP" et "Diagnostics de fiabilité", `DiagnosticBlock` généralise le motif
  de dégradation du Lot 5 (renommé `isDiagnosticStatus`) pour les 4 nouveaux
  diagnostics ; chaque graphe accompagné d'une phrase d'interprétation en
  langage clair, jamais un graphe brut sans explication.

**Vérifié** (repris du message des commits d'origine) : suite pytest
complète verte (195 tests à l'issue de ce lot — entraînements réels, pas
mockés, ~20 min) ; preuve structurelle d'anti-fuite de la courbe
d'apprentissage (préprocesseur cloné jamais déjà fit, données brutes en
entrée) ; round-trip complet worker → colonnes DB → réponse HTTP ;
rétrocompatibilité vérifiée sur un job sans ces colonnes. Frontend : `tsc
-b`, `vite build`, `vitest` verts (nouveau `theme/charts.test.ts`).

## Refonte UI : design system moderne (livré)

Livré par la même session parallèle, sur la même branche que le lot
ci-dessus, à la demande explicite de l'utilisateur en cours de session —
documenté ici a posteriori à partir du commit réel (validé par
l'utilisateur). Refonte visuelle calquée sur une maquette de référence
(v0/Vercel).

- [x] **`index.css`** — nouveau système de tokens sémantiques en OKLCH
  (`primary` passe du teal au bleu de marque ; `secondary`/`muted`/`accent`/
  `destructive`/`warning`/`success`/`border`/`ring`/`card`/`sidebar`),
  exposés en utilitaires Tailwind via `--color-*`. **Bug réel corrigé** : un
  commentaire contenant littéralement `*/` fermait prématurément le bloc CSS
  et cassait silencieusement le build (`vite build` échouait sans que `tsc`
  le détecte).
- [x] **Composants de base recolorisés** — `Button` (dégradé de marque sur
  le variant primaire, cohérent avec l'auth), `Badge` (puce de statut +
  pulse), `Card`, `Avatar`, `Heatmap` (cellules compactes, libellés pivotés
  au-delà de 6 colonnes — matrices larges illisibles signalées en usage
  réel), `Modal` (fond gris pâle pour faire ressortir les cartes internes),
  `Input`, `Tooltip`. `theme/charts.ts` aligné sur le bleu de marque.
- [x] **`AppShell.tsx`** — barre du haut remplacée par une sidebar fixe
  (façon maquette), groupée par pilier (ML supervisé actif, non
  supervisé/vision "Bientôt"), profil utilisateur en pied de sidebar,
  panneau glissant en mobile. *(Tokens `--color-sidebar-accent`/
  `--color-sidebar-muted-foreground` définis à ce lot mais pas encore
  appliqués partout dans la sidebar — complété par le Lot Nettoyage guidé des
  variables ci-dessus.)*
- [x] **Pages alignées sur les nouveaux tokens** — Dashboard (CTA "Nouvel
  entraînement", badges de statut à puce, bouton Supprimer sur chaque
  entraînement, absent avant ce commit) ; Datasets (grille plafonnée à 3
  colonnes — bouton Supprimer coupé par overflow-hidden sur une carte trop
  étroite à 4 colonnes, bug réel constaté) ; Training (pipeline en wizard
  horizontal à une étape visible, pastilles numérotées, récapitulatif
  honnête avant lancement — jamais de temps/coût estimé fabriqué) ;
  ModelResultModal/EvaluationCharts/PredictionForm (sections en grille de
  cartes plutôt qu'empilées) ; EdaModal (sections en cartes, nuages de
  points corrélés filtrés des valeurs manquantes — un point `null` faussait
  le domaine auto des axes, bug réel constaté ; histogramme à bornes de bin
  lisibles) ; ComingSoon (Clustering/Vision), Orientation, PillarCard,
  garde-fous (ClassRebalancing/DataQuality/FeatureEngineering),
  ExpertModePanel (sélection de modèles en tuiles), Login/Register/
  PasswordStrengthMeter.

**Vérifié** (repris du message du commit d'origine) : `tsc -b`, `vite
build`, `vitest` verts après chaque étape.

**Scope volontairement limité, signalé par la session d'origine** :
persistance de la progression du wizard Entraînement à la navigation (état
local React, pas encore de `sessionStorage`) — comportement ambigu à
clarifier avant d'y toucher, non traité depuis.

## Lot Nettoyage guidé des variables — détecter et exclure les colonnes inutiles (livré)

Déclenché par un audit expert du backend (lecture seule, validé avant tout
code) : les garde-fous Lot B détectaient déjà les colonnes constantes/quasi-
constantes et à cardinalité excessive avec une action textuelle ("retirez
cette colonne"), mais rien ne reliait cette recommandation à une action
concrète côté formulaire d'entraînement — l'utilisateur devait lire l'alerte
puis décocher la colonne manuellement, sans lien visuel entre les deux.
L'EDA autonome (avant choix d'une cible) n'exécutait en outre aucune de ces
détections. Deux lacunes supplémentaires identifiées au même audit :
colonnes dupliquées noyées dans l'alerte générique de colinéarité, et
colonnes numériques mal typées en texte (virgule décimale, séparateur de
milliers) totalement invisibles.

- [x] **`services/data_quality.py`** — `target_column` devient optionnel
  dans `analyze_data_quality()` : absent, les détections structurelles
  (constantes, cardinalité, doublons, numérique mal typé, valeurs
  manquantes, colinéarité, dataset trop petit) restent actives, seules
  fuite/déséquilibre (qui exigent une cible) sont omises — rétrocompatible,
  comportement inchangé quand une cible est fournie. Deux nouveaux
  détecteurs : `_detect_duplicate_columns` (hash `pandas.util.
  hash_pandas_object` puis `Series.equals` seulement entre colonnes de même
  hash — évite une comparaison O(k²) systématique sur un dataset à beaucoup
  de colonnes), niveau "attention" (contenu strictement identique, sans
  ambiguïté, contrairement à la colinéarité ≥0.9 restée en "info") ;
  `_detect_mistyped_numeric` (`_try_parse_numeric_text`/
  `_has_numeric_format_signal`), avec un garde-fou explicite avant tout
  parsing — une part suffisante de l'échantillon doit porter un signe de
  formatage numérique (virgule, séparateur de milliers) avant d'être
  considérée candidate, pour ne jamais confondre une colonne d'identifiants
  (ex. codes postaux à zéro non significatif) avec du numérique mal typé.
- [x] **`services/feature_engineering.py`** — `_suggest_column_exclusion`
  (nouveau, branché sur `colonne_constante`/`cardinalite_excessive`/
  `colonnes_dupliquees`) : suggestion d'exclusion, mais PAS une
  transformation de pipeline — son `transformation` (`{"type":
  "exclude_column", ...}`) n'entre jamais dans `spec["upstream"]` (absent de
  `_UPSTREAM_TRANSFORMATION_TYPES` par construction, lèverait une erreur
  explicite s'il y apparaissait), approuver cette suggestion revient
  simplement à décocher la colonne dans `TrainingJobCreate.feature_columns`,
  mécanisme qui existe depuis le Lot 3. `suggest_numeric_coercion` +
  `apply_numeric_coercion` (nouveau type upstream `numeric_coerce`,
  déterministe ligne à ligne comme `datetime_decompose`/`ratio`) : réutilise
  EXACTEMENT le même parseur que la détection, pour que suggestion affichée
  et conversion appliquée ne divergent jamais. Appliqué EN PREMIER dans
  `apply_upstream_feature_engineering` (avant décomposition datetime et
  ratio) : un ratio référençant une colonne mal typée doit voir sa forme
  déjà convertie.
- [x] **`api/routers/datasets.py`** — `GET /datasets/{id}/quality-check` :
  `target_column` devient optionnel, permet un appel dès l'exploration d'un
  dataset (page Données/EDA), avant même de choisir une cible pour un
  entraînement.
- [x] **`api/routers/training.py`** — `_KNOWN_UPSTREAM_TYPES` +
  `numeric_coerce`, validation de la colonne référencée par cette
  transformation au même titre que `datetime_decompose`/`ratio`.
- [x] **Frontend, `DataQualityWarnings.tsx`** — action "Exclure « colonne »"
  par alerte excluable (miroir de `_EXCLUSION_WARNING_CODES` côté backend,
  sans appel réseau supplémentaire : lit directement `warning.columns`) +
  bouton "Tout exclure" groupé ; devient utilisable sans cible (EDA) via
  `targetColumn` optionnel. `Training.tsx` : `excludeFeatures()` (retrait
  explicite, jamais un toggle — approuver deux fois la même suggestion reste
  sans effet) câblé sur la sélection de variables de l'étape 1.
  `FeatureEngineeringSuggestions.tsx` : filtre `exclusion_variable` (déjà
  proposée, plus utilement, dans le panneau qualité — l'afficher aussi ici
  aurait été un cul-de-sac, sa transformation n'étant jamais une entrée de
  pipeline) ; ajoute la branche manquante `numeric_coerce` dans la
  construction du payload (absente, une suggestion de conversion approuvée
  n'aurait silencieusement rien fait).
- [x] **Refonte visuelle associée** (au-delà du périmètre initial, cadrée en
  cours de session) : sidebar recolorisée (fond bleu de marque assombri,
  `--color-sidebar*` en OKLCH, teinte 258 cohérente avec `--color-primary` —
  remplace un blanc quasi invisible confondu avec le fond de page) ;
  `EdaModal.tsx` restructuré en onglets (Vue d'ensemble/Qualité des
  données/Corrélations/Distributions/Relation à la cible) avec bande de
  statistiques (`StatTile`, réutilisé du dashboard) — remplace un
  empilement vertical de 9 cartes identiques ; panneau qualité (ce lot)
  intégré comme onglet dédié. `ModelResultModal.tsx`/`ModelResultView`
  restructuré en onglets (Performance/Explicabilité/Fiabilité/Prédire/
  Détails), nouveau composant partagé `components/ui/SectionHeader.tsx`
  (icône colorée + titre, remplace les libellés gris uniformes dans les deux
  écrans). `Modal.tsx` gagne un prop `size` (`"md"`/`"xl"`, défaut inchangé)
  pour ces deux contenus riches.

**Vérifié** :

- Suite pytest complète verte (218/218, 26 nouveaux tests : détecteurs
  data_quality, target_column optionnel, suggestions d'exclusion/coercion,
  endpoints API).
- Frontend : `tsc -b`, `vite build` et `vitest run` verts (13/13).
- Non-régression structurelle : un job/dataset qui ne déclenche aucun des
  nouveaux garde-fous produit exactement les mêmes suggestions qu'avant ce
  lot (assertions de codes mises à jour uniquement là où un dataset de test
  déclenche réellement une nouvelle détection, ex. cardinalité excessive sur
  "ville").
- Rendu visuel réel **non vérifié** en conditions réelles — aucun outil
  d'interaction navigateur disponible dans cet environnement de travail ;
  revue visuelle à faire par l'utilisateur avant de considérer la refonte
  définitive.

**Scope volontairement limité** : pas de détection de colonnes numériques
mal typées AU-DELÀ de la virgule/séparateur de milliers (ex. devises avec
symbole, pourcentages en texte — non rencontrés dans les datasets réels de
l'utilisateur à ce jour) ; pas de comparaison inter-jobs (Lot D-bis,
toujours en attente) ; pas de registre de modèles versionné (Lot 9) ; pas de
durcissement SaaS (Lot 10, quotas de jobs concurrents notamment) — ces trois
derniers points restent priorisés dans l'audit backend qui a précédé ce lot.

## Refonte visuelle globale — sidebar, onglets, palette CVD-safe, cartes colorées (livré)

Frontend uniquement, 3 commits isolés. Retour utilisateur explicite après le
lot précédent : seule la sidebar plaisait, le reste des pages restait perçu
comme "tout blanc".

- [x] `index.css` — `--color-background` retinté (bleu-gris visible, teinte
  258 cohérente avec `--color-primary`/`--color-sidebar`) : le token
  n'avait, de fait, jamais été appliqué depuis la refonte sidebar
  (`AppShell` utilise `bg-background`, 0.985 de luminosité, quasi
  indissociable du blanc des cartes `bg-card` à 1.0).
- [x] Nouveaux `components/ui/Switch.tsx` (interrupteur pilule, extrait du
  motif déjà présent dans `ExpertModePanel.tsx`) et `components/ui/Tabs.tsx`
  (contrôle segmenté) — remplacent respectivement les cases à cocher des
  vrais réglages ON/OFF et le motif "bordure basse" dupliqué entre
  `EdaModal.tsx`/`ModelResultModal.tsx`.
- [x] `theme/charts.ts` — palette catégorielle (6 séries) RE-VALIDÉE avec le
  script du skill dataviz (`validate_palette.js`) : l'ancien ordre échouait
  la paire adjacente pink↔teal (ΔE 3.8 en deutéranopie, sous le seuil de 6),
  jamais vérifié avant. Nouvel ordre validé, bleu de marque conservé en
  tête. `Heatmap.tsx` — dégradés d'opacité ad hoc remplacés par des rampes à
  paliers discrets (séquentielle bleu, divergente bleu↔rouge + neutre gris),
  encre du texte calculée par palier.
- [x] `ColorIconBadge.tsx` — teinte "rose" (états d'échec, hors rotation
  déterministe par id) + helpers `accentValueTextClass`/`accentBorderClass`.
  `ModelResultModal.tsx` — `MetricCard` gagne une teinte par métrique,
  `Leaderboard` un podium visuel (gagnant en dégradé de marque + trophée,
  liseré or/argent pour #2/#3).
- [x] **Correction sur retour direct (capture d'écran)** : une première
  version colorait les cartes Dashboard/Datasets par STATUT réel plutôt que
  par identité — en usage réel, la quasi-totalité des datasets/entraînements
  partagent le même statut au même moment, ce qui rendait les grilles
  monochromes (pire qu'avant). Revenu à la coloration par identité
  (`accentColorForId`), le statut réel restant lisible via le `Badge`
  existant (texte + puce) — deux canaux séparés plutôt que confondus.
- [x] Table "Résumé par colonne" (`EdaModal.tsx`) retravaillée sur le même
  retour direct : badge de type coloré avec icône, taux de valeurs
  manquantes en mini barre de progression par sévérité, grands nombres
  formatés avec séparateur de milliers (`toLocaleString("fr-FR")` — une
  colonne identifiant peut avoir un écart-type dans les centaines de
  millions, illisible sans ce formatage), zébrage + survol de ligne.

**Vérifié** : `tsc -b`, `vite build`, `vitest` (13/13) verts après chaque
commit. Rendu visuel réel vérifié PARTIELLEMENT par l'utilisateur en cours
de lot (captures d'écran fournies en session, ayant motivé la correction
ci-dessus) — pas une revue exhaustive de chaque écran.

## Lot Explicabilité locale — pourquoi CETTE prédiction (livré)

Jusqu'ici, l'explicabilité SHAP (Lot 5, Lot Explicabilité globale) ne
répondait qu'à "quelles variables comptent EN MOYENNE pour ce modèle" —
jamais "pourquoi CE cas précis a reçu CETTE prédiction", la question la
plus naturelle pour un utilisateur qui vient de tester une prédiction
(`PredictionForm.tsx`, Lot 4a).

- [x] **`services/ml_explainability.py`** (nouveau) — `build_explainer`/
  `shap_values_per_class` déplacées depuis `services/ml_training.py` pour
  être PARTAGÉES avec l'inférence, sans risque qu'une copie diverge de
  l'autre sur la normalisation de la sortie SHAP (bug réel historique du
  Lot 3 : forme dépendante de la version SHAP/du backend d'arbre).
  `ml_training.py` conserve des alias locaux (`_build_explainer =
  build_explainer`) pour ne pas toucher ses nombreux appels existants.
  Nouvelles fonctions : `select_class_matrix` (choisit la matrice d'UNE
  classe dans la sortie normalisée), `normalize_base_value` (même
  normalisation défensive pour `explainer.expected_value`, qui porte la
  même ambiguïté de forme que `shap_values`).
- [x] **`services/ml_training.py`** — le bundle persisté gagne
  `explainer_kind` (routage à l'inférence, même famille que l'explicabilité
  globale) et `local_explain_background` (fond borné et déjà dense,
  uniquement pour les familles qui en ont besoin — linear/kernel ;
  `None` pour "tree", qui couvre tout le catalogue par défaut, donc aucun
  coût de bundle supplémentaire dans le cas courant).
- [x] **`services/ml_inference.py::explain_one`** — construit l'explainer
  adapté à la volée (même routage que l'entraînement), calcule les
  contributions SHAP de l'observation, bornées à
  `LOCAL_EXPLAIN_TOP_FEATURES` (10) avec le reste agrégé sous "Autres".
  Dégrade proprement (`status: "degraded"` + message FR) plutôt que de
  faire échouer la PRÉDICTION elle-même — un modèle entraîné avant ce lot
  n'a pas `explainer_kind` dans son bundle, cas rétrocompatible testé
  explicitement. Câblé dans `predict_one` : classification (explique la
  classe PRÉDITE, pas les K classes à la fois) et régression.
- [x] **Bug réel trouvé et corrigé en test** : pour une classification
  BINAIRE, certaines versions de SHAP renvoient un seul tableau de valeurs
  pour `explainer.shap_values(...)` — celui de `class_names[1]` (la "classe
  positive"), quelle que soit la classe réellement prédite pour
  l'observation. Sans correction, expliquer une observation prédite classe
  0 affichait des contributions au signe inversé (une variable qui pousse
  VERS la classe prédite semblait la pousser CONTRE). Détecté en vérifiant
  empiriquement la propriété fondamentale de SHAP (`base_value +
  sum(shap_values) == sortie du modèle`, ici en espace logit via sigmoïde)
  sur un modèle réel — pas supposée correcte. Corrigé par inversion de signe
  ciblée (`sign = -1` uniquement pour classe prédite = 0, sortie non listée,
  2 classes), verrouillé par un test qui vérifie LES DEUX classes prédites,
  pas seulement celle qui fonctionnait déjà par hasard.
- [x] **API** (`api/routers/training.py`) — `PredictionResponse.explanation`
  (nouveau, optionnel) : `LocalExplanation` (status/message/base_value/
  contributions/other_contribution), `LocalContribution`
  (feature/value/contribution).
- [x] **Frontend** — `components/training/LocalExplanation.tsx`
  (nouveau) : barres divergentes centrées sur 0 (rouge = pousse la
  prédiction vers le haut, bleu = vers le bas — même convention que le
  beeswarm SHAP déjà utilisé ailleurs, jamais une nouvelle convention
  concurrente), résumé "base → total" en langage clair. Intégré à
  `PredictionForm.tsx`, sous le résultat d'une prédiction.

**Vérifié** :

- Suite pytest complète verte (nouveaux tests : `test_ml_explainability.py`
  — 9 tests unitaires sur la normalisation partagée ; `test_inference.py` —
  reconstruction exacte en régression, reconstruction de la probabilité de
  la classe prédite en classification binaire (LES DEUX classes,
  verrouille le correctif de signe), non-plantage en classification
  multiclasse réelle (Iris), dégradation propre sur un bundle sans
  `explainer_kind`).
- La propriété de reconstruction SHAP (`base_value + Σ contributions ≈
  sortie du modèle`) est vérifiée QUANTITATIVEMENT sur des modèles réels
  entraînés de bout en bout, pas seulement testée structurellement — c'est
  cette vérification qui a révélé le bug de signe binaire ci-dessus.
- Frontend : `tsc -b`, `vite build`, `vitest` (13/13) verts.

**Scope volontairement limité** : le fond `local_explain_background` n'est
recalculé qu'à l'entraînement (pas de rétrocompatibilité pour les modèles
déjà entraînés avant ce lot — dégradation propre, pas de backfill) ;
explication locale non exposée pour les modèles CQR (intervalle de
confiance) eux-mêmes, seulement pour la prédiction centrale.

## Lot D-bis — comparaison inter-jobs (livré)

Le Lot D (leaderboard) comparait déjà les modèles D'UN MÊME job — ce lot
ajoute la comparaison ENTRE PLUSIEURS jobs (config, métriques), reporté du
Lot D pour ne pas le bâcler.

- [x] **`api/routers/training.py::GET /training/jobs/compare`** —
  `job_ids` en paramètres de requête répétés (`Query(..., min_length=2,
  max_length=8)`), isolé par organisation comme le reste : un id d'une
  autre organisation dans la liste est traité comme absent (404
  `TRAINING_JOB_INTROUVABLE`), jamais un indice d'existence croisée.
  Enregistré AVANT `GET /jobs/{job_id}` dans le routeur — FastAPI matche les
  routes dans l'ordre de déclaration, `/jobs/compare` après `/jobs/{job_id}`
  aurait été intercepté par le paramètre de chemin (`job_id="compare"`,
  échec de conversion en entier, 422 au lieu de la comparaison attendue).
- [x] `_differing_config_fields` — compare les champs de `config_json`
  (`test_size`/`optuna_trials`/`cv_folds`/`seed`/`cqr_alpha`/
  `class_rebalancing`/`model_ids`) entre tous les jobs demandés, calculé
  côté serveur (source unique de vérité) plutôt que recalculé côté
  frontend. `model_ids` comparé par ENSEMBLE (`frozenset`), pas par ordre —
  le même sous-ensemble de modèles choisi dans un ordre différent n'est pas
  une vraie différence de configuration.
- [x] `JobComparisonEntry` — dataset/cible/algorithme/statut/métriques
  complètes/config par job demandé, dans l'ORDRE de la requête (pas l'ordre
  SQL) pour que le frontend affiche les colonnes dans l'ordre de sélection
  de l'utilisateur.
- [x] **Frontend, nouvelle page `pages/TrainingHistory.tsx`** (route
  `/training/history`) — historique complet des entraînements (jusqu'ici
  seuls les 5 plus récents étaient visibles, sur le tableau de bord),
  sélection multiple par case à cocher, tableau de comparaison (métriques +
  configuration, lignes qui diffèrent surlignées). **Corrige au passage un
  point mort UX préexistant** : le lien "Voir tout" du tableau de bord
  pointait vers le formulaire d'entraînement (`/training`), qui n'affiche
  plus d'historique depuis le Lot E1-ter — aucun endroit ne permettait de
  consulter tous les entraînements passés. Nouvel item de navigation
  "Historique" dans la sidebar (`config/pillars.ts`).

**Vérifié** :

- 6 nouveaux tests (`tests/test_job_comparison.py`) : ordre des entrées
  respecté, détection des champs différents (et non-détection des champs
  identiques), `model_ids` comparé par ensemble, refus si moins de deux
  jobs valides, 404 sur un id inconnu, isolation entre organisations
  (un id d'une autre organisation dans la requête → 404, pas une fuite de
  métadonnées). Suite pytest complète verte.
- Frontend : `tsc -b`, `vite build`, `vitest` (13/13) verts.
- Rendu visuel réel **non vérifié** en conditions réelles — aucun outil
  d'interaction navigateur disponible dans cet environnement de travail.

**Scope volontairement limité** : pas de graphique de comparaison (courbes
superposées) — un tableau suffit pour ce volume de jobs comparés à la fois
(2 à 8) ; pas de sauvegarde d'une comparaison favorite.

## Lot 9 — registre de modèles versionné (livré)

L'artefact (bundle joblib) existait depuis le Lot 3, mais rien ne
distinguait "un modèle entraîné parmi d'autres" de "LE modèle sur lequel on
peut compter pour ce problème", et rien ne permettait de le récupérer hors
de la plateforme.

- [x] **`api/core/models.py::MLModel`** — `stage` (`"staging"`/`"production"`,
  `NULL` = jamais promu) et `promoted_at`, colonnes NULLABLE (rétrocompat
  par absence, même idiome que le reste du projet — jamais de backfill).
  Migration additive idempotente (`api/core/database.py`).
- [x] **`api/routers/training.py::POST /jobs/{id}/model/promote`** — règle
  du registre : UN SEUL modèle `"production"` à la fois par couple
  (dataset, cible) au sein d'une organisation. Promouvoir un nouveau modèle
  en production DÉMET automatiquement l'ancien pour LE MÊME couple
  dataset+cible (repasse en `"staging"`, jamais supprimé/écrasé) — deux
  jobs sur le même dataset mais des cibles différentes ne sont jamais en
  concurrence, ce sont deux problèmes distincts.
- [x] **`GET /jobs/{id}/model/export`** — l'artefact joblib complet
  (modèle + préprocesseur + CQR le cas échéant) en téléchargement direct
  (`FileResponse`), isolé par organisation comme le reste. Rechargeable via
  `joblib.load` dans un environnement Python équivalent — versions de
  scikit-learn/lightgbm/xgboost/catboost/shap non garanties au-delà de
  `backend/requirements.txt`, noté explicitement plutôt que promis à tort.
- [x] **`GET /models/registry`** — tous les modèles PROMUS de
  l'organisation (`stage IS NOT NULL`), tous datasets/cibles confondus :
  n'est PAS un doublon de l'historique complet (Lot D-bis), seulement ce
  qui a été explicitement retenu.
- [x] **Frontend** — `ModelRegistryControls` (`ModelResultModal.tsx`,
  onglet Détails) : badge de statut + actions Mettre en validation/
  Promouvoir en production/Retirer/Exporter l'artefact. `api/client.ts::
  exportModel` télécharge via `fetch` + lien éphémère (pas `request()`,
  qui suppose toujours une réponse JSON). Nouveau panneau "Registre de
  modèles" en tête de `TrainingHistory.tsx` (Lot D-bis) — liste les
  modèles promus, masqué s'il n'y en a aucun.

**Vérifié** :

- 11 nouveaux tests (`tests/test_model_registry.py`) : promotion vers
  chaque statut, démotion automatique du modèle production précédent pour
  le MÊME dataset+cible, absence de démotion pour une cible différente,
  retrait (`"none"`), rejet d'un statut invalide, rejet sur un job sans
  modèle, export retourne un fichier joblib valide et rechargeable, rejet
  si l'artefact a disparu du disque, isolation entre organisations
  (export ET registre). Suite pytest complète verte.
- Frontend : `tsc -b`, `vite build`, `vitest` (13/13) verts.
- Rendu visuel réel **non vérifié** en conditions réelles — aucun outil
  d'interaction navigateur disponible dans cet environnement de travail.

**Scope volontairement limité** : pas d'export ONNX — le pipeline peut
inclure un transformateur personnalisé (`RareCategoryFrequencyEncoder`,
Lot 4c) sans convertisseur ONNX standard ; l'export joblib est honnête
(fonctionne réellement) là où un export ONNX partiel aurait pu échouer
silencieusement pour certains pipelines. À reprendre dans un lot dédié si
un besoin réel d'interopérabilité hors Python apparaît. Pas de limite de
versions conservées (aucune purge automatique des anciens modèles).

## Lot 10 — durcissement SaaS, portée technique (livré)

Portée volontairement TECHNIQUE (garde-fous), pas commerciale — pas de
plans tarifaires, quotas de stockage ni facturation, décisions produit
hors périmètre d'un audit backend et non tranchées ici.

- [x] **`api/core/models.py::AuditLog`** (nouvelle table) — journal des
  actions sensibles d'une organisation : qui, quand, quoi. Pas un log
  applicatif générique (déjà couvert par `logging`, journaux serveur) :
  seulement ce qu'un `owner` voudrait pouvoir auditer après coup.
  `actor_id` avec `ondelete="SET NULL"` (pas de cascade) — une entrée
  survit à la suppression du compte de son auteur, la traçabilité de
  l'action ne doit jamais disparaître avec lui.
- [x] **`services/audit.py::log_action`** — écrit l'entrée SANS committer
  (l'appelant l'ajoute à la MÊME transaction que l'action auditée elle-même
  — suppression de dataset, d'entraînement, ajout de membre, promotion de
  modèle — pour qu'un rollback annule les deux ensemble, jamais un journal
  qui prétend qu'une action a eu lieu alors qu'elle a échoué).
- [x] **`GET /auth/team/audit-log`** (réservé au `owner`, même règle que le
  reste de la gestion d'équipe) — les 100 dernières actions de
  l'organisation, nom de l'auteur résolu, isolé comme le reste.
- [x] **Quota technique de jobs concurrents**
  (`Settings.max_concurrent_jobs_per_org`, défaut 3) — un seul worker RQ
  traite les jobs de TOUTES les organisations (`docker-compose.yml`) : sans
  limite, une organisation qui enfile beaucoup d'entraînements d'affilée
  peut affamer les autres. Vérifié dans `create_training_job` AVANT toute
  lecture du dataset (échec rapide), ne compte que les jobs `queued`/
  `running` — un job terminé ou en échec libère immédiatement le quota.
  429 `QUOTA_ENTRAINEMENTS_ATTEINT` avec un message actionnable.
- [x] **Frontend** — `AuditLogPanel` (`Dashboard.tsx`, section owner
  uniquement, à côté de la gestion d'équipe) : les 20 dernières actions,
  libellé en langage clair par type d'action. Le dépassement de quota
  s'affiche automatiquement via la bannière d'erreur déjà utilisée par le
  formulaire d'entraînement (aucun traitement spécial nécessaire côté UI,
  le message serveur est déjà actionnable).

**Vérifié** :

- Nouveaux tests (`tests/test_saas_hardening.py`) : chaque type d'action
  auditée (membre ajouté, dataset supprimé, entraînement supprimé, modèle
  promu) apparaît bien dans le journal avec le bon acteur et les bons
  détails ; accès restreint au owner (403 pour un membre) ; isolation entre
  organisations. Quota : blocage au-delà de la limite, jobs terminés/en
  échec qui ne comptent plus, isolation entre organisations (le quota
  atteint d'une organisation n'affecte jamais une autre).
- Frontend : `tsc -b`, `vite build`, `vitest` (13/13) verts.
- Rendu visuel réel **non vérifié** en conditions réelles — aucun outil
  d'interaction navigateur disponible dans cet environnement de travail.

**Scope volontairement limité** (hors périmètre technique de ce lot, à
cadrer séparément si besoin) : pas de plans tarifaires/facturation, pas de
quota de stockage (taille totale des datasets par organisation), pas
d'export du journal d'audit (CSV/PDF), pas de rétention limitée du journal
(aucune purge automatique).

## Lot Audit + durcissement — corrections de l'audit du 2026-08-14 (livré)

Audit expert complet (lecture seule, backend + frontend + legacy, voir
`AUDIT_ROADMAP.md`) suivi d'un lot correctif couvrant les points Critique et
la majorité des points Important. Portée volontairement large — validée
lot par lot via les tests, jamais un commit sans suite verte.

**Backend, critique** :
- [x] **H1 — isolation des tests** : la suite complète (257 tests) échouait
  intermittemment (2 tests) uniquement en run complet, jamais isolée ni sur
  des sous-groupes de fichiers testés explicitement. Diagnostiqué comme un
  épuisement de ressources (connexions SQLite non recyclées sur ~30 min,
  des dizaines d'entraînements réels), pas une fuite d'état déterministe.
  Mitigé par `poolclass=NullPool` dans `tests/conftest.py`.
- [x] **H2 — watchdog de jobs orphelins** (`services/job_watchdog.py`,
  nouveau) : un job `running` dont le worker a crashé (OOM, coupure)
  consommait indéfiniment un slot de quota. Réconciliation appelée avant le
  comptage du quota (`api/routers/training.py::create_training_job`) —
  `TrainingJob.progress_updated_at` (migration additive) sert de dernier
  signal de vie, seuil configurable (`stale_job_timeout_minutes`, défaut 40
  min, au-dessus du timeout RQ de 30 min).
- [x] **H3 — hard-fail JWT en production** : la clé par défaut n'était
  auparavant que journalisée en avertissement, jamais bloquante. Démarrage
  refusé (`RuntimeError`) si `environment=="production"` et
  `jwt_secret_key` reste la valeur par défaut du dépôt.
- [x] **H7 — erreurs déjà rédigées absorbées par le filet générique** :
  nouveau type dédié `TrainingAbortedError` (`services/ml_preprocessing.py`,
  sous-classe de `RuntimeError`) pour les erreurs de pré-condition
  utilisateur (dataset non prêt, classe absente après split groupé) —
  surfacées telles quelles. **Premier essai trop large** (catcher tout
  `RuntimeError`) cassait un test de non-régression existant en surfaçant
  aussi de vraies erreurs techniques (CatBoost) : corrigé avec un type
  dédié, pas un `RuntimeError` nu.
- [x] **H8 — classe absente après split groupé** : `GroupShuffleSplit` ne
  stratifie pas — une classe rare concentrée dans un seul groupe pouvait
  finir entièrement en test, provoquant une `ValueError` sklearn brute à
  l'encodage. Détecté explicitement dans `train_and_evaluate`
  (`services/ml_training.py`), message actionnable via `TrainingAbortedError`.
- [x] **H6 — seed non propagée à la CV** : `_make_cv` hardcodait
  `random_state=42`, indépendant du seed choisi par l'utilisateur. Corrigé,
  `GroupKFold` (déterministe, pas de `shuffle`) inchangé.

**Backend, important** :
- [x] **H11 — rate-limiting login** (`api/core/rate_limit.py`, nouveau) :
  fenêtre glissante Redis par IP cliente sur `/auth/login` uniquement
  (jamais `/auth/register`), 10 tentatives échouées / 15 min par défaut,
  échec ouvert si Redis est indisponible, compteur remis à zéro sur succès.
  Nécessite `tests/conftest.py::_fresh_rate_limit_counters` (nouveau,
  autouse) pour la même raison que H1 — sans ça, le compteur
  `login_attempts:testclient` se serait accumulé sur toute la suite.
- [x] **H17 — champ mort `ModelSpec.requires_scaling`** : jamais consommé
  (`build_preprocessor` scale déjà tout inconditionnellement) — supprimé du
  registre plutôt que branché, pour ne pas ajouter de complexité sans
  bénéfice réel.
- [x] **H5 — dérive documentaire** : `backend/ARCHITECTURE.md` était figé
  au Lot 4b pendant que ce fichier et `recap.md` continuaient d'avancer —
  remis à niveau (résumé, pas dupliqué).

**Frontend — refonte design + corrections UX** (guidée par retours
utilisateur directs sur captures d'écran, plusieurs itérations) :
- [x] Système de tokens sémantiques réellement appliqué partout (auparavant
  suivi seulement dans une minorité de composants) — passage systématique
  slate-\*/rose-\*/emerald-\* → tokens sur tous les fichiers `pages/` et
  `components/training|datasets|dashboard|pillars|ui`.
- [x] Fond de page reclarifié (bleu-gris terne → quasi blanc,
  `--color-background` 0.94→0.98 de luminosité) : la séparation
  canevas/carte se fait désormais par l'ombre (`Card.tsx`, `shadow-md`
  renforcée) plutôt que par un écart de couleur de fond.
- [x] **`PageHeader`** (nouveau, `components/ui/PageHeader.tsx`) : icône en
  dégradé de marque + fond teinté par pilier, remplace le titre flottant
  sans ancrage visuel sur Dashboard/Datasets/Training/TrainingHistory.
- [x] Cartes colorées sur **tous les onglets** de Résultats (Performance,
  Explicabilité, Fiabilité, Prédire, Détails) et Explorer (5 onglets) —
  fond teinté assorti à l'icône de section, sauf les deux cartes qui
  contiennent déjà des sous-cartes multicolores (Performance, Fiabilité des
  prédictions), laissées neutres pour ne pas se télescoper.
- [x] Modale de résultats/exploration agrandie (`Modal.tsx`, nouvelle taille
  `2xl`, 95vw/90vh) — perçue comme trop petite pour des grilles de
  graphiques.
- [x] **Bug réel trouvé et supprimé** (pas corrigé, supprimé sur demande
  explicite) : le graphe "Variance entre les découpages de validation
  croisée" (`Leaderboard`, `ModelResultModal.tsx`) affichait des valeurs
  aberrantes (jusqu'à 9 chiffres) malgré le clamp déjà en place côté
  utilitaire (`utils/cvScore.ts`) — cause non retrouvée dans le composant
  `BoxPlotChart` personnalisé, retiré plutôt que laissé cassé.
- [x] Premier vrai onboarding du produit (`components/HelpModal.tsx`,
  nouveau) — remplace la recherche/les notifications de l'AppShell
  (visuelles mais jamais câblées, honnêteté du même principe que les
  piliers "Bientôt").
- [x] Salutation à 3 tranches (Bonjour/Bon après-midi/Bonsoir).
- [x] **H4/D2** — confirmation à deux clics avant suppression de dataset
  (`Datasets.tsx`), motif extrait dans `hooks/useConfirmAction.ts`.
- [x] **H4/D3** — distinction erreur réseau / état vide sur Dashboard,
  Training, TrainingHistory (auparavant indiscernables, catch silencieux
  vers `[]`).
- [x] **H10** — labels de formulaire liés (`htmlFor`/`id`) sur le wizard
  d'entraînement (5 champs), `ExpertModePanel` (4 champs), `PredictionForm`
  (génération dynamique par variable).
- [x] **H14** — duplication réduite : `StatusBadge.tsx`
  (`JobStatusBadge`/`DatasetStatusBadge`, 3 implémentations identiques
  fusionnées), `Select.tsx` (6+ occurrences de la même classe recopiée),
  `useConfirmAction.ts` (motif à deux clics, 2 implémentations
  indépendantes fusionnées).
- [x] **H15** — contraste `text-warning`/`text-success` vérifié au
  contrastomètre (calcul WCAG direct sur les valeurs OKLCH, pas d'outil
  visuel dans cet environnement) : 2.57:1 et 3.40:1 sur fond blanc, sous le
  seuil AA texte normal (4.5:1). Luminosité réduite (teinte/chroma
  inchangés) → 4.82:1 sur les deux — bénéfice secondaire, améliore aussi le
  contraste du texte blanc sur `bg-success` plein (`Training.tsx`).
- [x] **H18** — grille de métriques (`ModelResultModal.tsx`) : repli
  `grid-cols-1` en dessous de `sm`, plus figée à 2 colonnes sur mobile.
- [x] **H9** — `eslint` réellement installé (`eslint.config.js`, nouveau —
  configuré dans `package.json` depuis des mois de lots mais jamais
  installé, `npm run lint` échouait immédiatement). Règles React Hooks
  limitées aux deux classiques (`rules-of-hooks`, `exhaustive-deps`) —
  `eslint-plugin-react-hooks` v7 embarque par défaut des règles orientées
  React Compiler (ex. `set-state-in-effect`) qui signalent comme erreur le
  pattern standard "charger des données dans un `useEffect`", largement et
  légitimement utilisé dans ce projet. État à l'installation : 0 erreur, 11
  avertissements mesurés (dette réelle mais non bloquante).
- [x] **Persistance de session** (`pages/Training.tsx`) — signalé comme
  "comportement ambigu à clarifier avant d'y toucher" dans ce fichier
  depuis le lot "Refonte UI : design system moderne", jamais traité :
  rafraîchir la page pendant qu'un entraînement tourne réellement côté
  serveur renvoyait silencieusement au formulaire de configuration.
  `sessionStorage` (pas `localStorage`, volontairement borné à l'onglet en
  cours) restaure la vue (progression/résultat/échec) au lieu de perdre
  l'état.
- [x] **Deep-linking** (H20/D12) — ouvrir un résultat d'entraînement
  (`Dashboard.tsx`) ou l'exploration/aperçu d'un dataset (`Datasets.tsx`)
  synchronise désormais l'URL (`?job=`, `?explore=`, `?preview=`) dans les
  deux sens via `useSearchParams` — un rafraîchissement ou un lien partagé
  rouvre la même vue.

**Vérifié** : suite pytest ciblée sur chaque fichier touché systématiquement
verte à chaque étape (jamais un commit sans validation) ; `tsc -b`,
`vite build`, `vitest run` (13/13) verts après chaque vague de changements
frontend ; `npm run lint` fonctionnel pour la première fois. Suite pytest
complète non re-vérifiée en un seul run après ce lot (coût ~30 min) — les
fichiers touchés ont chacun été validés isolément.

**Reporté** (voir `AUDIT_ROADMAP.md` pour le détail) : H12/H13 (découpage de
`ModelResultView`/`TrainingForm` en sous-composants, tests de composants
React — aucune infrastructure de test de composants dans ce dépôt,
`@testing-library/react` non installé) ; H19 (séparer Dashboard
activité/organisation) ; H21 (quota de stockage, export du journal
d'audit) ; H22 (couche Workspace/Projet, pas de besoin business confirmé) —
chantiers plus lourds, à traiter dans un lot dédié plutôt que dilués ici.

## Lot 11+12 — Clustering et profils de segments, ML non supervisé (livré)

Premier module du pilier "ML non supervisé". Module **délibérément séparé**
de `ml_registry.py`/`ml_training.py` — confirmé par l'audit du 2026-08-14
(AUDIT_ROADMAP.md) : le clustering n'a pas de `y`, les hypothèses binaires
classification/régression du moteur supervisé (encodage de la cible, score
de sélection, CQR) n'ont pas de sens pour un algorithme non supervisé.

- [x] **`services/clustering_registry.py`** (nouveau) — registre de 4
  algorithmes (`ClusterSpec`) : K-Means, K-Means rapide (MiniBatch, gros
  volumes, mode expert), clustering hiérarchique (linkage Ward), DBSCAN.
  `candidate_configs(n_samples, seed)` génère la grille de configurations à
  comparer par algorithme — k ∈ {2,3,4,5,6,8} pour les algorithmes à k
  explicite ; pour DBSCAN, `resolve_dbscan_eps` dérive le rayon de
  voisinage de la distribution réelle des distances au k-ème plus proche
  voisin (percentile 90, heuristique du "k-distance plot"), pas une valeur
  fixe. Sous-ensemble par défaut (`DEFAULT_ALGORITHM_IDS`) : K-Means,
  hiérarchique, DBSCAN — MiniBatchKMeans réservé au mode expert, même
  stratégie produit "B" que le Lot 5 côté supervisé.
- [x] **`services/clustering_training.py`** (nouveau) —
  `train_and_evaluate_clustering` : réutilise `build_preprocessor`
  (`ml_preprocessing.py`, générique, déjà indépendant de toute cible),
  compare toutes les configurations du registre filtré, sélectionne la
  meilleure sur le score de silhouette (Davies-Bouldin/Calinski-Harabasz
  calculés en recoupement, jamais pour classer). `_compute_cluster_metrics`
  exclut le bruit DBSCAN (label -1) du calcul des 3 métriques — l'inclure
  les fausserait. `_build_cluster_profiles` calcule, pour le clustering
  retenu, taille/moyenne/médiane/z-score par variable et top catégorie par
  variable catégorielle, sur les données D'ORIGINE (pas préprocessées) —
  variables différenciantes triées par |z-score|, jamais un texte inventé
  (règle reprise du skill senior-ai-saas-engineer, data-science.md).
  **Dégradation propre confirmée par test** : un `eps` DBSCAN résolu à 0.0
  (données quasi dégénérées) est écarté proprement plutôt que de lever
  l'`InvalidParameterError` sklearn brute — trouvé par un test dédié, pas
  supposé.
- [x] **`api/core/models.py`** — `ClusteringJob`/`ClusterModel`/
  `ClusterCandidateRecord`, tables dédiées (pas une extension de
  `TrainingJob`/`MLModel`), même conventions d'isolation
  (`organization_id`) et de progression (`progress_step`/
  `progress_percent`/`progress_updated_at`) que le supervisé. Tables
  neuves : pas de migration additive nécessaire, `create_all()` suffit.
- [x] **`services/job_watchdog.py`** — `reconcile_stale_jobs` généralisé
  (paramètre `model: Type[JobModel] = TrainingJob`) plutôt que dupliqué —
  `TrainingJob` et `ClusteringJob` partagent exactement les colonnes de
  progression nécessaires.
- [x] **`workers/clustering_worker.py`** (nouveau) — même structure que
  `training_worker.py` (session DB propre, progression persistée à chaque
  étape, `TrainingAbortedError`/`FeatureEngineeringSpecError`-like
  dégradation propre, `_user_safe_error_message` dupliqué volontairement
  plutôt qu'importé — indépendance des deux traductions si elles divergent
  un jour, même choix que la séparation des registres).
- [x] **`api/routers/clustering.py`** (nouveau) — `POST /clustering/jobs`
  (création + enfilement sur `training_queue`, même file que le supervisé —
  un seul worker physique), `GET /jobs`, `GET /jobs/{id}`, `GET
  /jobs/{id}/result`, `GET /jobs/{id}/candidates`, `DELETE /jobs/{id}`, `GET
  /algorithms-catalog`. **Quota partagé** avec le supervisé (compte
  `TrainingJob` + `ClusteringJob` actifs ensemble contre la même limite,
  `max_concurrent_jobs_per_org`) — un seul worker traite les deux types, une
  limite séparée par type aurait permis de le saturer en cumulant les deux.
- [x] **Frontend** — `pages/Clustering.tsx` (nouveau) : sélection dataset +
  variables (cases à cocher, pas de wizard multi-étapes comme le supervisé
  — la configuration d'un clustering est plus légère), lancement,
  progression, résultat (leaderboard implicite via le badge algorithme +
  cartes de profils de segments colorées, différenciantes en tête). Même
  persistance de session (`sessionStorage`) que `Training.tsx` — un
  rafraîchissement pendant le calcul ne perd pas la progression. Pilier
  "ML non supervisé" activé dans `config/pillars.ts` (`status: "active"`,
  route `/clustering`) — la page signale elle-même, honnêtement, que
  détection d'anomalies et réduction de dimension arrivent ensuite (pas
  promises dans le pilier sans être livrées).

**Vérifié** :

- 41 nouveaux tests (`test_clustering_registry.py` 8,
  `test_clustering_training.py` 14, `test_clustering_worker.py` 5,
  `test_clustering_api.py` 11, `test_security.py`/`test_auth.py` non
  comptés ici) : chaque algorithme du registre s'entraîne et prédit sans
  erreur, sélection correcte du nombre de groupes sur un cas construit (3
  groupes séparés + variable catégorielle corrélée — silhouette > 0,7,
  chaque segment découvert dominé à 100 % par sa vraie catégorie d'origine,
  pas seulement "ça ne plante pas"), variables différenciantes identifient
  le vrai signal, seed réellement propagée à l'estimateur (vérifié sur
  l'objet fitté, pas déduit d'un résultat statistique), quota partagé
  supervisé/clustering, isolation multi-tenant, dégradation propre sur
  dataset trop petit ou données dégénérées, aucune trace brute
  (chemin/traceback) dans un message d'erreur persisté.
- **Un bug de régression volontairement introduit par cette même session a
  été détecté par les tests de ce lot** : le premier essai du correctif H7
  (audit du même jour) attrapait tout `RuntimeError` pour lui donner un
  message plus clair — `test_worker_never_leaks_raw_traceback_on_training_failure`
  a immédiatement signalé que ça surfaçait aussi de vraies erreurs
  techniques CatBoost. Corrigé avec un type dédié (`TrainingAbortedError`)
  avant que ce lot ne réutilise le même mécanisme pour le clustering —
  preuve que le filet de test fonctionne.
- Régression : suite ciblée sur les fichiers touchés par le partage de
  quota (`test_training_api.py`, `test_saas_hardening.py`, `test_auth.py`)
  intégralement verte (39/39) après ce lot.
- Frontend : `tsc -b`, `vite build`, `npm run lint` (0 erreur), `vitest run`
  (13/13) verts.

**Scope volontairement limité** (planifié séparément, signalé honnêtement
dans la page `Clustering.tsx` elle-même) : détection d'anomalies tabulaire
(Isolation Forest, LOF — Lot 14) et réduction de dimension pour
visualisation (PCA/t-SNE/UMAP — Lot 13) ; pas de prédiction du cluster
d'une nouvelle observation (le bundle joblib persisté le permettrait, non
exposé par API dans ce lot) ; pas de comparaison inter-jobs de clustering
(pendant du Lot D-bis côté supervisé) ; rendu visuel réel non vérifié dans
un navigateur (même limite que tous les lots frontend précédents dans cet
environnement).

## Lot 13 — Réduction de dimension : PCA/t-SNE/UMAP (livré)

Deuxième module du pilier "ML non supervisé", même principe de séparation
que le clustering (Lot 11+12) : module backend dédié, aucune notion de
cible partagée avec le supervisé.

**Bug réel trouvé en testant le clustering en direct dans un navigateur,
avant même de commencer ce lot** : `frontend/vite.config.ts` proxyait
`/api`, `/auth`, `/datasets`, `/training` vers le backend en dev, mais
`/clustering` (ajouté au Lot 11+12) avait été oublié — toute requête
clustering en dev renvoyait 404. Corrigé, et `/dimensionality`/`/anomalies`
ajoutés par avance pour ne pas reproduire l'oubli avec les deux nouveaux
modules de cette session.

- [x] **UMAP — vérification d'installation Windows en tout premier**
  (risque explicitement accepté par l'utilisateur, à valider avant d'écrire
  le reste du lot) : `pip install umap-learn` installe par défaut
  `umap-learn==0.5.12`, qui force `scikit-learn>=1.6` et a upgradé
  silencieusement le `scikit-learn==1.3.2` épinglé du projet vers `1.9.0` —
  reverté immédiatement (risque de régression sur tout le pipeline
  supervisé/clustering déjà testé, joblib persistés avec 1.3.2). Résolu en
  épinglant `umap-learn==0.5.6` (`requirements.txt`), dont la contrainte
  réelle n'est que `scikit-learn>=0.22` — vérifié par un fit UMAP réel,
  aucun impact sur le reste des dépendances.
- [x] **`services/dimensionality_registry.py`** (nouveau) — `DimensionalitySpec`
  (id/label/family/`build_estimator`) pour `pca`/`tsne` (toujours présentes)
  et `umap` (ajoutée seulement si `umap-learn` s'importe, jamais un crash
  API sinon). Hyperparamètres clippés pour petits datasets (`perplexity`
  t-SNE < n_samples, `n_neighbors` UMAP < n_samples). `DISTANCE_FIDELITY_NOTES`
  — texte obligatoire par méthode (exigence du skill senior-ai-saas-engineer,
  data-science.md) : PCA fidèle aux distances globales, t-SNE/UMAP non.
- [x] **`services/dimensionality_training.py`** (nouveau) —
  `train_and_evaluate_dimensionality` : pas de leaderboard façon clustering
  (aucune métrique de qualité commune entre PCA/t-SNE/UMAP). La PCA est
  **toujours** calculée en plus de la méthode choisie (variance expliquée,
  loadings top 15 par magnitude) ; `sklearn.manifold.trustworthiness`
  calculée sur l'embedding principal ET sur la PCA de référence — mesure
  réelle et comparable de la qualité d'une projection, jamais un texte
  inventé. Échantillonnage déterministe (`sample_if_large`,
  `MAX_ROWS_FOR_EMBEDDING = 5000`, préservation de l'index de ligne
  d'origine à travers l'échantillonnage) — t-SNE quasi-quadratique et rendu
  `ScatterChart` poussif au-delà. Dégradation propre (`TrainingAbortedError`)
  si trop peu de lignes/variables exploitables.
- [x] **`api/core/models.py`** — `DimensionalityJob`/`DimensionalityModel`/
  `DimensionalityPoint` (tables dédiées). `DimensionalityPoint` : une ligne
  par point projeté (pas un JSON agrégé), même raisonnement que
  `ClusterCandidateRecord` — requêtable, borné par l'échantillonnage.
- [x] **`workers/dimensionality_worker.py`** (nouveau) — même structure que
  `clustering_worker.py`.
- [x] **`api/routers/dimensionality.py`** (nouveau) — `GET /algorithms-catalog`,
  `POST /jobs`, `GET /jobs`, `GET /jobs/{id}`, `GET /jobs/{id}/result`,
  `GET /jobs/{id}/points`, `GET /jobs/{id}/color-by?column=` (relit UNE
  colonne du dataset source à la demande pour les points déjà projetés —
  calcul léger synchrone, même pattern que `/datasets/{id}/histogram` ;
  changer la coloration du nuage de points devient instantané, sans
  relancer un calcul coûteux), `DELETE /jobs/{id}`.
- [x] **Frontend** — `pages/DimensionalityReduction.tsx` (nouveau) :
  formulaire dataset + colonnes + méthode, `ScatterChart` Recharts
  (première utilisation dans le projet, palette `CHART_SERIES_COLORS`),
  sélecteur de coloration (catégorielle ou numérique binnée en quantiles
  côté client, `utils/quantileBins.ts`), tableau des loadings PCA. Lien
  croisé "Visualiser en 2D" depuis un résultat de clustering
  (`Clustering.tsx`) — dataset et colonnes pré-remplis via query params,
  **aucun couplage backend** (pas de transmission de label de cluster,
  décision volontaire pour garder les deux moteurs indépendants).
  `components/ui/Table.tsx` (nouveau, générique) créé à cette occasion et
  immédiatement réutilisé pour combler un trou UX préexistant : le tableau
  des candidats comparés par le clustering était déjà exposé par l'API
  (Lot 11+12, `GET /clustering/jobs/{id}/candidates`) mais jamais affiché.
- [x] **Rigueur d'affichage corrigée sur retour direct** (après premier test
  utilisateur réel en navigateur) : la variance expliquée n'est définie que
  pour la PCA — retirée de l'affichage principal pour t-SNE/UMAP, isolée
  dans un bloc "Référence PCA" séparé et distinctement coloré (calculée en
  plus, à titre de repère, pas la méthode utilisée pour la projection
  affichée). Tableau des loadings PC1/PC2 masqué pour t-SNE/UMAP (n'a de
  sens que pour une projection linéaire). "Fidélité de la projection"
  renommée "Conservation des voisinages" — vérifié contre l'implémentation
  avant de renommer : `trustworthiness` mesure exactement si les
  voisinages proches sur la projection l'étaient déjà dans les données
  d'origine, le nouveau nom est donc plus exact, appliqué uniformément
  (PCA incluse, même métrique).

**Vérifié** : 34 tests (`test_dimensionality_registry.py` 6 avec
paramétrage, `test_dimensionality_training.py` 12 avec paramétrage,
`test_dimensionality_worker.py` 4,
`test_dimensionality_api.py` 12) — chaque méthode du registre s'exécute
bout en bout, la PCA identifie la vraie variable porteuse de signal parmi
du bruit (dataset construit : 2 variables corrélées + 1 indépendante — PC1
domine grâce à la corrélation partagée, PC2 domine sur la variable
indépendante, pas juste "ça ne plante pas"), `trustworthiness` toujours
dans [0, 1], seed propagée aux estimateurs vérifiée sur l'objet fitté,
échantillonnage exact et déterministe, endpoint `color-by` vérifié de bout
en bout (job réellement exécuté dans le test, pas mocké). Suite de
régression du supervisé + clustering rejouée intégralement après ce lot
(83 tests, aucune casse). `tsc -b`, `vite build`, `npm run lint` (0 erreur),
`vitest run` (33/33) verts. **Vérifié en conditions réelles dans un
navigateur par l'utilisateur** (contrairement aux lots précédents,
limitation levée cette fois) — un second aller-retour direct sur les
captures d'écran a mené aux corrections de rigueur ci-dessus.

## Lot 14 — Détection d'anomalies tabulaire : Isolation Forest + LOF (livré)

Troisième et dernier module du pilier "ML non supervisé" — les 3 modules
annoncés dans `config/pillars.ts` sont désormais tous actifs.

- [x] **`services/anomaly_registry.py`** (nouveau) — `AnomalySpec` pour
  `isolation_forest` et `lof`, `contamination="auto"` toujours (pas de
  paramètre à régler en mode guidé, cohérent avec le reste du produit).
  Pas de notion de "sélection d'algorithme" par l'utilisateur : les deux
  tournent systématiquement ensemble (voir plus bas), donc pas de
  `GET /algorithms-catalog` pour ce module — écart assumé et documenté par
  rapport au pattern clustering/réduction de dimension, il n'y a rien à
  choisir.
- [x] **`services/anomaly_training.py`** (nouveau) —
  `train_and_evaluate_anomalies` : Isolation Forest et LOF exécutés
  **toujours ensemble**, jamais un seul essai à l'aveugle (principe déjà
  établi par le reste du produit) — sans vérité terrain disponible, il n'y
  a pas de "gagnant" à élire comme au clustering. Score de **consensus**
  continu = moyenne des rangs percentiles de chaque algorithme (leurs
  scores bruts, `score_samples`/`negative_outlier_factor_`, ne sont pas sur
  la même échelle — seul le rang est comparable). Confiance par observation
  (`agreement` : `both`/`isolation_forest_only`/`lof_only`/`none`) déduite
  des deux flags booléens réels, jamais un nombre inventé. Pour chaque
  observation classée : déviation z-score par variable numérique (top 5
  |z|, même logique que `differentiating_variables` du clustering) et
  drapeau des valeurs catégorielles rares (fréquence < 5 % dans
  l'échantillon analysé) — calculé sur les données D'ORIGINE, pas
  préprocessées. Échantillonnage déterministe
  (`MAX_ROWS_FOR_ANOMALY = 20 000`, Isolation Forest/LOF restant efficaces
  à cette échelle contrairement à t-SNE). Top-N borné
  (`DEFAULT_TOP_N = 50`, `MAX_TOP_N = 200`).
- [x] **`api/core/models.py`** — `AnomalyJob`/`AnomalyModel`/
  `AnomalyObservationRecord` (tables dédiées). `AnomalyObservationRecord` :
  une ligne par observation classée (top-N seulement, borné), même
  raisonnement que `ClusterCandidateRecord`/`DimensionalityPoint`.
- [x] **`workers/anomaly_worker.py`** (nouveau) — même structure que les
  workers précédents.
- [x] **`api/routers/anomalies.py`** (nouveau) — `POST /jobs` (`top_n`
  validé 1..200 via `Field(ge=1, le=200)`), `GET /jobs`, `GET /jobs/{id}`,
  `GET /jobs/{id}/result`, `GET /jobs/{id}/observations`, `DELETE /jobs/{id}`.
- [x] **`services/job_quota.py`** (nouveau) — **corrige un oubli réel
  trouvé en construisant ce lot** : le quota partagé introduit au
  Lot 11+12 (`clustering.py` comptait `TrainingJob` + `ClusteringJob`
  ensemble) n'avait jamais été répercuté dans `training.py`, qui ne
  comptait toujours que les `TrainingJob` actifs — un entraînement
  supervisé pouvait donc être lancé sans tenir compte des clusterings déjà
  actifs, rendant le quota partagé contournable depuis un seul des deux
  côtés. Extrait en helper commun (`count_active_jobs`/
  `raise_if_quota_exceeded`) avant d'ajouter une 3ᵉ puis 4ᵉ table
  (réduction de dimension, anomalies) à la liste — pour ne plus jamais
  dupliquer ce bloc de comptage à la main. `training.py`, `clustering.py`,
  `dimensionality.py` et `anomalies.py` comptent désormais tous les 4 types
  de job ensemble.
- [x] **Frontend** — `pages/AnomalyDetection.tsx` (nouveau) : formulaire
  dataset + colonnes + nombre d'observations à classer (défaut 50, borné
  1..200 côté client aussi). Résultats : taux d'anomalies par méthode et en
  consensus (mis en avant), histogramme Recharts des scores de consensus,
  tableau des observations classées (`components/ui/Table.tsx`, badge de
  confiance coloré par niveau d'accord, variable la plus explicative en
  ligne), modale de détail par observation (déviations complètes,
  `components/ui/Modal.tsx`). Lien croisé bidirectionnel avec `Clustering.tsx`
  (dataset + colonnes pré-remplis via query params, même principe que le
  Lot 13). `config/pillars.ts` : 3ᵉ et dernier item de navigation du pilier
  non supervisé activé.

**Vérifié** : 30 tests (`test_anomaly_registry.py` 4,
`test_anomaly_training.py` 12, `test_anomaly_worker.py` 3,
`test_anomaly_api.py` 11) — dataset construit avec 5 observations injectées
loin du reste : les 5 apparaissent exactement en tête du classement par
consensus, les plus extrêmes avec `agreement="both"`, invariant
mathématique vérifié (taux de consensus toujours ≤ taux de chaque méthode
prise isolément, c'est une intersection), déviations numériques pointent
la bonne variable perturbée, valeurs catégorielles rares détectées,
dégradation propre sur données dégénérées (z-score à 0, jamais NaN), seed
propagée. Suite de régression complète du pilier non supervisé + supervisé
rejouée après ce lot (99 tests : clustering + réduction de dimension +
supervisé + sécurité/auth, aucune casse). `tsc -b`, `vite build`,
`npm run lint` (0 erreur), `vitest run` (33/33) verts. 376 tests au total
dans le dépôt à l'issue de ce lot.

## Lot 15, sous-lot A — Pilier Vision : infrastructure dataset image (livré)

Premier sous-lot du pilier Vision (4 sous-lots planifiés : A infra dataset →
B classification/transfer learning → C anomalies visuelles MVTec AD → D
Grad-CAM — voir `AUDIT_ROADMAP.md` section H). Fondation partagée par B et C
(les deux ont besoin d'ingérer un dataset d'images), construite une seule
fois. Aucune réutilisation du legacy Streamlit (`src/`, `orchestrators/`,
`monitoring/state_managers.STATE`) — module backend entièrement neuf, seule
l'infra générique (storage, audit) est réutilisée telle quelle.

- [x] **`services/vision_datasets.py`** (nouveau) — logique pure, sans
  dépendance HTTP, directement testable :
  - Détection de structure **stricte, jamais devinée en silence** —
    correctif préventif du bug #1 déjà documenté
    (`docs/legacy/ANALYSE_COMPLETE_COMPUTER_VISION.md` : un `y_train` MVTec
    AD mal chargé pouvait faire partir le legacy en mode supervisé par
    accident). Deux structures reconnues, aucune autre : dossiers de classes
    (`<classe>/*.png`, ≥2 classes) pour la classification, ou
    `train/good/` + `test/good/` + `test/<defaut>/` pour MVTec AD — `train/`
    ne doit contenir QUE des images normales (`good`), `test/` doit contenir
    `good` et au moins une catégorie de défaut (nécessaire pour calibrer le
    seuil au sous-lot C, correctif préventif du bug #12).
  - Validation systématique par image (Pillow `verify()` + réouverture pour
    lire les dimensions réelles — `verify()` invalide l'objet pour tout
    usage ultérieur) : images corrompues/tronquées exclues et reportées
    (jamais bloquant sur une image individuelle), doublons détectés par hash
    SHA-256 (conservés, signalés), images sous-dimensionnées (<20px)
    exclues, classes déséquilibrées signalées (ratio >10x) sans bloquer.
  - Sécurité : protection zip-slip (chemins `..`/absolus rejetés avant toute
    extraction), protection zip-bombe (cap sur la taille décompressée
    cumulée ET sur le nombre d'images, indépendamment de la taille
    compressée de l'archive).
- [x] **`api/core/models.py`** — `VisionDataset` (table dédiée :
  `structure_type`, `storage_dir`, `n_images`, `n_classes`,
  `class_distribution_json`, `validation_report_json`).
- [x] **`api/core/storage.py`** — `vision_dataset_dir`/
  `delete_vision_dataset_dir` : un dataset vision est un DOSSIER de
  nombreuses petites images (pas un fichier unique comme
  `dataset_file_path`), même isolation disque
  (`storage/datasets_vision/{organization_id}/{dataset_id}/`).
- [x] **`api/core/config.py`** — `max_vision_upload_size_mb` (500, plus haut
  que `max_upload_size_mb` des CSV : un dataset MVTec AD réel pèse
  facilement plusieurs centaines de Mo), `max_vision_dataset_images` (5000,
  protection zip-bombe indépendante de la taille).
- [x] **`api/routers/vision_datasets.py`** (nouveau) — upload **synchrone**
  (comme les datasets tabulaires, pas de tâche de fond : validation bornée
  par `max_vision_dataset_images`, de l'ordre de la seconde), `POST`/`GET`
  (liste)/`GET {id}` (détail + rapport de validation)/`DELETE`. Un ZIP dont
  la structure n'est pas reconnue reçoit quand même une entrée en base
  (`status="error"`, message diagnostiqué) plutôt qu'une simple 4xx — même
  convention que `POST /datasets`.
- [x] **`requirements.txt`** — `pillow==12.3.0` épinglé explicitement
  (c'était une dépendance transitive jusqu'ici, devient directe).

**Vérifié** : 22 tests ciblés verts (`test_vision_datasets_service.py` 15 :
structures valides classification/MVTec AD, structure non reconnue,
zip-slip, ZIP invalide, image corrompue exclue+reportée, doublons détectés,
image sous-dimensionnée exclue, classe sous le minimum rejetée, MVTec AD
sans `test/good` rejeté, `train/` avec dossier de défaut rejeté (fuite de
labels), trop d'images rejeté, déséquilibre signalé sans bloquer, fichiers
non-image ignorés ; `test_vision_datasets_api.py` 7 : upload valide,
extension refusée, archive vide refusée, structure invalide → `status:
error` avec message, isolation multi-tenant liste/détail, suppression,
rapport de validation exposé par l'API). **Suite de régression complète du
dépôt non rejouée dans cette session** (interrompue volontairement après
plus de 40 minutes — la suite complète prenait ~29 min lors du dernier
audit à 257 tests, le dépôt en compte désormais ~398 ; à revérifier avant
le sous-lot B ou via CI, pas de régression suspectée vu la nature additive
des changements — nouveau modèle/router/service, aucun fichier ML existant
modifié).

Frontend : aucun (pas de page dédiée pour ce sous-lot — intégration prévue
au wizard du sous-lot B). Pilier `vision` reste `status: "soon"` dans
`config/pillars.ts` jusqu'à ce qu'une page réelle existe.

## Lot 15, sous-lot B — Pilier Vision : classification d'images + transfer learning (livré, backend)

Deuxième sous-lot du pilier Vision. Premier module du dépôt à réellement
entraîner un réseau de neurones (PyTorch/CPU) — aucune notion commune avec
`ml_training.py` (Optuna, `ColumnTransformer`, SHAP n'ont aucun sens pour des
images), module séparé de bout en bout, même principe que
`anomaly_training.py`.

- [x] **`requirements.txt`** — `torch==2.13.0`/`torchvision==0.28.0` (roues
  CPU, `--extra-index-url https://download.pytorch.org/whl/cpu` ajouté en
  tête de fichier — évite de récupérer par erreur une roue CUDA de plusieurs
  Go, aucun GPU dans `docker-compose.yml`).
- [x] **`services/vision_classification_registry.py`** (nouveau) — 2
  backbones pré-entraînés ImageNet seulement (`mobilenet_v3_small` par
  défaut, `resnet18`), volontairement pas les 17 du legacy
  (`transfer_learning.py` : VGG/EfficientNet/ResNet50+) — impraticables en
  CPU dans le temps d'un job. `in_features` de la couche remplacée lu
  dynamiquement sur chaque backbone (jamais codé en dur — diffère entre
  `resnet.fc` et `mobilenet.classifier[-1]`). Gel/dégel du backbone par
  sous-modules explicites (`backbone_children`), pas un compteur de couches
  arbitraire comme le legacy.
- [x] **`services/vision_classification_training.py`** (nouveau) —
  `train_and_evaluate_classification` : pipeline `ImageFolder` (structure
  déjà garantie par le sous-lot A) → split stratifié 70/15/15 (sklearn,
  seed propagée) → augmentation légère train uniquement (flip/rotation/
  jitter) → normalisation ImageNet → boucle d'entraînement (transfert
  learning, backbone gelé par défaut, dégel optionnel à une époque donnée
  via `unfreeze_after_epoch`) → meilleure époque conservée (`val_loss`
  minimal, pas la dernière) → évaluation test (accuracy/precision/recall/F1
  macro, matrice de confusion) → exemples de prédictions **corrects ET
  erronés** (priorité aux erreurs, skill Computer Vision). Garde-fou de
  temps interne (`max_training_seconds`, 1500s par défaut — largement sous
  le timeout RQ de 1800s) : arrête proprement entre deux époques plutôt que
  de risquer un timeout RQ brutal sur un gros dataset CPU, honnêtement
  signalé (`model_card["time_capped"]`). Vérification stricte AVANT tout
  calcul coûteux : ≥2 classes, ≥6 images/classe (minimum pour un split
  stratifié fiable) — `TrainingAbortedError` réutilisée telle quelle
  (`services/ml_preprocessing.py`), pas un nouveau type.
- [x] **`api/core/models.py`** — `VisionClassificationJob`/
  `VisionClassificationModel` (tables dédiées, mêmes conventions
  `progress_*`/`status` que les autres jobs).
- [x] **`api/core/storage.py`** — `vision_classification_model_file_path`
  (`.pt`, `torch.save`, pas `.joblib`). **Bug réel trouvé et corrigé en
  écrivant les tests** : `vision_dataset_dir` (sous-lot A) ne nettoyait pas
  un dossier préexistant avant extraction — invisible en production
  (séquence Postgres jamais réutilisée) mais reproductible en test SQLite
  (séquence qui redémarre à 1 après `drop_all`/`create_all` entre deux
  tests) : un id de dataset réutilisé héritait silencieusement des fichiers
  MVTec AD d'un test précédent, mélangés aux nouvelles classes de
  classification (`ImageFolder` détectait alors 4 "classes" au lieu de 2).
  Corrigé : le dossier est maintenant TOUJOURS vidé avant d'être recréé.
- [x] **`services/job_quota.py`** — **simplification profitant de l'ajout
  d'un 5ᵉ type de job** : la liste `[TrainingJob, ClusteringJob,
  DimensionalityJob, AnomalyJob]` était recopiée à l'identique dans les 4
  routers existants (risque déjà matérialisé une fois au Lot 14). Extraite
  en constante unique `ALL_JOB_MODELS` (avec le 5ᵉ type inclus), importée
  par les 5 routers désormais — un seul endroit à modifier pour le
  prochain type de job (sous-lot C).
- [x] **`workers/vision_classification_worker.py`** (nouveau) — même
  structure que les workers précédents ; rejette explicitement un dataset
  de structure `mvtec_ad` (message diagnostiqué, pas un crash `ImageFolder`
  silencieux sur une structure inattendue).
- [x] **`api/routers/vision_classification.py`** (nouveau) — `GET
  /backbones`, `POST/GET/DELETE /jobs`, `GET /jobs/{id}/result`. Valide la
  structure du dataset (`classification`, pas `mvtec_ad`) en défense en
  profondeur (déjà vérifié par le worker).

**Vérifié** : 30 tests nouveaux, tous verts en isolation (8
`test_vision_classification_registry.py`, 6
`test_vision_classification_training.py` — entraînement réel non mocké sur
mini dataset synthétique, 4 `test_vision_classification_worker.py`, 12
`test_vision_classification_api.py`, dont le partage du quota avec un job
tabulaire). Un run concurrent accidentel de deux suites pytest partageant le
même fichier SQLite de test a produit 3 échecs illusoires (`no such table`,
422 non levé) — non réels, confirmés en rejouant isolément juste après ;
leçon retenue : ne jamais lancer deux suites pytest en parallèle sur ce
dépôt (fichier de test SQLite à chemin fixe, `tests/conftest.py`).
Entraînement CPU réel mesuré : ~10-25s pour 1-2 époques sur un dataset de 16
images (mobilenet_v3_small, poids ImageNet déjà en cache local) — nécessite
un accès réseau au premier téléchargement des poids pré-entraînés
(`~/.cache/torch/hub/checkpoints/`), à anticiper pour le déploiement
(pré-télécharger à la construction de l'image Docker, ou volume persistant
sur le worker).

Frontend : aucun pour ce sous-lot (backend seul) — page à construire dans un
lot dédié une fois les sous-lots C/D également livrés, ou dans un lot
frontend séparé à cadrer.

## Lot 15, sous-lot C — Pilier Vision : détection d'anomalies visuelles MVTec AD (livré, backend)

Troisième sous-lot du pilier Vision — le plus bugué du legacy (9 des 18 bugs
documentés dans `docs/legacy/ANALYSE_COMPLETE_COMPUTER_VISION.md` touchent
directement ce périmètre). Reconception, pas un simple portage : chaque bug
corrigé PENDANT l'écriture, pas ajouté après coup.

- [x] **`services/vision_localization.py`** (nouveau, réutilisable tel quel
  au sous-lot D Grad-CAM) — `generate_binary_mask` (correctif direct du bug
  **#14** : cette fonction n'existait PAS dans le legacy), `resize_map_to_original`
  (correctif des bugs **#10/#17** : la carte d'erreur est TOUJOURS
  réalignée à la taille réelle de l'image source avant toute superposition
  — testé explicitement : une zone d'erreur au coin haut-gauche reste au
  coin haut-gauche après resize), `encode_heatmap_png`/`encode_mask_png`
  (PNG base64 directement affichables, pas de floats bruts à interpréter
  côté consommateur).
- [x] **`services/vision_anomaly_registry.py`** (nouveau) — un seul
  `ConvAutoEncoder`, entièrement convolutif (pas de bottleneck dense — évite
  par construction toute la classe de bugs de calcul dynamique de
  `flat_features` du legacy), `IMAGE_SIZE` fixe (128) imposé par le
  pipeline de données, jamais un resize interne au modèle façon
  `auto_resize` du legacy (racine du bug **#5**, incohérence de format).
  Volontairement pas de VAE/denoising/PatchCore/Siamese (legacy en propose
  4+) — PatchCore notamment impraticable en CPU (recherche de plus proche
  voisin sur tout le train à chaque inférence).
- [x] **`services/vision_anomaly_training.py`** (nouveau) —
  `train_and_evaluate_anomaly_vision` : entraînement sur `train/good/`
  UNIQUEMENT (reconstruction MSE, split train/val 85/15), pas de
  normalisation ImageNet (reconstruction comparée directement en espace
  [0,1] — mélanger normalisé/non-normalisé est précisément le bug **#11**,
  évité en n'introduisant jamais de normalisation). Évaluation sur `test/`
  **systématique** : carte d'erreur calculée pour CHAQUE image de test
  pendant l'évaluation standard, jamais une fonction annexe jamais appelée
  (correctif direct des bugs **#8/#16**). Seuil de détection **calibré** par
  J de Youden sur la courbe ROC de `test/` (labels good/défaut toujours
  disponibles, structure garantie par le sous-lot A) — remplace le
  percentile fixe arbitraire du legacy (correctif des bugs **#7/#12**,
  vérifié par test : deux datasets différents produisent deux seuils
  différents). Même garde-fou de temps interne que le sous-lot B
  (`max_training_seconds`). 12 exemples conservés (triés par score
  décroissant), heatmap ET masque déjà encodés et réalignés.
- [x] **`api/core/models.py`** — `VisionAnomalyJob`/`VisionAnomalyModel`/
  `VisionAnomalyExampleRecord` (table dédiée pour les exemples, même
  raisonnement que `AnomalyObservationRecord` : une ligne par exemple,
  jamais un JSON agrégé contenant des PNG base64).
- [x] **`api/core/storage.py`** — `vision_anomaly_model_file_path`.
- [x] **`services/job_quota.py`** — `VisionAnomalyJob` ajouté à
  `ALL_JOB_MODELS` : un seul endroit modifié (le refactor du sous-lot B a
  payé immédiatement — sans lui, il aurait fallu retoucher les 5 routers
  existants un par un).
- [x] **`workers/vision_anomaly_worker.py`** (nouveau) — rejette
  explicitement un dataset de structure `classification` (message
  diagnostiqué).
- [x] **`api/routers/vision_anomalies.py`** (nouveau) — `GET /models`,
  `POST/GET/DELETE /jobs`, `GET /jobs/{id}/result`, `GET /jobs/{id}/examples`.

**Vérifié** : 32 tests nouveaux (7 `test_vision_localization.py` — dont un
test dédié de non-régression pour les bugs #10/#17, 3
`test_vision_anomaly_registry.py`, 6 `test_vision_anomaly_training.py`
— entraînement réel non mocké sur mini dataset MVTec AD synthétique, 4
`test_vision_anomaly_worker.py`, 12 `test_vision_anomaly_api.py`), tous
verts en isolation. **Deux leçons d'exécution retenues pendant ce
sous-lot** : (1) un run de suite complète lancé en tâche de fond plus tôt
dans la session avait été oublié actif — a de nouveau corrompu des tests
par accès concurrent au même fichier SQLite (mêmes symptômes qu'au
sous-lot B : 401 au lieu de 404, email "déjà enregistré" sur un
`_fresh_database` pourtant censé repartir de zéro) — confirmés faux en
l'arrêtant et en rejouant isolément ; (2) un vrai bug de fixture de test
(dataset de test à 6 images `train/good`, sous le minimum d'entraînement de
10) a été trouvé et corrigé au passage, distinct de la corruption.

Frontend : aucun pour ce sous-lot (backend seul, même choix que le
sous-lot B).

## Lot 15, sous-lot D — Pilier Vision : Grad-CAM (livré, backend) — backend du chantier Vision terminé

Dernier sous-lot backend planifié du pilier Vision. Équivalent visuel de
SHAP local (`services/ml_explainability.py`) : le legacy mentionnait
Grad-CAM dans sa documentation historique mais ne l'a jamais branché nulle
part (aucune trace dans les 4 audits `docs/legacy/`) — première
implémentation réelle du dépôt.

- [x] **`services/vision_classification_registry.py`** — `gradcam_target_layer`
  ajouté à `ClassificationBackboneSpec` (dernière couche convolutive avant
  pooling/classifier — `model.layer4` pour ResNet18, `model.features` pour
  MobileNetV3-Small), déclaré explicitement par backbone, jamais deviné
  dynamiquement.
- [x] **`services/vision_classification_training.py`** — `build_eval_transform()`
  rendue publique (était privée) : Grad-CAM doit voir exactement la même
  normalisation que l'entraînement, jamais une transformation reconstruite
  indépendamment.
- [x] **`services/vision_gradcam.py`** (nouveau) — `explain_classification_prediction` :
  hooks forward/backward sur la couche cible, pondération par la moyenne
  spatiale du gradient (Grad-CAM original, Selvaraju et al. 2017),
  réutilise intégralement `services/vision_localization.py` (sous-lot C)
  pour le réalignement à la taille d'image originale et l'encodage PNG — un
  seul mécanisme de superposition heatmap/image dans tout le module vision.
  **Bug réel trouvé et corrigé en testant** : avec un backbone gelé (mode
  par défaut du sous-lot B), l'entrée ET tous les paramètres du backbone
  ont `requires_grad=False` — le graphe autograd n'est alors jamais
  construit jusqu'à la couche cible, le hook backward capte un gradient
  vide/incorrect (silencieux, juste un `UserWarning` PyTorch facile à
  manquer). Corrigé par `input_tensor.requires_grad_(True)` avant le
  forward pass — les poids gelés ne sont jamais mis à jour (pas
  d'`optimizer.step()` dans ce module), seul le graphe est reconnecté.
  Vérifié explicitement par un test dédié (`std() > 0` sur la heatmap
  produite — une heatmap dégénérée/plate aurait laissé passer le bug
  silencieusement).
- [x] **`api/routers/vision_classification.py`** — `POST /jobs/{id}/explain`
  (upload image + `target_label` optionnel) : endpoint **synchrone**, seule
  exception délibérée à la règle "jamais de calcul ML dans la requête HTTP"
  de ce router — même raisonnement que `POST /training/jobs/{id}/predict`
  (inférence + SHAP local), un seul forward+backward pass de l'ordre de la
  seconde, pas un entraînement.

**Vérifié** : 11 tests nouveaux (6 `test_vision_gradcam.py` — dont un test
dédié de non-régression pour le bug du backbone gelé, un modèle réel
entraîné une seule fois et réutilisé par tous les tests du fichier
`scope="module"` pour ne pas répéter l'entraînement à chaque test ; 5 tests
`/explain` ajoutés à `test_vision_classification_api.py`), tous verts en
isolation.

Frontend : aucun (backend seul, comme B et C).

**Backend du chantier Vision (Lot 15, sous-lots A→D) terminé** — 97 tests
au total pour le pilier (`pytest -k vision`, mesuré directement, pas
recalculé à la main), tous verts en isolation. Le dépôt compte désormais
478 tests. Détection d'objets/annotation assistée restent hors périmètre
(Lot 16+, non cadré).

## Lot 15 — Frontend (pages React) + corrections trouvées en test réel

Frontend construit pour les sous-lots A/B/D (`pages/VisionClassification.tsx`
— upload dataset, wizard transfer learning, résultats, Grad-CAM) et C
(`pages/VisionAnomalies.tsx` — wizard MVTec AD, résultats, exemples avec
heatmap/masque), composant partagé `components/vision/VisionDatasetPicker.tsx`
(upload ZIP + sélection, factorisé une seule fois pour les deux modules),
`components/vision/VisionImage.tsx` (affichage authentifié d'une image de
dataset via blob URL — `<img src>` ne peut pas porter de Bearer token).
Pilier `vision` passé à `status: "active"` dans `config/pillars.ts`.

**Bugs réels trouvés en testant l'app réellement (serveurs + navigateur, pas
seulement pytest) — leçon déjà tirée pour le pilier supervisé (SHAP
multiclasse), confirmée à nouveau ici :**

- **`vite.config.ts` n'avait pas `/vision` dans le proxy de dev** — même
  oubli que Lot 13/14 (déjà signalé en commentaire dans ce fichier), corrigé
  en l'ajoutant explicitement.
- **`GET /vision/datasets/{id}/image` manquait entièrement** — les exemples
  de prédiction (classification) et d'anomalie ne référencent qu'un
  `relative_path`, aucun endpoint ne servait l'image elle-même. Ajouté
  (`api/routers/vision_datasets.py`), protection zip-slip par
  `resolve()`/`parents` vérifiée par test.
- **Structure MVTec AD réelle non reconnue** — un téléchargement officiel
  est zippé PAR CATÉGORIE avec un dossier englobant
  (`bottle/train/good/...`) et un dossier `ground_truth/` (masques pixel),
  ni l'un ni l'autre géré par le détecteur de structure du sous-lot A (testé
  jusque-là avec des fixtures synthétiques sans dossier englobant). Corrigé
  dans `services/vision_datasets.py::_detect_structure` : dossier englobant
  optionnel détecté et retiré (même mécanisme pour la classification —
  sélectionner un dossier parent puis "Compresser" est le geste le plus
  probable d'un utilisateur), `ground_truth/` ignoré silencieusement (aucune
  métrique pixel/IoU calculée dans ce lot). Le dossier MVTec AD officiel
  COMPLET (15 catégories à la fois) reçoit maintenant un message dédié
  ("zippez une seule catégorie") plutôt que l'erreur générique de
  classification, trompeuse dans ce cas précis.
- **`relative_path` utilisait des antislashs sous Windows**
  (`str(Path.relative_to(...))` nu) — invalide dans un contrat d'API
  JSON/URL portable ; le déploiement cible (Docker/Linux) ne l'aurait jamais
  révélé. Corrigé (`.as_posix()` explicite) dans les deux moteurs
  (classification et anomalies visuelles), testé.

Tests supplémentaires pour ces correctifs (image dataset, wrapper MVTec AD,
wrapper classification, collection multi-catégories rejetée,
`relative_path` portable). **106 tests au total pour le pilier Vision**
(`pytest -k vision`, mesuré directement), **487 tests dans le dépôt.**

## Lot 16 — Durcissement du pilier Vision (en cours)

Le pilier Vision (Lot 15) a été livré par une session parallèle pendant que
cette session travaillait sur le pilier non supervisé tabulaire (Lot 13/14).
Un audit à froid — vérifié directement dans le code, pas seulement dans ce
fichier — a montré un backend réellement soigné mais une expérience
utilisateur en retrait par rapport au pilier tabulaire déjà livré. Lot
découpé en sous-lots indépendants, même principe que Lot 15 A→D.

### Lot 16A — Heatmaps colorées et superposées (livré)

Retour utilisateur direct en testant l'app réellement : la heatmap
Grad-CAM/anomalies visuelles était en niveaux de gris, jamais la carte
rouge/bleu standard (Selvaraju et al. 2017), et affichée à côté de l'image
source plutôt que superposée.

- [x] **`services/vision_localization.py`** — `_apply_colormap` (palette
  "jet", approximation par fonctions triangulaires, aucune dépendance
  supplémentaire) appliquée dans `encode_heatmap_png` (désormais un PNG
  couleur, plus jamais niveaux de gris). Nouvelle fonction
  `overlay_heatmap_on_image` (alpha-blend sur l'image source) — remplace
  l'affichage côte à côte pour les deux usages qui partagent ce module.
  `generate_binary_mask`/`resize_map_to_original` INCHANGÉES (continuent
  d'opérer sur la carte brute float32, avant toute colorisation — le
  calibrage de seuil déjà testé ne dépend jamais du rendu visuel).
- [x] **`services/vision_gradcam.py`** — branché sur `overlay_heatmap_on_image`
  (l'image source `PIL.Image` était déjà un paramètre de la fonction).
- [x] **`services/vision_anomaly_training.py`** — le `with Image.open(...)`
  qui ne lisait que `.size` avant fermeture du fichier lit maintenant
  `.convert("RGB")` (décode entièrement en mémoire, reste valide après
  fermeture) pour pouvoir superposer.
- [x] **Frontend** — `GradCamPanel` (`VisionClassification.tsx`) et
  `AnomalyExampleCard` (`VisionAnomalies.tsx`) affichent l'image superposée
  unique (plus deux images côte à côte) avec une légende explicite
  (dégradé bleu→rouge + texte "Bleu = faible influence · Rouge = zones qui
  ont le plus influencé...").

**Vérifié** : 15 tests (`test_vision_localization.py`, 4 nouveaux —
non-régression du masque binaire/réalignement, palette réellement colorée
pas R=G=B, superposition de bonne taille, rejet si carte non réalignée,
blend visible dans la zone chaude). Suite `pytest -k vision` complète
rejouée (110 tests, aucune casse). `tsc -b`/`npm run lint`/`vitest run`
(33/33)/`vite build` verts.

### Lot 16B — Traçabilité sur le Dashboard (livré)

`created_by` était déjà porté par les 4 types de job résumé
(`TrainingJobSummary`/`ClusteringJobSummary`/`DimensionalityJobSummary`/
`AnomalyJobSummary`) mais jamais repris dans `ActivityItem` du flux
"Dernière activité" — aucune traçabilité "qui a fait quoi" visible côté
Dashboard alors que la donnée existait déjà.

- [x] `frontend/src/pages/Dashboard.tsx` — `ActivityItem.createdBy`
  ajouté, propagé dans les 4 `.forEach`, affiché dans le sous-titre de
  chaque ligne d'activité. **Aucun changement backend.**

**Vérifié** : `tsc -b`/`npm run lint`/`vitest run` (33/33) verts.

### Lot 16C — Exploration complète des datasets Vision (livré)

Le rapport de qualité (images corrompues, doublons, sous-dimensionnées,
déséquilibre de classes >10x) est calculé intégralement depuis le Lot 15
sous-lot A (`VisionDatasetReport`) et déjà exposé en entier par l'API
(`VisionDatasetDetail.validation_report`) — le vrai manque était l'absence
de galerie d'images et l'exposition partielle côté UI (seuls les
`warnings` étaient affichés, jamais le détail des fichiers concernés).

- [x] **`api/routers/vision_datasets.py`** — nouvel endpoint `GET
  /vision/datasets/{id}/images?class_name=` : liste les fichiers d'une
  classe/bucket en parcourant `storage_dir` à la demande (pas de nouvelle
  table, les images sont déjà extraites sur disque à l'upload), plafonné à
  `MAX_GALLERY_IMAGES_PER_CLASS = 60` (`total` reste le compte réel).
  Même protection anti-traversée de répertoire que `GET .../image`.
- [x] **Frontend** — `components/vision/VisionDatasetExplorer.tsx`
  (nouveau, modale 2 onglets façon `EdaModal.tsx`) : "Aperçu" (galerie de
  miniatures par classe, `VisionImage.tsx` déjà existant réutilisé tel
  quel) et "Qualité" (rapport de validation complet — fichiers corrompus,
  doublons, sous-dimensionnés, pas seulement les avertissements résumés).
  Bouton "Explorer" ajouté dans `VisionDatasetPicker.tsx`, disponible dès
  qu'un dataset est sélectionné (upload ou choix dans la liste).

**Vérifié** : 6 tests nouveaux (`test_vision_datasets_api.py`, 17 au
total dans ce fichier) — liste correcte par classe, plafond respecté avec
`total` réel préservé, 404 classe inconnue, rejet de traversée de
répertoire, isolation multi-tenant. `tsc -b`/`npm run lint`/`vitest run`
(33/33)/`vite build` verts.

**Reste à faire** (voir le plan de session) : wizard par étapes + mode
expert complet (16D), pages "Mes données Vision"/Historique (16E),
catalogue élargi + indicateur "lent" + note d'évolution GPU (16F),
métriques binaire vs multiclasse (16G).

## Prochains lots (résumé — détail complet dans le diagnostic de migration et les échanges de cadrage)

| Lot | Contenu | Livrable testable |
| --- | --- | --- |
| 6-8 | Upload / entraînement / évaluation vision (détection d'anomalies) | Parité fonctionnelle côté vision |
| 9 | Registre de modèles unifié (versioning, export) | Remplace les 3 mécanismes de persistance de l'app historique |
| 10 | Durcissement SaaS (erreurs, audit, quotas) | Prêt pour un client pilote |

Ce fichier sera complété à chaque lot livré avec le détail réel (fichiers
créés, endpoints exposés, décisions techniques prises en cours de route) —
même format que le `workflow.md` de CIAM, sourcé fichier par fichier. Voir
aussi [`../recap.md`](../recap.md) pour une synthèse lisible de l'ensemble,
mise à jour au même rythme.
