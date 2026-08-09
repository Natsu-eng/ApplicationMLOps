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

## Prochains lots (résumé — détail complet dans le diagnostic de migration et les échanges de cadrage)

| Lot | Contenu | Livrable testable |
| --- | --- | --- |
| E | Sélection expert des modèles (exposer `subset`/activation individuelle en API + UI) | Un utilisateur avancé choisit d'activer SVM/KNN/linéaire/ExtraTrees/Naive Bayes |
| 6-8 | Upload / entraînement / évaluation vision (détection d'anomalies) | Parité fonctionnelle côté vision |
| 9 | Registre de modèles unifié (versioning, export) | Remplace les 3 mécanismes de persistance de l'app historique |
| 10 | Durcissement SaaS (erreurs, audit, quotas) | Prêt pour un client pilote |

Ce fichier sera complété à chaque lot livré avec le détail réel (fichiers
créés, endpoints exposés, décisions techniques prises en cours de route) —
même format que le `workflow.md` de CIAM, sourcé fichier par fichier. Voir
aussi [`../recap.md`](../recap.md) pour une synthèse lisible de l'ensemble,
mise à jour au même rythme.
