# AUDIT COMPLET — DataLab Pro

**Périmètre** : `E:\mlops\app-analyse` — backend FastAPI, frontend React/TS/Vite, moteurs ML supervisé / non supervisé / Vision, legacy Streamlit.
**Date** : 16 août 2026
**Méthode** : lecture du code réel (routers, services, workers, modèles ORM, pages et composants React, tests), pas de la documentation. Chaque affirmation de ce rapport est rattachée à un fichier et, quand c'est utile, à une ligne. Ce qui n'a pas été vérifié dans le code est signalé comme tel.
**Aucun fichier n'a été modifié.**

---

## A. Résumé exécutif

DataLab Pro n'est pas un prototype. C'est un produit réellement construit, avec un niveau d'ingénierie ML nettement au-dessus de ce que l'on trouve habituellement dans un projet de cette taille : anti-fuite par groupe jusque dans la validation croisée, préprocesseur refit dans chaque fold, sélection sur le score CV et jamais sur le test, intervalles conformes Mondrian, SHAP routé par famille de modèle, leaderboard persisté, 174+ tests backend qui entraînent de vrais modèles. La discipline méthodologique du moteur supervisé est le vrai actif du projet.

Le problème n'est pas la qualité du ML. Il est ailleurs, et il est structurel :

1. **Trois défauts méthodologiques réels subsistent, tous dans le pilier Vision** — un seuil d'anomalie calibré puis évalué sur le même jeu de test, des doublons d'images détectés mais conservés avant un split aléatoire, une augmentation figée en dur. Le pilier Vision a le backend d'un prototype soigné, pas la rigueur du pilier tabulaire.
2. **L'infrastructure ne supporte pas le mot « SaaS »**. Un seul worker RQ en `SimpleWorker` (pas de fork) traite les jobs de toutes les organisations, un à la fois ; le stockage est un `bind-mount` local ; il n'y a pas de migrations de schéma versionnées ; pas de pagination ; pas d'observabilité ; pas de CI. Ce n'est pas « à améliorer plus tard » : c'est ce qui décidera si la plateforme tient au deuxième client.
3. **La traçabilité s'arrête au modèle**. Le lignage réel est `Dataset → Job → Model`. Il n'y a **aucune persistance des prédictions**, aucun hash de contenu de dataset, aucune version de dataset, aucun rollback. Le « Model Registry » est un champ `stage` sur `MLModel` — utile, mais ce n'est pas un registre.
4. **L'aide à la décision existe avant l'entraînement, presque pas après.** Dix détecteurs de qualité de données côté amont (excellent), mais aucun verdict après coup : rien ne dit à l'utilisateur « ce modèle surapprend », « cette métrique est celle à regarder », « voici quoi faire ensuite ». C'est précisément la promesse produit énoncée, et c'est le plus gros écart entre l'intention et le code.
5. **Le dépôt mélange deux projets.** ~1,5 Mo de Python Streamlit mort (`src/`, `ui/`, `monitoring/`, `orchestrators/`, `helpers/`, `utils/`, `tests/`, `pytest.ini`, `requirements.txt` racine) cohabite avec le projet actif. Le legacy contient pourtant des capacités **supérieures** à la nouvelle plateforme sur la Vision (PatchCore, réseaux siamois, taxonomie de défauts, augmentation configurable, MLflow) — il faut décider explicitement ce qu'on récupère, puis supprimer le reste.

**Verdict court** : moteur ML de très bon niveau, produit à mi-chemin, infrastructure de démo. Dans cet ordre exact.

---

## B. État réel de DataLab

### B.1 Inventaire vérifié

| Zone | Contenu réel |
| --- | --- |
| `backend/api/` | 1 `main.py`, 7 modules `core/`, **9 routers** |
| `backend/services/` | **28 modules**, dont `ml_training.py` (1 268 lignes / 67 Ko) et `data_quality.py` (34 Ko) |
| `backend/workers/` | **6 workers RQ** + `run_worker.py` |
| `backend/tests/` | **42 fichiers pytest**, ~380 Ko (dont `test_ml_training.py`, 59 Ko) |
| Modèle de données | **22 tables** ORM dans un seul fichier `core/models.py` (49 Ko) |
| `frontend/src/` | **17 pages**, ~40 composants, 2 contextes, **1 hook**, 6 utilitaires (5 testés) |
| Legacy Streamlit | `src/` (55 `.py`, ~1,2 Mo), `ui/` (7 fichiers, 170 Ko), `monitoring/` (MLflow, 120 Ko), `orchestrators/`, `helpers/`, `utils/`, `notebooks/`, `pipeline_visio/`, `env/` |
| Documentation | `backend/workflow.md` **(147 Ko)**, `recap.md` (43 Ko), `AUDIT_ROADMAP.md` (35 Ko), `backend/ARCHITECTURE.md` (19 Ko), `AUDIT_PILIER2_ET_REFONTE_UX.md`, `docs/legacy/` (10 fichiers) |
| CI/CD | **Absent** — aucun `.github/`, aucun pipeline |

### B.2 Classification fonctionnelle (état réel, pas documenté)

**Socle / plateforme**

| Fonctionnalité | État | Preuve |
| --- | --- | --- |
| Auth JWT + bcrypt, multi-tenant par `organization_id` | ✅ | `core/security.py`, filtrage systématique dans les 9 routers |
| Hard-fail clé JWT par défaut en production | ✅ | `core/security.py:25-36` |
| Rate-limiting login (Redis, fenêtre glissante, échec ouvert) | ✅ | `core/rate_limit.py`, `routers/auth.py:174-203` |
| Journal d'audit (`AuditLog`) | 🟡 | Table + `log_action` OK, mais **10 actions couvertes, toutes des suppressions sauf 2** ; pas d'export, pas de purge |
| Gestion d'équipe | 🟡 | Lister + ajouter un membre. **Le owner saisit lui-même le mot de passe du membre** (`auth.py:283`) — pas d'invitation par e-mail |
| Désactiver / supprimer un membre | 🔴 | Champ `User.actif` présent, docstring d'`AuditLog` mentionne « désactivation de membre » — **aucun endpoint** |
| Réinitialisation de mot de passe / vérification e-mail | 🔴 | Absent |
| Révocation de token / `logout` réel | ❌ | `POST /auth/logout` (`auth.py:246`) ne fait rien ; le token reste valide 24 h |
| Quota de jobs concurrents par organisation | ✅ | `services/job_quota.py`, `ALL_JOB_MODELS` partagé par les 6 types de job |
| Watchdog de jobs orphelins | ✅ | `services/job_watchdog.py`, `progress_updated_at` |
| Migrations de schéma | ❌ | `_add_column_if_missing` maison, 11 colonnes listées à la main (`core/database.py:98-117`), sur 2 tables. Pas d'Alembic |
| Pagination des listes | 🔴 | Aucune, sur aucun endpoint (seul `/auth/team/audit-log` a une `limit`) |
| Observabilité (métriques, tracing, erreurs) | 🔴 | `logging.basicConfig` texte, aucun ID de corrélation, aucun Sentry/Prometheus/OTel |
| Stockage objet | 🔴 | Disque local uniquement (`core/storage.py`), bind-mount partagé backend/worker |

**ML supervisé**

| Fonctionnalité | État |
| --- | --- |
| Catalogue 9 modèles piloté par registre | ✅ `services/ml_registry.py` |
| Recherche Optuna TPE, CV, sélection sur CV | ✅ `ml_training.py:259-355` |
| Split/CV groupés anti-fuite + vérification explicite | ✅ `ml_preprocessing.py:89-107` |
| Préprocesseur refit dans chaque fold | ✅ `ml_training.py:321-332` |
| Suppression des doublons exacts avant split | ✅ `ml_preprocessing.py:64-70` |
| Métriques + IC bootstrap 95 % | ✅ `_bootstrap_ci` |
| CQR Mondrian (régression), préprocesseur CQR dédié | ✅ `_compute_cqr` |
| SHAP barres + beeswarm, permutation, calibration, courbe d'apprentissage, dégradation propre | ✅ |
| Leaderboard persisté (tous candidats, variance inter-folds) | ✅ `ModelCandidate` |
| Comparaison inter-jobs | ✅ `GET /training/jobs/compare` |
| Rééquilibrage de classes par pondération (jamais d'office) | ✅ |
| Garde-fous qualité de données (10 détecteurs) | ✅ `services/data_quality.py` |
| Suggestions d'ingénierie de variables approuvées par l'utilisateur | ✅ `services/feature_engineering.py` |
| Prédiction unitaire + explication locale (waterfall) | ✅ `POST /training/jobs/{id}/predict` |
| Promotion staging/production, export `.joblib` | ✅ |
| **Prédiction par lot / API de scoring externe** | 🔴 Absent |
| **Persistance des prédictions** | 🔴 Absente — aucune table |
| **Verdict/aide à la décision post-entraînement** | 🔴 Absent (voir §E.3) |
| **Détection de dérive / monitoring en production** | 🔴 Absent |

**ML non supervisé**

| Fonctionnalité | État |
| --- | --- |
| K-Means / DBSCAN / hiérarchique, multi-k, leaderboard | ✅ `clustering_training.py`, `ClusterCandidateRecord` |
| Profils de segments (z-scores vs population) | ✅ |
| Budget de bruit DBSCAN (`MAX_SELECTABLE_NOISE_RATIO = 0.5`) | ✅ |
| PCA / t-SNE / UMAP + `trustworthiness` + note de fidélité obligatoire | ✅ `dimensionality_training.py` |
| Isolation Forest + LOF **toujours ensemble**, score de consensus par rangs | ✅ `anomaly_training.py` |
| Explication par observation (déviations numériques, modalités rares) | ✅ |
| **Interprétation guidée du résultat** | ✅ `utils/clusterQuality.ts` — le seul pilier qui en a une |
| Échantillonnage au-delà de 5 000 (embedding) / 20 000 (anomalies) lignes | 🟡 Correct mais silencieux dans l'UI au-delà d'un simple drapeau |
| Prédiction du cluster d'une nouvelle observation | 🟠 Artefact persisté, aucun endpoint |

**Computer Vision**

| Fonctionnalité | État |
| --- | --- |
| Upload ZIP, détection stricte de structure (classification / MVTec AD), zip-slip + zip-bomb | ✅ `vision_datasets.py` |
| Validation : corrompues, sous-dimensionnées, doublons (SHA-256), déséquilibre >10× | ✅ (rapport) |
| Galerie d'exploration par classe | ✅ `VisionDatasetExplorer.tsx` (Lot 16C) |
| Transfer learning 2 backbones (MobileNetV3-S, ResNet18), gel/dégel | ✅ |
| Budget de temps interne + meilleure époque conservée + `time_capped` honnête | ✅ |
| Grad-CAM (endpoint synchrone), heatmaps colorées superposées | ✅ |
| Autoencodeur MVTec AD + masque binaire + réalignement à la taille d'origine | ✅ |
| **Doublons conservés puis split aléatoire** | ❌ **Fuite** (§G.1) |
| **Seuil calibré sur le test, métriques rapportées sur le même test** | ❌ **Biais** (§G.2) |
| Augmentation de données | 🟡 Figée en dur (`RandomHorizontalFlip`, `RandomRotation(10)`, `ColorJitter`) — non configurable, non recommandée, non expliquée |
| Pondération de classes en vision | 🔴 Absente (`CrossEntropyLoss()` nu) alors que le déséquilibre est détecté à l'upload |
| Early stopping / scheduler LR | 🔴 Absents |
| Split train/val/test paramétrable | 🔴 Figé 70/15/15 |
| Analyse résolution / format / EXIF / images quasi-dupliquées | 🔴 Absente |
| Wizard par étapes + mode expert Vision | 🔴 Absents (16D non fait) |
| ROC / AUC / seuil en classification binaire d'images | 🔴 Absents (16G non fait) |
| Détection d'objets, segmentation, annotation | 🔴 Hors périmètre assumé |
| CNN from scratch, PatchCore, siamois, taxonomie de défauts | ⚠️ Existent **uniquement** dans le legacy Streamlit |

**Frontend**

| Fonctionnalité | État |
| --- | --- |
| 17 pages, routage protégé, thème clair/sombre/système | ✅ |
| Wizard d'entraînement 5 étapes, retour arrière autorisé | ✅ `pages/Training.tsx` |
| Reprise du job actif après rafraîchissement (`sessionStorage`) | ✅ |
| Deep-linking `?job=` / `?explore=` / `?preview=` | ✅ |
| États loading / error / empty distingués | ✅ sur Dashboard, Datasets, Training ; 🟡 ailleurs |
| Confirmation d'action destructive (`useConfirmAction`) | ✅ |
| **Gestion globale du 401** | ❌ Absente — voir §D.4 |
| **Notifications / toasts** | 🔴 Absents (messages en ligne uniquement) |
| **Tests de composants / E2E** | 🔴 Absents (5 tests d'utilitaires purs) |
| Page morte | ⚠️ `pages/ComingSoon.tsx` (4,2 Ko) n'est importée nulle part |

---

## C. Architecture backend

### C.1 Ce qui est bien conçu

**Le découpage router → service → worker est rigoureux et tenu.** Aucun calcul ML n'a lieu dans une requête HTTP pour un job ; les deux exceptions (prédiction unitaire, Grad-CAM) sont explicitement justifiées et légitimes. Les moteurs (`services/*_training.py`) sont des fonctions pures testables sans HTTP ni base.

**La décision de ne jamais étendre `ml_training.py`** pour le clustering / la réduction de dimension / les anomalies / la vision, mais de créer des modules et des tables dédiés, est le bon choix. Une branche `task_type == "clustering"` de plus dans `_make_cv`/`LabelEncoder`/CQR aurait cassé silencieusement le supervisé. Cette discipline est visible partout.

**Les commentaires de code documentent le *pourquoi*, y compris les bugs réels trouvés en usage.** C'est rare et précieux (`_compute_permutation_importance` et le `Pool` CatBoost en lecture seule, `.as_posix()` sur Windows, la réutilisation d'id de dataset SQLite). Ne perdez pas cette pratique.

**L'isolation multi-tenant est réellement appliquée**, pas seulement déclarée : chaque `_get_org_*` filtre par `organization_id`, et le chemin disque le répète en défense en profondeur.

### C.2 Problèmes structurels

**C.2.1 — Le worker est le goulot d'étranglement absolu.**
`workers/run_worker.py` utilise `SimpleWorker` : les jobs s'exécutent **dans le process du worker**, sans fork. Combiné à `docker-compose.yml` qui ne déclare **qu'un seul conteneur `worker`**, cela donne :

- un entraînement à la fois pour **toutes** les organisations confondues ;
- un `MemoryError` ou un segfault LightGBM/torch **tue le worker entier**, pas seulement le job (le `restart: unless-stopped` relance, mais le job reste `running` jusqu'à ce que le watchdog le récupère 40 min plus tard) ;
- `TimerDeathPenalty` est un timeout par thread : il **ne peut pas interrompre** du code natif bloquant (boucle C de LightGBM, forward torch). Le timeout de 1 800 s est donc théorique sur les cas qui en auraient le plus besoin.

Le `max_concurrent_jobs_per_org = 3` est un garde-fou d'équité, pas de capacité : trois organisations à 3 jobs = 9 jobs pour 1 slot d'exécution.

**C.2.2 — Pas de migrations versionnées.**
`_add_column_if_missing` (`core/database.py:56-70`) est astucieux et idempotent, mais c'est un cul-de-sac : impossible de renommer, supprimer, backfiller, ou revenir en arrière. Les 11 appels sont listés à la main dans `init_db()` et ne couvrent que `ml_models` (9) et `training_jobs` (2) — les 20 autres tables n'ont **aucun** chemin d'évolution. Toute base créée avant l'ajout d'une colonne à `ClusterModel` ou `VisionAnomalyModel` cassera silencieusement.

**C.2.3 — Relecture intégrale du dataset à chaque requête.**
`read_dataframe(...)` est appelé sans cache dans : `/preview`, `/eda`, `/histogram`, `/quality-check`, `/feature-engineering-suggestions`, `/feature-by-target`, **et** `POST /training/jobs` (uniquement pour détecter le type de tâche). Sur un CSV de 200 Mo, chaque clic dans l'assistant reparse le fichier entier. La docstring de `/eda` affirme que « pas besoin de mise en cache pour ce volume d'usage » — c'est vrai pour une démo, faux pour un SaaS.

**C.2.4 — Aucune pagination, requêtes N+1.**
`GET /training/jobs` retourne **tous** les jobs de l'organisation. `_to_summary` accède à `job.dataset`, `job.created_by`, `job.model` sans `joinedload` → 3 requêtes par job, plus un `json.loads(metrics_json)` par job. Idem pour les 5 autres types de job. Le Dashboard appelle **8 endpoints de liste complets** au montage (`Dashboard.tsx:194-200`).

**C.2.5 — Upload synchrone entièrement en mémoire.**
`await file.read()` charge 200 Mo (tabulaire) ou 500 Mo (ZIP vision) en RAM dans le process API, avant décompression (plafonnée à ×4, soit 2 Go décompressés potentiels sur `_validate_and_copy_images`, qui ouvre en plus une **seconde** `ZipFile` sur les mêmes octets). Pas de reprise, pas de multipart, pas de progression.

**C.2.6 — `routers/training.py` : 913 lignes.**
Recommandé comme à scinder par l'audit interne (`AUDIT_ROADMAP.md`, section G) et jamais fait. Le fichier mélange schémas Pydantic, catalogue de modèles, jobs, leaderboard, registre, comparaison, prédiction et export. Même remarque pour `core/models.py` (22 tables, 49 Ko, un seul module).

**C.2.7 — Sécurité, points ouverts.**
- CORS : `allow_credentials=True` avec `allow_methods=["*"]`/`allow_headers=["*"]` — acceptable puisque les origines sont explicites, mais à resserrer.
- Aucune limite de taille sur les corps JSON (`/predict` accepte un dict arbitraire).
- Aucun rate-limiting hors `/auth/login` (ni sur `/register`, ni sur les uploads, ni sur `/explain` qui charge un modèle torch à chaque appel).
- `torch.load(..., weights_only=False)` (`vision_classification.py:315`) : le commentaire justifie correctement (fichier écrit par notre worker), mais c'est une bombe à retardement le jour où un artefact viendra d'un import utilisateur. À contraindre par un `weights_only=True` + métadonnées séparées.
- `load_bundle` et `torch.load` **sans cache** : chaque prédiction / chaque Grad-CAM relit l'artefact du disque.

### C.3 Capacité d'évolution vers le modèle SaaS cible

| Brique cible | Prête ? | Commentaire |
| --- | --- | --- |
| Utilisateurs | ✅ | Table `User`, 2 rôles |
| Organisations | ✅ | `Organization`, isolation réelle |
| **Projets / Workspaces** | 🔴 | Absents. Pool plat par organisation. Ajout = `project_id` sur 8 tables + migration → **impossible sans Alembic** |
| Datasets | 🟡 | Pas de version, pas de hash de contenu, pas de schéma figé |
| Expériences / Runs | 🟡 | `TrainingJob` fait office de run, mais pas de notion d'« expérience » regroupant des runs |
| Modèles / Versions | 🟡 | 1 job = 1 modèle ; `stage` mais pas de numéro de version, pas de rollback |
| Jobs | ✅ | 6 types, statut, progression, watchdog, quota |
| Permissions | 🔴 | 2 rôles globaux, rien par ressource |
| Audit logs | 🟡 | 10 actions (8 suppressions), pas d'export, pas de rétention |
| Quotas | 🟡 | Jobs concurrents uniquement ; pas de stockage, pas de calcul, pas de plan |
| Stockage objet | 🔴 | Disque local — verrou mono-nœud |
| Monitoring | 🔴 | Rien |

**Conclusion** : l'architecture *peut* évoluer, mais deux verrous doivent sauter d'abord — **les migrations** et **le stockage objet**. Tant qu'ils tiennent, chaque évolution du modèle de données est un risque de casse silencieuse et chaque montée en charge est bloquée à une machine.

---

## D. Architecture frontend

### D.1 Ce qui est bien fait

- **React 19 + Vite 6 + TS 5.8 + Tailwind 4 + Recharts 3** : stack moderne, à jour, cohérente. Pas d'axios (fetch natif), pas de dépendance superflue.
- **`api/client.ts` typé de bout en bout**, un seul point d'accès, `ApiError` qui porte le code métier du backend. C'est propre.
- **Le wizard d'entraînement** (`Training.tsx`) est la meilleure page du produit : 5 étapes, retour arrière autorisé (`maxReachedStep`), récapitulatif avant lancement, reprise après rafraîchissement, phases `configure/progress/results/failed` explicites.
- **Distinction erreur vs vide** réellement implémentée (`datasetsError` ≠ `datasets.length === 0`).
- **Deep-linking par URL** sur les modales de résultat.
- **Le langage est celui de l'utilisateur, pas du data scientist** : « Prédire une valeur ou une catégorie », « Colonne de regroupement », « Structure plutôt faible ». C'est un différenciateur produit réel.

### D.2 Absence de gestion d'état et de cache

Aucune bibliothèque de données (pas de TanStack Query, SWR, Zustand, Redux). Conséquences mesurables dans le code :

- **Chaque page refait ses appels au montage.** Naviguer Dashboard → Datasets → Dashboard relance 8 requêtes.
- **Polling non coordonné** : `setInterval(3000)` dupliqué dans `Training.tsx`, `VisionClassification.tsx`, `VisionAnomalies.tsx`, `Clustering.tsx`, `AnomalyDetection.tsx`, `DimensionalityReduction.tsx`. Six implémentations du même mécanisme.
- **Le polling ne s'arrête pas quand l'onglet est masqué** (pas de `visibilitychange`), et redémarre à chaque changement de `activeJob` (dépendance `[phase, activeJob]` → l'intervalle est recréé à chaque tick réussi).
- **Pas de cache**, donc `VisionImage` refait un `fetch` + `createObjectURL` pour chacune des 60 miniatures d'une galerie, à chaque montage.

### D.3 Formulaires

Aucune bibliothèque de formulaire, aucune validation déclarative. `TrainingForm` porte **14 `useState`** et une validation ad hoc (`step1Valid`). Le mode expert en ajoute 5. C'est encore lisible, mais c'est exactement le point où un formulaire supplémentaire fait basculer le fichier dans l'ingérable — et `ModelResultModal.tsx` (32 Ko) et `EdaModal.tsx` (26,6 Ko) montrent où ça mène.

### D.4 Le trou de sécurité/UX du 401

Le JWT expire à 24 h. Il n'y a **aucun refresh token** et **aucune interception du 401** dans `request()` (`client.ts:58-71`). `clearToken()` n'est appelé que depuis `AuthContext` (échec explicite du `/auth/me` au démarrage) et depuis `logout()`.

Conséquence concrète : un utilisateur dont le token expire en cours de session voit ses actions échouer une par une avec des messages d'erreur métier trompeurs, sans jamais être renvoyé vers l'écran de connexion. C'est un bug de première catégorie en usage réel, et il est invisible en développement.

Deuxième point : le token est en `localStorage` — accessible à tout script injecté. Un cookie `httpOnly`/`SameSite=Strict` serait plus sûr, au prix d'un CSRF token.

### D.5 Analyse page par page

| Page | Constat |
| --- | --- |
| **Orientation** (`/`) | 3 cartes de piliers avec exemples concrets. Bonne idée. Mais c'est un écran de choix *à chaque visite*, pas un onboarding : aucune mémoire du dernier pilier utilisé, aucun état « vous n'avez encore rien fait, commencez ici ». |
| **Dashboard** | Riche et honnête (activité des 6 types de job, qui a lancé quoi). Mais : 8 appels au montage, aucune pagination, aucun filtre, aucune recherche, aucun tri. Devient inutilisable au-delà de ~50 jobs. |
| **Datasets** | Upload, aperçu, EDA (modale de 26 Ko), suppression confirmée. Manque : versionner un dataset, renommer, taguer, voir quels modèles en dépendent avant de le supprimer (la suppression cascade sur les jobs **et** les modèles — sans avertissement chiffré). |
| **Training** | La référence du produit. Manque : estimation de durée avant lancement, possibilité d'annuler proprement un job en cours (le `DELETE` supprime, il n'annule pas), et un verdict à l'arrivée. |
| **Résultats** (`ModelResultModal`, 32 Ko) | Contenu remarquable : métriques + IC, matrice de confusion, ROC/PR, calibration, courbe d'apprentissage, SHAP barres + beeswarm, permutation, leaderboard, CQR, prédiction, explication locale. **Mais aucune hiérarchie de lecture** : tout est au même niveau. Un débutant ne sait pas par où commencer, et rien ne lui dit ce que ces courbes veulent dire *pour son cas*. |
| **Historiques** (3 pages distinctes : supervisé, non supervisé, vision) | Fragmentation. Trois écrans qui font la même chose sur trois entités différentes. Aucun filtre, aucune recherche, aucune pagination. |
| **Vision (Datasets / Classification / Anomalies / Historique)** | Les 4 pages existent. Mais la page Classification est un **formulaire à plat** (backbone, époques, batch, LR, dropout, gel) là où le tabulaire a un wizard guidé — l'asymétrie est frappante, et elle expose des hyperparamètres bruts à un utilisateur qu'on prétend ne pas être expert. |
| **Profile** | Profil + mot de passe + équipe + journal d'audit. Correct. |
| **ComingSoon.tsx** | Code mort. |

### D.6 Design system

`index.css` (13 Ko) définit des tokens sémantiques OKLCH (`--background`, `--foreground`, `--primary`, `--sidebar-*`, `--success`, `--warning`, `--destructive`) avec support `data-theme` clair/sombre/système, et `theme/charts.ts` centralise les couleurs de graphiques. Le contraste a été vérifié mathématiquement sur `text-warning`/`text-success` (relevé dans `AUDIT_ROADMAP.md` H15).

**C'est une vraie base**, plus solide que ce que l'audit interne laissait entendre. Ce qui manque est en dessous : voir §J.

### D.7 Accessibilité

Positif : `Modal.tsx` a rôle, focus trap et Échap ; les champs du wizard ont des `label`/`htmlFor` ; les boutons icône ont `aria-label`.
Manquant : pas de `skip to content`, pas de gestion du focus au changement de route, `role="status"`/`aria-live` absent sur les zones de progression et d'erreur (un lecteur d'écran n'apprend jamais que l'entraînement est terminé), contraste non vérifié systématiquement, tableaux sans `scope`/`caption`.

---

## E. ML supervisé

### E.1 Le pipeline réel

```
Upload CSV/XLSX/Parquet/JSON  →  extract_schema (nom + dtype)
   ↓  (à la demande, à chaque requête, sans cache)
EDA : stats, corrélations num. + catégorielles, manquants, outliers, paires corrélées
Quality-check : 10 détecteurs (fuite cible, déséquilibre, cardinalité, constantes,
                petit dataset, manquants élevés, colinéarité, colonnes dupliquées,
                numérique mal typé, transparence colonne de groupe)
Suggestions FE : datetime_decompose, ratio, numeric_coerce, frequency_encoding, imputation
   ↓  approbation explicite de l'utilisateur
POST /training/jobs → validation serveur des colonnes → détection du type de tâche → RQ
   ↓  worker
FE amont déterministe → remove_exact_duplicates → split (groupé si demandé, vérifié)
   ↓
Pour chaque modèle du catalogue : Optuna(TPE) × cross_validate(Pipeline(preproc, model))
   ↓  sélection sur le score CV (ROC-AUC pondérée / R²)
Fit final sur tout le train → métriques + IC bootstrap → évaluation (CM, ROC/PR ou résidus)
   ↓
SHAP (routé par famille) → permutation → calibration → learning curve → CQR (régression)
   ↓
joblib bundle + MLModel + N × ModelCandidate  (une seule transaction)
```

### E.2 Ce qui est réellement solide

**L'anti-fuite est traité sérieusement, à cinq endroits distincts** : split groupé avec vérification explicite du chevauchement (`DataLeakageError`), `GroupKFold` en CV, `GroupShuffleSplit` pour le split interne du CQR, préprocesseur cloné et refit dans chaque fold via `Pipeline`, et `RareCategoryFrequencyEncoder` qui n'apprend ses fréquences qu'au `fit`. C'est plus rigoureux que beaucoup de plateformes commerciales.

**La sélection ne se fait jamais sur le test.** Le score test est reporté, pas utilisé. `_headline_metric` affiche `cv_score` et non l'accuracy — la correction documentée (accuracy trompeuse sur dataset déséquilibré) est réelle dans le code.

**`_classification_selection_score` gère le cas dégradé proprement** : `predict_proba` → `decision_function` → accuracy, avec repli sur `ValueError` quand une classe manque dans un fold. Comparer 9 modèles hétérogènes sur la même échelle est un problème réel, et il est résolu.

**Le CQR Mondrian par strates** est au-dessus du split conformal simple, et le préprocesseur CQR est correctement distinct de celui du modèle principal — une erreur classique évitée.

**Chaque diagnostic porte son statut `ok`/`degraded` + message français**, et aucun ne peut faire échouer l'entraînement. C'est la bonne posture.

### E.3 Le manque central : aucune aide à la décision après l'entraînement

C'est le plus gros écart entre l'objectif énoncé et le code.

Une recherche exhaustive sur `frontend/src/` des termes `surapprentissage|overfit|recommand|verdict|conseil|prochaine étape|fiabilité` ne renvoie **qu'une seule occurrence**, dans un libellé de l'étape 3 du wizard. Le pilier non supervisé, lui, a `utils/clusterQuality.ts` : `assessSilhouetteQuality()` (avec les précautions d'usage correctes) et `buildRecommendationExplanation()` (qui recoupe le gagnant avec deux autres métriques). **Le supervisé n'a pas d'équivalent.**

Toutes les données nécessaires sont pourtant déjà calculées et persistées :

| Question de l'utilisateur | Donnée disponible | Exploitée ? |
| --- | --- | --- |
| Mon modèle surapprend-il ? | `delta_r2`, `accuracy` train/test, `learning_curve_json` (écart train/val) | 🔴 Graphes affichés, aucun verdict |
| Mon modèle est-il fiable ? | `r2_bootstrap` / `accuracy_bootstrap` (IC 95 %), `fold_scores` (variance inter-folds) | 🔴 Affiché brut |
| Quelle métrique regarder ? | `task_type`, déséquilibre de classes détecté en amont | 🔴 Toutes affichées au même niveau |
| Le gagnant est-il vraiment meilleur ? | `ModelCandidate.selection_score` + `fold_scores_json` de chaque candidat | 🔴 Classement affiché, écart non qualifié (un écart de 0,003 avec un écart-type de 0,04 = égalité) |
| Mes probabilités sont-elles honnêtes ? | `calibration_json` | 🔴 Courbe affichée, écart à la diagonale non chiffré |
| Plus de données aideraient-elles ? | `learning_curve_json` (plateau ou non) | 🔴 Non interprété |
| Mes intervalles tiennent-ils ? | `empirical_coverage` vs `target_coverage` | 🔴 Deux nombres, pas de « couverture conforme / insuffisante » |
| Et maintenant ? | — | 🔴 Rien |

**Ce n'est pas un chantier de recherche. C'est une couche de règles déterministes sur des nombres déjà en base**, du même type que `clusterQuality.ts` — quelques centaines de lignes, testables, sans risque. C'est le meilleur rapport impact/effort de tout ce rapport.

### E.4 Autres points de vigilance ML supervisé

- **`detect_task_type` force la lecture complète du dataset dans la requête HTTP** de création de job (`training.py:482`), uniquement pour inspecter une colonne. Le schéma (`columns_json`) et un échantillon suffiraient.
- **Le rééquilibrage se limite à `sample_weight`.** Décision correctement documentée et défendable (SMOTE est fuyant s'il est mal placé), mais il faudra un jour du rééchantillonnage **dans les folds** pour les cas très déséquilibrés.
- **`class_rebalancing` n'est pas rejoué à l'inférence** — c'est correct (il n'affecte que la perte), et c'est bien expliqué. Bon point.
- **Le seuil de décision en classification binaire n'est jamais ajustable** : `predict` utilise 0,5 implicitement, alors que la courbe PR et la calibration sont calculées. Sur un problème déséquilibré c'est le levier le plus utile, et il est absent.
- **Aucune métrique métier / matrice de coût.** Un bureau d'études se moque du F1 ; il veut « combien coûte un faux négatif ». Rien ne permet de l'exprimer.
- **`_bootstrap_ci` fait 500 rééchantillonnages × 2 métriques** en régression, en pur Python. Négligeable sur 10 k lignes, coûteux sur 1 M.
- **`progress_cb` commit en base à chaque essai Optuna** : 4 modèles × 20 essais = 80+ `COMMIT` par job. Fonctionnel, mais bruyant sur Postgres partagé.

---

## F. ML non supervisé

### F.1 Points forts réels

- **Isolation Forest et LOF tournent systématiquement ensemble** avec un score de consensus par **rangs percentiles** (les scores bruts ne sont pas comparables) et un drapeau `agreement`. Refuser d'élire un gagnant sans vérité terrain est intellectuellement honnête et rare.
- **Les explications d'anomalie sont statistiques, jamais générées** : z-scores des variables déviantes, modalités catégorielles rares. Aucun texte inventé.
- **PCA est toujours calculée en plus de la méthode choisie**, avec variance expliquée et loadings — l'utilisateur garde une lecture interprétable même quand il demande t-SNE.
- **`trustworthiness` + note obligatoire** sur la non-fidélité des distances globales en t-SNE/UMAP. C'est le piège classique de ces méthodes, et il est traité.
- **Budget de bruit DBSCAN** (`MAX_SELECTABLE_NOISE_RATIO = 0.5`) : une configuration qui classe la moitié des points en bruit ne peut pas gagner.
- **`clusterQuality.ts`** est le modèle à généraliser aux autres piliers.

### F.2 Limites

| Point | Constat |
| --- | --- |
| Choix des variables | L'utilisateur choisit les colonnes à la main. Aucune suggestion, aucun avertissement sur les échelles hétérogènes ou les colonnes quasi-constantes (le `quality-check` existe mais n'est pas branché sur le clustering) |
| Traitement des catégorielles | Non explicité dans l'UI — l'utilisateur ne sait pas si sa colonne texte est one-hotée, ignorée, ou ce que ça implique sur la distance |
| Échantillonnage | 5 000 lignes (embedding) / 20 000 (anomalies). Le drapeau `sampled` existe, mais l'UI ne dit pas *ce que ça change* pour l'interprétation |
| Choix automatique de `k` | Balayage sur silhouette : correct. Mais pas de méthode du coude, pas de gap statistic, pas de stabilité par ré-échantillonnage — le `k` retenu peut être instable sans que rien ne le signale |
| Réutilisation | L'artefact du clustering est persisté mais **aucun endpoint ne permet d'assigner un nouveau point** à un cluster. Le résultat est un rapport, pas un modèle |
| Contamination | `IsolationForest`/`LOF` : le taux de contamination n'est pas exposé à l'utilisateur alors qu'il détermine directement le nombre d'anomalies retournées |
| Cas limites | Dataset entièrement catégoriel, colonnes toutes constantes, moins de 3 lignes : gérés côté dimensionality (`n_samples_used < 3`), non vérifiés ailleurs |

---

## G. Computer Vision

C'est le pilier le plus récent et le plus fragile. Le backend est propre ; la méthodologie et l'UX ne sont pas au niveau du tabulaire.

### G.1 🔴 Fuite de données par doublons d'images

`services/vision_datasets.py` calcule un SHA-256 par image, détecte les doublons exacts, les compte… et **les conserve** : *« conservées, à revoir manuellement »* (ligne 339). Elles sont toutes copiées sur disque (ligne 278).

Puis `vision_classification_training.py:133-138` fait un `train_test_split` stratifié **aléatoire** sur l'ensemble des images. Deux copies bit-à-bit de la même image peuvent donc se retrouver, l'une en entraînement, l'autre en test. Le modèle est alors évalué sur des images qu'il a mémorisées.

**Ce que ça rend incohérent** : le pilier tabulaire supprime les doublons exacts avant le split (`remove_exact_duplicates`, `ml_preprocessing.py:64`) et l'expose dans la fiche modèle (`duplicates_removed`). Le pilier vision détecte exactement le même problème et ne fait rien. Il n'y a aucune raison méthodologique à cette asymétrie.

**Impact** : métriques de test surestimées, d'autant plus que le dataset est petit — c'est-à-dire précisément le cas d'usage visé (contrôle qualité, petits jeux industriels).

### G.2 🔴 Seuil d'anomalie calibré et évalué sur le même jeu de test

`services/vision_anomaly_training.py` :

```
235:  fpr, tpr, roc_thresholds = roc_curve(all_true_labels, all_scores)   # scores du TEST
236:  youden_j = tpr - fpr
237:  threshold = float(roc_thresholds[int(np.argmax(youden_j))])          # seuil choisi SUR le test
240:  predicted_labels = [1 if s > threshold else 0 for s in all_scores]   # même test
241-244: accuracy / precision / recall / F1                                # rapportés sur le même test
```

Le J de Youden est un vrai progrès sur le percentile arbitraire du train qu'il remplace (bugs #7/#12 du legacy). Mais choisir le seuil optimal *sur* le jeu d'évaluation puis rapporter les métriques ponctuelles sur ce même jeu produit un **biais optimiste systématique**. Le `roc_auc` reste valide (indépendant du seuil) ; `test_accuracy`, `test_precision`, `test_recall`, `test_f1` ne le sont pas.

**Correction** : découper `test/` en calibration/évaluation (stratifié par catégorie de défaut, en conservant `good` des deux côtés), ou calibrer sur une part de `train/good` + un sous-ensemble de défauts réservé. Et, tant que ce n'est pas fait, l'écrire noir sur blanc dans l'UI.

### G.3 🟠 Le workflow guidé demandé n'existe pas

Sur les 27 étapes du workflow Vision idéal que vous décrivez, voici l'état réel :

| # | Étape | État |
| --- | --- | --- |
| 1-3 | Import ZIP, vérification de structure, détection des classes | ✅ strict et bien fait |
| 4 | Équilibre des classes | 🟡 détecté (ratio > 10×), affiché en avertissement, **jamais actionnable** |
| 5 | Visualisation d'échantillons | ✅ galerie par classe (Lot 16C) |
| 6 | Résolution des images | 🔴 seul un minimum de 20 px est vérifié ; aucune distribution de résolutions |
| 7 | Formats | 🟡 filtre d'extensions ; aucun rapport (RGB/RGBA/niveaux de gris, EXIF, profondeur) |
| 8 | Doublons | 🟡 détectés, **conservés** → §G.1 |
| 9 | Images corrompues | ✅ exclues et reportées |
| 10 | Split train/val/test | 🟡 fait, **figé à 70/15/15**, non paramétrable, non affiché avant lancement |
| 11 | Data augmentation | 🟡 **figée en dur**, non configurable, non expliquée, non recommandée |
| 12 | Normalisation | ✅ ImageNet en classification, volontairement absente en autoencodeur (correct) |
| 13-14 | Choix du modèle / transfer learning | 🟡 2 backbones, gel par défaut. Aucune aide au choix |
| 15 | Hyperparamètres | ⚠️ **exposés bruts** (LR, dropout, batch, époques, `unfreeze_after_epoch`) sans garde-fou ni explication |
| 16 | Entraînement | ✅ + budget de temps + meilleure époque conservée |
| 17 | Monitoring | 🟡 polling 3 s, barre de progression, `Époque i/N`. Pas de courbe live |
| 18-20 | Évaluation, matrice de confusion, P/R/F1 | ✅ macro |
| 21 | Courbes d'apprentissage | ✅ train/val loss + accuracy par époque |
| 22-23 | Prédictions et erreurs du modèle | ✅ **très bien** : erreurs priorisées sur les succès (`incorrect[:12] + correct[:12]`) |
| 24 | Grad-CAM | ✅ superposé et coloré (16A) |
| 25 | Sauvegarde | ✅ `.pt` isolé par organisation |
| 26 | Versioning | 🔴 aucun (pas de `stage`, pas de registre, pas d'export) |
| 27 | Réutilisation | 🟡 Grad-CAM sur une image uploadée ; **pas de prédiction simple**, pas de scoring par lot |

**Les questions de décision que vous listez** (faut-il augmenter ? laquelle ? faut-il équilibrer ? CNN ou transfer learning ? quel backbone ? quand arrêter ? y a-t-il surapprentissage ? quelle métrique ?) : **aucune n'a de réponse dans le produit**. Toutes sont calculables à partir de ce qui est déjà en base.

### G.4 Autres manques Vision

- **Pas de pondération de classes** dans la perte alors que le déséquilibre est détecté à l'upload — le déséquilibre est signalé puis ignoré.
- **Pas d'early stopping** : seul le meilleur `val_loss` est conservé (bon), mais l'entraînement va au bout des époques demandées.
- **Pas de scheduler de learning rate.**
- **Métriques macro uniquement** : pas de per-classe, pas de ROC/AUC/seuil en binaire (16G identifié, non fait).
- **`MAX_EXAMPLES_PER_KIND = 12`** : sur un dataset à 20 classes, 12 erreurs ne sont pas représentatives.
- **Pas de registre / promotion / export** pour les modèles vision, contrairement au tabulaire.
- **Le rechargement `torch.load` à chaque `/explain`** sans cache.

### G.5 Ce que contient le legacy et que la nouvelle plateforme n'a pas

Le portage a été volontairement minimaliste (2 backbones au lieu de 17, justifié par l'absence de GPU). Mais le legacy contient des capacités qui n'ont pas d'équivalent :

| Module legacy | Taille | Ce qu'il apporte | Récupérable ? |
| --- | --- | --- | --- |
| `src/models/computer_vision/anomaly_detection/patch_core.py` | 14 Ko | **PatchCore** — état de l'art MVTec AD, très supérieur à un autoencodeur convolutif | ⭐ Oui, à prioriser |
| `src/data/image_augmentation.py` | 16 Ko | Augmentation **configurable** par stratégie | ⭐ Oui, répond directement à §G.3 #11 |
| `src/explorations/image_exploration_plots.py` | 43 Ko | EDA d'images (résolutions, formats, distributions) | ⭐ Oui, répond à §G.3 #6-7 |
| `src/evaluation/computer_vision_metrics.py` | 37 Ko | Métriques CV étendues (per-classe, seuils, top-k) | Oui, en partie |
| `src/models/computer_vision/anomaly_detection/anomaly_type_classifier.py` + `config/anomaly_taxonomy.py` | 17 Ko | **Typologie de défauts** — valeur métier forte en contrôle qualité | À évaluer |
| `src/models/computer_vision/cross_validator.py` | 4 Ko | Validation croisée sur images | Oui |
| `src/models/computer_vision/classification/cnn_models.py` | 11 Ko | CNN from scratch | Faible priorité (CPU) |
| `src/models/computer_vision/anomaly_detection/siamese_networks.py` | 4 Ko | Few-shot / similarité | À évaluer |
| `monitoring/mlflow_collector.py` + `mlflow_vision_tracker.py` | 39 Ko | **Intégration MLflow** — tracking d'expériences | ⭐ Décision d'architecture (voir §O) |

**Attention** : ces modules sont ⚠️ *legacy Streamlit*. Aucun n'est utilisé par la plateforme actuelle. Aucun ne doit être considéré comme fonctionnel avant portage et test. Les 9 bugs critiques documentés dans `docs/legacy/ANALYSE_COMPLETE_COMPUTER_VISION.md` sont là pour le rappeler.

---

## H. Stockage / modèles / versioning

### H.1 Où va quoi, réellement

| Objet | Emplacement | Problème |
| --- | --- | --- |
| Datasets tabulaires | `storage/datasets/{org_id}/{id}{ext}` | Fichier unique, aucune version, aucun hash |
| Datasets images | `storage/datasets_vision/{org_id}/{id}/` | Dossier d'images ; `vision_dataset_dir()` fait un `rmtree` avant recréation |
| Modèles supervisés | `storage/models/{org_id}/{job_id}.joblib` | Bundle pickle (modèle + préproc + CQR + fond SHAP) |
| Clustering / dimension / anomalies | `storage/models/{org_id}/{type}/{job_id}.joblib` | Idem |
| Modèles vision | `storage/models/{org_id}/vision_*/{job_id}.pt` | `state_dict` torch + métadonnées |
| Graphiques, courbes, matrices | **Base de données**, colonnes `TEXT` JSON | Voir ci-dessous |
| Heatmaps / masques d'anomalie | **Base de données**, PNG **base64** dans `TEXT` | Voir ci-dessous |
| Rapports | N'existent pas | Aucun export PDF/HTML |
| Logs | `stdout` uniquement | Aucune persistance, aucune structure |
| Prédictions | **Nulle part** | Aucune trace |

**Deux anomalies de stockage à corriger** :
1. **Des PNG base64 en base de données** (`VisionAnomalyExampleRecord.heatmap_png`, `mask_png`). Sur `MAX_EXAMPLES` images × 2 PNG, la table grossit en Mo par job, les dumps deviennent lourds, et le cache de Postgres est pollué par des blobs. Ces images appartiennent au stockage objet, avec une URL en base.
2. **Le bundle `joblib` est un pickle**. Chargé par `load_bundle` sans validation. Tant qu'il vient du worker, c'est acceptable ; le jour où un utilisateur importe un modèle, c'est une exécution de code arbitraire.

### H.2 Le « Model Registry » n'en est pas un

Ce qui existe (`Lot 9`) :
- `MLModel.stage` ∈ {`NULL`, `staging`, `production`} + `promoted_at` ;
- règle « un seul `production` par (dataset, cible) », l'ancien repasse en `staging` ;
- `GET /training/models/registry` (modèles promus) ;
- `GET /training/jobs/{id}/model/export` (`.joblib`) ;
- `model_card_json` avec seed, folds, essais Optuna, **versions des librairies** (`_training_environment_versions`) — très bon point.

Ce qui manque pour que ce soit un registre :

| Attendu | Présent ? | Détail |
| --- | --- | --- |
| Numéro de version (v1, v2, v3) | 🔴 | Il n'y a que des `job_id` |
| Nom de modèle logique regroupant des versions | 🔴 | Le regroupement implicite est `(dataset_id, target_column)`, jamais nommé ni exposé |
| Métadonnées | ✅ | `model_card` complet |
| Artefacts | ✅ | Bundle joblib |
| Hyperparamètres | 🟡 | Config du job oui ; **les hyperparamètres Optuna retenus ne sont jamais persistés séparément** — ils sont dans le pickle, pas requêtables |
| Métriques | ✅ | |
| Dataset utilisé | 🟡 | `dataset_id` seulement — **pas de hash de contenu, pas de version**. Le dataset peut être supprimé (cascade) |
| Seed | ✅ | |
| Environnement | ✅ | Versions des 8 librairies clés |
| Date / utilisateur | 🟡 | `created_at` oui ; `created_by` sur le job, **pas sur le modèle** |
| Comparaison de versions | 🟡 | `/jobs/compare` compare des **jobs**, pas des versions d'un même modèle |
| Promotion | ✅ | |
| Archivage | 🔴 | Pas d'état `archived` |
| Rollback | 🔴 | Remettre l'ancien en production est possible manuellement, mais rien ne trace la séquence |
| Historique de transitions | 🔴 | Seul `AuditLog` garde `model.promoted`, sans état précédent |
| Registre pour clustering / dimension / anomalies / **vision** | 🔴 | **Aucun** — seul le supervisé a `stage` |

### H.3 Reproductibilité : bonne, mais incomplète

Ce qui est capturé : seed, folds, essais Optuna, versions des librairies, spec de feature engineering, colonnes, `test_size`, rééquilibrage demandé **et** appliqué.
Ce qui manque : **le hash du dataset** (le fichier peut être remplacé — même id, contenu différent, aucune détection), la version du code de la plateforme (pas de commit SHA dans le `model_card`), et les hyperparamètres finaux sous forme requêtable.

---

## I. Traçabilité

### I.1 La chaîne réelle

```
Dataset (id, pas de version, pas de hash)
   └─→ TrainingJob (config, FE, seed, créé par, statut, progression)
          ├─→ ModelCandidate × N        (leaderboard : algo, score, folds)
          └─→ MLModel (1-1)             (métriques, SHAP, CQR, stage, environnement)
                 └─→ ???                 ⛔ RIEN
```

**La chaîne s'arrête au modèle.** `POST /training/jobs/{id}/predict` calcule une prédiction, une explication SHAP locale, un intervalle CQR — et **ne persiste rien**. Il n'existe aucune table `Prediction`.

Conséquences, toutes réelles :
- impossible de répondre à « quelle prédiction ce modèle a-t-il produite le 12 mars, avec quelles entrées ? » ;
- impossible de mesurer la dérive (pas de distribution des entrées observées) ;
- impossible d'auditer une décision prise sur la base du modèle ;
- impossible de savoir si un modèle promu en production est réellement utilisé ;
- impossible de facturer à l'usage.

Pour un produit qui se veut « orienté aide à la décision » et « traçable », c'est le maillon manquant le plus lourd de conséquences après §E.3.

### I.2 Journal d'audit

`AuditLog` couvre **10 actions**, vérifiées sur les 10 sites d'appel de `log_action` : `dataset.deleted`, `training_job.deleted`, `clustering_job.deleted`, `dimensionality_job.deleted`, `anomaly_job.deleted`, `vision_dataset.deleted`, `vision_classification_job.deleted`, `vision_anomaly_job.deleted`, `model.promoted`, `member.added`. Le champ `actor_id` est `SET NULL` pour survivre à la suppression du compte — bien vu.

**Huit des dix sont des suppressions.** Le journal répond donc à « qui a supprimé quoi » et à rien d'autre. Manquent : création de dataset, lancement de job, connexion/déconnexion, changement de rôle, export de modèle, appel de prédiction. Et : pas d'export CSV/JSON, pas de rétention, pas de filtre par action ou par période, `limit` par défaut à 100.

---

## J. UX/UI

### J.1 Parcours d'un nouvel utilisateur, étape par étape

**Inscription** — `/register`. Formulaire nom, e-mail, mot de passe, nom d'organisation, avec `PasswordStrengthMeter`. Pas de vérification d'e-mail, pas de CGU, pas de « mot de passe oublié ». *Où il se perd* : nulle part, c'est court. *Ce qui manque* : rien ne dit ce qu'il va se passer ensuite.

**Onboarding** — **Inexistant.** L'utilisateur atterrit sur `/` (Orientation), 3 cartes. Il n'y a ni visite guidée, ni dataset d'exemple, ni « voici votre première analyse en 3 clics ». Le bouton Aide ouvre `HelpModal` (6,9 Ko de texte). *C'est le premier écart majeur avec toute plateforme SaaS moderne* : aucun « temps jusqu'à la première valeur » n'est conçu.

**Création de projet** — **N'existe pas.** Il n'y a pas de projet. Tout tombe dans le pool de l'organisation.

**Import du dataset** — `/datasets`, bouton d'upload. Fonctionne. *Ce qui manque* : pas de glisser-déposer visible, pas de barre de progression pendant l'upload (synchrone, gelé sur un gros fichier), aucun exemple téléchargeable, aucune indication du format attendu avant l'erreur.

**Préparation** — Le wizard en 5 étapes est bon. *Ce qui manque* : à l'étape 1, l'utilisateur doit deviner quelle colonne prédire — aucune suggestion (le backend a pourtant tout ce qu'il faut : cardinalité, type, corrélations). À l'étape 4, « Mode expert » expose Optuna/folds/seed/CQR à quelqu'un à qui on vient de dire qu'il n'a pas besoin d'être expert : le repli par défaut est correct, l'existence même de l'étape ajoute une décision inutile en mode guidé.

**Entraînement** — Barre de progression + étape en cours, honnête (« lancé il y a X », « vous pouvez quitter la page »). *Ce qui manque* : **aucune estimation de durée** (or `n_lignes × n_modèles × n_essais × folds` la rend estimable), aucun bouton **Annuler** (seulement Supprimer), et **aucune notification** quand c'est fini — si l'utilisateur ferme l'onglet, rien ne le prévient.

**Résultats** — Riches. *Où il se perd* : tout de suite. Une modale avec métriques, IC, matrice de confusion, ROC, PR, calibration, courbe d'apprentissage, SHAP barres, SHAP beeswarm, permutation, leaderboard, CQR, formulaire de prédiction, explication locale. **Rien n'est hiérarchisé, rien n'est interprété.** Le débutant voit un mur de graphiques ; l'expert doit chercher.

**Décision** — L'étape n'existe pas. Voir §E.3.

**Modèle** — Promotion staging/production disponible depuis la modale. *Ce qui manque* : aucune explication de ce que « production » **fait** (réponse : rien, aucune conséquence système) ; c'est une étiquette.

**Historique** — Trois pages séparées, sans filtre ni recherche ni pagination.

**Nouvelle expérience** — Bouton « Nouvel entraînement ». *Ce qui manque* : **impossible de repartir d'une configuration existante** (« relancer avec 50 essais au lieu de 20 »). L'utilisateur ressaisit tout. C'est le geste le plus fréquent en pratique.

### J.2 Comptage de clics — « premier modèle entraîné »

Inscription (1) → Orientation (1) → Datasets (1) → Upload (2) → Training (1) → dataset (1) → cible (1) → Continuer ×4 → Lancer (1) = **~13 interactions**, dont 4 « Continuer » purement mécaniques. Comparable au marché, mais **sans** le dataset d'exemple qui permet, chez tous les concurrents, d'atteindre un résultat en moins de 5 clics.

### J.3 Feedback, erreurs, confirmations

| Aspect | État |
| --- | --- |
| États de chargement | ✅ Présents et distincts sur les pages principales |
| États d'erreur vs vide | ✅ Distingués (correction H4 réelle) |
| États vides avec appel à l'action | 🟡 « Aucun dataset prêt — importez-en un » : bon. Ailleurs, souvent un simple « — » |
| Confirmation destructive | ✅ `useConfirmAction` (double-clic) — élégant, mais **non découvrable** et **non accessible au clavier** (`onMouseLeave` pour annuler) |
| Notifications | 🔴 Aucun système de toast ; tout est en ligne, dans le contexte |
| Erreurs de polling | ⚠️ Silencieuses par conception — si le backend tombe, la barre de progression reste figée sans explication |
| Fin de job | 🔴 Aucune notification (ni navigateur, ni e-mail, ni badge) |
| Messages d'erreur backend | ✅ Excellents — français, actionnables, codes métier structurés. C'est un point fort réel |

### J.4 Recommandations de design system, par composant

Le socle de tokens OKLCH est bon. Ce qui manque est une couche au-dessus :

**Espacement & rythme** — Aucune échelle documentée ; les valeurs Tailwind sont choisies au cas par cas (`p-5`, `p-8`, `gap-2.5`, `py-1.5`). Fixer une échelle à 4 pt et 3 densités de carte (`compact` / `default` / `spacious`), puis l'appliquer.

**Typographie** — Aucune échelle nommée. On trouve `text-[11px]`, `text-xs`, `text-sm`, `text-base` mélangés. Définir 6 rôles (`display`, `title`, `subtitle`, `body`, `caption`, `overline`) et interdire les tailles arbitraires en `[]`.

**Cards** — `Card.tsx` fait 848 octets : un conteneur nu. Il lui faut des variantes (`elevated`, `outlined`, `interactive`, `stat`) et un slot `header`/`footer` normalisé, sinon chaque page réinvente son en-tête (c'est déjà le cas).

**Tables** — `Table.tsx` (2,6 Ko) n'a **ni tri, ni pagination, ni sélection, ni colonne figée, ni état vide, ni skeleton**. Chaque page qui liste des jobs réimplémente sa propre boucle. C'est le composant à refaire en premier : il débloque à lui seul les pages Historique et Dashboard.

**Badges / StatusBadge** — Existent et sont unifiés. Bon. Ajouter une variante `dot` pour les statuts en cours (point pulsant) plutôt qu'un spinner par ligne.

**Boutons** — `Button.tsx` (1,5 Ko) : variantes `primary`/`secondary`, tailles `sm`/`md`. Manquent : `destructive`, `ghost`, `link`, et surtout un état `loading` intégré (aujourd'hui chaque appelant gère `isSubmitting ? "Lancement…" : "Lancer"`).

**Tabs** — `Tabs.tsx` (1,5 Ko) sans synchronisation d'URL. La modale de résultats a 5+ onglets non deep-linkables — on ne peut pas partager « regarde l'onglet Explicabilité de ce modèle ».

**Formulaires** — `Input`/`Select` sont des primitives nues (601 et 669 octets). Il manque un `Field` qui compose label + aide + erreur + état requis, sinon chaque formulaire les recompose (déjà le cas partout).

**Graphiques** — `theme/charts.ts` centralise les couleurs : très bien. Manquent : un état vide de graphique, un skeleton, une légende repliable réutilisable (aujourd'hui `IsolatableLegend` vit dans `EvaluationCharts.tsx`), et une hauteur normalisée (on trouve 240 px en dur à plusieurs endroits).

**Navigation / sidebar** — Filtrée par pilier, `Objectifs` et `Tableau de bord` épinglés : bonne décision. Manquent : indicateur de jobs en cours dans la sidebar, fil d'Ariane, et **mémoire du dernier pilier** (revenir sur `/` force à rechoisir).

**Headers** — `PageHeader` (eyebrow, titre, description, icône, couleur, action) est bien conçu et réutilisé. À généraliser aux pages Vision, qui l'utilisent inégalement.

**KPI** — `StatTile` existe. Manquent : tendance (delta vs période précédente), état de chargement, et un lien cliquable systématique vers la vue détaillée.

**Modales** — `Modal.tsx` est accessible. Mais `ModelResultModal` (32 Ko) dans une modale est une erreur de conteneur : **ce contenu mérite une page** (`/training/jobs/:id`), deep-linkable, imprimable, partageable.

**Responsive** — Sidebar mobile en panneau glissant : OK. Mais les modales de 32 et 26 Ko et les graphiques Recharts à hauteur fixe ne sont pas pensés pour un écran étroit. Aucune page n'a de test à 375 px.

---

## K. Architecture SaaS

| Dimension | Aujourd'hui | Ce qu'il faut |
| --- | --- | --- |
| Multi-tenant | Organisation, isolation en base + sur disque | ✅ Bon socle |
| Hiérarchie | Organisation → Utilisateur | Organisation → **Projet** → ressources |
| Rôles | `owner` / `member` | + `admin`, `viewer`, et permissions par projet |
| Cycle de vie du compte | Inscription, connexion | + invitation par e-mail, mot de passe oublié, vérification, désactivation, SSO |
| Session | JWT 24 h, `localStorage`, pas de refresh, `logout` factice | Refresh token, révocation, cookie `httpOnly` |
| Quotas | Jobs concurrents (3/org) | + stockage, temps de calcul, nombre de datasets/modèles, rétention |
| Facturation | Aucune | Plans, compteurs d'usage, limites souples/dures |
| Stockage | Disque local partagé | S3/MinIO + URLs présignées |
| Calcul | 1 worker `SimpleWorker` | N workers, files par priorité, autoscaling, isolation par job |
| Base | Postgres 15, migrations maison | + Alembic, pooling, réplicas de lecture, sauvegardes testées |
| Observabilité | `print` structuré | Logs JSON + ID de corrélation, métriques Prometheus, traces, Sentry, alertes |
| Déploiement | `docker compose up` | CI (tests + lint + build), images versionnées, migrations au déploiement, healthchecks, rollback |
| Sauvegarde / restauration | Aucune | Postgres + stockage objet, restauration **testée** |
| Conformité | Aucune | RGPD (export, suppression, rétention), CGU, DPA, journalisation des accès |
| Statut / SLA | Aucun | Page de statut, incidents |

---

## L. Benchmark concurrentiel

*Contexte 2026, vérifié par recherche : le paysage a bougé.* Google a consolidé Vertex AI sous **Gemini Enterprise Agent Platform** — Model Garden, Custom Training, AutoML, Model Registry, Endpoints et Pipelines sont désormais rangés sous un menu « Models » de la plateforme d'agents ; l'entrée « Vertex AI » a disparu de la console. AWS a de son côté regroupé Canvas et Autopilot dans **SageMaker Unified Studio**. Weights & Biases opère sous CoreWeave. **La conséquence stratégique pour DataLab : les hyperscalers réorientent massivement leurs investissements vers les agents et le GenAI, et laissent le ML tabulaire/vision guidé se banaliser.** C'est une fenêtre, pas une menace.

### L.1 Comparaison qualitative

| Critère | Vertex/Gemini EAP | SageMaker | Azure ML | Dataiku | Databricks | H2O | DataRobot | W&B | Hugging Face | **DataLab** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Onboarding débutant | Faible | Très faible | Moyen | Bon | Faible | Moyen | **Très bon** | Faible | Bon | **Bon** |
| Workflow guidé pas à pas | Faible | Faible | Moyen | Bon | Faible | Moyen | **Très bon** | N/A | Moyen | **Bon (tabulaire) / Faible (vision)** |
| AutoML | Fort | Fort | Fort | Fort | Fort (glass-box) | **Très fort** | **Très fort** | N/A | Moyen | Moyen (9 modèles, Optuna) |
| Aide à la décision | Faible | Faible | Faible | Moyen | Moyen | Moyen | **Très fort** | Faible | Faible | **Faible (amont bon, aval nul)** |
| Explicabilité | Moyen | Moyen | Moyen | Bon | Bon | **Très fort** | **Très fort** | Faible | Faible | **Fort** |
| Anti-fuite / rigueur méthodo | Moyen | Moyen | Moyen | Bon | Bon | Bon | Fort | N/A | Faible | **Très fort (tabulaire)** |
| Model registry / versioning | Fort | Fort | Fort | Fort | **Très fort** (MLflow 3 + Unity Catalog) | Fort | Fort | **Très fort** | Moyen | **Faible** |
| Lignage / traçabilité | Fort | Fort | Fort | **Très fort** | **Très fort** | Moyen | Fort | Fort | Faible | **Faible** |
| Collaboration | Fort | Fort | Fort | **Très fort** | **Très fort** | Moyen | Fort | **Très fort** | Fort | **Faible** |
| Monitoring / dérive | Fort | Fort | Fort | Fort | Fort | Moyen | **Très fort** | Fort | Faible | **Nul** |
| Computer Vision | Fort | Fort | Fort | Moyen | Moyen | Moyen | Moyen | Fort | **Très fort** | **Faible** |
| Déploiement / serving | **Très fort** | **Très fort** | Fort | Fort | Fort | Fort | Fort | Moyen | **Très fort** | **Nul** |
| Simplicité globale | Faible | Très faible | Faible | Moyen | Faible | Moyen | Bon | Moyen | Bon | **Très bon** |
| Coût d'entrée | Élevé | Élevé | Élevé | Très élevé | Très élevé | Élevé | **Très élevé** | Moyen | Faible | **Faible** |

### L.2 Ce qu'ils font mieux que DataLab

1. **Le déploiement.** Tous, sans exception, transforment un modèle en endpoint appelable. DataLab s'arrête à un `.joblib` téléchargeable. **C'est le plus gros écart fonctionnel du tableau.**
2. **Le lignage et le versioning.** MLflow 3 + Unity Catalog (Databricks) et W&B sont à des années-lumière : versions nommées, alias, lignage dataset↔run↔modèle↔déploiement, comparaison de versions.
3. **Le monitoring en production.** Dérive des données, dérive des prédictions, alertes. DataLab n'a rien, et ne peut rien avoir tant que les prédictions ne sont pas persistées.
4. **La collaboration.** Commentaires, partage de rapports, projets, revues. DataLab a une organisation plate.
5. **L'AutoML industriel.** DataRobot et H2O testent des centaines de modèles avec blending et ensembling ; DataLab en teste 4 par défaut.
6. **La Vision.** Hugging Face AutoTrain permet d'entraîner un classifieur d'images sans code, avec un catalogue de modèles entier ; DataLab a 2 backbones CPU.
7. **L'échelle.** Tous ont du calcul élastique. DataLab a un worker.

### L.3 Ce que DataLab fait déjà mieux

1. **La rigueur méthodologique anti-fuite.** Split groupé vérifié, préprocesseur refit par fold, sélection sur CV. La plupart des AutoML grand public laissent l'utilisateur produire un modèle fuyant sans jamais l'avertir. C'est un vrai différenciateur, et il est défendable commercialement.
2. **Les garde-fous de données en langage clair.** 10 détecteurs, chacun avec `title` / `explanation` / `action` en français. Aucune plateforme du tableau ne fait aussi bien pour un non-spécialiste.
3. **Les intervalles de confiance conformes (CQR Mondrian).** Une prédiction accompagnée d'un intervalle avec couverture garantie, c'est ce dont un bureau d'études a besoin. Les concurrents donnent un point.
4. **L'honnêteté du produit.** `explainability_status: degraded` avec un message, `time_capped: true`, « repère indicatif, pas un seuil universel », suppression des éléments d'UI jamais câblés. Cette culture est rare et se voit.
5. **Le refus d'élire un gagnant en détection d'anomalies** sans vérité terrain. Intellectuellement supérieur à ce que fait la concurrence.
6. **La simplicité et le coût.** Pas de compte cloud, pas de facturation à la seconde, pas de vocabulaire MLOps imposé.

### L.4 Ce que DataLab peut faire mieux — la stratégie

**Ne copiez pas l'AutoML.** Vous perdrez : DataRobot a 15 ans d'avance et des ressources sans commune mesure. Le nombre de modèles testés n'est plus un argument.

**Le positionnement défendable est : « la plateforme qui vous dit ce que votre modèle vaut, et pourquoi ».** Toutes les plateformes produisent des modèles. Presque aucune n'explique honnêtement leurs limites à un non-spécialiste. Vous avez déjà la moitié du chemin (§E.2, §L.3) — il manque la moitié aval (§E.3).

**Quatre différenciateurs à construire, dans cet ordre :**

1. **Le « Rapport de confiance ».** Une page (et un PDF) par modèle qui répond en langage clair : ce modèle surapprend-il, ses probabilités sont-elles honnêtes, l'écart avec le 2ᵉ est-il significatif, ses intervalles tiennent-ils, plus de données aideraient-elles, sur quoi ne faut-il **pas** l'utiliser. Tout est déjà calculé. **Personne ne fait ça bien.**
2. **La quantification d'incertitude comme produit de première classe.** Vous avez le CQR Mondrian. Étendez-le à la classification (ensembles conformes : « ce modèle prédit A ou B, il ne sait pas trancher ») et affichez-le au premier plan, pas dans un onglet.
3. **La détection de fuite comme argument commercial.** Vous avez déjà les briques. Rendez-le visible : un badge « Analyse anti-fuite : 4 contrôles passés » sur chaque modèle. Un bureau d'études qui a déjà été brûlé par un modèle à 99 % en test et 60 % en réel comprendra immédiatement.
4. **La verticalisation contrôle qualité industriel.** Vous avez MVTec AD, Grad-CAM, la localisation de défauts, et une taxonomie de défauts dans le legacy. Aucun généraliste du tableau n'est bon là-dessus. C'est un marché où le CQR, l'anti-fuite et l'explicabilité sont exactement les bons arguments.

### L.5 Ce qu'il ne faut **pas** copier

- Le vocabulaire MLOps (feature store, pipeline DAG, experiment tracking exhaustif) : votre public ne le parle pas, et c'est votre avantage.
- Les notebooks intégrés : vous concurrenceriez Jupyter sur son terrain, pour rien.
- Le « glass box » à la Databricks (générer des notebooks) : incompatible avec un utilisateur non développeur.
- Un catalogue de 100 algorithmes : coût de maintenance élevé, gain marginal nul face à 4 boosters bien réglés.
- Le déploiement multi-cloud, Kubernetes, le serving GPU autoscalé : bien au-delà de ce qui est développable.
- La collaboration temps réel type Google Docs.

---

## M. Fonctionnalités manquantes (synthèse)

**Bloquantes pour « SaaS »** : migrations versionnées · stockage objet · workers multiples · pagination · gestion du 401 · observabilité · CI · sauvegardes · mot de passe oublié · désactivation de membre.

**Bloquantes pour « aide à la décision »** : verdict post-entraînement · interprétation des courbes · seuil de décision ajustable · métrique métier / matrice de coût · comparaison qualifiée des candidats · rapport exportable.

**Bloquantes pour « traçabilité »** : persistance des prédictions · versions de dataset + hash · versions de modèle numérotées · lignage complet · registre pour les 5 autres types de modèle · rollback · commit SHA dans la fiche modèle.

**Bloquantes pour « produit »** : onboarding · datasets d'exemple · projets/workspaces · notifications de fin de job · relancer depuis une config existante · annuler un job · filtres/recherche/tri sur les historiques · export PDF/HTML.

**Bloquantes pour « Vision au niveau du tabulaire »** : suppression des doublons · calibration hors test · augmentation configurable et recommandée · pondération de classes · early stopping · EDA d'images · wizard guidé · métriques binaires · registre vision · prédiction simple.

---

## N. Risques techniques

| # | Risque | Probabilité | Impact | Détail |
| --- | --- | --- | --- | --- |
| R1 | **Métriques Vision publiées et fausses** | Certaine (déjà le cas) | Critique | §G.1 + §G.2. Un client prend une décision industrielle sur une performance surestimée |
| R2 | **Impossible de faire évoluer le schéma sans casser** | Élevée | Critique | Ajouter `project_id`, renommer, backfiller : `_add_column_if_missing` ne sait pas faire |
| R3 | **Le worker meurt et emporte le job** | Élevée | Élevé | `SimpleWorker` sans fork + OOM sur gros dataset ; 40 min avant que le watchdog n'agisse |
| R4 | **Timeout inopérant sur code natif** | Moyenne | Élevé | `TimerDeathPenalty` par thread ; un `fit` LightGBM qui part en vrille bloque le worker unique |
| R5 | **Perte totale des données** | Moyenne | Critique | Aucune sauvegarde des artefacts (bind-mount) ni de Postgres |
| R6 | **Effondrement de performance à la montée en charge** | Certaine | Élevé | Pas de pagination, N+1, relecture de dataset à chaque requête, 8 appels au montage du Dashboard |
| R7 | **Session expirée = produit qui semble cassé** | Certaine | Moyen | §D.4 |
| R8 | **Régression invisible** | Élevée | Élevé | Pas de CI ; 174 tests qui ne tournent que si quelqu'un y pense ; aucun test frontend |
| R9 | **Dette du legacy** | Certaine | Moyen | `pytest.ini` + `tests/` à la racine peuvent ramasser du code mort ; 1,5 Mo de Python qui désoriente |
| R10 | **Dérive documentaire** | Élevée | Moyen | 4 documents concurrents dont un `workflow.md` de 147 Ko ; `AUDIT_ROADMAP.md` affirme encore « aucune page frontend Vision » alors qu'il y en a 4 |
| R11 | **Pickle non fiable** | Faible aujourd'hui | Critique si import | `joblib.load` + `torch.load(weights_only=False)` |
| R12 | **Fichiers monolithiques** | Certaine | Moyen | `training.py` 913 l., `models.py` 22 tables, `ml_training.py` 1 268 l., `ModelResultModal.tsx` 32 Ko |
| R13 | **Table Postgres saturée de PNG base64** | Moyenne | Moyen | §H.1 |
| R14 | **Le owner connaît le mot de passe de ses membres** | Certaine | Moyen | `auth.py:283` — problème de conformité autant que de sécurité |

---

## O. Architecture cible

**Principe directeur : la plus petite architecture qui lève les verrous, pas la plus impressionnante.** Pas de Kubernetes, pas de microservices, pas de feature store, pas de service mesh.

### O.1 Modèle de données cible

```
Organization
 └─ Project                                    ← NOUVEAU (le seul ajout structurel)
     ├─ Dataset ──── DatasetVersion            ← NOUVEAU (hash de contenu + schéma figé)
     ├─ Experiment                             ← NOUVEAU (regroupe des runs sur un même objectif)
     │   └─ Run  (= TrainingJob/Clustering/Vision… unifiés par un champ `kind`)
     │        ├─ RunCandidate  (leaderboard, existe déjà)
     │        └─ ModelVersion                  ← ÉTEND MLModel : numéro, alias, stage, lineage
     │             ├─ Artifact (URI objet)     ← remplace file_path
     │             └─ Prediction               ← NOUVEAU (le maillon manquant)
     ├─ AuditLog (existe)
     └─ Quota / Usage                          ← NOUVEAU
```

`Project` est le seul ajout qui touche tout le monde — raison de plus pour introduire Alembic **avant**.

### O.2 Infrastructure cible

```
            ┌──────────────┐
Navigateur ─┤ Nginx / TLS  ├─┬─→ API FastAPI (2+ instances, sans état)
            └──────────────┘ │        │
                             │        ├─→ Postgres (+ Alembic, + sauvegarde testée)
                             │        ├─→ Redis (files : `fast`, `training`, `vision`)
                             │        └─→ Stockage objet S3/MinIO (URLs présignées)
                             │
                             └─→ SSE `/events` (remplace le polling)

Workers RQ (Worker classique, fork) :
   worker-fast     × 2   → EDA, quality-check, aperçu (courts)
   worker-training × N   → supervisé + non supervisé
   worker-vision   × M   → torch, mémoire dédiée, GPU plus tard

Observabilité : logs JSON + request-id · /metrics Prometheus · Sentry
```

**Décisions clés et pourquoi :**

- **Alembic** — non négociable, prérequis de tout le reste.
- **MinIO plutôt que S3 direct** — API S3, auto-hébergeable, migration ultérieure vers S3 sans changer le code. Débloque le multi-nœud, les URLs présignées (les images vision cessent de transiter par l'API) et les PNG hors base.
- **`Worker` classique au lieu de `SimpleWorker`** — récupère l'isolation par fork sous Linux (le commentaire du fichier justifie `SimpleWorker` par la portabilité Windows ; garder `SimpleWorker` **en dev uniquement**, via une variable d'environnement).
- **Trois files séparées** — un EDA de 3 s ne doit jamais attendre derrière un entraînement vision de 20 min. C'est le gain de réactivité perçue le moins cher du rapport.
- **SSE plutôt que WebSocket** — unidirectionnel, suffisant, trivial derrière Nginx, supprime 6 implémentations de polling.
- **Un service de scoring, pas un service par modèle** — un endpoint `POST /v1/models/{alias}/predict` avec cache LRU des bundles, authentifié par clé d'API. C'est ce qui transforme DataLab en plateforme.
- **MLflow : non.** Vous avez déjà un registre embryonnaire, un leaderboard et un lignage partiels, intégrés à votre modèle multi-tenant. Ajouter MLflow créerait deux sources de vérité. Reprenez ses **concepts** (experiment / run / model version / alias), pas sa dépendance. Le code MLflow du legacy sert de référence, pas de cible.
- **Ce qu'on n'ajoute PAS** : Kubernetes, Kafka, feature store, service mesh, base vectorielle, orchestrateur de DAG.

### O.3 Extension NLP / GenAI

L'architecture par registre (`ModelSpec`, `models_for_task`) et par tables dédiées supporte un 4ᵉ pilier sans refonte : `services/nlp_registry.py` + `NlpJob`/`NlpModel` + un worker dédié. **Mais ne l'ouvrez pas avant que les trois piliers existants soient au niveau** — un 4ᵉ pilier à moitié fait aggraverait l'écart déjà visible entre tabulaire et vision.

---

## P. Roadmap

### 🔴 CRITIQUE — avant toute nouvelle fonctionnalité

| # | Problème | Impact | Solution | Zone | Dépend de |
| --- | --- | --- | --- | --- | --- |
| C1 | Doublons d'images conservés avant split | Métriques Vision fausses | Dédupliquer avant le split (garder 1 par hash), exposer `duplicates_removed` comme en tabulaire | `vision_datasets.py`, `vision_classification_training.py` | — |
| C2 | Seuil MVTec calibré et évalué sur le même test | Métriques Vision fausses | Scinder `test/` en calibration/évaluation stratifiées ; en attendant, l'afficher explicitement dans l'UI | `vision_anomaly_training.py` | — |
| C3 | Pas de migrations versionnées | Toute évolution du schéma est un risque | Introduire Alembic, générer la migration initiale depuis l'état courant, geler `_add_column_if_missing` | `core/database.py` | — |
| C4 | Pas de CI | Aucun filet | GitHub Actions : `pytest` + `tsc` + `eslint` + `vitest` + `vite build` sur chaque push | racine | — |
| C5 | 401 non géré côté frontend | Produit qui semble cassé | Intercepter le 401 dans `request()` → `clearToken()` + redirection `/login?expired=1` | `api/client.ts` | — |
| C6 | Aucune sauvegarde | Perte de données possible | `pg_dump` planifié + sauvegarde du volume `storage`, **restauration testée** | infra | — |
| C7 | `SimpleWorker` mono-process | Un job à la fois, crash contagieux | `Worker` classique sous Linux (env var), `replicas: 2+` | `run_worker.py`, compose | — |

### 🟠 IMPORTANT — rapidement

| # | Problème | Impact | Solution | Zone | Dépend de |
| --- | --- | --- | --- | --- | --- |
| I1 | **Aucune aide à la décision post-entraînement** | La promesse produit n'est pas tenue | Module `services/model_verdict.py` + composant `ModelVerdict` : surapprentissage, fiabilité, écart au 2ᵉ, calibration, couverture CQR, prochaine action | back + front | — |
| I2 | Prédictions non persistées | Traçabilité rompue, monitoring impossible | Table `Prediction` (entrées, sortie, intervalle, modèle, utilisateur, date) + rétention | `models.py`, `training.py` | C3 |
| I3 | Pas de pagination, N+1 | Effondrement à l'échelle | Pagination par curseur sur les 6 listes + `joinedload` + endpoint agrégé pour le Dashboard | routers | — |
| I4 | Relecture du dataset à chaque requête | Latence, CPU | Cache Parquet par `dataset_id` + LRU en mémoire du worker ; sortir `detect_task_type` du chemin HTTP | `services/datasets.py` | — |
| I5 | Stockage local | Verrou mono-nœud | MinIO + URIs ; migrer les PNG base64 hors base | `core/storage.py` | C3 |
| I6 | Files non séparées | EDA bloquée derrière un entraînement | 3 files RQ | `job_queue.py`, compose | C7 |
| I7 | Aucune observabilité | Diagnostic impossible en production | Logs JSON + request-id, `/metrics`, Sentry | `main.py` | — |
| I8 | Vision : pas de pondération de classes, pas d'early stopping, pas de scheduler | Modèles vision sous-optimaux sur données déséquilibrées | `CrossEntropyLoss(weight=…)`, patience, `ReduceLROnPlateau` | `vision_classification_training.py` | — |
| I9 | Vision : augmentation figée | L'utilisateur ne contrôle rien | Presets configurables (`aucune` / `légère` / `standard` / `forte`) + recommandation fondée sur la taille du dataset | idem | — |
| I10 | Wizard Vision absent | Asymétrie flagrante avec le tabulaire | Porter le pattern `Training.tsx` (5 étapes + mode expert) | `VisionClassification.tsx` | — |
| I11 | Pas de notification de fin de job | L'utilisateur doit surveiller | SSE + notification navigateur + e-mail optionnel | back + front | — |
| I12 | Mot de passe : pas d'oubli, owner qui saisit celui du membre | Sécurité, conformité | Invitation par jeton à usage unique + « mot de passe oublié » | `auth.py` | — |
| I13 | Aucun test frontend | Régressions invisibles | `@testing-library/react` + 1 test par page critique + 1 parcours Playwright | front | C4 |
| I14 | Pas d'annulation de job | Frustration | `POST /jobs/{id}/cancel` (drapeau relu par le callback de progression) | routers + workers | — |

### 🟡 PRIORITAIRE — pour atteindre un vrai niveau SaaS

| # | Sujet | Solution |
| --- | --- | --- |
| P1 | Versions de modèle réelles | `ModelVersion` : numéro, alias, `archived`, historique de transitions, rollback ; étendre aux 5 autres types |
| P2 | Versions et hash de dataset | `DatasetVersion` + SHA-256 ; le modèle référence une version, pas un id |
| P3 | Endpoint de scoring | `POST /v1/models/{alias}/predict` + clés d'API + cache LRU des bundles + scoring par lot |
| P4 | Projets / Workspaces | `project_id` sur les ressources, sélecteur de projet dans le shell |
| P5 | Onboarding | 3 datasets d'exemple, parcours en 3 clics, état vide guidé, mémoire du dernier pilier |
| P6 | Refonte des tables | `Table` avec tri, pagination, recherche, sélection, états vides — débloque Dashboard et Historiques |
| P7 | Historique unifié | Une page, filtres par type/statut/dataset/auteur/date |
| P8 | Rapport exportable | PDF/HTML du modèle avec verdict, métriques, graphiques, fiche modèle |
| P9 | Résultats en page, plus en modale | `/training/jobs/:id` avec onglets deep-linkables |
| P10 | Quotas de stockage et de calcul | Compteurs + limites souples/dures par organisation |
| P11 | Registre Vision | `stage`, promotion, export, prédiction simple |
| P12 | Seuil de décision ajustable | Curseur sur la courbe PR, persisté avec le modèle, appliqué à l'inférence |
| P13 | Journal d'audit complet | Créations, connexions, exports, prédictions + export CSV + rétention |
| P14 | Nettoyage du legacy | Décider ce qu'on récupère (§G.5), déplacer le reste dans `legacy/` ou un dépôt d'archive, retirer `pytest.ini`/`requirements.txt` racine |
| P15 | Consolidation documentaire | Un `ARCHITECTURE.md` à jour + un `CHANGELOG.md` ; archiver `workflow.md` (147 Ko) |

### 🟢 AMÉLIORATION — ensuite

Détection de dérive · matrice de coût métier · prédictions conformes en classification · PatchCore porté du legacy · EDA d'images (résolutions, formats) · taxonomie de défauts · SSO/SAML · i18n · thèmes par organisation · commentaires et partage · webhooks · SDK Python client · rétention automatique · page de statut · RGPD (export/suppression) · GPU optionnel pour la Vision.

---

## Q. Plan d'exécution détaillé

L'ordre suivant est dicté par les dépendances réelles trouvées dans le code, pas par l'ordre des piliers.

**Phase 0 — Arrêter l'hémorragie de crédibilité (1 semaine)**
C1, C2 (les deux fuites Vision) puis C4 (CI) et C5 (401).
*Pourquoi d'abord* : C1/C2 rendent des chiffres déjà affichés faux ; C4 protège tout ce qui suit ; C5 se corrige en 20 lignes pour un gain immédiat.
*Fait quand* : un test prouve qu'aucun hash d'image n'apparaît des deux côtés du split ; un test prouve que le seuil est calibré hors du jeu d'évaluation ; la CI est verte sur un push.

**Phase 1 — Débloquer l'évolution (1 à 2 semaines)**
C3 (Alembic) puis C6 (sauvegardes) et C7 (workers).
*Pourquoi ici* : tout le reste ajoute des colonnes et des tables. Sans Alembic, chaque phase suivante devient un risque.
*Fait quand* : une migration a été appliquée et annulée sur une copie de la base de production ; une restauration a été testée ; deux workers tournent en parallèle.

**Phase 2 — L'aide à la décision (2 semaines) ← le lot à plus fort impact**
I1 en entier : `services/model_verdict.py` (règles déterministes sur des nombres déjà en base, testées), `ModelVerdict` en tête de la vue Résultats, puis extension au clustering et à la vision.
*Pourquoi ici* : ne dépend d'aucune infrastructure, tient la promesse produit, et se démontre en 30 secondes à un prospect.
*Fait quand* : chaque modèle affiche un verdict en langage clair avec, pour chaque affirmation, la donnée qui la fonde.

**Phase 3 — Tenir la charge (2 semaines)**
I3 (pagination + N+1), I4 (cache dataset), I6 (files séparées), I7 (observabilité).
*Fait quand* : le Dashboard répond en moins de 300 ms avec 500 jobs ; un EDA reste sous 2 s pendant qu'un entraînement tourne.

**Phase 4 — Traçabilité (2 à 3 semaines)**
I2 (prédictions), P1 (versions de modèle), P2 (versions et hash de dataset), I5 (MinIO).
*Pourquoi après la phase 1* : ce sont exactement les changements de schéma qui exigent Alembic.
*Fait quand* : depuis une prédiction, on remonte au modèle, au run, à la version de dataset et à son hash.

**Phase 5 — Hisser la Vision au niveau du tabulaire (3 semaines)**
I8, I9, I10 puis P11 (registre vision) et les métriques binaires (16G).
*Fait quand* : le parcours Vision et le parcours tabulaire ont la même structure, les mêmes garde-fous et le même verdict.

**Phase 6 — Produit et UX (3 semaines)**
P5 (onboarding), P6 (tables), P7 (historique unifié), P9 (résultats en page), I11 (notifications), I14 (annulation), I12 (mots de passe).
*Fait quand* : un utilisateur qui n'a jamais vu le produit obtient un modèle avec verdict en moins de 5 minutes, sans aide.

**Phase 7 — Devenir une plateforme (4 semaines)**
P3 (endpoint de scoring), P4 (projets), P10 (quotas), P8 (rapport PDF), P13 (audit complet).
*Fait quand* : un modèle promu en production est appelable depuis un système tiers avec une clé d'API.

**Phase 8 — Consolidation continue**
P14 (nettoyage du legacy), P15 (documentation), I13 (tests frontend), puis les 🟢.

**Ce qu'il ne faut pas faire pendant ce temps** : ouvrir un pilier NLP/GenAI, ajouter des algorithmes au catalogue, ajouter des backbones vision, refondre le design.

---

## R. Les 10 améliorations à plus fort impact

1. **Le verdict post-entraînement** (I1) — tient la promesse produit, zéro dépendance, tout est déjà calculé. *Le meilleur rapport impact/effort du rapport.*
2. **Corriger les deux fuites Vision** (C1, C2) — des chiffres faux sont pires qu'une fonctionnalité absente.
3. **Alembic** (C3) — débloque littéralement tout le reste.
4. **Persister les prédictions** (I2) — ferme la chaîne de traçabilité et ouvre monitoring, audit et facturation.
5. **Pagination + N+1 + cache dataset** (I3, I4) — le produit reste utilisable au-delà de la démo.
6. **Workers multiples + files séparées** (C7, I6) — supprime le goulot d'étranglement n°1.
7. **CI** (C4) — 174 tests existants deviennent enfin un filet.
8. **Wizard Vision + augmentation configurable + pondération de classes** (I8, I9, I10) — supprime l'asymétrie la plus visible du produit.
9. **Endpoint de scoring** (P3) — fait passer DataLab d'un outil d'analyse à une plateforme.
10. **Onboarding avec datasets d'exemple** (P5) — divise par cinq le temps jusqu'à la première valeur, le seul chiffre qui compte en SaaS.

---

## S. Verdict final

**Vous avez construit le moteur avant la voiture, et le moteur est excellent.**

La qualité méthodologique du pilier tabulaire — anti-fuite à cinq niveaux, sélection sur CV, CQR Mondrian, SHAP routé par famille, dégradation propre, 174 tests qui entraînent de vrais modèles — place DataLab **au-dessus de la plupart des plateformes commerciales** sur la rigueur. La culture d'honnêteté du code (statuts dégradés affichés, éléments d'UI non câblés retirés, « repère indicatif, pas un seuil universel ») est un actif rare qu'il faut protéger.

Mais trois choses empêchent aujourd'hui d'appeler ce produit un SaaS :

**Un.** L'infrastructure est celle d'une démonstration. Un worker, un disque, pas de migrations, pas de pagination, pas d'observabilité, pas de CI, pas de sauvegarde. Ce n'est pas de la dette technique — c'est l'absence des fondations sur lesquelles tout le reste doit reposer. Elle se comble en trois à quatre semaines de travail ciblé, et il faut le faire **maintenant**, avant d'ajouter quoi que ce soit.

**Deux.** La promesse « aide à la décision » n'est tenue qu'à moitié. Avant l'entraînement, DataLab est meilleur que le marché : dix détecteurs de qualité en langage clair, des suggestions approuvées explicitement, jamais rien appliqué en silence. Après l'entraînement, il ne dit **rien** : douze visualisations remarquables et aucune phrase qui explique ce qu'elles signifient pour l'utilisateur. Le pilier non supervisé montre pourtant la voie avec `clusterQuality.ts`. C'est deux semaines de travail, sans risque, et c'est ce qui distinguerait le produit de tous ses concurrents.

**Trois.** Le pilier Vision n'a pas la rigueur du reste. Des doublons détectés puis conservés avant un split aléatoire, un seuil calibré sur le jeu qui sert à l'évaluer, une augmentation figée en dur, aucun registre : ce sont exactement les erreurs que le pilier tabulaire évite avec soin. L'incohérence est d'autant plus visible que le même dépôt contient les deux.

**La bonne nouvelle est la forme de ce travail restant.** Aucun des problèmes listés ici n'exige de repenser l'architecture. Les modules sont bien séparés, les moteurs sont purs et testables, le modèle de données est cohérent, le design system a de vrais tokens. Ce sont des ajouts et des corrections ciblées sur une base saine — pas une refonte.

**Et le contexte de marché vous est favorable.** Les hyperscalers réorientent leurs investissements vers les agents et le GenAI (Vertex AI est devenu Gemini Enterprise Agent Platform ; SageMaker se replie dans Unified Studio), et laissent le ML tabulaire et vision guidé se banaliser. Le créneau « la plateforme qui vous dit honnêtement ce que votre modèle vaut, et pourquoi » est ouvert, et vous avez déjà la moitié difficile — la rigueur méthodologique. Il vous manque la moitié visible : la traduire en phrases que votre utilisateur comprend.

**Recommandation** : ne construisez rien de neuf avant d'avoir terminé les phases 0, 1 et 2 (environ cinq à six semaines). Ensuite seulement, la plateforme pourra grandir sans se casser.

---

### Sources (benchmark, contexte 2026)

- [Vertex AI Is Now Gemini Enterprise Agent Platform: What Changed in 2026](https://gcpstudyhub.com/blog/vertex-ai-replaced-by-gemini-enterprise-agent-platform)
- [Vertex AI release notes — Google Cloud Documentation](https://docs.cloud.google.com/vertex-ai/docs/core-release-notes)
- [Automated ML, no-code, or low-code — Amazon SageMaker AI](https://docs.aws.amazon.com/sagemaker/latest/dg/use-auto-ml.html)
- [Amazon SageMaker Autopilot — AutoML](https://aws.amazon.com/sagemaker/ai/autopilot/)
- [No-code Machine Learning — Amazon SageMaker Canvas](https://aws.amazon.com/sagemaker/ai/canvas/)
- [MLflow 3 release](https://mlflow.org/releases/3/)
- [Manage model lifecycle in Unity Catalog — Databricks](https://docs.databricks.com/aws/en/machine-learning/manage-model-lifecycle/)
- [What is AutoML? — Databricks](https://docs.databricks.com/aws/en/machine-learning/automl/)
- [Weights & Biases: The AI developer platform](https://wandb.ai/site/)
- [AutoTrain — Hugging Face documentation](https://huggingface.co/docs/autotrain/main/index)
- [Top DataRobot Competitors & Alternatives 2026 — Gartner Peer Insights](https://www.gartner.com/reviews/market/data-science-and-machine-learning-platforms/vendor/datarobot/alternatives)
- [Comparison of AI Data Analytics Platforms: DataRobot vs H2O.ai vs Google AutoML](https://www.dsstream.com/post/comparison-of-ai-data-analytics-platforms-datarobot-vs-h2o-ai-vs-google-automl)
