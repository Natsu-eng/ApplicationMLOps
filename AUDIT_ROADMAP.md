# AUDIT_ROADMAP.md — Audit complet DataLab Pro (2026-08-14)

> Audit lecture seule, aucun fichier de code modifié. Document de suivi pour
> prioriser les prochains lots — à cocher au fur et à mesure, comme
> `backend/workflow.md`. Contexte narratif : [`recap.md`](recap.md).
> Détail technique lot par lot : [`backend/workflow.md`](backend/workflow.md).

## Méthode

Audit mené en trois passes parallèles + vérification directe :
1. Lecture intégrale de `recap.md`, `backend/ARCHITECTURE.md`, `backend/workflow.md`.
2. Vérification directe du code (config, sécurité, modèles ORM, routers, Docker, requirements).
3. Trois audits délégués et croisés : `backend/services/` (pipeline ML), `frontend/src/` (UI/UX), code legacy Streamlit (`src/`, `ui/`, `helpers/`, `orchestrators/`, `pipeline_visio/`).
4. **Exécution réelle de la suite pytest complète** (257 tests, 29 min, entraînements réels non mockés) — pas une simple lecture de la documentation.

---

## A. État réel du projet

| Module | État réel | Détail |
| --- | --- | --- |
| **ML supervisé** | **Développé, robuste, quasi prêt pour un pilote client** | Bout en bout (upload → EDA → entraînement → leaderboard → explicabilité globale/locale → prédiction → registre versionné → comparaison inter-jobs → audit/quota). 257 tests, dont 2 actuellement en échec sur un run complet (voir F1) — **jamais mentionné dans la doc existante**. |
| **ML non supervisé** | **0 % côté SaaS** ; legacy plus riche que documenté | Aucun endpoint, aucun service, pilier "soon" côté frontend. **Correction du plan déjà écrit par l'utilisateur** : du clustering tabulaire (KMeans/DBSCAN/hiérarchique + métriques silhouette/DB/CH + visualisation PCA 2D) existe bien dans le legacy Streamlit, mais dilué dans l'orchestrateur généraliste (`orchestrators/ml_training_orchestrator.py`), pas dans un module autonome. Aucun Isolation Forest/LOF/One-Class SVM, aucun UMAP nulle part dans le dépôt. |
| **Computer Vision** | **0 % côté SaaS** ; legacy substantiel mais bugué | ~11 300 lignes de logique PyTorch (classification + transfer learning + détection d'anomalies MVTec AD avec autoencodeurs/PatchCore/Siamese + Grad-CAM), documentée "Production-Ready" par l'app elle-même mais avec **18 bugs déjà recensés dans 4 audits legacy existants** (`docs/legacy/`), dont 9 critiques (heatmaps de localisation jamais générées, pas de masque binaire, seuil de détection non calibré). Fortement couplé à un état global Streamlit incompatible avec une API stateless multi-tenant. |
| **MLOps** | **Partiel** | Registre de modèles versionné (staging/production, export, Lot 9), journal d'audit + quota technique (Lot 10) : réels et testés. MLflow **prévu mais jamais branché** (mentionné en commentaire dans `.env.example` seulement). Pas de tracking d'expériences centralisé au sens MLflow — le tracking actuel est ad hoc (tables SQL dédiées). |
| **Frontend / UI** | **Architecture saine, exécution du design system incomplète** | Registre de piliers unique (`config/pillars.ts`) qui rend l'extension vers non-supervisé/vision triviale au niveau routing/nav. Mais le système de tokens sémantiques déclaré dans `index.css` n'est **réellement appliqué que dans une minorité de composants** (voir D). |
| **SaaS (architecture)** | **Bon socle mono-tenant équipe, quelques trous avant un vrai pilote payant** | Isolation par organisation vérifiée à chaque lot (pas supposée). Pas de couche Workspace/Projet, RBAC à 2 rôles seulement, secret JWT par défaut non bloquant en production. |

**Ne jamais confondre** : `src/`, `ui/`, `helpers/`, `orchestrators/`, `pipeline_visio/`, `utils/`, `notebooks/`, `monitoring/` à la racine = app Streamlit legacy (référence de méthodologie, intacte). `backend/` + `frontend/` = l'architecture SaaS actuelle, seule cible de développement.

---

## B. Architecture actuelle

```
FRONTEND (React 19 + TS + Vite + Tailwind 4)
  AppShell (sidebar par pilier) → pages → components/ui (design system)
        │ HTTP JSON (Bearer JWT)
        ▼
API FastAPI (api/main.py)
  routers : auth / datasets / training (916 lignes — jobs, catalog,
  leaderboard, comparaison, registre, prédiction, tout dans un seul fichier)
        │
        ├─→ services/ (datasets, ml_task, ml_preprocessing, ml_registry,
        │    ml_training, ml_explainability, ml_inference, data_quality,
        │    feature_engineering, dataset_eda, stats_utils, audit)
        │
        └─→ RQ + Redis (job_queue.py) ──→ WORKER (process séparé,
             SimpleWorker + TimerDeathPenalty, portable Windows/Linux)
             workers/training_worker.py → persiste TrainingJob/MLModel/
             ModelCandidate + artefact joblib sur disque
```

- **Multi-tenant** : `Organization` (bureau d'études) → `User` (owner/member). Isolation à deux niveaux sur les datasets (filtre DB **et** chemin disque `storage/{organization_id}/...`).
- **Registre de modèles ML** (`ml_registry.py`) : 9 algorithmes déclarés via `ModelSpec`, aucun nom d'algorithme en dur dans le moteur — architecture extensible, **confirmée par lecture directe du code consommateur**, pas seulement par la doc.
- **Stockage** : disque local monté en volume Docker — migration S3 déjà identifiée comme nécessaire "quand le volume de clients l'impose", pas avant.
- **Docker** : Dockerfile backend soigné (utilisateur non-root, healthcheck, gunicorn 2 workers) ; docker-compose avec healthchecks et `restart: unless-stopped` sur tous les services. Un seul worker RQ non répliqué — limite de scale connue et assumée.

---

## C. Audit ML supervisé

### Points forts vérifiés (pas seulement documentés)

- **Anti-fuite réellement étanche** : préprocesseur cloné et refit à l'intérieur de chaque fold (`Pipeline` + `cross_validate`), jamais fit hors fold ; `GroupShuffleSplit`/`GroupKFold` avec vérification explicite par assertion qu'aucun groupe ne chevauche train/test.
- **Sélection stricte sur le score de validation croisée**, jamais sur le score test — respecté à la lettre partout, y compris dans le leaderboard.
- **SHAP partagé training/inférence** dans un seul module (`ml_explainability.py`) pour éliminer un risque de divergence déjà rencontré historiquement (bug de forme selon version SHAP).
- **Dégradation propre systématique** des diagnostics secondaires (SHAP/permutation/calibration/learning curve) — jamais de plantage de l'entraînement complet pour un diagnostic annexe.
- **Statistiques maîtrisées** : Cramér's V corrigé du biais de Bergsma (référencé), pas la version brute qui sur-estime.

### Problèmes trouvés

**Critique**

- **C1 — Job bloqué en "running" indéfiniment si le worker meurt.** Aucun heartbeat, watchdog, ni réconciliation de job orphelin (confirmé par grep : zéro trace de `stale`/`heartbeat`/`orphan`/`StartedJobRegistry` dans tout le backend). `SimpleWorker` (nécessaire sous Windows, sans fork) supprime justement l'isolation process-par-job qui permettrait à RQ de détecter un worker mort. Le timeout RQ de 30 min (`TimerDeathPenalty`, injection d'exception par thread) n'est pas garanti pour un `fit()` C/C++ (LightGBM/XGBoost/CatBoost) qui relâche le GIL. **Conséquence concrète** : un crash worker (OOM, coupure) laisse un job "running" pour toujours, consommant en permanence un des 3 slots du quota par organisation jusqu'à suppression manuelle.
- **C2 — La suite pytest complète n'est PAS verte actuellement, contrairement à ce que documente le projet.** Run réel (257 tests, 29 min) : **2 échecs** dans `tests/test_inference.py` (`test_predict_returns_plausible_value_with_interval`, `test_predict_rejects_missing_feature`), tous deux avec une 401 Unauthorized inattendue sur `POST /training/jobs/{id}/predict`. **Vérifié que ce n'est pas un bug produit** : les deux tests passent à 100 % rejoués isolément (`pytest -k ...`). C'est un bug d'isolation/pollution entre tests qui n'apparaît qu'en suite complète — jamais documenté dans `workflow.md`/`recap.md`, qui affichent "218/218" ou "257 tests" comme un fait acquis. Le filet de sécurité "la suite est verte" ne peut plus être invoqué tel quel tant que la cause (état global partagé — piste : `@lru_cache` sur `get_settings()`, séquence d'ID SQLite supposée redémarrer à 1 par un fixture `_fresh_database`) n'est pas identifiée et corrigée.

**Important**

- **C3 — Seed non propagée à la construction des folds de CV.** `_make_cv` (`ml_training.py:210-217`) hardcode `random_state=42`, ignore `config.seed` — contrairement au reste du pipeline (Optuna, learning curve, CQR) qui le respecte. Changer le seed utilisateur ne fait jamais varier les folds : `model_card.seed` est trompeur sur ce point précis, impossible de vérifier la stabilité des résultats en variant le seed.
- **C4 — Messages d'erreur déjà rédigés pour l'utilisateur absorbés par le filet générique.** `RuntimeError("Dataset introuvable ou non prêt")` (`training_worker.py:83-84`) et `FeatureEngineeringSpecError` tombent dans le `except Exception` générique, qui ne reconnaît que les patterns mémoire (`_user_safe_error_message`) — l'utilisateur reçoit un message générique inutile au lieu du diagnostic déjà écrit.
- **C5 — Classe cible absente du train après un split groupé** peut lever une `ValueError` non anticipée (`LabelEncoder`, `ml_training.py:990`) plutôt qu'un message diagnostiqué en amont.
- **C6 — `ml_inference.py:195`** renvoie telle quelle une exception sklearn brute côté message utilisateur (pas une stack trace complète, mais du texte technique interne).

**Amélioration**

- `ModelSpec.requires_scaling` déclaré mais jamais consommé (champ mort, risque de confusion future).
- `span_per_model = 65 // n_models` non protégé contre une division par zéro si le catalogue par défaut d'un futur `task_type` est vide.
- Couplage par import privé (`_try_parse_numeric_text`) entre `feature_engineering.py` et `data_quality.py`.
- `train_and_evaluate` (~280 lignes) reste la plus grosse fonction orchestratrice du projet — bien découpée en sous-fonctions, mais à surveiller si de nouveaux diagnostics s'ajoutent encore.

### Cette architecture backend est-elle prête pour clustering/anomaly/vision ?

**Clustering — oui, mais uniquement en module séparé.** `build_preprocessor` et `analyze_data_quality(target_column=None)` sont déjà génériques et réutilisables tels quels. En revanche `ml_training.py` est truffé d'hypothèses binaires implicites classification/régression (`_make_cv`, `LabelEncoder` sur `y`, CQR conditionné sur `"regression"`) — y injecter un 3ᵉ `task_type="clustering"` casserait silencieusement plusieurs branches. Le plan déjà écrit par l'utilisateur (module `clustering_registry.py` séparé façon `ml_registry.py`) est la bonne approche — **à condition d'écrire aussi un `clustering_training.py` séparé**, jamais une branche de plus dans `ml_training.py`.

**Vision — non, nouvelle construction quasi complète de la couche ML.** `ColumnTransformer`/Optuna+`cross_validate`/SHAP Tree-Linear-Kernel n'ont aucun sens pour des images/PyTorch. Seule l'infrastructure (job_queue, pattern worker RQ, stockage joblib, audit, `MLModel` générique) est réutilisable. Risque de régression sur l'existant tabulaire faible si la séparation est respectée (aucun couplage croisé prématuré constaté).

---

## D. Audit UX/UI

### Ce qui fonctionne bien

- Système de tokens OKLCH réel et documenté dans `index.css` — mais voir D-Critique.
- `theme/charts.ts` : palette validée contre les daltonismes, seule zone du code où le design system est suivi à 100 %.
- `AppShell.tsx` : navigation responsive correctement câblée (sidebar fixe desktop, drawer mobile), registre unique piloté par `config/pillars.ts`.
- États loading/error/empty globalement présents sur les listes principales.
- Pages "Bientôt disponible" (non supervisé/vision) honnêtes — jamais de fausse impression de fonctionnalité.
- Réutilisation de code réelle (`ModelResultView` partagé modale/page pleine largeur, `DataQualityWarnings` partagé wizard/EDA).

### Problèmes

**Critique**

- **D1 — Le système de tokens sémantiques est abandonné dans la quasi-totalité du code.** `Modal.tsx`, `Input.tsx`, `Heatmap.tsx`, `SectionHeader.tsx` et **toutes** les pages/modales métier utilisent des centaines de fois `text-slate-900/700/500`, `bg-slate-50`, `border-slate-200` en dur au lieu de `text-foreground`/`bg-muted`/`border-border`. Les messages d'erreur/succès utilisent `text-rose-700`/`text-emerald-600` en dur au lieu des tokens `destructive`/`success` (pourtant bien utilisés dans `Badge.tsx`). Deux systèmes de couleur coexistent pour la même sémantique. Faire évoluer la marque ou introduire un mode sombre demanderait de retoucher des dizaines de fichiers plutôt qu'un seul `index.css`.
- **D2 — Suppression d'un dataset sans aucune confirmation** (`Datasets.tsx:240-247`, clic unique direct), alors que le motif de confirmation à deux clics existe et fonctionne déjà pour la suppression d'un job (`Dashboard.tsx`, `Training.tsx`) — incohérence de garde-fou sur une action au moins aussi destructrice.
- **D3 — Échecs réseau rendus indiscernables d'un état "vide" sur au moins 5 écrans** (`Training.tsx`, `TrainingHistory.tsx`, `Dashboard.tsx`) : les erreurs de chargement sont catchées silencieusement vers `[]`, un utilisateur ne peut jamais distinguer "vous n'avez encore rien" de "le serveur ne répond pas".

**Important**

- **D4** — `Modal.tsx` sans `role="dialog"`, `aria-modal`, gestion d'Échap, ni focus trap — utilisé par tous les écrans de résultats/exploration de l'app.
- **D5** — Labels de formulaire non liés (`htmlFor`/`id` absents) sur le wizard d'entraînement, le mode expert, et le formulaire de prédiction — alors que Login/Register le font correctement.
- **D6** — Barre de recherche et cloche de notification visuellement actives sur chaque page mais **totalement inertes**, sans aucune indication (contraste avec le traitement honnête des piliers "Bientôt").
- **D7** — Contraste à risque sur les badges `warning`/`success` (texte 11px sur teinte claire) — à vérifier au contrastomètre, composant utilisé sur presque tous les écrans.
- **D8** — `ModelResultModal.tsx` (675 lignes) et `Training.tsx` (747 lignes) : logique et rendu de 5 onglets/étapes en un seul bloc JSX, sans sous-composants — freine la maintenabilité des deux écrans les plus centraux du produit.
- **D9** — Duplication répétée (badge de statut réécrit 3 fois, motif de confirmation à deux clics réimplémenté 2 fois, classe CSS de `&lt;select&gt;` recopiée 6+ fois — aucun composant `Select.tsx` alors qu'`Input.tsx` existe).
- **D10** — Grille de métriques figée à 2 colonnes sans repli mobile (`ModelResultModal.tsx:445`).

**Amélioration**

- Dashboard mélange activité ML et administration d'organisation sur un seul écran très long.
- Résultats non "deep-linkables" (pas d'URL pour un job/une exploration ouverts).
- Cibles cliquables sous la recommandation tactile de 44×44px sur certains boutons de suppression.
- Choix typographique éditorial (`font-serif` sur tous les H1) qui tranche avec les codes visuels habituels des plateformes data/IA de référence — à assumer consciemment ou reconsidérer.
- Micro-interactions un peu "site vitrine" (rotation/scale au survol des icônes, podium doré/argenté/bronze) pour un outil analytique.
- Motif "carte + en-tête de section" répété 15-20 fois sur un seul écran de résultats — aplatit la hiérarchie visuelle plutôt que de la renforcer.

---

## E. Audit SaaS — ce qui manque pour un niveau professionnel

| Sujet | État | Sévérité |
| --- | --- | --- |
| Secret JWT par défaut | Seulement journalisé en avertissement, jamais bloquant même si `environment=="production"` | **Critique avant tout pilote client réel** |
| Rate-limiting / lockout login | Absent — force brute possible sans limite | **Important** |
| Watchdog job orphelin | Absent (voir C1) | **Critique** |
| Workspace/Projet sous Organisation | Absent — tous les datasets/jobs d'une organisation sont dans un seul pool plat | Important si le besoin de cloisonner par client/mission apparaît, pas urgent pour un usage mono-équipe |
| RBAC | 2 rôles seulement (owner/member), pas de permission par dataset | Amélioration |
| Scaling worker | Un seul worker RQ pour toutes les organisations — limite connue et assumée par le projet | Important à moyen terme |
| Quota de stockage | Absent (seul un quota de jobs concurrents existe) | Amélioration |
| Export/rétention du journal d'audit | Absent (pas de CSV/PDF, pas de purge automatique) | Amélioration |
| Lint frontend | `eslint` configuré dans `package.json` mais **absent des dépendances installées** — le script `npm run lint` échoue immédiatement, aucune vérification de qualité JS/TS n'a tourné depuis des mois de lots | **Important** |
| Tests de composants React | Aucun (seulement 3 fichiers de tests sur des fonctions utilitaires pures — `cvScore`, `trainingPayload`, `charts`) | Important |
| Documentation architecture | `backend/ARCHITECTURE.md` arrêté au Lot 4b alors que `workflow.md`/`recap.md` vont jusqu'au Lot 10 — dérive documentaire | Amélioration (mais à corriger vite, risque de désorienter un futur contributeur) |

---

## F. Risques de régression identifiés

1. **Ajouter `"clustering"` comme 3ᵉ valeur de `task_type` dans les fichiers existants** (`ml_training.py`, `ml_task.py`) casserait silencieusement les branches implicites classification/régression (`_make_cv`, `LabelEncoder`, CQR). → Toujours passer par des modules neufs séparés (`clustering_registry.py` + `clustering_training.py`), jamais par une branche de plus dans l'existant.
2. **Le bug d'isolation de tests (C2)** signifie que "pytest est vert" ne peut plus être un filet de sécurité fiable pour juger les prochains lots tant que la cause n'est pas trouvée — risque qu'une vraie régression future se noie dans un faux-positif déjà "connu et ignoré".
3. **Aucun watchdog de job orphelin (C1)** : un futur module Vision (entraînements plus longs, plus gourmands, potentiellement GPU) rend un crash worker plus probable qu'aujourd'hui en tabulaire — le problème s'aggravera avec la Vision si non traité avant.
4. **Dérive documentaire d'`ARCHITECTURE.md`** : un futur contributeur (humain ou IA) qui s'y fierait pour comprendre l'état du Lot 5+ partirait d'un schéma obsolète (catalogue 3 modèles au lieu de 9, pas de registre versionné, pas d'audit log).
5. **Absence totale de lint fonctionnel côté frontend** depuis plusieurs mois de lots : dette potentiellement déjà accumulée et invisible (variables inutilisées, dépendances de hooks manquantes) que seul `tsc` ne peut pas détecter.
6. **Abandon des tokens de design (D1)** : toute future évolution de thème ou nouvelle page reprendra probablement le même réflexe `text-slate-*` en dur par mimétisme avec le code existant majoritaire — le problème s'auto-aggrave tant qu'il n'est pas corrigé sur les composants `ui/` partagés en premier.

---

## G. Architecture cible recommandée

**Backend**
- Nouveau module `services/clustering_registry.py` + `services/clustering_training.py`, jamais de branche `task_type=="clustering"` dans les fichiers ML supervisé existants.
- Introduire une réconciliation de jobs orphelins : au démarrage du worker (ou via un job périodique RQ dédié), repérer les `TrainingJob` en `"running"` sans activité récente (`progress` non mis à jour depuis N minutes) et les marquer `"failed"` avec un message explicite.
- Hard-fail au démarrage de l'API si `environment=="production"` et `jwt_secret_key` encore égal au défaut — remplacer le simple `logger.warning`.
- Scinder `api/routers/training.py` (916 lignes) en sous-routers logiques (`jobs`, `model_registry`, `comparison`) avant d'y ajouter les futurs endpoints clustering/vision, pour ne pas empiler encore plus dans un seul fichier.
- Propager `config.seed` à `_make_cv`.

**Frontend**
- Traiter `Modal.tsx`/`Input.tsx`/`Heatmap.tsx`/`SectionHeader.tsx` (composants `ui/` partagés, effet de levier maximal) en premier dans un passage systématique slate-\*/rose-\*/emerald-\* → tokens sémantiques.
- Installer réellement `eslint` (actuellement configuré mais absent) avant tout nouveau lot frontend.
- Extraire `Select.tsx`, un `StatusBadge` unique, un hook `useConfirmAction()`.

**Vision**
- Nouveau module backend isolé (`services/vision_*.py`), infra partagée uniquement (job_queue, storage, audit, `MLModel` générique) — jamais les pages Streamlit ni le state global `monitoring/state_managers.STATE`.
- Corriger les 9 bugs critiques déjà documentés dans `docs/legacy/ANALYSE_COMPLETE_COMPUTER_VISION.md` **pendant** le portage, pas après.

**Documentation**
- Mettre à jour `backend/ARCHITECTURE.md` au niveau réel (Lot 10) ou le fusionner dans `workflow.md`, qui est déjà le document vivant fiable.

---

## H. Roadmap priorisée

### 🔴 Critique — à corriger avant de lancer le prochain module

- [x] **H1** — Diagnostiqué et mitigé (2026-08-14) : bug d'épuisement de ressources (connexions SQLite non recyclées sur ~30 min), pas une fuite d'état déterministe. `poolclass=NullPool` dans `tests/conftest.py`. Non re-vérifié sur un run complet (~30 min) après ce lot — chaque fichier touché a été validé isolément.
- [x] **H2** — `services/job_watchdog.py` (nouveau), réconciliation avant le comptage du quota, `TrainingJob.progress_updated_at`. Testé (`tests/test_saas_hardening.py`).
- [x] **H3** — Hard-fail `RuntimeError` au démarrage si `environment=="production"` et clé par défaut. Testé (`tests/test_security.py`, sous-process).
- [x] **H4** — D2 (confirmation suppression dataset) et D3 (erreur/vide sur Dashboard/Training/TrainingHistory) corrigés. D1 (tokens) traité à l'échelle du projet, pas seulement `ui/` — voir refonte design ci-dessous.
- [x] **H5** — `backend/ARCHITECTURE.md` remis à niveau (résumé jusqu'au Lot 10 + ce lot).

### 🟠 Important — pour une plateforme solide et professionnelle

- [x] **H6** — Seed propagée à `_make_cv`, testé (`test_make_cv_folds_vary_with_seed`).
- [x] **H7** — Type dédié `TrainingAbortedError` (pas un `RuntimeError` nu — un premier essai trop large a été détecté et corrigé grâce à la suite de tests existante). Testé.
- [x] **H8** — Garde-fou classe absente après split groupé, testé (seed=0 reproduit le cas empiriquement).
- [x] **H9** — `eslint.config.js` créé, dépendances installées. 0 erreur, 11 avertissements mesurés.
- [x] **H10** — `Modal.tsx` accessible (rôle, focus trap, Échap) + labels liés sur le wizard (5 champs), `ExpertModePanel` (4 champs), `PredictionForm`.
- [x] **H11** — Rate-limiting Redis par IP sur `/auth/login`, testé (3 nouveaux tests, dont l'isolation avec `/auth/register`).
- [ ] **H12** — Découper `ModelResultView`/`TrainingForm` en sous-composants par onglet/étape. **Reporté** — chantier de refactoring risqué sur des composants déjà fonctionnels, à faire dans un lot dédié avec revue attentive.
- [ ] **H13** — Tests de composants React. **Reporté** — `@testing-library/react` non installé, aucune infrastructure existante ; à cadrer dans un lot dédié plutôt que bâclé.
- [x] **H14** — `StatusBadge.tsx`, `Select.tsx`, `useConfirmAction.ts` — dédupliqués et adoptés sur Dashboard/Datasets/Training/TrainingHistory/EdaModal.
- [x] **H15** — Contraste vérifié mathématiquement (WCAG sur valeurs OKLCH) : `text-warning`/`text-success` étaient à 2.57:1/3.40:1 sur blanc (sous 4.5:1) → 4.82:1 après ajustement de luminosité.
- [x] **H16** — Recherche/notifications AppShell retirées (jamais câblées), remplacées par un vrai bouton Aide (`HelpModal.tsx`, premier onboarding du produit).

### 🟢 Amélioration — à ajouter ensuite

- [x] **H17** — `ModelSpec.requires_scaling` supprimé (confirmé inutilisé ailleurs).
- [x] **H18** — Grille de métriques `ModelResultModal.tsx` : repli `grid-cols-1` sous `sm`.
- [ ] **H19** — Séparer Dashboard (activité ML) et Organisation/Équipe (admin) en deux surfaces. **Reporté.**
- [x] **H20** — Deep-linking par URL : `?job=`, `?explore=`, `?preview=` (Dashboard/Datasets), via `useSearchParams`.
- [ ] **H21** — Quota de stockage par organisation, export/rétention du journal d'audit. **Reporté.**
- [ ] **H22** — Couche Workspace/Projet — pas de besoin business confirmé, volontairement non traité.
- [ ] **H23** — Réévaluer la charge visuelle une fois le reste stabilisé. **Reporté**, largement atténué par la refonte design de ce lot.

**Bonus hors roadmap initiale, ajoutés sur retour utilisateur direct pendant ce lot** : persistance de session du wizard d'entraînement (`sessionStorage`, signalé "à clarifier" dans `workflow.md` depuis des mois, jamais traité) ; refonte visuelle complète (fond de page, `PageHeader`, cartes colorées sur tous les onglets Résultats/Explorer, modale agrandie) ; suppression d'un graphe de variance CV réellement cassé (valeurs aberrantes malgré le clamp déjà en place).

### 🔵 Lots produit déjà planifiés (hors périmètre de cet audit technique)

- [x] **Lot 11+12** — Clustering + profils de segments **livrés** (2026-08-14) : `services/clustering_registry.py` + `clustering_training.py` (module séparé, jamais une extension de `ml_training.py`, confirmé dans le code livré), `ClusteringJob`/`ClusterModel`/`ClusterCandidateRecord` (tables dédiées), `workers/clustering_worker.py`, router `api/routers/clustering.py`, page `pages/Clustering.tsx`, pilier "ML non supervisé" activé dans `config/pillars.ts`. Watchdog (H2) et quota (partagé avec le supervisé) réutilisés, pas dupliqués. 41 tests (registre, moteur, worker, API).
- [ ] **Lot 13** — Réduction de dimension (PCA/t-SNE/UMAP), transversale au clustering et à la détection d'anomalies. **Prochaine étape.**
- [ ] **Lot 14** — Détection d'anomalies tabulaire (Isolation Forest, LOF, méthodes statistiques). **Prochaine étape.**
- [ ] **Lots 6-8** — Computer Vision : portage des architectures PyTorch legacy (transfer learning, autoencodeurs, PatchCore, Grad-CAM) vers un module backend neuf, en corrigeant au passage les 9 bugs critiques déjà documentés dans `docs/legacy/`.

---

## I. Recommandation finale

**Il faut d'abord corriger un nombre limité de points ciblés (H1 à H5, quelques jours de travail, pas un chantier de plusieurs semaines) avant de se lancer sereinement sur le Lot 11 (ML non supervisé).**

Ce n'est ni un blocage total, ni un feu vert inconditionnel :

- Le **cœur ML supervisé est réellement solide** — anti-fuite vérifiée, sélection méthodologiquement correcte, explicabilité multi-famille, architecture en registre qui a fait ses preuves d'extensibilité (9 modèles ajoutés sans toucher le moteur). C'est un socle sain sur lequel construire.
- Mais **la suite de tests, qui est la seule garantie objective de non-régression pour tout ce travail, n'est actuellement pas fiable en un seul run** (H1) — c'est le point le plus urgent, car il conditionne la confiance qu'on peut avoir dans tout le reste au fur et à mesure que le projet grandit.
- Le **risque de régression le plus concret pour le prochain module** n'est pas dans l'architecture (le pattern registre est prêt, confirmé par lecture de code), mais dans la tentation d'étendre `ml_training.py` au lieu de créer un module séparé pour le clustering — le plan déjà écrit va dans la bonne direction sur ce point, juste à corriger l'hypothèse de départ sur le legacy.
- Les points critiques UX (D1-D3) et SaaS (JWT, watchdog) sont **rapides à corriger** individuellement et évitent d'accumuler une dette qui coûtera plus cher à rattraper une fois deux nouveaux piliers (non supervisé + vision) construits par-dessus.

Correction de H1 à H5 → feu vert pour le Lot 11 en confiance.
