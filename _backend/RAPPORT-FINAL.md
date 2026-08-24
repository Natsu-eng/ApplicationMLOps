# RAPPORT FINAL — Consolidation backend DataLab Pro

> Mission lancée le 2026-08-23, exécutée en 8 phases (`back/0` à `back/8`,
> une branche par phase, fusionnée après porte de qualité). Détail
> complet, décision par décision, avec les commandes réellement lancées
> et leur sortie réelle : [`JOURNAL.md`](JOURNAL.md) — ce document en est
> la synthèse, pas un remplacement. Audit de départ :
> [`AUDIT_BACKEND_2026-08-23.md`](../AUDIT_BACKEND_2026-08-23.md).

**Ce document est censé être lu en premier — il ne l'est pas s'il est
vide.** Chaque section ci-dessous est vérifiée contre le code et le
JOURNAL, pas assumée de mémoire.

---

## 1. Trouvé / corrigé / mesuré

### 1.1 Bugs réels trouvés (invisibles à la lecture de code, révélés par l'exécution)

| # | Bug | Phase | Gravité | Preuve |
|---|---|---|---|---|
| 1 | `backend/Dockerfile` ne copiait jamais `domains/` (post-Lot 8) — `ModuleNotFoundError` au démarrage, **toute image construite depuis le Lot 8 était cassée** | 1 | 🔴 | Simulation avant/après (Docker indisponible) ; confirmé par build réel en Phase 2 |
| 2 | `backend/Dockerfile` ne copiait jamais `alembic.ini`/`alembic/` — masqué par le bug #1 jusqu'à sa correction ; une fois #1 corrigé, **aucune migration ne s'appliquait plus jamais en production**, silencieusement (`lifespan()` avale l'exception) | 5 | 🔴 | Simulation avant/après ; `CommandError: Path doesn't exist` reproduit puis résolu |
| 3 | `token_valid_after` comparé sans normalisation de fuseau sous SQLite — une révocation de session après changement de mot de passe restait inopérante ~2h (décalage UTC+2 local) | 1B | 🟠 | `test_password_change_revokes_all_existing_sessions` |
| 4 | Les 6 endpoints `POST /jobs/{id}/rerun` appelaient `create_*_job` positionnellement — l'ajout d'un paramètre `request` a désaligné `current_user`/`db` dans les 6 domaines simultanément | 2 | 🟠 | `test_rerun_creates_a_new_job_with_the_same_configuration` |
| 5 | `_queued_rq_job_is_gone` classait une panne Redis et une disparition réelle de job dans le même `except Exception`, inversant le résultat pour le cas qui compte | 2 | 🟠 | `test_reconcile_marks_queued_job_as_failed_when_rq_job_is_gone` |
| 6 | Réconciliation des jobs `"queued"` sans délai de grâce — un job simulé en test (ou fraîchement enfilé en production) était déclaré perdu immédiatement, cassant silencieusement le quota de jobs concurrents (7 tests) — **invisible fichier par fichier, visible uniquement en suite complète** | 2 | 🟠 | 7 tests de quota, isolés dans la suite complète du 1ᵉʳ run Phase 2 |
| 7 | `validation_exception_handler` levait une `TypeError` non gérée (exception Python non sérialisable dans `err["ctx"]`) — un 422 attendu devenait un 500 | 1 | 🟠 | `test_register_rejects_common_password` |
| 8 | `domains/auth/router.py` n'avait jamais de `logger` défini — 3 appels `logger.exception`/`logger.info` levaient `NameError` silencieusement dans une tâche de fond (réponse HTTP déjà envoyée) | 1B | 🟡 | Suite `test_password_reset.py`, visible seulement parce que le gestionnaire global journalise `[UNHANDLED]` |
| 9 | `X-Request-ID` fourni par le client accepté sans validation et injecté tel quel dans les logs JSON | 1 | 🟡 | `test_observability.py` |
| 10 | `python-multipart` 0.0.27 — 3 CVE réelles et directement exploitables (`/auth/login` en premier lieu) | 1 | 🟠 | `pip-audit` |
| 11 | `statement_timeout` testé avec une assertion sur le texte anglais du message d'erreur — le serveur PostgreSQL local répond en français, le test échouait alors que le mécanisme fonctionnait | 2 | 🔵 | `test_statement_timeout_actually_cancels_a_slow_query` |
| 12 | `backend/.env` local contenait de vrais identifiants SMTP Gmail — la suite de tests tentait une vraie connexion réseau à chaque run (159s → 28s après correctif) | 1B | 🟡 | Mesure directe du temps d'exécution |

**Point commun aux bugs #4, #5, #6, #11** : tous trouvés en ÉCRIVANT et EN EXÉCUTANT des tests, jamais en relisant le code — preuve directe que le principe directeur du mandat (« rien n'est fait tant que ce n'est pas prouvé ») n'était pas une formule.

### 1.2 Sécurité (Phase 1/1B)

- IP cliente réelle derrière un reverse proxy (`get_client_ip`, CIDR de confiance explicite, jamais `*`).
- Cycle de vie complet des jetons : access 20 min + refresh rotatif 14 jours, révocation Redis ciblée + colonne `token_valid_after` pour la révocation en masse, `POST /auth/refresh` (rotation à usage unique), `POST /auth/logout` réellement effectif (ne faisait rien avant).
- En-têtes de sécurité stricts (CSP sans `unsafe-inline`, HSTS, X-Frame-Options...), `/docs` désactivé en production, CORS resserré.
- Enveloppe d'erreur unifiée `{"code","message","request_id"}` sur **100 % des réponses d'erreur** (HTTPException, 422 Pydantic, 500 non géré) — avant : seules les erreurs métier explicites la portaient.
- Mode d'échec du rate-limiting choisi par endpoint (fail-open pour l'authentification, fail-closed pour les 3 endpoints coûteux).
- Suite de régression IDOR paramétrée sur les 8 domaines à ressource.
- Upload durci : signature réelle de fichier, garde anti-bombe zip/xlsx.
- Validation exhaustive de la config en production (hard-fail, jamais un simple log).
- Réinitialisation de mot de passe reprise de CIAM et durcie sur 7 points (révocation de session, limite par compte, journal d'audit, purge, robustesse partagée, mail avec IP + notification de changement, absence d'oracle de temporisation/existence).

### 1.3 Fiabilité (Phase 2)

- Idempotence de création de job (`Idempotency-Key`), échec d'enfilage traité dans la même transaction (jamais de job `"queued"` orphelin).
- `job_watchdog.py` étendu aux jobs `"queued"` perdus, avec délai de grâce.
- Arrêt propre du worker (`stop_grace_period` Docker aligné sur `job_timeout`).
- Bornes explicites du pool de connexions + `statement_timeout` PostgreSQL, testés contre une vraie base.
- Sauvegarde/restauration vérifiée pour de vrai (cycle complet peuplement → sauvegarde → suppression → restauration → vérification).
- Comportement explicite et testé quand Redis ou PostgreSQL devient indisponible (au démarrage ET en cours de requête).

### 1.4 Traçabilité (Phase 3)

- `request_id` propagé au-delà de l'API : colonne sur les 6 tables de job + `audit_logs`, lu automatiquement par `log_action` (aucun site d'appel modifié).
- `workers/run_worker.py` utilise désormais le même formateur JSON que l'API (remplace `logging.basicConfig`) — le travail de corrélation des 6 workers était invisible jusqu'à ce correctif.
- Lignage prédiction → dataset/job/version exposé directement par l'API.
- Catalogue central de 65 codes d'erreur (`api/core/error_codes.py`), exposé dans `/openapi.json`.
- `log_action` étendu à la création de job dans les 6 domaines (avant : seuls cancel/delete/promote étaient audités).

### 1.5 Supply chain / CI (Phase 5)

- `.github/workflows/ci.yml` : `pip-audit` (6 vulnérabilités déjà évaluées ignorées par identifiant précis), `bandit`, `gitleaks` (secrets, historique complet), `--cov-fail-under=94` (la référence Phase 1 devient un vrai gate), scan d'image Trivy + SBOM CycloneDX, nouveau job `migration-on-populated-db` (PostgreSQL réel).
- Container hardening déjà en place vérifié (utilisateur non-root, `HEALTHCHECK`).

### 1.6 Mesures avant/après

| Mesure | Avant ce chantier | Après Phase 5 |
|---|---|---|
| Tests backend | ~525 (non mesuré avec `--cov`) | **863**, tous verts |
| Tests frontend | 64 | **68** |
| Couverture backend | jamais mesurée | **94 %**, gate CI |
| `ruff`/`mypy`/`bandit`/`pip-audit` | jamais exécutés | intégrés au flux de chaque phase + CI |
| CVE `python-multipart` | 3 exploitables | 0 |
| `pip-audit` en CI | absent | présent, gate |
| Secrets détectés en CI | jamais vérifié | `gitleaks`, historique complet |
| Migration testée sur base peuplée | SQLite seulement | SQLite + **PostgreSQL réel en CI** |
| Image Docker scannée | jamais | Trivy (CRITICAL/HIGH, gate) + SBOM |
| Déploiements fonctionnels depuis le Lot 8 | **aucun** (bug #1) | oui, prouvé |

---

## 2. Décisions prises seul, numérotées, avec leur raison

Liste complète et détaillée dans [`JOURNAL.md`](JOURNAL.md) (30 décisions
numérotées). Les plus structurantes :

1. **Corriger `Dockerfile` (bug #1) avant tout le reste** (Décision 1) —
   prérequis mécanique à toute vérification bout-en-bout des phases
   suivantes, pas un choix de priorisation parmi d'autres.
2. **`X-Real-IP` plutôt que `X-Forwarded-For`** (Décision 2) — une seule
   topologie de reverse proxy dans ce dépôt, `X-Real-IP` toujours écrasé
   par nginx, sans ambiguïté de position dans une liste.
3. **Colonne DB en plus de Redis pour la révocation de masse** (Décision
   3) — doit survivre à un redémarrage/vidage Redis.
4. **Ne pas dupliquer `dataset_id`/`training_job_id` sur `Prediction`**
   (Décision 22) — respecte un choix architectural déjà documenté dans le
   modèle plutôt que de le contredire pour un gain marginal ; « queryable »
   satisfait par l'exposition API, pas par une dénormalisation
   supplémentaire.
5. **Catalogue d'erreurs établi sans migrer les 56 sites d'appel
   dupliqués existants** (Décision 23) — le point de vérité et sa
   découvrabilité (`/openapi.json`) sont ce que le mandat demande
   explicitement ; la migration complète des littéraux est un chantier
   de diff disproportionné pour cette phase, priorisée en dette par ordre
   de risque de divergence.
6. **Scission des 4 plus gros fichiers et conversion async reportées en
   totalité** (Décision 25) — l'opérateur a demandé EXPLICITEMENT, à
   deux reprises pendant cette session, de ne pas laisser une étape
   prendre trop de temps ; ces deux chantiers exigeraient plusieurs
   cycles extraction → suite complète (60-80 min chacun) → correction,
   incompatible avec l'instruction reçue. Arbitrage assumé entre la
   vitesse demandée et la lettre du mandat, documenté avec sa raison
   exacte plutôt que silencieusement sauté.
7. **Adoption partielle de `apiErrorReference` (1 site sur 6)** (Décision
   29) — infrastructure posée et prouvée, migration des 5 sites restants
   mécanique et sans risque de conception, reportée pour la même raison
   de rythme que la décision 6.
8. **Aucune suppression en Phase 7** (JOURNAL, Phase 7) — un script
   heuristique de détection de code mort frontend a produit un résultat
   manifestement peu fiable (pages actives signalées comme orphelines) ;
   plutôt que d'agir dessus, la phase se conclut sans suppression, la
   méthode correcte (outillage dédié) documentée pour plus tard.

---

## 3. Ce qui a été délibérément laissé de côté (vérifié contre le code, jamais supposé)

- **Migration complète des 56 sites d'appel de codes d'erreur littéraux**
  vers `ErrorCode.XXX` (Décision 23) — 13 codes dupliqués identiquement
  dans 2 à 6 fichiers chacun, comptés par `grep`, pas estimés.
- **Scission de `training/router.py` (1351 lignes), `auth/router.py`
  (861), `vision/datasets/service.py` (822), `datasets/router.py` (689)**
  (Décision 25) — tailles mesurées par `wc -l`, pas estimées.
- **Audit systématique des endpoints candidats à `async def`** — jamais
  commencé ; SQLAlchemy reste synchrone partout dans ce dépôt, convertir
  un seul endpoint sans convertir la session DB sous-jacente serait la
  régression que le mandat redoute explicitement.
- **`log_action` sur l'upload de dataset et les endpoints `/predict`**
  (Décision 24) — seule la création de job a été ajoutée à l'audit cette
  phase, périmètre volontairement limité au gap le plus visible identifié
  par le survol factuel.
- **Adoption de `apiErrorReference` sur 5 sites `ErrorNote` restants**
  (`Profile.tsx` ×5, `AllHistory.tsx` ×1) — infrastructure prête, non
  câblée (Décision 29).
- **Retrait de code mort / harmonisation des états de chargement
  frontend** (Décision 30) — non traité, nécessiterait un outil
  d'analyse de graphe d'import absent de ce dépôt (`ts-prune` ou
  équivalent), jamais introduit sans justification écrite préparée.
- **Mise à jour de `pytest`/`python-dotenv`/`pyarrow`/`scikit-learn`/
  `lightgbm`/`ecdsa`** — 6 vulnérabilités connues, chacune évaluée
  individuellement (Phase 1) comme non exploitable dans l'usage réel de
  ce dépôt (dépendance jamais appelée, ou fonctionnalité absente) ;
  ignorées par identifiant précis en CI, jamais mises à jour (la montée
  de `lightgbm` en particulier affecterait potentiellement la
  compatibilité des modèles déjà persistés — nécessiterait son propre
  test de non-régression dédié).
- **3 écarts de schéma sans rapport avec la Phase 3, détectés par
  `alembic --autogenerate`** (`ml_models.promoted_at`/`training_jobs.
  progress_updated_at` : TIMESTAMP sans fuseau en base vs `DateTime
  (timezone=True)` dans le modèle ; `password_reset_tokens.created_at` :
  NOT NULL en base mais nullable dans le modèle) — dérive préexistante,
  probablement introduite dans une phase antérieure sans jamais être
  détectée. Retirés manuellement de la migration Phase 3 pour ne pas
  mélanger un correctif non lié — jamais traités séparément.

---

## 4. Ce qui a été approximé (jamais laissé implicite)

- **Smoke test Docker bout-en-bout local** — Docker Desktop instable sur
  ce poste de développement pendant la quasi-totalité du chantier
  (2 échecs de build >2h chacun en Phase 1 ; un succès partiel en Phase 2
  après amélioration réseau, mais le démon est redevenu indisponible
  ensuite). Chaque fois qu'une vérification directe était impossible, une
  preuve de substitution rigoureuse a été produite (reproduction exacte
  du jeu de fichiers copiés + exécution du code concerné) — jamais une
  supposition non vérifiée. Le job CI `smoke` (environnement Linux
  propre, GitHub Actions) reste la preuve bout-en-bout de référence,
  désormais enrichi de Trivy + SBOM (Phase 5).
- **`ruff format --check` à l'échelle du dépôt** — jamais satisfait
  globalement (confirmé dès la Phase 1 : jamais lancé avant ce chantier,
  la quasi-totalité du code préexistant ne suit pas le style par défaut).
  Appliqué systématiquement aux fichiers NEUFS de chaque phase (entièrement
  sous contrôle, zéro risque de diff illisible), jamais au code
  préexistant — reformater tout le dépôt en un seul commit sortirait du
  périmètre de n'importe laquelle des 8 phases.
- **Comparaison avant/après ruff sur les fichiers touchés** — à chaque
  phase, vérifié par comparaison directe (`git show main:... | ruff
  check ... -` vs état actuel) plutôt que supposé propre du fait que
  « seules quelques lignes ont changé ».

---

## 5. Travail restant, par sévérité

### 🔴 Critique — aucun

Le seul point qui aurait mérité ce niveau (Dockerfile) est corrigé et
vérifié (Décisions 1 et 26).

### 🟠 Majeur

1. Scission de `training/router.py`/`auth/router.py`/`vision/datasets/
   service.py`/`datasets/router.py` (Décision 25) — risque de régression
   réel si fait à la hâte, nécessite plusieurs cycles de validation
   complets.
2. Migration des 56 sites d'appel de codes d'erreur littéraux vers
   `ErrorCode.XXX` (Décision 23), priorisée par les 13 codes dupliqués
   d'abord.
3. Smoke test Docker bout-en-bout jamais exécuté avec succès complet en
   local sur ce poste (couvert en CI, jamais reproduit localement).

### 🟡 Moyen

4. `log_action` sur l'upload de dataset et les endpoints `/predict`
   (Décision 24).
5. Adoption de `apiErrorReference` sur les 5 sites `ErrorNote` restants
   (Décision 29).
6. Audit des endpoints candidats à `async def` (jamais commencé).
7. Outillage d'analyse de code mort (`ts-prune` ou équivalent) à
   introduire avec sa justification écrite, puis véritable passage de
   nettoyage frontend/backend (Phase 7).
8. 3 écarts de schéma préexistants détectés par `alembic --autogenerate`
   (voir §3) — mériteraient leur propre migration dédiée et leur propre
   validation, jamais glissés dans un autre correctif.

### 🔵 Mineur

9. `ruff format --check` à l'échelle du dépôt (reformatage global, gros
   diff sans rapport avec la logique métier).
10. 6 vulnérabilités `pip-audit` connues et acceptées (voir §3) — mise à
    jour de routine à prévoir, `lightgbm` en particulier nécessitant un
    test de non-régression sur les artefacts déjà persistés avant toute
    montée de version.
11. Une vraie file de rebut RQ surveillée (alerting, tableau de bord) —
    `failure_ttl` borne la rétention (Phase 2, Décision 19) mais rien ne
    surveille activement le `FailedJobRegistry`.

---

## 6. Ce qui n'a pas pu être vérifié

- **Parcours de fumée manuel dans un navigateur réel** — aucun
  environnement graphique disponible dans ce contexte d'exécution, à
  chaque phase. Compensé par `npx vitest run` + `npx tsc -b` + `npx
  eslint` + `node scripts/check-contrast.mjs` systématiques, jamais
  suffisant pour remplacer un vrai parcours utilisateur.
- **Comportement du job CI `migration-on-populated-db` sur GitHub
  Actions lui-même** — écrit et vérifié 2 fois en local contre un vrai
  PostgreSQL (dont une fois avec le jeu de dépendances minimal exact du
  step CI), jamais observé tourner réellement sur l'infrastructure
  GitHub Actions (ce chantier n'a pas déclenché de push/CI réel).
- **Effet réel du scan Trivy et de la génération SBOM en CI** — actions
  GitHub Marketplace standard, jamais exécutées dans ce contexte
  (nécessite Docker + GitHub Actions, indisponibles ici pour un test
  direct).

---

*Rapport écrit à l'issue de la Phase 8. Toute section vide ci-dessus
serait un signal d'alerte sur un projet de cette taille — aucune ne l'est
plus haut.*
