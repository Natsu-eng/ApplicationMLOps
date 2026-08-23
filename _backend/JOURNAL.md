# JOURNAL.md — Consolidation backend DataLab Pro

> Décisions prises seules, numérotées, avec leur raison. Chaque point de la
> porte de qualité est consigné ici avec la commande réellement lancée et sa
> sortie — jamais un résumé de ce qu'elle aurait donné.

---

## Phase 1 — Sécurité

### Décision 1 — Corriger `backend/Dockerfile` avant tout le reste

**Constat** (voir `AUDIT_BACKEND_2026-08-23.md`, §0) : `backend/Dockerfile`
n'a plus été touché depuis le commit `bf5c594` (Lot 3), soit **avant**
`bd512fd` (Lot 8, monolithe modulaire `domains/`). Il copie encore
`services/` (mort depuis le Lot 8, ne contient plus qu'un `__pycache__` non
tracké) et jamais `domains/`, alors que `api/main.py` importe
`from domains.<x>.router import ...` pour 11 domaines. Résultat :
`gunicorn api.main:app` lève `ModuleNotFoundError: No module named
'domains'` au démarrage — l'image de production ne peut pas tourner depuis
le Lot 8.

**Décision** : corriger immédiatement (`COPY domains/ ./domains/` remplace
`COPY services/ ./services/`), avant toute autre correction de sécurité.
**Raison** : la porte de qualité de CHAQUE phase de ce chantier exige
`docker compose up -d --build` + `python -m scripts.smoke_test_docker` au
vert (voir le mandat). Sans ce correctif, aucune phase ne peut être
vérifiée bout-en-bout — ce n'est pas un choix de priorisation parmi
d'autres, c'est un prérequis mécanique à tout le reste. Ne relève d'aucun
des 4 cas d'arbitrage produit (pas de migration destructrice, pas de
rupture d'API, pas une faille distante à elle seule, pas une contradiction
d'exigences) : c'est un bug d'infrastructure pur, correctif d'une ligne,
sans ambiguïté.

**Vérifié** : `git log --oneline -- backend/Dockerfile` (dernier commit
`bf5c594`) vs `git log --oneline --all --diff-filter=A --
backend/domains/auth/router.py` (introduit par `bd512fd`) — confirme l'ordre
chronologique. `git show 10e15fd:backend/Dockerfile` (dernier commit avant
Lot 8 à toucher `Dockerfile`, en fait `bf5c594` est le vrai dernier — need
double check ci-dessous) confirme l'absence de `domains/` dans toutes les
versions historiques du fichier.

**Suite** : reconstruire l'image (`docker compose build backend`) et lancer
`docker compose up -d` + `python -m scripts.smoke_test_docker` pour
confirmer la résolution — build relancé en arrière-plan, résultat consigné
en fin de phase avec le reste de la porte de qualité.

**⚠️ Limitation d'environnement rencontrée** : `docker compose build backend`
a été tenté deux fois (une fois avant ce correctif pour confirmer le bug,
une fois après pour le vérifier) — les deux tentatives ont échoué après
plus de 2h chacune, non pas à cause du code mais d'un réseau extrêmement
lent et instable sur ce poste (~20-30 kB/s, coupures répétées ; `xgboost`
seul pèse 297 Mo, jamais téléchargé en entier avant épuisement des
tentatives de reprise de `pip`). **Impossible d'obtenir une image Docker
construite dans cet environnement** — voir mémoire
`feedback_test_docker_environment.md`.

**Vérification de substitution, honnête et rigoureuse** (pas une
supposition) : reproduction EXACTE du jeu de fichiers que copie le
`Dockerfile` (`api/`, `domains/`, `workers/`, rien d'autre) dans un
répertoire temporaire isolé, puis `python -c "import api.main"` avec le
même interpréteur (le `.venv` local a déjà toutes les dépendances
installées, donc ce test isole précisément la variable qui a changé — le
jeu de fichiers copiés — de celle qui n'a pas changé — les dépendances) :
- **Avant correctif** (jeu `api/` + `services/` vide + `workers/`, sans
  `domains/`) : `ModuleNotFoundError: No module named 'domains'` — confirme
  le bug tel que diagnostiqué en lecture de code.
- **Après correctif** (jeu `api/` + `domains/` + `workers/`) : import
  réussi.

Preuve suffisante que le correctif résout le problème diagnostiqué ; ne
remplace pas un vrai `docker compose up` + smoke test bout-en-bout (healthcheck
réseau, volumes, variables d'environnement Docker, nginx) — repris dès que
la construction de l'image aboutira (build laissé en arrière-plan,
opportuniste, ou à relancer sur un réseau plus stable). Consigné aussi en
Phase 8, section « ce qui a été approximé ».

### Décision 2 — IP cliente réelle (§A.6)

`api/core/rate_limit.py::get_client_ip` (nouveau) : ne fait confiance à
`X-Real-IP` (fixé par nginx à `$remote_addr`, jamais dérivé d'un en-tête
client) que si le pair TCP direct appartient à `trusted_proxy_cidrs`
(défaut `172.16.0.0/12`, plage standard des réseaux bridge Docker — jamais
`*`, conformément à la consigne). Câblé dans `rate_limit_dependency` (donc
`/register`, upload, `/explain`) et dans `/auth/login` (logique dupliquée
avant ce correctif). Testé : `tests/test_client_ip.py` (5 tests) — le cas
qui aurait le plus de conséquences si régressé est
`test_two_distinct_real_clients_behind_same_proxy_get_distinct_keys`.

**Pourquoi `X-Real-IP` et pas `X-Forwarded-For`** : un seul reverse proxy
dans cette topologie (nginx). `X-Forwarded-For` (`$proxy_add_x_forwarded_for`
côté nginx) AJOUTE à une valeur déjà fournie par le client — ambigu sur
quelle position de la liste faire confiance selon le nombre de sauts.
`X-Real-IP` (`$remote_addr`) est toujours écrasé par nginx, jamais dérivé
d'un en-tête client : pas d'ambiguïté avec un seul saut.

### Décision 3 — Cycle de vie des jetons (§A.1, §A.2, §A.4)

Jeton access court (20 min, dans la fourchette 15-30 min demandée) + jeton
refresh rotatif (14 jours), tous deux avec `jti`. Révocation Redis
(`api/core/token_store.py`) pour la révocation ciblée (logout d'une
session) et colonne `User.token_valid_after` (migration
`a1c9d4e27b6f`) pour la révocation en masse (changement de mot de passe) —
choisi plutôt qu'énumérer chaque jeton émis, qui n'existe nulle part côté
serveur pour un JWT stateless.

**Pourquoi une colonne DB en plus de Redis** : une révocation en masse doit
survivre à un redémarrage/vidage Redis — un `token_valid_after` en base est
la seule source de vérité qui ne peut pas disparaître silencieusement.

`POST /auth/refresh` (nouveau) : rotation à usage unique, testée
explicitement (`test_refresh_token_is_single_use_rotation`) — une seconde
présentation du même refresh token échoue toujours, y compris s'il vient
d'être utilisé avec succès (vol + rejeu ne prolonge jamais une session).

`POST /auth/logout` : révoque réellement le jeton access de la requête
(TTL Redis = durée de vie restante, jamais plus) et le refresh token fourni
en corps — avant ce correctif, ne faisait rien. `PATCH /auth/me/password`
révoque désormais TOUTES les sessions (access via `token_valid_after`,
refresh via le set Redis par utilisateur).

Frontend (`frontend/src/api/client.ts`, `AuthContext.tsx`) : renouvellement
transparent (`withAuthRetry` + `tryRefreshAccessToken`, promesse partagée
pour éviter de consommer le refresh token rotatif plusieurs fois en
parallèle), `logout()` devient asynchrone et appelle réellement
`POST /auth/logout` (ne le faisait pas avant ce correctif — le bouton
"Déconnexion" ne faisait que vider le `localStorage`).

Testé : `tests/test_token_lifecycle.py` (9 tests, backend) +
`npx tsc -b`/`npx vitest run` (frontend, aucune régression sur les tests
existants). `docker compose ps`/smoke test Docker reportés à la fin de la
porte de qualité (build en cours).

### Décision 5 — Surface exposée : en-têtes de sécurité, CSP, `/docs` gaté, CORS resserré, enveloppe d'erreur unifiée (Axe E)

`api/main.py` : `SecurityHeadersMiddleware` (HSTS, X-Content-Type-Options,
X-Frame-Options, Referrer-Policy, Permissions-Policy, CSP stricte —
`script-src 'self'` sans exception, `style-src 'self' 'unsafe-inline'`
pour les graphiques Recharts qui posent des styles inline). CSP exemptée
sur `/docs`/`/redoc`/`/openapi.json` (Swagger UI charge ses assets par
CDN) — sans conséquence en production, ces trois routes y sont désormais
désactivées (`docs_url=None`, etc.) : vérifié en direct que `/docs`
répondait 200 sans authentification quel que soit l'environnement avant ce
correctif.

CORS : les origines de dev (`localhost:5173`) ne sont plus incluses
inconditionnellement — uniquement `settings.frontend_url` en production.
Vérifié en direct AVANT correctif qu'un preflight avec
`Origin: http://localhost:5173` était accepté avec
`Access-Control-Allow-Credentials: true`.

**Script inline retiré de `frontend/index.html`** (prévention du flash de
thème) : la CSP stricte (`script-src 'self'`, sans `unsafe-inline`)
l'aurait bloqué silencieusement. Déplacé vers `frontend/public/theme-init.js`,
servi par la même origine — comportement inchangé, juste plus de script
inline dans le HTML.

**Enveloppe d'erreur unifiée** (`http_exception_handler`,
`validation_exception_handler`, `unhandled_exception_handler` dans
`api/main.py`) — avant ce correctif, seules les `HTTPException` levées
explicitement par le code métier portaient `{"detail": {"code",
"message"}}` ; les 404 sans route, les 422 de validation Pydantic et un 500
non géré échappaient à cette convention (vérifié en direct :
`GET /api/does-not-exist` renvoyait `{"detail":"Not Found"}`, sans
`request_id`). Les trois gestionnaires globaux unifient l'enveloppe
partout, ajoutent toujours `request_id` dans le CORPS (pas seulement l'
en-tête `X-Request-ID`), et un 500 non prévu ne renvoie plus jamais
`str(exc)` au client — seulement le `request_id` pour le support,
l'exception complète part dans les logs.

**`X-Request-ID` client validé** (`observability.py::_is_valid_request_id`)
— vérifié en direct qu'un `X-Request-ID: <script>...` fourni par le client
était accepté tel quel et injecté dans les logs JSON sans validation.
Seul un UUID valide est désormais accepté, sinon un nouveau est généré —
même règle que si rien n'avait été fourni. `request.state.request_id`
ajouté en plus du `ContextVar` : nécessaire pour que les 3 gestionnaires
d'erreur globaux (dont celui pour `Exception`, qui s'exécute hors de la
pile de middlewares utilisateur où le `ContextVar` a déjà été réinitialisé)
puissent quand même lire le bon `request_id`.

Testé : `npx tsc -b`/`npx vitest run` (frontend, verts après le retrait du
script inline). Pas de nouveau test backend dédié à ce correctif précis
(en-têtes/CSP) — à couvrir en Phase 5 (CI) via un test d'intégration
dédié plutôt qu'ajouté à la hâte ici ; noté dans « ce qui reste à faire ».

### Décision 6 — Démarrage : validation exhaustive de la config en production (Axe D)

`api/core/database.py`, même pattern que `security.py` pour `JWT_SECRET_KEY`
(hard-fail, pas un log) : refuse de démarrer en production si
`DATABASE_URL` pointe vers SQLite, ou contient encore `CHANGE_ME` (le
placeholder documenté dans `.env.example`/`docker-compose.yml`). Testé en
sous-process (`tests/test_database_startup.py`, même technique que
`tests/test_security.py` — la validation a lieu à l'import du module,
donc invisible à une simple ré-importation dans le process de test déjà
démarré).

### Décision 7 — Mode d'échec du rate-limiting choisi par endpoint (§4)

`api/core/rate_limit.py::is_rate_limited`/`rate_limit_dependency` acceptent
désormais `fail_open: bool`. Conservé ouvert (comportement historique) pour
`/login`/`/register`/l'inscription — la disponibilité de l'authentification
prime. Basculé fermé (`RateLimitBackendUnavailable` → 503, jamais confondu
avec un 429 de limite réellement atteinte) sur les 3 endpoints coûteux déjà
identifiés par l'audit initial : upload tabulaire, upload vision (extraction
ZIP synchrone), `/explain` (charge un modèle torch à chaque appel). Testé :
`tests/test_rate_limit_fail_mode.py` (Redis simulé injoignable des deux
façons).

### Décision 8 — Bug réel trouvé en testant : `token_valid_after` mal comparé sous SQLite

En écrivant `tests/test_token_lifecycle.py::test_password_change_revokes_all_existing_sessions`,
le test échouait : un jeton access émis AVANT un changement de mot de passe
restait accepté après. Cause : SQLite ne conserve pas le fuseau horaire des
colonnes `DateTime(timezone=True)` — `user.token_valid_after` relu depuis la
base est un datetime NAÏF (alors qu'il a été écrit en UTC), et
`naive_dt.timestamp()` l'interprète comme une heure LOCALE (ce poste étant
en UTC+2, la comparaison se retrouvait décalée de 2h, rendant la révocation
inopérante pendant cette fenêtre). **Bug déjà connu et déjà corrigé
ailleurs dans ce dépôt** (`domains/shared/job_watchdog.py::_as_aware_utc`,
`domains/training/services/prediction_retention.py`) — le même correctif
(normaliser en UTC avant `.timestamp()`) a été dupliqué dans
`domains/auth/router.py` plutôt que factorisé (éviterait un couplage
`auth`→`shared/job_watchdog.py` sans rapport, cohérent avec la duplication
déjà acceptée entre les deux fichiers existants). **Preuve concrète que
« rien n'est fait tant que ce n'est pas prouvé » (principe 1) n'est pas une
formule** : ce bug n'aurait jamais été vu sans le test qui reproduit le
scénario exact demandé par la mission (Phase 1B, point 1 : « l'utilisateur
qui change son mot de passe parce qu'il se croit compromis doit pouvoir
chasser quiconque »). Confirmé en PostgreSQL non nécessaire ici : le
correctif fonctionne dans les deux cas (`dt.tzinfo is not None` → no-op).

### Décision 9 — Suite de régression IDOR consolidée (Axe B)

`tests/test_idor_regression.py` (9 tests) — mission : « un test paramétré
qui, pour CHAQUE route à ressource, vérifie qu'un utilisateur d'une autre
organisation reçoit 404 ». L'audit délégué (Phase 0) a déjà vérifié les 110
routes manuellement, sans trouver d'IDOR. Plutôt que dupliquer 110
assertions individuelles (chaque domaine délègue à UN SEUL helper interne
`_get_org_dataset`/`_get_org_job`, donc un test par route n'apporterait
aucune garantie supplémentaire par rapport à un test par domaine), ce
fichier teste l'endpoint « détail » de CHAQUE domaine à ressource
(datasets, training, clustering, dimensionality, anomalies, vision
datasets, vision classification, vision anomalies) — un proxy fidèle de
toutes les routes du même domaine qui partagent le même helper, et surtout
un garde-fou qui couvre AUTOMATIQUEMENT un futur domaine ajouté au backend,
contrairement aux tests d'isolation déjà dispersés dans chaque
`test_<domaine>_api.py` (qui existent et restent en place, non
supprimés). Un test dédié vérifie aussi explicitement l'absence de 403
(qui confirmerait l'existence de la ressource à l'attaquant).

Réutilise les créateurs de zip de test déjà écrits (`_classification_zip_bytes`,
`_mvtec_zip_bytes`) plutôt que de les dupliquer une troisième fois — import
direct depuis les modules de test existants (mode d'import non-package de
pytest sur ce dépôt, `pythonpath = .`, confirmé dans `pytest.ini`).

## Phase 1B — Réinitialisation de mot de passe

### Décision 10 — Mécanisme repris de CIAM, durci sur les 7 points prévus

`api/core/models.py::PasswordResetToken` (migration `c274f8e19a3b`),
`api/core/mailer.py`, `domains/auth/router.py` (`POST
/auth/password-reset/request`/`confirm`) — jeton `secrets.token_urlsafe(32)`
haché SHA-256, usage unique, un seul actif par compte, réponse `request`
strictement neutre (204 sans corps, y compris rate-limité), envoi mail en
tâche de fond (`BackgroundTasks`).

Les 7 durcissements demandés, tous implémentés :
1. **Révocation de toutes les sessions à la confirmation** —
   `user.token_valid_after = now()` + `revoke_all_refresh_tokens()`.
   Impossible chez CIAM (JWT stateless), possible ici grâce au cycle de vie
   construit en Phase 1.
2. **Limite par compte en plus de l'IP** — `password_reset_email:<email>`,
   même réponse neutre dans les deux cas. IP réelle via `get_client_ip`
   (Phase 1, pas `request.client.host`).
3. **Journalisé dans `AuditLog`** — `auth.password_reset_requested`,
   `auth.password_reset_confirmed`, `auth.login`, `auth.logout`,
   `auth.password_changed` (ces 3 derniers ajoutés au passage, fermant une
   partie du constat I1 de l'audit Phase 0).
4. **Purge des jetons expirés** — `_purge_expired_password_reset_tokens`,
   même patron que `prediction_retention.py` (purge à la demande, pas de
   scheduler dédié), appelée à chaque `request`.
5. **Robustesse du mot de passe partagée** — `api/core/password_policy.py`
   (nouveau, un seul point de vérité), appelée par `RegisterRequest`,
   `TeamMemberCreate`, `change_own_password`, et `confirm_password_reset` —
   les 4 chemins où un mot de passe est choisi.
6. **Mail avec date/IP + second mail de notification** —
   `send_password_reset_email` inclut l'IP demandeuse ;
   `send_password_changed_notification_email` (nouveau, absent de CIAM)
   envoyé après un changement EFFECTIF, qu'il vienne d'une réinitialisation
   ou d'un changement volontaire (`change_own_password` aussi mis à jour).
7. **Testé** — `tests/test_password_reset.py` (10 tests), dont
   `test_request_reset_returns_204_for_known_and_unknown_email_identically`
   (corps identique) et `test_request_reset_response_time_does_not_leak_account_existence`
   (ordre de grandeur du temps de réponse, marge ×10 pour rester robuste en
   CI sans être un test de charge).

**Config** : mêmes noms de variable que CIAM (`SMTP_HOST`/`SMTP_PORT`/
`SMTP_USER`/`SMTP_PASSWORD`/`PASSWORD_RESET_EXPIRE_MINUTES`) — même bloc
`.env` valable des deux côtés. Canal optionnel au démarrage ; avertissement
(pas un hard-fail) si absent en production (`api/main.py::lifespan`) — un
hard-fail aurait été disproportionné (le reste de l'API n'a besoin d'aucune
config mail pour fonctionner), mais un silence total aurait été pire qu'un
avertissement pour un opérateur qui déploie sans y penser.
`backend/.env.example` mis à jour avec ce bloc — et, au passage, les 19
autres variables de durcissement Phase 1/1.4 qui manquaient déjà (constat
D de l'audit Phase 0), toutes documentées avec leur valeur par défaut réelle.

### Décision 11 — Bugs réels trouvés en écrivant les tests (Phase 1B)

Deux bugs distincts, tous deux révélés uniquement par l'exécution réelle de
`tests/test_password_reset.py`, jamais visibles à la lecture du code :

1. **`NameError: name 'logger' is not defined`** — `domains/auth/router.py`
   n'avait jamais eu de `logger` défini (le fichier n'en avait pas besoin
   avant la Phase 1B). Les 3 nouveaux appels `logger.exception`/`logger.info`
   du bloc réinitialisation levaient une `NameError` **à l'intérieur d'une
   tâche de fond** — invisible pour l'appelant HTTP (déjà répondu 204), donc
   silencieuse en usage normal, visible seulement parce que le gestionnaire
   d'erreur global (Phase 1, Axe E) journalise désormais tout `[UNHANDLED]`.
   Corrigé (`import logging` + `logger = logging.getLogger("datalab.auth")`).
2. **`backend/.env` local contient de VRAIS identifiants SMTP Gmail** — la
   suite de tests a réellement tenté de se connecter à `smtp.gmail.com`
   (échec d'authentification, mais requête réseau réelle, lente,
   dépendante d'Internet) parce que `conftest.py` ne surchargeait pas
   `SMTP_*` comme il le fait déjà pour `DATABASE_URL`/`JWT_SECRET_KEY`.
   **Une suite de tests ne doit jamais dépendre d'un service tiers réel**,
   quel que soit le `.env` local du poste qui l'exécute — corrigé en
   ajoutant `SMTP_HOST`/`SMTP_USER`/`SMTP_PASSWORD` vides à la liste des
   surcharges de `conftest.py`. Temps d'exécution de la suite passé de
   159 s à 28 s après ce correctif — mesure directe de l'impact.

Ces deux bugs n'auraient pas été détectables par une revue de code seule
(le premier ne se déclenche que si l'exception attrapée est réellement
levée ; le second dépend d'un `.env` local que le code lui-même ne peut
pas connaître) — confirmation supplémentaire du principe 1 (« rien n'est
fait tant que ce n'est pas prouvé »).

### Décision 12 — Frontend : 3 écrans + garde-fou de cohérence session

`frontend/src/pages/ForgotPassword.tsx` (nouveau), `ResetPassword.tsx`
(nouveau), `Profile.tsx::ChangePasswordForm` (modifié) — voir aussi
`Login.tsx` (lien « Mot de passe oublié ? », bannière `password_changed=1`).

Point non prévu explicitement par la mission mais découlant directement de
la Phase 1 : après un changement de mot de passe volontaire, le jeton
access de la session EN COURS est immédiatement invalidé côté serveur
(`token_valid_after`). Sans action côté client, l'utilisateur resterait sur
l'écran Profil avec un jeton déjà mort, jusqu'au prochain appel API qui
échouerait en 401 de façon peu compréhensible. Corrigé : `ChangePasswordForm`
appelle `logout()` puis redirige vers `/login?password_changed=1`
immédiatement après un changement réussi — cohérent avec l'avertissement
affiché avant validation (« toutes vos sessions ouvertes seront fermées, y
compris celle-ci »), qui serait resté un simple texte sans cette
conséquence réellement appliquée.

Testé : `npx tsc -b` et `npx eslint` verts (0 erreur, 1 avertissement
préexistant sans rapport sur `AuthContext.tsx`).

### Décision 4 — `/feedback` : rôle documenté aligné sur le rôle appliqué (Axe B)

`GET /feedback` passe de `get_current_user` à `require_owner`, conforme au
commentaire du module qui promettait déjà cet accès réservé depuis le
Lot 10. Pas un IDOR (le filtre `organization_id` était déjà correct), un
rôle mal appliqué. Aucun test existant ne couvrait cet endpoint (vérifié
avant modification), aucune régression possible.
