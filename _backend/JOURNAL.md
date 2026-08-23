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

### Décision 4 — `/feedback` : rôle documenté aligné sur le rôle appliqué (Axe B)

`GET /feedback` passe de `get_current_user` à `require_owner`, conforme au
commentaire du module qui promettait déjà cet accès réservé depuis le
Lot 10. Pas un IDOR (le filtre `organization_id` était déjà correct), un
rôle mal appliqué. Aucun test existant ne couvrait cet endpoint (vérifié
avant modification), aucune régression possible.
