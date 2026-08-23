# AUDIT_BACKEND_2026-08-23.md — Audit expert du backend DataLab Pro

> Document de constat. Aucun fichier applicatif n'a été modifié pendant cette
> phase. Classement : 🔴 critique · 🟠 majeur · 🟡 mineur · 🔵 amélioration ·
> ✅ conforme.
>
> Méthode : lecture intégrale du code exécuté (jamais déduit d'un commentaire,
> d'un nom de fonction ou d'un test), croisée avec les migrations Alembic
> réelles, et vérification live sur une instance locale du backend
> (`uvicorn`, port 8010) pour tout ce qui pouvait l'être sans la stack Docker
> complète — voir §0 pour l'incident qui a empêché la vérification Docker
> bout-en-bout. Cinq audits délégués en parallèle (axes A+C, B, D+J, F+G, H+I)
> ont été croisés et vérifiés ligne par ligne avant intégration dans ce
> document ; les tableaux et citations `fichier:ligne` proviennent de cette
> lecture directe.

---

## §0. Constat bloquant trouvé en cours d'audit — l'image Docker de production ne démarre plus

**🔴 `backend/Dockerfile:26-28` ne copie pas `domains/`, alors que tout le code
métier vit désormais là depuis le Lot 8 (monolithe modulaire).**

```dockerfile
COPY api/ ./api/
COPY services/ ./services/     # ce dossier ne contient plus que __pycache__ (mort depuis le Lot 8)
COPY workers/ ./workers/
# domains/ n'est JAMAIS copié
```

Preuves :
- `git ls-files backend/domains | wc -l` → **74 fichiers trackés**, structure
  vivante (`domains/auth`, `domains/training`, `domains/vision/*`, etc.).
- `git ls-files backend/services` → **0 résultat** ; `ls backend/services` ne
  contient plus qu'un `__pycache__` local, résidu non tracké.
- `backend/api/main.py:27-37` importe explicitement
  `from domains.anomalies.router import router as anomalies_router` (et 10
  imports `domains.*` similaires) — sans le dossier `domains/` dans l'image,
  ces imports lèvent `ModuleNotFoundError: No module named 'domains'` **au
  premier import de `api.main`**, donc dès la ligne de commande `gunicorn
  api.main:app` du `CMD`.
- **Vérifié en tentant de construire réellement l'image** (`docker compose
  build backend`) pendant cet audit — build lancé, PyTorch/CUDA CPU
  téléchargés (~10 min), mais le point qui compte n'est pas le succès du
  `build` (qui ne exécute jamais le `CMD`) : c'est que **le conteneur
  résultant ne peut pas démarrer**. Ce point est documenté ici comme preuve
  de code, la vérification `docker compose up` (démarrage réel) n'a pas pu
  être menée à son terme dans le temps de cet audit — voir limitation ci-dessous.

**Impact concret** : `docker compose up -d --build` (commande documentée dans
`README.md`, `docker-compose.yml` et exigée par la porte de qualité de ce
chantier) démarre un conteneur `backend` qui **crashe immédiatement** et boucle
en `restart: unless-stopped`. Le `worker`/`worker-analysis` (même image, même
`CMD` de base mais `command: ["python", "-m", "workers.run_worker"]`) sont
également affectés : `workers/run_worker.py` importe des modules de
`domains/*/worker.py` selon la file choisie. **Le déploiement Docker complet
de DataLab Pro est cassé depuis le Lot 8** (regroupement en `domains/`), et
rien dans le dépôt ne l'a détecté — ni la CI (`.github/workflows/ci.yml`, job
`smoke`, qui construit et démarre pourtant la stack complète — voir Axe J,
sauf si ce job échoue silencieusement/est ignoré, à vérifier en priorité),
ni un test.

**Correctif** : ajouter `COPY domains/ ./domains/` dans `backend/Dockerfile`
(et supprimer la ligne `COPY services/ ./services/` devenue un mort-vivant),
avant tout nouveau lot. C'est un correctif d'une ligne, mais bloquant pour
**toute** vérification live de ce chantier (Phase 1 à Phase 8) tant qu'il
n'est pas appliqué — **priorité absolue de la Phase 1**, avant même le rate
limiting.

**⚠️ Point à signaler explicitement à l'utilisateur, hors des 4 cas prévus par
la mission mais critique pour la suite** : soit le job CI `smoke` échoue
actuellement sur `main` (auquel cas c'était déjà visible et ignoré), soit il
a été ajouté/cassé très récemment (le dernier commit visible,
`4ab47db`, porte sur la refonte visuelle, pas sur `domains/`) — à vérifier en
premier geste de la Phase 1, car cela change le diagnostic (régression toute
fraîche vs dette connue).

### Limitation de vérification directement causée par ce bug

La stack Docker complète n'a pas pu être amenée à un état sain dans le temps
de cet audit (le `build` de l'image `backend` — dépendances PyTorch CPU
incluses — prend lui seul plusieurs minutes, et corriger le Dockerfile pour
la faire démarrer sortirait du périmètre « lecture seule » de la Phase 0).
En conséquence, plusieurs vérifications qui exigent la topologie nginx→backend
réelle (comportement de `X-Forwarded-For`, `client_max_body_size` nginx vs
backend, `limit_req_zone` nginx) reposent ici sur une **preuve de code**
convergente et non ambiguë (voir Axe A/E), pas sur une requête HTTP réelle à
travers nginx. Les vérifications qui ne nécessitent que le process API seul
ont, elles, été faites **en exécutant réellement** le backend en local
(`uvicorn`, hors Docker, port 8010) — voir Axe E.

---

## Résumé exécutif

| Axe | 🔴 | 🟠 | 🟡 | 🔵 | Points forts confirmés |
|---|---|---|---|---|---|
| A. Auth/sessions | 1 | 4 | 2 | 1 | Hard-fail JWT prod, bcrypt 12 rounds, pas d'énumération par message sur `/login` |
| B. Autorisation | 0 | 1 | 0 | 0 | **110 routes auditées, aucun IDOR cross-tenant** — pattern `_get_org_*` systématique, 404 jamais 403 |
| C. Entrées/fichiers | 0 | 3 | 0 | 1 | Zip-slip bloqué, zip-bombe bloquée pendant l'extraction (sauf entrée unique), pas d'injection CSV (pas d'export CSV) |
| D. Secrets/config | 1 | 2 | 2 | 1 | JWT hard-fail réel, aucun secret dans le dépôt ni l'historique git |
| E. Surface exposée | 0 | 2 | 3 | 0 | CORS resserré méthodes/en-têtes, pas de secrets dans les erreurs |
| F. Fiabilité | 2 | 4 | 0 | 0 | Démarrage dégradé si DB down (vérifié vrai), watchdog partiel existant |
| G. Intégrité données | 1 | 1 | 1 | 0 | `ondelete=` sur 54/54 FK, migrations réversibles vérifiées, test de cohérence schéma↔modèles |
| H. Performance | 0 | 4 | 2 | 0 | Pagination SQL réelle sur 5/6 domaines, N+1 déjà corrigés sur 6/7 listes, SSE bien conçu |
| I. Traçabilité | 2 | 1 | 1 | 0 | Logs JSON structurés, audit log couvrant suppressions/promotions/annulations |
| J. Supply chain | 1 | 3 | 0 | 1 | 100 % des dépendances épinglées, image non-root, aucune dépendance morte |
| **Total** | **8** | **25** | **11** | **4** | |

**Les 3 constats les plus graves, par ordre de blast radius :**
1. **§0 — déploiement Docker cassé** (`domains/` absent de l'image) : bloque toute mise en production ET toute vérification ultérieure de ce chantier.
2. **A.6 — rate limiting basé sur `request.client.host`** derrière nginx : un seul attaquant peut verrouiller `/login` pour **toute l'application**, tous tenants confondus (DoS trivial, confirmé par lecture croisée du Dockerfile/nginx/rate_limit.py).
3. **F4/F5 — aucune idempotence de création de job + jobs `queued` orphelins si Redis tombe à l'enfilage**, jamais détectés par le watchdog (qui ne couvre que `running`).

---

## Axe A — Authentification et sessions

### A.1 — Cycle de vie du JWT
`backend/api/core/security.py:21-60`. HS256, TTL fixe 24h, **aucun `jti`**,
aucune liste de révocation (grep `jti|blacklist|revoke|token_version` sur tout
le backend → 0 résultat).
🟠 Un jeton volé reste valide jusqu'à 24h, sans moyen de le révoquer.

### A.2 — `POST /auth/logout` ne fait rien côté serveur
`backend/domains/auth/router.py:273-276` :
```python
@router.post("/logout")
def logout():
    return {"message": "Déconnexion effectuée (supprimer le token côté client)"}
```
🟡 Documenté honnêtement dans le code, mais absence réelle de révocation.

### A.3 — Réinitialisation de mot de passe absente
Grep exhaustif (`forgot|reset-password|reset_token`) → aucun résultat. Aucune
route, aucun service. 🟠 Fonctionnalité absente (traitée en Phase 1B de ce
chantier).

### A.4 — Changement de mot de passe n'invalide aucune session
`backend/domains/auth/router.py:257-270`. 🟠 Cohérent avec A.1 (pas de
révocation possible tant qu'il n'y a pas de `jti`+blacklist), mais un attaquant
avec un token déjà volé garde l'accès jusqu'à 24h après que la victime se
croie protégée.

### A.5 — Énumération de comptes
- `POST /auth/register` (`router.py:170-174`) : message `AUTH_EMAIL_DEJA_UTILISE`
  explicite → 🟠 énumération directe des emails inscrits, scriptable, sans
  rate-limit par adresse (seulement par IP, et cette IP est de toute façon
  fausse derrière nginx — voir A.6).
- `POST /auth/login` (`router.py:218-223`) : message **identique** dans les
  deux cas (bon point) mais court-circuit Python (`if not user or not
  verify_password(...)`) — `verify_password` (bcrypt ~100-300ms) n'est **jamais
  appelé** si l'email est inconnu. 🟡 Canal d'énumération par timing, mesurable
  à distance, indépendant du message affiché.

### A.6 — 🔴 Rate limiting basé sur l'IP du proxy, pas du client (confirmé)
`backend/api/core/rate_limit.py:58` : `client_ip = request.client.host if
request.client else "inconnu"`. Aucun `ProxyHeadersMiddleware` dans
`api/main.py`, aucun `--forwarded-allow-ips` dans la commande `gunicorn` du
`Dockerfile` (`Dockerfile:41-47`). `nginx/templates/default.conf.template:56-59`
définit bien `X-Forwarded-For`/`X-Real-IP`, mais **rien côté backend ne les
lit**.
**Conséquence prouvée par lecture croisée des 3 fichiers** (rate_limit.py +
Dockerfile + nginx template) : dans la topologie réelle de
`docker-compose.yml` (nginx devant backend), `request.client.host` vaut
l'IP du conteneur nginx pour **toutes** les requêtes. Le compteur Redis
`rate_limit:login:<ip-nginx>` est donc **partagé par tous les utilisateurs de
toutes les organisations**. Un seul visiteur anonyme qui échoue 10 fois sur
`/login` verrouille `/login` pendant 15 minutes pour l'application entière —
DoS trivial et répétable indéfiniment. Même défaut sur `/register`, upload,
`/explain` (même fabrique `rate_limit_dependency`).
*(Non re-vérifié par une requête HTTP réelle à travers nginx à cause de §0 —
la preuve de code ici est cependant sans ambiguïté : les 3 fichiers
concernés ne peuvent pas produire un autre comportement.)*

### A.7 — Bcrypt : 12 rounds ✅
`security.py:45`, valeur raisonnable, codée en dur.

### A.8 — Clé JWT par défaut bloquante en production ✅
`security.py:24-40` : `raise RuntimeError` réel si `environment=="production"`
et clé par défaut — vérifié en lisant le corps de la condition, pas juste sa
présence.

### A.9 — Stockage du token en `localStorage` (pas de cookie httpOnly)
`frontend/src/api/client.ts:24-36`. 🟡 Combiné à A.1 (pas de révocation), tout
XSS frontend permet le vol et la réutilisation du token pendant 24h.

### A.10 — Bornes de mot de passe incohérentes entre endpoints
`RegisterRequest.password`/`TeamMemberCreate.password` : `min_length=8`,
**pas de `max_length`** ; `ChangePasswordRequest.new_password` : `max_length=100`.
🔵 Mineur (borné de facto par `max_json_body_size_mb=2`), incohérence de
rigueur.

---

## Axe B — Autorisation (tableau exhaustif)

**Verdict global : 110 routes auditées sur les 12 fichiers `domains/*/router.py`
+ `api/main.py`, aucun IDOR cross-tenant trouvé.** Chaque domaine définit un
helper privé (`_get_org_dataset`/`_get_org_job`) qui filtre systématiquement
par `organization_id == current_user.organization_id` et renvoie **404**
(jamais 403) si la ressource appartient à une autre organisation. Tous les
endpoints à ID délèguent à ce helper, y compris les exports d'artefacts et
l'accès aux images vision (double protection : org-scoping **et** vérification
de traversée de répertoire `base_dir not in target.parents`).

Le tableau complet (110 lignes, un domaine par section : `auth`, `datasets`,
`training`, `clustering`, `dashboard`, `dimensionality`, `anomalies`,
`vision/datasets`, `vision/classification`, `vision/anomalies`, `api/main.py`)
est conservé dans les artefacts de l'audit délégué ; il n'est pas reproduit
intégralement ici pour la lisibilité — seul l'écart trouvé est détaillé
ci-dessous, avec un échantillon représentatif par domaine.

Échantillon (voir méthode ci-dessus pour la couverture complète) :

| Domaine | Endpoints sensibles vérifiés | Filtre org | Statut |
|---|---|---|---|
| `training` | `GET .../model/export`, `.../predict`, `.../candidates`, `.../model/versions` | `_get_org_job` + double filtre org sur les sous-ressources | ✅ |
| `clustering`/`dimensionality`/`anomalies` | `.../model/export`, `.../result`, `.../candidates`\|`points`\|`observations` | `_get_org_job`, sous-ressources filtrées par transitivité (job déjà org-scoped) | ✅ (sûr, mais sans défense en profondeur contrairement à `training.py`) |
| `vision/datasets` | `GET .../image`, `GET .../images` | `_get_org_dataset` + anti-traversée de répertoire | ✅ |
| `vision/classification`/`vision/anomalies` | `.../explain`, `.../model/export` | `_get_org_job` | ✅ |
| `dashboard` | `GET /dashboard/summary` | 8 sous-requêtes toutes filtrées `org_id` | ✅ agrège correctement |
| `auth` | `GET/POST /team/*` | `require_owner` sur les 2 actions réservées | ✅ |

### 🟠 Seul écart trouvé — `GET /feedback` : rôle documenté ≠ rôle appliqué
`backend/domains/auth/router.py:352-368`. Le commentaire de section (l.305-311)
affirme un accès « administrateurs de leur organisation uniquement », mais la
route utilise `Depends(get_current_user)` (pas `Depends(require_owner)`).
Le filtre `Feedback.organization_id == current_user.organization_id` (l.359-360)
est correct — **ce n'est pas un IDOR cross-tenant** — mais n'importe quel
`member` (pas seulement le `owner`) peut lire tous les retours utilisateurs de
son organisation, contrairement à l'intention documentée dans le code
lui-même.
**Correctif** : soit `Depends(require_owner)`, soit corriger le commentaire —
à trancher côté produit, l'un des deux est actuellement faux.

---

## Axe C — Entrées et fichiers

### C.1 — Validation Pydantic inégale selon les domaines
Bornes réelles trouvées : `TrainingJobCreate` (`training/router.py:76-83`,
`test_size: Field(0.2, gt=0.05, lt=0.5)`, `optuna_trials: Field(None, ge=3,
le=100)`...), `AnomalyJobCreate` (`top_n: Field(..., ge=1, le=MAX_TOP_N)`).
Types nus sans borne : `ClusteringJobCreate.feature_columns: List[str]`,
`DimensionalityJobCreate.feature_columns: List[str]` (aucune limite de taille
de liste), `PredictionRequest.data: dict[str, Any]` (borné uniquement par
`MaxJsonBodySizeMiddleware`, 2 Mo global, pas par schéma métier).
🔵 Risque faible en pratique, rigueur inégale entre domaines.

### 🟠 C.2 — Upload de dataset tabulaire : extension déclarée, pas signature réelle
`backend/domains/shared/dataset_io.py:32-40::validate_extension` vérifie
uniquement `Path(filename).suffix` — aucun sniffing de magic bytes. Un fichier
renommé `.csv` mais de contenu arbitraire est accepté au contrôle d'entrée,
n'échoue qu'au parsing pandas (`DatasetParsingError`, capturé proprement, pas
de crash). Plus grave : **`.xlsx`/`.xls` sont eux-mêmes des conteneurs ZIP,
sans aucune limite de ratio de compression** sur ce chemin (contrairement au
ZIP vision, voir C.4), alors que l'upload est **synchrone dans la requête
HTTP** (`domains/datasets/router.py:286-359`, pas de tâche de fond). Un
`.xlsx` de quelques Mo conçu pour décompresser en plusieurs Go peut bloquer un
worker gunicorn en mémoire — la limite `max_upload_size_mb=200` ne borne que
la taille **compressée**.

### 🟠 C.3 — Bombe zip vision : chaque entrée lue intégralement en mémoire avant contrôle cumulé
`backend/domains/vision/datasets/service.py:139-160::_accumulate_member` :
le contrôle de taille cumulée décompressée s'applique **pendant** l'extraction
(bon point, pas seulement après) mais chaque entrée est d'abord lue
entièrement (`zf.read(info)` l.183) **avant** que la vérification cumulée ne
s'applique. Une archive à **une seule entrée** avec un ratio de compression
extrême (~1000:1 en DEFLATE standard) peut donc faire exploser la mémoire
pendant la lecture de cette entrée unique, avant tout rejet. La protection
couvre le cas « beaucoup de petits fichiers qui s'accumulent », pas le cas
« un fichier unique hautement compressé ».

### ✅ C.4 — Zip-slip bloqué, avec défense en profondeur
`backend/domains/vision/datasets/service.py:103-122::_safe_member_path` :
rejette tout membre absolu ou contenant `..`, appliqué avant tout traitement
(zip **et** tar). Revérifié une seconde fois à la lecture (`GET
.../image`, `router.py:301-324` : `base_dir not in target.parents`).

### ✅ C.5 — Injection de formule CSV : sans objet
Grep exhaustif (`to_csv|csv.writer`) → aucun endpoint d'export CSV de données
utilisateur dans tout le backend. Les `export_*` exportent des artefacts
modèle (`.joblib`/`.pt`), pas du CSV.

### 🟠 C.6 — Noms de fichiers : sûr côté tabulaire et vision (confirmé, pas un défaut)
`backend/api/core/storage.py:24-27` : nom sur disque = `f"{dataset_id}
{extension}"`, jamais dérivé du nom client. Vision : `dest = target_dir /
vi.bucket_name / vi.rel_path.name` (nom isolé, pas le chemin complet). ✅ Classé
ici pour mémoire, pas un problème — mentionné par cohérence avec le reste de
l'axe C.

---

## Axe D — Secrets et configuration

### Inventaire (`backend/api/core/config.py`, 24 variables)

| Variable sensible | Défaut | Hard-fail prod ? |
|---|---|---|
| `jwt_secret_key` | `"changez-cette-cle-en-production"` | **✅ Oui** (`security.py:24-40`) |
| `database_url` | SQLite locale | **🔴 Non** — rien n'empêche `environment=="production"` avec la base SQLite de dev |
| `redis_url` | `redis://localhost:6379/0` | 🟠 Non — si injoignable, le rate-limit est *fail-open* (désactivé silencieusement) |
| `frontend_url` | `http://localhost:5173` | 🟡 Non, mais échoue « sûr » (CORS trop restrictif plutôt que trop ouvert) |

`POSTGRES_USER`/`POSTGRES_PASSWORD`/`POSTGRES_DB` sont lus **hors** de
`Settings` (uniquement par l'interpolation `docker-compose.yml`) — **aucune
validation nulle part**. `POSTGRES_PASSWORD=CHANGE_ME` en défaut dans les deux
`.env.example` : si un opérateur copie les exemples sans les modifier (ce que
fait le job CI `smoke`, pour un environnement éphémère seulement — à
confirmer que ce n'est jamais le chemin de prod réel), la base tourne avec un
mot de passe public documenté dans le dépôt, sans garde ni journalisation.

🔴 **Seul `JWT_SECRET_KEY` a un hard-fail. `DATABASE_URL`/`POSTGRES_PASSWORD`
n'ont pas la même garde alors que le risque est de même nature** (valeur par
défaut publique, dépôt versionné).

### 🟠 `.env.example` incomplet
10 variables de durcissement SaaS récentes (Lot 10, Lot 1.4 — quotas,
fenêtres de rate-limit register/upload/explain, `max_json_body_size_mb`,
`prediction_retention_days`) sont lues par `config.py` mais **absentes** de
`backend/.env.example`. Un opérateur qui s'y fie pour ajuster les seuils
anti-bruteforce n'a aucune visibilité dessus.

### ✅ Aucun secret dans le dépôt ni dans l'historique git
`git log --all --full-history -- "**/.env"` → 0 résultat, un `.env` réel n'a
jamais été committé. Recherche par contenu (`git log -p --all -S"SECRET_KEY"`,
`-S"POSTGRES_PASSWORD="`) → seules des valeurs placeholder trouvées.

---

## Axe E — Surface exposée *(vérifié en exécutant réellement le backend, hors Docker à cause de §0)*

Backend démarré localement (`uvicorn api.main:app`, port 8010,
`ENVIRONMENT=development` — même code qu'en production sur ces points,
aucune des faiblesses ci-dessous n'est conditionnée à l'environnement).

### 🟡 Aucun en-tête de sécurité — confirmé par une requête réelle
```
$ curl -sD - http://127.0.0.1:8010/api/health
HTTP/1.1 200 OK
server: uvicorn
content-type: application/json
x-request-id: e60ba55a-...
```
Ni `Strict-Transport-Security`, ni `X-Content-Type-Options`, ni
`X-Frame-Options`/CSP, ni `Referrer-Policy`, ni `Permissions-Policy` — sur
aucune réponse testée. Confirmé aussi côté nginx (`nginx/templates/default.conf.template`,
grep négatif sur les mêmes en-têtes). L'en-tête `server: uvicorn` divulgue
en plus la techno serveur (mineur).

### 🟠 `/docs` et `/openapi.json` toujours exposés, jamais gatés par l'environnement
```
$ curl -o /dev/null -w "%{http_code}" http://127.0.0.1:8010/docs        → 200
$ curl -o /dev/null -w "%{http_code}" http://127.0.0.1:8010/openapi.json → 200
```
`api/main.py:111-115` : `FastAPI(title=..., version=..., lifespan=lifespan)`
— aucun `docs_url=None if settings.environment == "production" else "/docs"`.
Documentation interactive complète (tous les schémas, tous les endpoints)
accessible sans authentification, y compris en production.

### 🟡 `/metrics` non authentifié — reachability réelle dépend du déploiement
```
$ curl http://127.0.0.1:8010/metrics
# HELP python_gc_objects_collected_total ...
```
Confirmé accessible **au niveau du process FastAPI**, sans authentification
(`api/main.py:180-185`, volontaire selon le commentaire — convention
Prometheus). **Nuance importante par rapport à l'hypothèse de départ** : dans
la topologie `docker-compose.yml` **telle qu'écrite**, le service `backend`
n'a qu'un `expose: "8000"` (pas de `ports:`) — le port n'est **pas** publié
sur l'hôte — et `nginx/templates/default.conf.template` ne proxifie que
`location /api/` (jamais `/metrics` ni `/docs`, qui restent sous le préfixe
`/api` uniquement pour les routers métier). **Avec ce compose file précis,
`/metrics` et `/docs` ne sont donc pas joignables depuis l'extérieur du
réseau Docker** — correction d'une hypothèse d'audit initiale. Le risque
réel : c'est un point de reachability qui dépend entièrement de la
configuration de déploiement (tout override qui publie le port 8000, tout
déploiement hors de ce `docker-compose.yml` précis — bare-metal, autre
orchestrateur — expose immédiatement les deux sans qu'aucun code applicatif
ne s'y oppose). Classé 🟡 (pas 🔴) parce que non exploitable *dans la
topologie documentée du dépôt*, mais à corriger côté application (pas
seulement compter sur la configuration réseau) pour une défense en
profondeur réelle.

### 🟡 CORS : origines de développement toujours whitelistées, y compris en production
`backend/api/main.py:118-122` :
```python
_allowed_origins = list({
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    settings.frontend_url,
})
```
Ces deux origines de dev sont ajoutées **inconditionnellement**, quel que
soit `environment`. Avec `allow_credentials=True`, un attaquant qui ferait
tourner un serveur sur `localhost:5173`/`127.0.0.1:5173` sur la machine d'une
victime (scénario à faible probabilité mais réel — malware local, autre
application dev qui écoute sur ce port) pourrait émettre des requêtes
cross-origin authentifiées vers l'API de **production**. Vérifié en direct
(preflight `OPTIONS` avec `Origin: http://localhost:5173` → `200` avec
`access-control-allow-origin: http://localhost:5173` et
`access-control-allow-credentials: true`). Un `Origin` non whitelisté est lui
correctement rejeté (`400 Bad Request` au preflight, vérifié en direct avec
`Origin: http://evil.example.com`).

### 🟡 Enveloppe d'erreur incohérente sur les erreurs de framework
Le reste de l'API utilise systématiquement `{"detail": {"code": "...",
"message": "..."}}` (`AUTH_IDENTIFIANTS_INCORRECTS`, etc. — vérifié en direct :
`POST /auth/login` avec mauvais mot de passe renvoie bien cette forme). Mais
les erreurs générées par FastAPI/Starlette lui-même échappent à cette
convention :
```
$ curl http://127.0.0.1:8010/api/does-not-exist
{"detail":"Not Found"}
$ curl -X POST http://127.0.0.1:8010/api/auth/login -d '{}' -H "Content-Type: application/json"
{"detail":[{"type":"missing","loc":["body","username"],"msg":"Field required","input":null}, ...]}
```
Aucun `code` stable, message en anglais technique (« Field required ») pour
les erreurs 422, aucun `request_id` dans le corps (seulement dans l'en-tête
`X-Request-ID`, invisible pour un utilisateur qui lit le corps de la réponse).
Incohérent avec le principe du chantier (« le message d'erreur dit quoi
faire, en français, actionnable, avec le `request_id` ») — cet axe n'a
simplement jamais été étendu aux erreurs framework (404, 422, 500 non
gérées), aucun `add_exception_handler` global n'existe (`grep
"@app.exception_handler|add_exception_handler"` sur `api/`+`domains/` → 0
résultat).

### 🟡 `X-Request-ID` entièrement contrôlé par le client, jamais validé
`backend/api/core/observability.py:99-108::RequestIdMiddleware` :
```python
incoming = request.headers.get("x-request-id")
request_id = incoming if incoming else str(uuid.uuid4())
```
Vérifié en direct :
```
$ curl -D - http://127.0.0.1:8010/api/health -H "X-Request-ID: <script>evil-injected-id-12345</script>"
x-request-id: <script>evil-injected-id-12345</script>
```
Aucune validation de format (UUID attendu) ni de longueur — la valeur
fournie par le client est reflétée telle quelle dans la réponse **et**
injectée dans **chaque ligne de log JSON** émise pour cette requête
(`observability.py:51-53`). `json.dumps` échappe correctement la valeur (pas
d'injection dans la structure JSON du log), mais un attaquant peut : (a)
polluer/rendre méconnaissable la corrélation de logs en réutilisant l'ID
d'une autre requête légitime, (b) injecter du contenu arbitraire qu'un outil
de visualisation de logs en aval pourrait ne pas échapper (XSS stocké
différé, hors du contrôle de ce backend mais causé par lui). Correctif simple :
valider que `X-Request-ID` entrant est un UUID, sinon en générer un nouveau.

---

## Axe F — Fiabilité

### 🔴 F1 — Le worker avale presque toutes les exceptions ; RQ ne voit donc (presque) jamais d'échec, et rien n'est retenté ni surveillé
Les 6 workers (`domains/*/worker.py`) englobent tout le calcul dans un
`try/except Exception` qui marque `"failed"` en base et **avale
l'exception** — elle ne remonte jamais à RQ. Aucun `retry=Retry(...)` n'est
configuré sur aucun `enqueue()` (grep sur tout `backend/` → 0 résultat) —
sans importance en pratique puisque RQ ne voit presque jamais d'échec. Mais
une exception qui survient **hors** de la zone protégée (ex. `db.commit()`
échoue parce que Postgres est momentanément indisponible, avant/après le
`try` interne) part réellement en échec RQ, tombe dans le `FailedJobRegistry`
par défaut de RQ (rétention ~1 an) et **n'est jamais revu par personne** —
aucune alerte, aucun dashboard, aucune purge (grep `FailedJobRegistry` → 0
résultat applicatif).

### 🟠 F2 — Job « empoisonné » : pas de boucle infinie, mais par absence de config plutôt que par garde-fou explicite
Confirmé : aucun retry configuré nulle part → un payload qui fait toujours
échouer la logique métier se termine « failed » une fois, sans reboucler.
Nuance : si le payload crashe le *process* worker lui-même (OOM, crash natif
LightGBM/torch), RQ ne le ré-enfile pas automatiquement ; le job reste
`started` côté Redis indéfiniment sans être nettoyé par RQ — seul le
watchdog DB (F3) le referme, au mieux 40 minutes plus tard.

### 🟠 F3 — `job_watchdog.py` : ni process séparé, ni cron ; portée limitée à `"running"`
`backend/domains/shared/job_watchdog.py::reconcile_stale_jobs` est appelée
**synchroniquement à chaque `POST .../jobs`** (les 6 routers), juste avant la
vérification de quota — aucun scheduler, aucun `rq-scheduler`. Détecte les
jobs `status=="running"` dont `progress_updated_at` date de plus de
`stale_job_timeout_minutes` (40 min). **Ne couvre jamais `"queued"`** — voir
F5. Marque `"failed"` mais **ne nettoie aucun fichier sur disque** (aucun
accès à `storage/` dans cette fonction) : un artefact `.joblib` partiellement
écrit par un worker mort reste indéfiniment orphelin sur disque.

### 🔴 F4 — Aucune idempotence sur la création de job
Grep `Idempotency|idempotenc` sur tout `backend/` → 0 résultat. Le flux
(`db.add(job); db.commit()` puis `enqueue()` inconditionnel) ne détecte
aucun doublon. Un double-clic frontend ou une requête retentée après un
timeout réseau crée deux `TrainingJob` et deux jobs RQ distincts,
consommant deux slots du quota (`max_concurrent_jobs_per_org=3` par défaut).

### 🔴 F5 — Redis down à l'enfilage : le job reste `"queued"` pour toujours, invisible au watchdog
```python
job = TrainingJob(..., status="queued")
db.add(job); db.commit()                      # déjà committé
rq_job = training_queue.enqueue(...)           # lève si Redis down, non intercepté
```
(`domains/training/router.py:647-655`, motif identique dans les 5 autres
domaines). Si `enqueue()` lève, le client reçoit un 500 générique non
structuré (incohérent avec le reste de l'API), **et le `TrainingJob` reste en
base avec `status="queued"`, `rq_job_id=NULL`, pour toujours** — F3 montre
que le watchdog ne traite jamais `"queued"`. Ce job orphelin occupe un slot
de quota indéfiniment, sans qu'aucun mécanisme du dépôt ne le détecte. À
comparer avec `api/core/rate_limit.py:20-31`, où le même risque (Redis down)
est traité en échec ouvert explicite — le patron existe dans le projet mais
n'a pas été appliqué ici.

### 🟠 F6 — Pas de handler SIGTERM custom, non aligné avec `stop_grace_period` (absent)
Aucun handler de signal applicatif (`grep SIGTERM|signal\.` → rien
d'applicatif). RQ 1.16.2 installe ses handlers par défaut (1er signal =
arrêt chaud, 2e = arrêt froid), mais `docker-compose.yml` ne définit **aucun
`stop_grace_period`** → Docker envoie SIGTERM puis SIGKILL après 10s par
défaut. Les jobs longs (`job_timeout=1800`) sont donc quasi systématiquement
tués brutalement lors d'un `docker compose down`/redéploiement pendant un
entraînement réel, laissant le job `"running"` jusqu'à détection par le
watchdog (au mieux 40 min plus tard, et seulement au prochain `POST
.../jobs` de la **même organisation**).

### 🟠 F7 — Aucun `statement_timeout` PostgreSQL
Grep exhaustif → 0 résultat, ni au niveau `create_engine`, ni en
`connect_args`. Une requête lente ou bloquée (verrou) peut occuper une
connexion indéfiniment.

### 🟠 F8 — Pool SQLAlchemy non dimensionné pour la topologie réelle
`api/core/database.py:35-39` : `create_engine()` sans `pool_size`/
`max_overflow` → défauts SQLAlchemy (5+10=15 connexions/process). Topologie
réelle : `backend` (2 workers gunicorn) + `worker` (2 réplicas) +
`worker-analysis` (1 réplica) = 5 process × 15 = **75 connexions** au repos ;
le commentaire de `docker-compose.yml:26-28` documente lui-même `--scale
worker=3 --scale worker-analysis=2` comme scénario supporté (7 process × 15 =
105), au-dessus du `max_connections=100` par défaut de `postgres:15-alpine`
(jamais reconfiguré). Combiné à l'absence de `statement_timeout` (F7), une
requête lente immobilise une connexion sur un pool déjà juste.

### ✅ F9 — Démarrage API malgré Postgres down : vérifié vrai
`api/main.py:94-108::lifespan` — `init_db()` en `try/except`, `yield` toujours
exécuté, `GET /api/health` reflète l'état réel via `check_connection()`.
Conforme à ce que documente `ARCHITECTURE.md` §10.
🟡 Note documentaire annexe : `ARCHITECTURE.md:286` affirme « migrations
idempotentes plutôt qu'Alembic » — **faux**, le projet utilise bien Alembic
(8 révisions réelles) depuis le Lot 1.1 ; phrase jamais mise à jour.

---

## Axe G — Intégrité des données

### ✅ G1/G2 — `ondelete=` explicite sur 54/54 FK, cohérent avec le schéma migré
`api/core/models.py` (lu intégralement, 959 lignes) : les 54 `ForeignKey(...)`
ont toutes un `ondelete=` explicite (`CASCADE` sur les FK internes à
l'organisation, `SET NULL` sur les FK vers `users.id` qui doivent survivre à
la suppression de l'auteur — avec justification explicite en commentaire
pour `AuditLog.actor_id`). Vérifié cohérent dans les 8 fichiers de migration
Alembic (`sa.ForeignKeyConstraint(..., ondelete=...)` identique).

### 🔴 G2-bis — Ces contraintes ne sont jamais réellement appliquées en SQLite (dev + toute la suite de tests)
Grep exhaustif `PRAGMA foreign_keys` sur tout `backend/`, y compris
`tests/conftest.py` → **0 résultat**. SQLite désactive l'application des FK
par défaut tant que `PRAGMA foreign_keys=ON` n'est pas exécuté par
connexion — ni le moteur applicatif ni le moteur de test ne le font. En dev
et dans **toute** la suite de tests automatisés, un `DELETE`/`db.delete(org)`
laisserait des lignes orphelines : le `CASCADE` documenté en G1 n'est
**exercé qu'en production (Postgres)**, jamais par les tests du dépôt.

### 🟠 G3 — Fuite disque confirmée : suppression d'un dataset ne nettoie pas les artefacts de modèles
`domains/datasets/router.py:673-684::delete_dataset` supprime le fichier du
dataset lui-même, puis repose **entièrement sur `ON DELETE CASCADE`** pour
les lignes `TrainingJob`/`MLModel` associées — mais ne supprime **jamais**
les fichiers `storage/models/{org}/{job_id}.joblib` sur disque. À comparer
avec `domains/training/router.py:1280-1289::delete_training_job`, qui lui
nettoie explicitement le fichier avant `db.delete(job.model)` — l'auteur
savait que le `CASCADE` seul ne suffit pas pour le disque, mais ce même soin
n'a pas été répliqué sur le chemin « suppression de dataset ». `GET
/{dataset_id}/usage` informe bien l'utilisateur du nombre de jobs impactés
avant confirmation, mais cela ne change rien au nettoyage serveur.

### 🟡 G4 — Suppression d'utilisateur : chemin applicatif inexistant, donc non exercé
Aucune route de suppression physique d'un `User` (seule la désactivation via
`actif=False` existe). Le design ORM/migration (`SET NULL` sur les FK
concernées) est cohérent mais **jamais exercé** par un chemin API réel.

### ✅ G5 — Migrations réversibles, vérifié réel (pas des `downgrade()` vides)
8 migrations inspectées, chaque `downgrade()` fait un vrai travail inverse
(y compris un cas non trivial documentant explicitement une irréversibilité
de **données**, tout en supprimant correctement colonnes/contraintes).
`tests/test_alembic_migration.py::test_downgrade_actually_drops_all_tables`
exécute réellement `command.downgrade(cfg, "base")` et vérifie la disparition
des 22 tables.

### ✅ G6 — Test de cohérence modèle ↔ schéma migré, réel et strict
`api/core/database.py:66-89::_schema_matches_metadata` compare `Base.metadata`
contre le schéma réel (`sqlalchemy.inspect`) table par table et colonne par
colonne, **appelée à chaque démarrage**, refuse de démarrer
(`SchemaMismatchError`) en cas d'écart. 5 tests dédiés, dont un qui
reproduit l'incident réel de production déjà connu (colonne stampée `head`
sans être physiquement créée).

---

## Axe H — Performance et montée en charge

### 🟠 H1 — Pool SQLAlchemy vs threadpool AnyIO — voir F8, même cause racine
Doublon volontaire avec F8 : la conséquence performance (contention/attente
30s puis `TimeoutError`) est distincte de la conséquence fiabilité déjà
notée.

### 🟠 H2 — Threadpool AnyIO par défaut (40/process), jamais dimensionné, alors que quasi tous les endpoints sont synchrones
Tous les endpoints métier sont des `def` (pas `async def`), sauf les 6
endpoints SSE. Aucun `CapacityLimiter` configuré (grep vide). Avec
`gunicorn --workers 2`, débit max théorique ≈ **80 requêtes bloquantes
simultanées**, tous endpoints confondus (upload, EDA, prédiction). Au-delà,
les requêtes suivantes attendent en file côté AnyIO avant même d'atteindre le
handler.

### 🟠 H3 — Cache LRU des datasets non borné en octets, dupliqué par process
`domains/shared/dataset_io.py:61-63::@lru_cache(maxsize=64)` borne le
**nombre** d'entrées, jamais leur poids mémoire. Un CSV proche de
`max_upload_size_mb=200` pèse 2-5× sa taille disque une fois en DataFrame
pandas — pire cas non plafonné par construction. **Process-local** : les 2
workers gunicorn + 3 process RQ ont chacun leur propre cache indépendant, le
même dataset peut être chargé plusieurs fois simultanément sans partage.
`read_dataset_dataframe()` retourne en plus une `.copy()` à chaque hit — pic
mémoire = original + copie pendant l'appel.

### 🟠 H4 — N+1 réel et absence totale de pagination sur `GET /vision-datasets`
`domains/vision/datasets/router.py:285-293` : aucun `joinedload` alors que
`_to_summary` accède à `dataset.uploaded_by.nom` → une requête SQL par
dataset. C'est exactement le pattern déjà corrigé sur **tous les autres**
endpoints de liste du backend (training/datasets/clustering/anomalies/
dimensionality/vision classification/vision anomalies ont tous leur
`joinedload` équivalent, vérifié un par un) — ce endpoint vision, ajouté
après ce correctif, ne l'a jamais reçu. C'est aussi le **seul** endpoint de
liste de tout le backend sans `limit`/curseur : renvoie systématiquement la
totalité des datasets vision de l'organisation.

### 🟡 H5 — `GET /dashboard/summary` : ~20 aller-retours SQL séquentiels
Pas du N+1 (chaque requête est indexée et correcte), mais rien n'agrège les
~20 requêtes (2 `COUNT` + 6+6 `COUNT` par type de job + 6 `_recent`) en 1-2
requêtes groupées. Sur une base chargée, la latence est la somme de 20
aller-retours réseau.

### 🟡 H6 — Aucun index composite malgré un pattern de requête dominant à 2 prédicats
85 index simple-colonne déclarés et **vérifiés créés en base** (comptage
croisé exact avec les 8 migrations), mais le pattern dominant
(`WHERE organization_id = ? AND status IN (...)`, `WHERE organization_id = ?
ORDER BY id DESC`) bénéficierait d'index composites — dégradation
progressive avec le volume, pas un incident brutal.

### ✅ H7 — Pagination par curseur réelle au niveau SQL (sauf H4)
`api/core/pagination.py::paginate_by_id()` applique `.filter(id < cursor)
.limit(limit + 1)` côté SQL sur tous les domaines sauf l'exception déjà
notée en H4 et `GET /training/jobs/{id}/predictions` (limite simple sans
curseur, dette assumée et documentée dans le code).

### ✅ H8 — SSE bien conçu : session DB courte par tick
`domains/shared/job_events.py:35-56` : chaque tick ouvre/ferme sa propre
session via `asyncio.to_thread`, jamais la session de la requête FastAPI
gardée ouverte. Timeout dur 1h. Seul risque résiduel : chaque tick consomme
un slot du même pool de 40 threads que les endpoints sync classiques (lié à
H2).

### 🟡 H9 — `pandas.read_csv`/`read_excel` en entier, sans streaming
Pas de `chunksize` nulle part. Pic mémoire de plusieurs centaines de Mo à
quelques Go pour un fichier proche de la limite. Double occupation mémoire
transitoire à l'upload (`content = await file.read()` puis re-parsing pandas).

---

## Axe I — Traçabilité et transparence

### 🟠 I1 — Actions sensibles non journalisées dans `AuditLog`
Vérification route par route (toutes les routes mutantes des 8 fichiers
routers qui utilisent `log_action`) :

| Action | Journalisée ? |
|---|---|
| Connexion réussie/échouée | ❌ Non |
| Changement de mot de passe | ❌ Non |
| Mise à jour du profil | ❌ Non |
| Upload dataset tabulaire/vision | ❌ Non |
| Création de n'importe quel job (6 domaines) | ❌ Non (seuls `cancel`/`delete` le sont) |
| Prédiction clustering (non persistée du tout, contrairement à `Prediction` supervisé) | ❌ Non |
| Suppression dataset/job, promotion de modèle, ajout de membre | ✅ Oui |

Impact concret : un incident « qui s'est connecté sur ce compte, quand ? »
est **invérifiable** au-delà de la fenêtre de rate-limit Redis (qui expire).

### 🔴 I2 — `request_id` s'arrête à la couche HTTP, jamais propagé vers RQ ni vers `AuditLog`
`AuditLog` (`models.py:349-387`) n'a **aucune colonne `request_id`**. Les
jobs RQ sont enfilés avec `enqueue(run_training_job, job.id, job_timeout=1800)`
— seul `job.id` traverse, jamais le `request_id` de la requête HTTP
d'origine. Les workers tournent dans des process séparés : même un
`ContextVar` partagé ne traverserait pas la frontière de process (confirmé :
grep `request_id` sur tout `backend/` → uniquement `api/main.py`,
`observability.py`, son test). Un incident « l'entraînement du job 42 a
échoué » ne peut pas être relié à la requête HTTP d'origine autrement que par
recoupement manuel `job_id` + fenêtre temporelle.

### 🔴 I3 — Aucune documentation OpenAPI des réponses d'erreur
`grep -rn "responses=" domains api` → **0 résultat** sur tout le backend.
Chaque route lève des `HTTPException` structurées, mais Swagger/OpenAPI ne
documente que le cas de succès. Un consommateur de l'API ne peut découvrir
les codes d'erreur possibles qu'en lisant le code Python.

### ✅ I4 — Journalisation structurée JSON, corrélée par requête HTTP — vérifié en direct
`observability.py::JsonFormatter` + `RequestIdMiddleware` — confirmé en
exécutant réellement le backend (§ Axe E) : chaque réponse porte
`X-Request-ID`, présent aussi dans les lignes de log. Limite : ce bénéfice
ne s'étend pas aux workers RQ (I2), et `X-Request-ID` est spoofable côté
client sans validation (voir Axe E).

### 🟢 I5 — Chaîne de filiation prédiction → dataset : possible en un JOIN, supervisé seulement
FK vérifiées : `Prediction.ml_model_id → MLModel.id` (NOT NULL, indexé) →
`MLModel.training_job_id → TrainingJob.id` (NOT NULL) →
`TrainingJob.dataset_id → Dataset.id` (NOT NULL, indexé) →
`Dataset.content_hash` (SHA-256). Techniquement reconstructible par un
simple `JOIN`, mais **aucun endpoint ne l'expose** (capacité SQL latente, pas
une fonctionnalité consultable), et cette chaîne n'existe que pour le
supervisé — les 3 piliers non supervisés n'ont aucune table de
prédiction/assignation persistée.

### 🟡 I6 — Messages d'erreur incohérents entre couches
Bon pattern dans `domains/training/worker.py::_user_safe_error_message`
(liste blanche, messages actionnables en français). Pattern plus faible
répété 10+ fois dans `datasets/router.py` et 1 fois dans chaque autre
router : `f"Impossible de lire le fichier : {exc}"` concatène directement le
message brut d'exception pandas/openpyxl (souvent en anglais) dans la
réponse client.

---

## Axe J — Chaîne d'approvisionnement et hygiène

### ✅ J1 — 100 % des dépendances épinglées, aucune dépendance morte
33 dépendances dans `backend/requirements.txt`, toutes en `==`. Vérifié
package par package (y compris les usages indirects : `python-dotenv` via
`pydantic-settings`, `email-validator` via `EmailStr`, `openpyxl`/`pyarrow`
via `pandas`, `sentry-sdk` en import différé conditionnel) — aucune
dépendance déclarée mais inutilisée.

### 🟠 J2 — Pas de multi-stage build
Un seul `FROM python:3.12-slim` (`Dockerfile`) : `build-essential`/
`libpq-dev` (nécessaires seulement à la compilation) restent dans l'image
finale de production — surface d'attaque et poids inutiles (compilateurs
disponibles au runtime).

### 🔴 J3 — Aucun contrôle de sécurité en CI, confirmé absent
`.github/workflows/ci.yml` : 3 jobs (`backend` = pytest+Redis,
`frontend` = tsc/lint/contraste/vitest/build, `smoke` = stack Docker complète
+ scénario réel). **Confirmé absents** : `pip-audit`, `bandit`, scan de
secrets, scan d'image Docker, SBOM, seuil de couverture (la couverture n'est
même pas mesurée, `pytest` tourne sans aucun flag `--cov`).

**⚠️ Lien direct avec §0** : le job `smoke` construit et démarre la stack
Docker complète — si `main` est vert sur ce job aujourd'hui, cela
contredirait directement le constat §0 (image cassée). À vérifier en tout
premier geste de la Phase 1 : soit ce job échoue déjà silencieusement/est
ignoré en pratique, soit `domains/` a disparu du `Dockerfile` très
récemment, après le dernier run vert.

### 🟠 J4 — Conteneurs non durcis (`docker-compose.yml`)
Pas de `user:` (bien que l'image elle-même tourne déjà en non-root via
`USER appuser` — double emploi, pas un trou), pas de `read_only`, pas de
`cap_drop`, pas de `security_opt`.

### ✅ J5 — Aucun résidu problématique tracké par git
`catboost_info/`, `.venv/`, `*.db` : présents sur disque mais **gitignorés**,
confirmé absents de `git ls-files` — pas un problème de dépôt. Aucun fichier
`.py` vide suspect (les seuls fichiers de 0 octet trackés sont des
`__init__.py`, convention normale). `domains/datasets/services/preview.py`
fait 929 octets, pas vide (corrige une hypothèse de départ).

**Résidus locaux non trackés, hors périmètre backend** (voir aussi le
`.gitignore` — ne couvre pas tout) : `src/`, `ui/`, `helpers/`,
`orchestrators/`, `utils/`, `monitoring/` existent toujours sur le disque de
développement (dossiers de l'ancienne app Streamlit, retirés de git au
commit `e905f2c`), avec des `__pycache__` datés d'aujourd'hui — signe qu'ils
ont été exécutés localement récemment. Non trackés par git, donc **aucun
impact sur le dépôt versionné** ; à signaler pour information (Phase 7,
nettoyage local), pas un constat de sécurité.

### 🔵 J6 — `pip-audit`/`bandit` non exécutables dans cet environnement
Ni l'un ni l'autre n'est installé ; aucune liste de CVE n'a été inventée en
leur absence — à ajouter en CI (J3) plutôt qu'à exécuter ponctuellement.

---

## Ce qui est déjà bon — confirmé, à ne pas régresser

- **Isolation multi-tenant** : 110 routes auditées, zéro IDOR, pattern
  `_get_org_*` systématique, 404 jamais 403 (Axe B).
- **JWT_SECRET_KEY** : hard-fail réel en production, pas un simple log
  (Axe A.8, D).
- **Zip-slip** : bloqué avec défense en profondeur (double vérification),
  zip-bombe bloquée pendant l'extraction pour le cas dominant
  (accumulation), format vérifié par signature binaire réelle, pas par
  extension (Axe C).
- **Aucun secret dans le dépôt ni son historique git**, sur 221 commits
  inspectés (Axe D).
- **Contraintes FK avec `ondelete=` explicite sur 54/54 relations**,
  cohérentes entre `models.py` et les 8 migrations réelles ; migrations
  réversibles vérifiées (pas des `downgrade()` vides) ; test de cohérence
  schéma↔modèles strict et exécuté à chaque démarrage (Axe G).
- **Pagination SQL par curseur réelle** sur la quasi-totalité des listes,
  N+1 déjà corrigés sur 6/7 endpoints de liste, SSE conçu pour ne jamais
  garder une session DB ouverte (Axe H).
- **Logs JSON structurés avec corrélation `request_id`** au niveau HTTP,
  100 % des dépendances épinglées, aucune dépendance morte, image Docker en
  utilisateur non-root (Axes I, J).
- **Démarrage dégradé si la base est indisponible** (vérifié vrai, pas
  supposé) — le healthcheck reflète l'état réel plutôt que de faire planter
  le process (Axe F).

---

## Ce que je n'ai pas pu vérifier, et pourquoi

- **Comportement réel du rate-limit/`X-Forwarded-For` à travers nginx** :
  la stack Docker complète n'a pas pu être amenée à un état sain dans le
  temps de cet audit à cause de §0 (image `backend` cassée). La preuve
  retenue est une lecture croisée non ambiguë de 3 fichiers
  (`rate_limit.py`, `Dockerfile`, `nginx/templates/default.conf.template`),
  mais aucune requête HTTP réelle n'a traversé nginx pour le confirmer.
- **`ON DELETE CASCADE` réel contre un vrai Postgres** : le code le déclare
  et le migre correctement, mais aucun test du dépôt ne l'exerce contre
  Postgres — seulement contre SQLite sans `PRAGMA foreign_keys=ON`, qui ne
  l'applique pas (G2-bis). Non vérifiable sans la stack Docker.
- **Comportement réel d'un SIGTERM pendant un job en cours** (F6) : dépend
  du timing exact entre l'arrêt « chaud » de RQ et le `SIGKILL` Docker à
  10s — non simulable par lecture statique, nécessite un test de charge
  réel contre la stack.
- **Débit réel soutenu (req/s), consommation mémoire réelle du cache LRU,
  nombre de connexions Postgres réellement ouvertes en production** (Axe H)
  : tous dépendent de la charge réelle et de la distribution des tailles de
  dataset, non observables par lecture de code.
- **Vulnérabilités CVE réelles des dépendances épinglées** : `pip-audit`
  non installé dans cet environnement, aucune liste inventée en son
  absence.
- **État réel du job CI `smoke` sur `main`** (vert ou rouge aujourd'hui) :
  non exécuté dans cet audit (aurait nécessité de déclencher un run GitHub
  Actions) — signalé comme premier geste de vérification pour la Phase 1
  (voir §0).

---

## Note méthodologique — écarts avec la liste de constats déjà vérifiés fournie en amont

Les 10 constats fournis en amont de cet audit ont tous été revérifiés
indépendamment. Neuf sont confirmés tels quels. Un est nuancé :

- **Constat n°5 (`/metrics` sans authentification et hors `/api`)** :
  confirmé exact au niveau applicatif (endpoint non authentifié, vérifié en
  direct), mais **la conclusion sur la reachability externe est corrigée** —
  dans la topologie exacte de `docker-compose.yml` du dépôt (port backend
  non publié via `ports:`, nginx ne proxifiant que `/api/`), `/metrics`
  n'est **pas** joignable depuis l'extérieur du réseau Docker. Reclassé 🟡
  (défense en profondeur manquante côté application) plutôt que 🟠/🔴
  (exposition réseau directe) — voir Axe E pour le détail complet et la
  nuance sur les déploiements hors de ce compose file précis.

Un onzième constat, non présent dans la liste fournie, a été trouvé en
cours d'audit et documenté en §0 : **l'image Docker de production ne peut
plus démarrer** (`domains/` absent du `Dockerfile`) — probablement la
découverte la plus importante de cette phase, car elle bloque toute
vérification live pour le reste du chantier tant qu'elle n'est pas corrigée.
