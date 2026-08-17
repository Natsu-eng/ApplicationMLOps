# Décisions — DataLab Pro

Une entrée par décision non triviale prise en exécutant le plan de l'audit
(`AUDIT_DATALAB_2026-08-16.md`), lot par lot, en mode autonome (voir cadrage
du 2026-08-16). Format : la question, l'option retenue, les options
écartées et pourquoi, ce qui devrait amener à la remettre en cause.

---

## Lot 0

### D0.1 — Ordre d'exécution du Lot 0 : CI (0.4) avant les correctifs (0.1/0.2/0.3)

**Question** : dans quel ordre traiter les quatre sous-lots de Lot 0 ?
**Retenu** : 0.4 (CI) en premier, puis 0.1, 0.2, 0.3 — demandé explicitement
par l'utilisateur, pour que chaque correctif soit couvert par la CI dès son
premier commit.
**Écarté** : l'ordre C1/C2/C4/C5 de l'audit (§Q, Phase 0) — logique mais ne
protège pas les trois premiers commits.
**Remise en cause si** : jamais, c'est une contrainte du cadrage, pas un
choix technique.

### D0.2 — CI : service Redis requis, pas seulement optionnel

**Question** : la CI backend a-t-elle besoin d'un service Redis réel, ou
suffit-il que la suite tourne sans (comme le suggérait l'audit avec
« si nécessaire ») ?
**Vérifié, pas supposé** : `is_rate_limited()` (`api/core/rate_limit.py`)
est **fail-open** — si Redis est injoignable, il retourne toujours
`False` (non limité). Le test
`tests/test_auth.py::test_login_blocked_after_too_many_failed_attempts`
attend un 429 après dépassement du seuil : sans Redis réellement
accessible, ce test échouerait puisque la limite ne se déclencherait
jamais. Confirmé en lançant la suite en local avec Redis actif (passe) —
le raisonnement par le code montre qu'il échouerait sans.
**Retenu** : service `redis:7-alpine` dans le job backend de la CI (même
image que `docker-compose.yml`).
**Écarté** : mocker `redis_conn` dans ce test spécifique — changerait le
sens du test (il vérifie un comportement bout-en-bout, pas juste la
fonction `is_rate_limited`).
**Postgres, en revanche, n'est pas nécessaire** : `tests/conftest.py`
réécrit `DATABASE_URL` vers un SQLite temporaire avant tout import de
`api.core.*`, donc la suite ne touche jamais une vraie base Postgres.
**Remise en cause si** : un futur test dépend explicitement d'un
comportement SQL propre à Postgres (ex. contraintes, types JSON natifs)
absent de SQLite — pas le cas aujourd'hui.

### D0.4 — Dédoublication Vision (0.1) : quelle copie survit, et jusqu'où

**Question** : face à une image bit-à-bit dupliquée, laquelle garder ? Et
que faire d'un doublon entre deux classes/catégories différentes ?
**Retenu** :
- Même catégorie, même split (ou classification, qui n'a pas de split à
  l'ingestion) → doublon bénin, une seule copie survit (la première par
  ordre alphabétique du chemin — déterministe, reproductible).
- Même catégorie, présente à la fois dans `train/` et `test/` (MVTec AD) →
  fuite train/test, la copie de `train/` survit TOUJOURS, jamais celle de
  `test/` — le jeu d'évaluation doit rester non vu.
- Catégories DIFFÉRENTES (ex. deux classes de classification, ou "good" vs
  "scratch" en MVTec AD) → conflit d'étiquette, pas un doublon : les DEUX
  copies sont exclues, aucun arbitrage arbitraire qui fausserait la classe
  survivante.
**Écarté** : hachage perceptuel pour attraper les quasi-doublons (image
recadrée/recompressée) — hors périmètre du Lot 0, documenté comme limite
connue dans `VisionDatasetReport.duplicate_detection_note` et affiché à
l'utilisateur plutôt que caché.
**Conséquence assumée** : la déduplication peut faire passer une classe
sous le seuil minimum d'images alors qu'elle le respectait avant — le
message d'erreur le dit explicitement (`vision_datasets.py`) plutôt que de
laisser l'utilisateur deviner pourquoi un import qui passait avant est
soudain refusé.
**Effet de bord découvert en testant** : plusieurs fixtures de tests
existantes (`test_vision_datasets_service.py`, `test_vision_datasets_api.py`,
`test_vision_classification_api.py`, `test_vision_anomaly_api.py`)
généraient des images "différentes" en réalité bit-à-bit identiques (même
couleur unie, ou même graine `np.random.default_rng(0)` réinitialisée à
chaque appel) — invisible avant ce correctif puisque les doublons étaient
conservés sans conséquence. Corrigé en ajoutant un paramètre `variant` aux
générateurs de PNG de test, jamais en affaiblissant la déduplication pour
les faire passer.
**Remise en cause si** : le hachage perceptuel devient un besoin produit
réel (retours utilisateurs sur des quasi-doublons non détectés).

### D0.5 — Calibration MVTec AD (0.2) : seuil du split, fraction, sens de n_test

**Question** : à partir de quel nombre d'images par catégorie un split
calibration/évaluation stratifié est-il fiable ? Quelle fraction pour
chaque côté ?
**Retenu** : `MIN_IMAGES_PER_CATEGORY_FOR_CALIBRATION_SPLIT = 6`, split
50/50. Le 50/50 est un choix délibéré plutôt qu'un split asymétrique
(ex. 40/60) : avec un seuil pair (6, 8, 10...) et une catégorie de taille
paire, 50/50 donne un partage exact sans ambiguïté d'arrondi — un split
asymétrique aurait pu faire tomber une petite catégorie sous le plancher
par arrondi même après le filtre. 3 images par catégorie de chaque côté
reste un plancher bas (pas une garantie de puissance statistique), mais
c'est la limite en dessous de laquelle un seuil ou une métrique par
catégorie devient un artefact de hasard plutôt qu'une mesure.
**Vérifié, pas supposé** : le repli (mêmes indices des deux côtés, biaisé
mais signalé) est le cas FRÉQUENT sur MVTec AD réel — plusieurs catégories
officielles ont des sous-types de défaut à moins de 10 images de test,
donc sous le plancher de 6 une fois divisées par 2. Implémenté et testé en
conséquence (`test_calibration_falls_back_and_flags_bias_on_small_dataset`),
pas traité comme un cas dégradé secondaire.
**`roc_auc` calculé sur l'évaluation seule**, pas sur tout `test/` : l'audit
notait que `roc_auc` reste valide indépendamment du seuil (donc calculable
sur l'ensemble complet), mais calculer TOUTES les métriques rapportées
(roc_auc compris) depuis le même sous-ensemble évaluation les rend
cohérentes entre elles et avec le principe "jamais évalué sur ce qui a
servi à calibrer".
**`n_test` inchangé, `n_calibration`/`n_evaluation` ajoutés séparément** :
nouvelles colonnes nullable sur `vision_anomaly_models` (via
`_add_column_if_missing`, le mécanisme déjà en place — Alembic n'existe pas
encore, Lot 1), `NULL` sur les modèles entraînés avant ce correctif.
**Exemples UI toujours tirés de l'évaluation**, jamais de la calibration —
même en repli biaisé, où `evaluation_idx == tous les indices` (le repli ne
crée pas de sous-ensemble calibration distinct, donc aucun risque
qu'un exemple "calibration" soit présenté comme non vu).
**Remise en cause si** : les catégories MVTec AD réelles utilisées en
production s'avèrent systématiquement plus grandes que prévu (auquel cas
un plancher plus élevé donnerait un split plus fiable sans coûter le
repli) — à observer une fois en usage réel.

### D0.6 — 401 global (0.3) : logique de décision pure, effet de bord centralisé

**Question** : comment intercepter un 401 sur les 4 chemins `fetch`
distincts (`request`, `uploadFile(WithFields)`, `exportModel`,
`VisionImage.tsx`) sans dupliquer la logique, et rester testable dans un
environnement Vitest sans `jsdom` ?
**Retenu** : `handleUnauthorized()` (effet de bord : `clearToken()` +
`window.location.href`) exporté depuis `api/client.ts`, appelé aux 4
endroits ; sa décision de rediriger ou non est extraite en fonction pure
`shouldRedirectToLogin(pathname)`, seule testée (`client.test.ts`).
**Écarté** : dupliquer la vérification `res.status === 401` avec une
logique propre à chaque appelant — un seul point de vérité sur "que
signifie un 401", cohérent avec le reste du client API.
**`/auth/login` naturellement exclu** : passe par `requestForm()`, jamais
relié à `handleUnauthorized()` — et de toute façon `/auth/login` ne renvoie
jamais 401 dans ce backend (identifiants incorrects → 400, trop de
tentatives → 429, voir `routers/auth.py`), vérifié dans le code avant
d'écrire cette décision.

## Lot 1

### D1.1 — Migration initiale Alembic autogénérée contre SQLite, pas Postgres

**Question** : contre quelle base générer la révision initiale (`alembic
revision --autogenerate`) — un Postgres jetable, ou SQLite ?
**Retenu** : SQLite vierge (`tests/test_alembic_migration.py` le
revérifie). Les opérations Alembic (`op.create_table`, `op.create_index`...)
sont compilées par dialecte AU MOMENT de l'exécution, pas à la génération —
le même script produit du DDL Postgres correct quand `alembic upgrade
head` tourne réellement contre Postgres (vérifié : la révision a été
appliquée avec succès contre le Postgres réel de développement via le
chemin `stamp`, voir D1.2).
**Écarté** : instancier un Postgres jetable juste pour l'autogénération —
coût d'infra superflu, aucun gain de fidélité pour une révision qui ne
fait que du CREATE TABLE (pas d'ALTER, où des différences de dialecte
pourraient réellement compter).

### D1.2 — Chemin de migration d'une base existante : `stamp`, jamais un rejeu

**Contexte** : ce dépôt a une vraie base Postgres de développement locale,
déjà peuplée (2 organisations, 3 utilisateurs, 13 datasets, 13
entraînements — données de sessions de test précédentes), créée par
l'ancien `Base.metadata.create_all()`. Occasion de tester le chemin
"base déjà en service" en conditions réelles, pas seulement en théorie.
**Retenu** : `run_migrations()` détecte ce cas (tables présentes, pas de
table `alembic_version`) et exécute `alembic stamp head` — marque la
révision comme appliquée SANS rejouer son SQL. Testé avec de vraies
données dans `test_alembic_migration.py` (schéma créé hors Alembic, une
ligne insérée, migration appliquée, donnée toujours là, aucune exception).
**Écarté** : un `upgrade head` inconditionnel — testé explicitement qu'il
échoue sur une base existante (`CREATE TABLE` sur une table déjà là), ce
qui justifie la détection plutôt que de la supposer nécessaire.

### D1.3 — Sauvegarde/restauration testée par SCHÉMA jetable, pas base jetable

**Découvert en testant, pas supposé** : le rôle applicatif `datalab` n'a
PAS le privilège `CREATEDB` (`rolcreatedb = false`, vérifié par requête
directe sur `pg_roles`) — cohérent avec le principe de moindre privilège
qu'on attendrait d'un déploiement réel, mais ça invalidait mon premier
essai (créer une base Postgres jetable pour le cycle de test
sauvegarde/restauration).
**Retenu** : isolation par SCHÉMA (`CREATE SCHEMA`/`DROP SCHEMA`, que le
rôle applicatif peut faire dans sa propre base) plutôt que par base
séparée. `pg_dump --schema=...` restreint le dump au schéma de test ;
`pg_restore --clean --if-exists` ne touche que les objets présents dans le
dump (jamais `public`, jamais les données réelles). Le test simule une
vraie perte (DROP SCHEMA) avant de prouver la restauration, pas juste un
aller-retour dump/restore sur un schéma toujours intact.
**Effet de bord corrigé en testant** : `restore_storage()` extrayait
l'archive en supposant que le nom de dossier interne fixe de l'archive
("storage/", voir `backup_storage`) correspondait au nom du dossier cible
demandé par l'appelant — faux dès que `target_dir` porte un autre nom
(exactement le cas du test). Corrigé pour extraire le contenu directement
dans `target_dir`, quel que soit son nom.
**Remise en cause si** : le rôle applicatif obtient un jour `CREATEDB` pour
une autre raison — l'isolation par base séparée deviendrait alors plus
simple et plus proche d'une vraie restauration complète (base entière, pas
un schéma), et vaudrait la peine d'être reconsidérée.

### D1.4 — Mode worker piloté par variable d'environnement, repli sur détection plateforme

**Retenu** : `RQ_WORKER_MODE=fork|simple`, explicite en priorité ; à
défaut, détecté depuis `sys.platform` ("simple" est le SEUL mode
fonctionnel sous Windows, jamais un choix arbitraire — `os.fork()` n'y
existe pas). `docker-compose.yml` fixe `RQ_WORKER_MODE=fork` explicitement
pour le service `worker` (jamais un repli implicite en production, même
si la détection plateforme donnerait le même résultat sous Linux).
**`deploy.replicas: 2`** dans `docker-compose.yml` plutôt que documenter
uniquement `--scale` : le défaut doit déjà être multi-worker sans action
manuelle — l'audit demandait explicitement "passe le service worker à
plusieurs répliques dans docker-compose.yml", pas seulement rendre le
scaling possible.
**Écarté** : détection automatique sans variable d'environnement (juste
`sys.platform`) — l'audit demande explicitement un pilotage par variable
d'environnement, et un override explicite reste utile pour forcer "simple"
en diagnostic même sous Linux.

### D1.5 — Rate-limiting étendu : compteurs indépendants par action, mêmes limites configurables

**Retenu** : `rate_limit_dependency(action, max_attempts, window_seconds)`
généralise le mécanisme de `/auth/login` (déjà en place) à `/register`, aux
deux endpoints d'upload (tabulaire et vision — compteurs INDÉPENDANTS,
`dataset_upload` vs `vision_dataset_upload`, pour qu'épuiser l'un ne
bloque pas l'autre) et à `/explain`. Limites par défaut définies dans
`Settings` (10/h, 30/h, 30/h, 20/h respectivement), même convention que
`login_rate_limit_max_attempts` existant.
**Connu et accepté** : les limites sont capturées comme valeurs fixes à
l'import du module (`_upload_rate_limit = rate_limit_dependency(...)` au
niveau module), pas relues dynamiquement par requête — cohérent avec le
mécanisme `/auth/login` déjà en place (même limite), documenté dans les
tests (qui exercent la vraie valeur par défaut plutôt que de la
monkeypatcher, ce qui ne fonctionnerait pas ici).

### D1.6 — torch.load : `weights_only=True` suffit, vérifié empiriquement, pas de restructuration

**Question** : l'audit suggère de restructurer l'artefact vision
(`state_dict` d'un côté, métadonnées JSON de l'autre) pour permettre
`weights_only=True`.
**Vérifié avant de décider** : un test direct (dict avec `state_dict`
PyTorch + `backbone_id`/`class_names`/`dropout_rate` en types Python
simples, sauvegardé puis rechargé) montre que `weights_only=True` charge
cet artefact SANS AUCUNE modification de structure — l'allowlist par
défaut de torch 2.13 couvre déjà les tenseurs, `OrderedDict` et les types
Python de base.
**Retenu** : juste changer `weights_only=False` → `True`, sans toucher au
format de l'artefact. Plus simple que ce que l'audit envisageait, et
vérifié plutôt que supposé.
**`joblib.load` (bundle tabulaire)** : pas d'équivalent `weights_only` en
pickle/joblib — documenté comme risque connu et accepté (frontière de
confiance : fichier écrit uniquement par notre worker), pas de correctif
technique possible sans changer complètement de format de sérialisation
(hors périmètre de ce lot).

### D1.7 — JWT en cookie httpOnly : coût présenté, pas implémenté (Arrêt B)

Conformément au cadrage : ce point touche toute la chaîne d'authentification
(migration `localStorage` → cookie `httpOnly`/`SameSite=Strict`, jeton CSRF
à introduire, `client.ts` et `AuthContext.tsx` à revoir, tests d'auth à
adapter). Coût et impacts présentés séparément dans le rapport de fin de
lot — implémentation non commencée, en attente d'arbitrage.

### D1.8 — Incident réel : stamp à l'aveugle + colonnes manquantes en base, correctif à deux niveaux

**Trouvé (retour utilisateur, incident réel)** : `GET
/vision/anomalies/jobs` renvoyait 500 en production locale.
`_to_summary()` (`api/routers/vision_anomalies.py`) accède à
`job.result`, qui charge paresseusement `VisionAnomalyModel` — donc TOUTES
ses colonnes mappées, y compris `n_calibration`/`n_evaluation` (ajoutées
au modèle au Lot 0.2, avant l'introduction d'Alembic au Lot 1.1). Vérifié
directement sur la base : `alembic_version = 594bce594adf` (donc marquée
`head`) mais `vision_anomaly_models` sans ces deux colonnes physiquement —
la version antérieure de `run_migrations()` (D1.2) les avait stampées
"conformes" sans jamais vérifier le schéma réel, exactement le risque que
l'utilisateur avait signalé au cadrage.

**Retenu — niveau systémique** :
- `_schema_matches_metadata()` compare désormais le schéma réel
  (`inspect()`) à `Base.metadata` table par table ET colonne par colonne
  avant tout `stamp` sur une base pré-Alembic — divergence détectée =
  `SchemaMismatchError` avec la liste exacte des écarts, jamais un stamp
  à l'aveugle (corrige D1.2, qui ne vérifiait que l'absence
  d'`alembic_version` combinée à la présence de `organizations`).
- **Filet ajouté au-delà de ce que D1.2 couvrait** : une vérification
  identique s'exécute maintenant APRÈS `stamp` ET après `upgrade`, pas
  seulement avant un stamp — nécessaire parce que l'incident réel s'est
  produit sur une base DÉJÀ marquée `head` : chaque redémarrage suivant
  tombait dans la branche `upgrade head` (no-op, déjà à `head`) sans
  jamais revérifier le schéma réel. Sans ce filet, le correctif n'aurait
  protégé QUE les futures transitions pré-Alembic → Alembic, pas rattrapé
  une base déjà mal stampée par l'ancien code.
- Tests ajoutés : `test_pre_alembic_database_with_missing_column_is_refused_not_stamped`
  (variante colonne du test table existant) et
  `test_already_stamped_database_with_drifted_schema_is_refused_at_every_startup`
  (reproduit l'incident exact : base déjà à `head`, colonne manquante
  malgré tout).

**Retenu — niveau immédiat** : migration de rattrapage
(`2744196bc3c7_rattrapage_vision_anomaly_models_n_.py`) ajoutant
`n_calibration`/`n_evaluation` à `vision_anomaly_models`, puis appliquée
à la base de développement réelle (`alembic upgrade head`) — vérifié
après coup que les deux colonnes existent et que les données existantes
sont intactes (aucun `DROP`/`TRUNCATE`).
**Idempotente par construction** : `594bce594adf` (révision initiale)
crée déjà ces colonnes pour toute base NEUVE — un `add_column`
inconditionnel dans la migration de rattrapage cassait donc toute base
neuve traversant la chaîne complète (`duplicate column name`, constaté en
test avant correction). `upgrade()`/`downgrade()` vérifient l'état réel
via `sa.inspect()` avant d'agir, dans les deux sens : no-op sur une base
neuve (déjà conforme), rattrapage réel sur une base drifted.

**Trouvé, non traité (hors périmètre)** : l'autogénération a aussi
détecté un changement de type sur `ml_models.promoted_at` et
`training_jobs.progress_updated_at` (`TIMESTAMP` → `DateTime(timezone=True)`)
sur la base réelle — sans rapport avec cet incident (pas une colonne
manquante, un type divergent que `_schema_matches_metadata()` ne
détecte d'ailleurs pas, puisqu'elle ne compare que les NOMS de colonnes).
Non inclus dans la migration de rattrapage pour ne pas mélanger deux
correctifs sans lien. À traiter séparément si ce type de divergence
s'avère un jour bloquant (aujourd'hui SQLAlchemy lit/écrit les deux
représentations sans erreur).
**Remise en cause si** : ce type de divergence (type de colonne, pas
présence/absence) cause un jour un bug réel — étendrait
`_schema_matches_metadata()` au-delà des noms de colonnes, avec le coût
de complexité que ça implique (faux positifs sur des équivalences de
type bénignes selon le dialecte SQL).

### D0.3 — Terminologie "MVTec AD" dans le pilier anomalies visuelles

**Trouvé, non traité (hors périmètre Lot 0)** : `structure_type ==
"mvtec_ad"` et les messages d'erreur associés (`vision_datasets.py`,
`vision_anomaly_training.py`) nomment explicitement "MVTec AD" alors que
la structure détectée (train/good + test/good + test/<défaut>) est
générique — elle fonctionne pour n'importe quel dataset d'anomalies
visuelles construit ainsi, pas seulement le jeu de données industriel
"MVTec AD" au sens strict. Le correctif de calibration (D0.x, sous-lot
0.2) est lui-même générique et s'applique correctement quel que soit le
contenu réel du dataset.
**Pourquoi non traité ici** : renommer la terminologie touche les
messages d'erreur, le `structure_type` persisté en base (valeur déjà
utilisée par des enregistrements existants), les labels frontend et la
documentation — plus large que "arrêter les métriques fausses" (Lot 0).
**À traiter** : lors du Lot 6A (wizard Vision) ou Lot 7 (produit), au
moment de retravailler l'UX du pilier anomalies visuelles.

## Lot 4 — Tenir la charge (Phase 3 de l'audit, correctifs I3, I4 et I6)

### D4.1 — Pagination rétrocompatible par absence, jamais une enveloppe JSON

**Question** : `GET /training/jobs` et les 5 endpoints équivalents
renvoient TOUJOURS la totalité des jobs de l'organisation
(AUDIT_DATALAB_2026-08-16.md §C.2.4, R6 — "effondrement de performance à
la montée en charge"). Comment ajouter une vraie pagination par curseur
sans casser le Dashboard ni les 3 pages Historique, qui attendent
aujourd'hui un tableau JSON à plat ?
**Retenu** : `limit`/`cursor` optionnels (`api/core/pagination.py`,
partagé par les 6 routers de job + `GET /datasets`) — absents (défaut) :
comportement STRICTEMENT inchangé, tout est renvoyé, aucun appelant
existant cassé. La page suivante est signalée par un en-tête
`X-Next-Cursor`, jamais dans le corps JSON : la forme de la réponse
(`List[XSummary]`) reste identique que la pagination soit utilisée ou
non — pas d'enveloppe `{items, cursor}` qui aurait forcé une migration de
tous les appelants dans ce même lot.
**Écarté** : migrer les pages Historique vers une UI de pagination réelle
dans ce lot — c'est `P6` (refonte de `Table.tsx`, tri/pagination/
recherche/sélection), une phase produit séparée et postérieure dans le
plan d'exécution de l'audit (§Q). Ce lot livre la CAPACITÉ backend,
robuste et testée ; l'adoption frontend page par page est un chantier
distinct.
**Remise en cause si** : `P6` révèle qu'un en-tête HTTP est un mauvais
support pour le curseur (ex. proxy qui le filtre) — passer à un champ
dans le corps JSON à ce moment-là, pas préventivement.

### D4.2 — Curseur basé sur `id`, jamais `created_at` : bug réel trouvé en testant

**Trouvé en testant** : les 6 endpoints de job (et `GET /datasets`)
triaient par `created_at DESC`. SQLite stocke `func.now()` avec une
précision à la SECONDE — un test créant 7 jobs en rafale (as réel en
usage : plusieurs jobs lancés depuis un script, ou une simple rafale de
clics) leur donne souvent le même `created_at`, rendant l'ordre non
déterministe. Le curseur (`WHERE id < cursor`) suppose que l'ordre de
tri correspond à l'ordre décroissant des `id` — faux dès qu'il y a des
égalités de `created_at`, ce qui a fait sauter/dupliquer des lignes entre
deux pages dans `test_cursor_advances_to_the_next_page_without_overlap_or_gap`.
**Retenu** : tri par `id DESC` partout où une pagination existe
désormais (les 6 listes de jobs, `GET /datasets`, l'agrégat Dashboard) —
`id` auto-incrémenté encode l'ordre de création SANS ambiguïté possible
(contrairement à un horodatage à résolution limitée), équivalent en
pratique à `created_at DESC` pour toute table où les lignes ne sont
jamais réordonnées après coup (vrai ici).
**Pourquoi ce n'est pas anecdotique** : ce bug existait DÉJÀ (silencieusement)
avant ce lot — sans pagination, l'ordre de retour n'avait pas besoin
d'être stable puisque TOUT était renvoyé d'un coup ; il devient visible
et cassant seulement quand on doit garantir qu'une page suivante
reprend exactement là où la précédente s'est arrêtée.
**Remise en cause si** : un jour `id` cesse d'être strictement corrélé à
l'ordre de création (ex. import en masse avec des `id` explicites) —
pas le cas actuellement, aucune table de ce projet n'assigne `id`
manuellement.

### D4.3 — `joinedload` sur les 6 listes de jobs + `GET /datasets`

**Trouvé, vérifié** : `_to_summary()` de chacun des 6 routers de job
accède à `job.dataset`/`job.vision_dataset`, `job.created_by` et
`job.model`/`job.result` — 3 requêtes SQL supplémentaires PAR JOB sans
`joinedload` (N+1, AUDIT_DATALAB_2026-08-16.md §C.2.4). `GET /datasets`
a le même défaut sur `dataset.uploaded_by` — pas nommé explicitement
dans les "6 listes" de l'audit (qui ne visait que les jobs), mais
exactement la même classe de bug, dans la même zone fonctionnelle,
servant directement le risque R6 que ce correctif existe pour éliminer
— corrigé en même temps plutôt que laissé de côté par une lecture trop
littérale du périmètre.
**Retenu** : `.options(joinedload(...))` sur les 3 relations de chacun
des 7 endpoints — un seul aller-retour SQL désormais, quel que soit le
nombre de lignes.
**Écarté** : étendre `joinedload` à `GET /auth/team/audit-log`
(`AuditLog.actor`) ou `GET /vision/datasets` — hors du périmètre I3
(pas des listes de JOB), pas de risque N+1 signalé pour ces deux-là dans
l'audit ; à vérifier séparément si un jour ils deviennent lents en
pratique.

### D4.4 — Endpoint agrégé `GET /dashboard/summary` : réutilise les schémas existants, jamais une forme dupliquée

**Question** : `Dashboard.tsx` appelait 8 endpoints de liste complets à
chaque montage (`AUDIT_DATALAB_2026-08-16.md` ligne 170-171 : "Le
Dashboard appelle 8 endpoints de liste complets au montage"). Comment
construire l'agrégat sans dupliquer la définition de `TrainingJobSummary`
et des 5 schémas équivalents ?
**Retenu** : `api/routers/dashboard.py` importe directement les fonctions
`_to_summary` et classes `XJobSummary` de chacun des 6 routers de job
(alias à l'import pour éviter la collision de nom `_to_summary` entre
modules) — un seul endroit fait foi sur "à quoi ressemble un résumé de
job supervisé/clustering/...", jamais une seconde définition qui
pourrait diverger de l'original. Réutilise aussi `count_active_jobs`/
`ALL_JOB_MODELS` (`services/job_quota.py`), déjà le point de vérité sur
"tous les types de job confondus" pour le quota.
**Comptages via `COUNT(*)` SQL**, jamais en chargeant les lignes pour les
compter côté Python — `recent_*` se limite à 6 lignes par pilier (assez
pour dominer le tri final à 6, tous piliers confondus, sans jamais
ramener des milliers de lignes juste pour en garder 6).
**Test explicite** (`test_summary_recent_supervised_matches_list_training_jobs_shape`) :
vérifie que `recent_supervised[0]` de l'agrégat est BYTE-POUR-BYTE
identique à l'entrée correspondante de `GET /training/jobs` — la
réutilisation de `_to_summary` n'est pas qu'une intention documentée,
elle est vérifiée.
**Écarté** : dupliquer la logique de fusion/tri/troncature à 6 déjà
présente côté frontend (`Dashboard.tsx`, `useMemo` sur `activity`) —
l'agrégat renvoie 6 candidats PAR PILIER (36 au total dans le pire cas),
le frontend continue de fusionner/trier/tronquer à 6 exactement comme
avant, code inchangé au-delà de la SOURCE des données (1 champ de
l'agrégat au lieu de 6 états séparés).
**Remise en cause si** : le nombre de piliers augmente au point où
36 lignes remontées par montage devient un volume non négligeable —
réduire `_RECENT_PER_PILLAR` à ce moment-là, pas préventivement.

### D4.5 — `Dashboard.tsx` : dégradation par pilier abandonnée, assumé

**Trouvé, assumé** : le Lot 2A (branche séparée, non fusionnée à ce
jour) avait ajouté une dégradation indépendante par pilier au Dashboard
(un pilier en échec n'empêchait plus les autres de s'afficher). Ce lot
étant basé sur `main` (avant le Lot 2A), et remplaçant les 8 appels par
UN seul, cette dégradation fine n'a plus de sens : un succès ou un échec
est désormais forcément global (une seule requête HTTP, une seule
réponse).
**Retenu** : accepté comme compromis délibéré, pas un oubli — les 8
requêtes touchaient de toute façon la même base de données (pas des
systèmes indépendants), le risque de panne partagée entre elles était
déjà largement corrélé en pratique ; le gain de performance (1 requête
au lieu de 8, N+1 éliminé) l'emporte pour la page la plus visitée du
produit.
**Remise en cause si** : au moment de fusionner ce lot avec le Lot 2A
(branches actuellement séparées), le conflit sur `Dashboard.tsx` devra
être résolu à la main — décider À CE MOMENT-LÀ si la dégradation par
pilier vaut la peine d'être réintroduite par-dessus l'agrégat (ex. un
endpoint agrégé qui répond quand même partiellement si une SEULE requête
SQL interne échoue), pas maintenant, par anticipation d'un conflit qui
n'existe pas encore.

### D4.6 — Cache dataset : LRU en mémoire keyed par `(chemin, extension, mtime)`, jamais Parquet sur disque

**Question** : I4 (AUDIT_DATALAB_2026-08-16.md §I4) demande un « cache
Parquet par `dataset_id` + LRU en mémoire du worker » pour éviter de
relire le fichier dataset à chaque requête (preview/eda/histogram/
quality-check/feature-engineering-suggestions/feature-by-target
appellent chacun `read_dataframe` séparément depuis la même page, plus
une lecture par job créé côté router ET une seconde côté worker RQ).
**Retenu** : un seul mécanisme, `services/datasets.py::read_dataset_dataframe`
— `functools.lru_cache(maxsize=64)` sur une fonction privée
`_read_cached(path_str, extension, mtime_ns)`, la clé inclut le `mtime_ns`
du fichier (un `stat()`, négligeable face à un `pd.read_csv`/`read_excel`).
Retourne toujours une copie (`.copy()`) : aucun appelant ne peut muter
l'entrée partagée. Remplace `read_dataframe` sur tous les points d'appel
qui lisent un dataset DÉJÀ persisté (6 endpoints `datasets.py`, 1 chacun
dans `training.py`/`clustering.py`/`dimensionality.py` (×2)/`anomalies.py`,
et les 4 workers RQ) — jamais l'upload (`POST /datasets`), qui lit un
fichier qui vient d'être écrit, sans dataset encore en cache.
**Écarté** : un cache Parquet matérialisé sur disque (ce que l'audit
suggère littéralement) — ajoute un second fichier à gérer (invalidation,
nettoyage à la suppression du dataset, cohérence si l'upload échoue à
mi-chemin) pour un gain marginal : le fichier original est déjà local
(pas de S3/MinIO à ce stade, I5 non traité), le coût dominant n'est pas
le format CSV/Excel mais la RÉPÉTITION de la lecture — un LRU en mémoire
règle ça sans nouvel état sur disque à faire vivre.
**Limite assumée** : le cache est par PROCESS — l'API et chaque worker RQ
ont chacun le leur, pas de partage entre eux. Suffisant pour l'usage
visé (une page EDA qui enchaîne 6 requêtes vers le même process API ;
un worker qui traite plusieurs jobs successifs sur le même dataset) ;
un cache partagé inter-process demanderait Redis ou un fichier Parquet
partagé — reporté avec I5 (stockage partagé) si le besoin se confirme.
**Remise en cause si** : passage à plusieurs instances API derrière un
load-balancer sans affinité de session, où le gain par-process devient
marginal (chaque instance reconstitue son propre cache) — c'est alors
que le Parquet partagé ou un cache Redis prendrait tout son sens.

### D4.7 — `detect_task_type` : sortie du chemin HTTP non traitée dans ce lot

**Question** : I4 demande aussi de « sortir `detect_task_type` du chemin
HTTP » — `POST /training/jobs` lit le dataset ENTIER (`read_dataset_dataframe`,
training.py:483) juste pour déduire `task_type` de la colonne cible quand
`body.task_type` est absent, avant même de créer le job.
**Retenu** : non traité ici, documenté comme limite connue. Le cache
(D4.6) absorbe déjà le coût de RÉPÉTITION (2ᵉ création de job sur le même
dataset = lecture en cache), ce qui couvre le cas dominant en pratique.
Le coût réel restant — lire tout le fichier pour une seule colonne — ne
peut être éliminé sans lecture partielle spécifique à chaque format
(colonne unique en CSV/Parquet vs Excel/JSON qui n'offrent pas cette
primitive aussi simplement), un chantier plus large que ce lot.
**Écarté** : bricoler une lecture partielle seulement pour le CSV
(format le plus courant) — traiterait les formats de façon asymétrique
sans justification produit, pour un gain qui ne se matérialise que sur
les tout premiers jobs d'un dataset jamais encore lu.
**Remise en cause si** : la création de job devient mesurablement lente
en pratique sur de gros datasets (plusieurs centaines de Mo) — à ce
moment, ajouter une lecture partielle par format (ex.
`pd.read_csv(path, usecols=[target_column])`) deviendrait justifié.

### D4.8 — I6 : 3 files RQ par coût CPU/durée typique, pas par pilier produit

**Question** : I6 (AUDIT_DATALAB_2026-08-16.md §I6, dépend de C7 — déjà
traité au Lot 1.3) demande de séparer les files RQ pour qu'un job court
n'attende plus derrière un entraînement long. Une seule `training_queue`
partagée par les 6 types de job (supervisé, clustering, dimensionnalité,
anomalies tabulaires, classification vision, anomalies vision) — même
avec 2 répliques de worker (C7), deux entraînements longs simultanés
suffisent à occuper les deux workers, laissant un clustering de
quelques secondes attendre en file derrière eux. Comment découper les
"3 files RQ" que demande l'audit ?
**Retenu** : découpage par coût CPU/durée typique, pas par pilier
produit — `training_queue` (supervisé, recherche Optuna, le plus long),
`vision_queue` (classification + anomalies vision, torch CPU-only,
également long), `analysis_queue` (clustering + dimensionnalité +
anomalies tabulaires, pas de recherche d'hyperparamètres, nettement plus
courts en pratique). `job_timeout` aligné : 1800s pour les deux files
longues (inchangé), 600s pour `analysis_queue` (les 3 routers
concernés). Un service Docker Compose dédié par groupe de files
(`worker` → `training,vision`, replicas 2 ; `worker-analysis` →
`analysis`, replicas 1), piloté par la variable d'environnement
`RQ_QUEUES` lue par `workers/run_worker.py::_resolve_queues` — absente
(dev local), le worker écoute les 3 files, comme avant ce correctif.
**Écarté** : découper par pilier produit (ex. une file "tabulaire" et
une file "vision") — n'aurait pas résolu le problème initial, un
entraînement supervisé tabulaire (long) et un clustering tabulaire
(court) auraient continué à se gêner sur la même file.
**Écarté aussi** : donner une priorité RQ (`Worker([queue_prioritaire,
queue_secondaire])`) au lieu de workers dédiés par file — RQ ne fait
QUE choisir dans quel ordre un worker LIBRE pioche parmi ses files ; un
worker déjà occupé par un job long ne libère rien avant la fin de ce
job, quelle que soit la priorité. Seule une capacité dédiée
(`worker-analysis`, jamais partagée avec les jobs longs) garantit
qu'un job court ne soit jamais bloqué par un job long.
**Vérifié, pas supposé** : ~150 endroits (6 routers + 13 patches de
tests + 5 docstrings de worker) référençaient `training_queue` par ce
nom précis — chacun vérifié individuellement avant renommage (aucun
remplacement automatique aveugle) pour confirmer quelle nouvelle file
lui correspond.
**Remise en cause si** : le split observé en usage réel diverge de
l'hypothèse "clustering/dimensionnalité/anomalies tabulaires sont
courts" (ex. anomalies sur un dataset de plusieurs millions de lignes) —
`job_timeout=600` deviendrait alors trop court, à ajuster par mesure
réelle plutôt que par supposition.
