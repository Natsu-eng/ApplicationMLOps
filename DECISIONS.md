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

## Lot 5 — Traçabilité (Phase 4 de l'audit, correctif I2)

Branche `lot-5-tracability`, basée sur `main` (avant les Lots 2A/3/4,
sur des branches séparées non encore fusionnées — voir D4.5/D4.8/D4.9
sur `lot-4-perf` pour le même avertissement de réconciliation à venir).
Phase 4 complète de l'audit (I2, P1, P2, I5) traitée en plusieurs
correctifs séquentiels sur cette même branche, même découpage que
Lot 4 (I3/I4/I6/I7).

### D5.1 — Table `Prediction` dédiée, jamais un JSON sur `MLModel`

**Question** : I2 (AUDIT_DATALAB_2026-08-16.md §I2) demande de persister
chaque prédiction (`POST /training/jobs/{id}/predict`) — jusqu'ici
perdue sitôt la réponse HTTP envoyée, rendant impossible d'investiguer
après coup "le modèle a mal prédit pour ce dossier". Une table dédiée ou
un journal générique (réutiliser `AuditLog`) ?
**Retenu** : table `Prediction` dédiée (`api/core/models.py`) —
`organization_id`, `ml_model_id`, `requested_by_id`, `input_json`,
`output_json`, `created_at`. `output_json` capture prédiction +
probabilités + intervalle CQR (tout ce que l'audit désigne par "sortie"
et "intervalle"), **jamais** `explanation` (SHAP local) : recalculable à
la demande depuis le bundle + `input_json` (voir
`services/ml_inference.py::explain_one`), volumineuse, pas une donnée
qui fait foi — persister une valeur recalculable à l'identique aurait
été de la duplication sans bénéfice de traçabilité.
**Écarté** : réutiliser `AuditLog` (`services/audit.py`) — explicitement
scopé aux "actions sensibles" (suppression, promotion, ajout de membre),
volontairement minimal ; une prédiction est le chemin ROUTINE de
l'application, pas une action de gouvernance. Le détourner aurait rendu
`AuditLog` bruyant (potentiellement des centaines de lignes par heure)
pour un usage qui n'est pas le sien.
**Vérifié, pas supposé** : migration Alembic (`0860f4355873`) générée
par autogénération contre une base neuve, testée upgrade → vérification
du schéma réel (colonnes + index) → downgrade → vérification que la
table disparaît → ré-upgrade, avant tout commit (voir méthodologie de
session, `_as_aware_utc` déjà établi dans `job_watchdog.py`).

### D5.2 — Rétention par purge à la demande, jamais un scheduler dédié

**Question** : I2 demande "+ rétention" — sans borne, `predictions`
grossirait indéfiniment, et `input_json` peut contenir des données
personnelles saisies par l'utilisateur (aucune raison de les garder
indéfiniment).
**Retenu** : même principe que `services/job_watchdog.py::reconcile_stale_jobs`
(déjà en place, Lot AUDIT_ROADMAP.md H2) — une purge à la demande,
appelée juste avant d'enregistrer une nouvelle prédiction de la MÊME
organisation (`services/prediction_retention.py::purge_old_predictions`,
`Settings.prediction_retention_days = 90`). Une prédiction jamais
réutilisée est donc purgée au plus tard à la prochaine prédiction de sa
propre organisation — pas de process séparé, pas de dépendance nouvelle
(Celery beat ou équivalent), cohérent avec "pas de scheduler dédié" déjà
choisi pour les jobs orphelins.
**Écarté** : un script de purge lancé par un cron externe (comme
`backend/scripts/smoke_test_docker.py` sur la branche
`fix-api-prefix-routing`) — aurait exigé une tâche cron/Task Scheduler
en plus de l'application elle-même, jamais garantie de tourner (pas
d'infrastructure de planification existante dans ce projet) ; la purge à
la demande, elle, s'exécute à coup sûr dès qu'une organisation redevient
active.
**Vérifié, pas supposé** : comparaison de dates faite en PYTHON, jamais
par un filtre SQL sur `created_at` — un filtre SQL direct aurait paru
fonctionner en local (SQLite) tout en étant fiable différemment en
production (PostgreSQL), le même écart de fuseau horaire que
`job_watchdog.py::_as_aware_utc` documente déjà. Constaté concrètement
en écrivant `test_old_predictions_are_purged_on_the_next_prediction`
(`TypeError: can't compare offset-naive and offset-aware datetimes` —
sur l'assertion du TEST, pas sur la purge elle-même, qui utilisait déjà
`_as_aware_utc` en interne).
**Remise en cause si** : le volume de prédictions par organisation
devient assez élevé pour que "charger tous les id/created_at de
l'organisation à chaque prédiction" devienne mesurablement coûteux — un
index composite `(organization_id, created_at)` ou un vrai job planifié
deviendrait alors justifié.

### D5.3 — Historique `GET /training/jobs/{id}/predictions` : `limit` simple, pas encore le curseur du Lot 4

**Question** : comment exposer l'historique des prédictions d'un job ?
**Retenu** : `GET /training/jobs/{job_id}/predictions`, un paramètre
`limit` simple (défaut 50, max 500), tri par `id` décroissant (même
raison qu'ailleurs dans le projet : `id` est sans ambiguïté, jamais
`created_at`, voir D4.2 sur `lot-4-perf`).
**Écarté** : la pagination par curseur de `api/core/pagination.py`
(Lot 4/I3) — ce module vit sur `lot-4-perf`, une branche distincte non
fusionnée au moment de ce lot ; le dupliquer ici aurait créé deux
implémentations concurrentes du même mécanisme. La rétention (D5.2)
borne déjà la taille de cette table dans le temps, rendant un simple
`limit` suffisant pour l'instant.
**Remise en cause si** : au moment de fusionner `lot-4-perf` et
`lot-5-tracability`, harmoniser cet endpoint sur `paginate_by_id` comme
les autres listes — à traiter EXPLICITEMENT à ce moment (même avertissement
que D4.5 pour `Dashboard.tsx`), pas anticipé ici.

### D5.4 — Pas de surface frontend pour I2 dans ce lot

**Question** : faut-il un onglet "Historique des prédictions" dans
`ModelResultModal.tsx` pour ce correctif ?
**Retenu** : non — la colonne "Fichiers concernés" d'I2 dans l'audit ne
liste que `models.py`/`training.py` (contrairement à I1, explicitement
"back + front"). Le backend est prêt (persistance + endpoint), la
surface frontend est un chantier séparé, cohérent avec le traitement de
P6 (pagination UI) différé au Lot 4 (D4.1).
**Remise en cause si** : un utilisateur/le produit demande explicitement
cette vue — à ce moment, un nouvel onglet dans `ModelResultModal.tsx`
consommant `GET /training/jobs/{id}/predictions` (déjà prêt côté API).
