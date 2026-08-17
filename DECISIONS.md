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

## Lot 6A — Hisser la Vision au niveau du tabulaire (Phase 5 de l'audit, correctifs I8, I9, expansion du registre, généralisation de l'ingestion, D0.3)

Branche `lot-6a-vision-wizard`, basée sur `main`. Périmètre étendu en
cours de lot, à la demande explicite de l'utilisateur : au-delà de
I8/I9/I10 (les 3 items Phase 5 concernant l'entraînement/le wizard), le
registre de backbones et toute l'ingestion de dataset Vision (formats
d'archive, import de dossier, détection de structure) ont été
généralisés dans le même lot, jugés directement liés ("hisser la Vision
au niveau du tabulaire" couvre aussi la robustesse de l'ingestion, pas
seulement l'entraînement).

### D6A.1 — I8 : pondération de classes + arrêt anticipé + scheduler, tous 3 activés par défaut

**Question** : I8 demande 3 mécanismes manquants (pondération de
classes, early stopping, scheduler de taux d'apprentissage) — activés
par défaut ou seulement optionnels ?
**Retenu** : les 3 activés par défaut (`class_weighting=True`,
`early_stopping_patience=3`, `use_lr_scheduler=True`) — ce sont des
bonnes pratiques ML standard, pas des choix de style contestables ; les
laisser désactivés par défaut aurait reproduit le problème qu'I8 signale
("pas de pondération de classes") pour quiconque ne change pas les
défauts. Exposés en configuration (jamais imposés en dur) pour rester
désactivables si un besoin réel s'en présente.
**Vérifié** : pondération par fréquence inverse calculée UNIQUEMENT sur
le split d'entraînement (jamais validation/test) — testé explicitement
(`test_class_weights_only_uses_the_training_split`). Généralise à N
classes nativement (`CrossEntropyLoss(weight=...)` accepte un poids par
classe quel que soit leur nombre) — testé avec un dataset à 3 classes
(`test_training_handles_three_classes`), le produit supportait déjà le
multiclasse nativement (`ImageFolder`/métriques macro), vérifié plutôt
que supposé.
**Décision de conception** : `_should_stop_early` extraite en fonction
pure plutôt que la logique inline dans la boucle — la dynamique réelle
d'un entraînement sur un dataset synthétique minuscule n'est pas
déterministe (bruit de convergence), rendre la DÉCISION testable
séparément de l'entraînement réel évite des tests fragiles.
**Remise en cause si** : un utilisateur signale qu'un des 3 défauts nuit
à un cas d'usage réel (ex. pondération de classes qui dégrade un dataset
déjà équilibré) — ajuster le défaut, pas seulement documenter qu'il faut
le désactiver.

### D6A.2 — I9 : 4 presets d'augmentation, recommandation fondée sur la classe la plus petite

**Question** : quels seuils pour la recommandation "fondée sur la taille
du dataset" ?
**Retenu** : fondée sur la classe la plus PETITE (`min_class_size`),
jamais le total d'images — un dataset de 1000 images dont une classe à 5
reste, pour cette classe, un cas de sur-apprentissage à haut risque ; le
total masquerait ce déséquilibre. Seuils empiriques (< 20 → "forte", <
50 → "standard", < 150 → "légère", sinon "aucune"), documentés comme
indicatifs et non une science exacte. "standard" reproduit exactement
l'augmentation historique (seule option avant ce lot) — défaut
`augmentation_preset` inchangé pour quiconque ne choisit pas
explicitement un autre preset.
**Retenu aussi** : la recommandation n'est JAMAIS appliquée
automatiquement — un champ `recommended_augmentation_preset` sur
`GET /vision/datasets/{id}`, l'utilisateur choisit toujours explicitement
le preset final à l'entraînement.
**Remise en cause si** : les seuils s'avèrent mal calibrés en usage réel
— ajuster `_RECOMMENDATION_THRESHOLDS`, un seul point de vérité.

### D6A.3 — Registre de backbones étendu à 7 (au lieu de 2), jamais les 17 du legacy

**Question** (posée explicitement à l'utilisateur, scope conflictuel
avec une contrainte déjà documentée — voir le docstring pré-existant du
registre citant "aucun GPU... seuls des backbones légers") : fallait-il
étendre le registre de backbones vision, et jusqu'où ?
**Retenu**, réponse de l'utilisateur : ajouter "une poignée" de plus,
CPU-praticables — resnet34, mobilenet_v3_large, efficientnet_b0,
shufflenet_v2, densenet121 (7 au total avec les 2 existants). Chaque
backbone déclare `build_model`/`backbone_children`/`gradcam_target_layer`
(même contrat que l'existant) — `_resnet_backbone_children` et
`_mobilenet_backbone_children` réutilisées telles quelles quand la
structure interne coïncide (générique par NOM d'attribut, pas par
architecture), jamais dupliquées inutilement.
**Écarté** : les 17 backbones du legacy (VGG16/19, ResNet101/152...) —
le garde-fou de temps (`max_training_seconds`) empêcherait un blocage,
mais un backbone trop lourd compléterait très peu d'époques dans ce
budget, qualité douteuse en pratique — pas une vraie option utilisable,
juste une entrée de menu trompeuse.
**Vérifié** : les 3 tests déjà PARAMÉTRÉS sur `CLASSIFICATION_BACKBONE_REGISTRY`
(forward pass, freeze/unfreeze, support Grad-CAM) couvrent les 7
backbones automatiquement, sans modification. Ajouté en plus : 2 tests
Grad-CAM bout-en-bout (pas seulement "target existe") sur les motifs
`gradcam_target_layer` génuinement nouveaux (`.layer4` famille resnet,
`.conv5` shufflenet) — `.features` (mobilenet/efficientnet/densenet)
déjà couvert bout-en-bout par le backbone par défaut.

### D6A.4 — Ingestion Vision généralisée : zip/tar/tar.gz + import de dossier + classification pré-découpée, jamais de rétrogradation vers l'ancien flux zip-only

**Question** (posée explicitement à l'utilisateur) : pourquoi l'upload
Vision n'acceptait-il QUE .zip ? Réponse demandée : "la meilleure option
moderne", formats d'archive élargis ET import de dossier.
**Retenu** :
1. `services/vision_datasets.py` refactorisé autour d'une représentation
   UNIQUE et format-agnostique (`_ExtractedMember` : chemin relatif +
   contenu déjà en mémoire) — toute la logique de détection de
   structure/validation/déduplication ignore désormais complètement
   d'où viennent les octets. Trois sources alimentent cette liste :
   `_extract_zip_members`, `_extract_tar_members` (zip/tar/tar.gz/tgz,
   format détecté par SIGNATURE BINAIRE réelle, jamais par l'extension
   déclarée — défense en profondeur), `_members_from_uploaded_files`
   (dossier, chemins relatifs portés par `webkitRelativePath` côté
   navigateur).
2. `POST /vision/datasets` accepte désormais `files: List[UploadFile]` —
   1 fichier ⇒ tenté comme archive, plusieurs ⇒ import de dossier (même
   endpoint, jamais deux routes séparées à maintenir en parallèle).
3. Frontend (`VisionDatasets.tsx` ET `VisionDatasetPicker.tsx`, les DEUX
   points d'entrée d'upload Vision) : bouton "Importer un dossier" en plus
   du glisser-déposer archive, `<input webkitdirectory>` posé
   impérativement (attribut non standard, absent du typage JSX React).
**Vérifié en conditions réelles** (pas seulement des fixtures
synthétiques — demande explicite de l'utilisateur, dossier de test
fourni) : un vrai `tile.zip` MVTec AD (zip sans dossier englobant,
347 images, `ground_truth/` correctement exclu), un vrai dossier
`capsule/` DÉPLIÉ SUR DISQUE walké directement (462 fichiers réels,
351 images valides après exclusion de 111 fichiers non-dataset —
masques ground_truth + license.txt/readme.txt), un vrai dossier
`carpet/` (488 fichiers, 397 images valides) — tous corrects, zéro
faux positif/négatif.
**Écarté** : le glisser-déposer d'un DOSSIER entier (traversée de
`DataTransferItem.webkitGetAsEntry()`, récursive) — chantier séparé,
plus complexe qu'un bouton `<input webkitdirectory>` (bien supporté,
standard de facto) ; le glisser-déposer reste réservé aux archives.

### D6A.5 — Classification pré-découpée en train/test(/val) reconnue et fusionnée par classe, jamais rejetée

**Question** (trouvée en vérifiant avec les vrais fichiers de
l'utilisateur, pas anticipée à l'origine) : un dataset structuré
`train/<classe>/`, `test/<classe>/` (parfois aussi `val/<classe>/`) SANS
dossier "good" — donc pas une structure normal/défaut malgré la même
profondeur à 3 niveaux — était REJETÉ ("Structure de classification
invalide : doit être directement sous <classe>/"), alors qu'il s'agit
d'un dataset de classification légitime, juste pré-découpé en splits.
**Retenu** : `_detect_structure` distingue désormais les deux cas par le
contenu de `train/` : exactement `{"good"}` ⇒ structure normal/défaut
(validation stricte inchangée), n'importe quoi d'autre (ou vide) ⇒
classification pré-découpée — les 3 splits sont FUSIONNÉS par nom de
classe (même sac que la classification simple). Généralisé à train/test
**et** val (pas seulement train/test) pour les deux structures.
**Conséquence assumée** : le split d'origine (train/test/val) n'est PAS
respecté — `services/vision_classification_training.py` refait de toute
façon son propre split aléatoire stratifié à l'entraînement (aucune
notion de "respecter un split figé" n'existe ailleurs dans ce produit) ;
inventer cette notion aurait été un chantier séparé, hors périmètre ici.
**Bug réel trouvé en implémentant** : fusionner des splits par nom de
classe peut faire collisionner deux fichiers SOURCE distincts partageant
le même nom final (`train/chat/0.png` et `test/chat/0.png` copiés vers
la même classe) — la copie sur disque écrasait silencieusement l'un des
deux. Corrigé par désambiguïsation (`_2`, `_3`... suffixé) dans
`_validate_and_copy_images`, jamais un écrasement silencieux — testé
explicitement (`test_pre_split_classification_resolves_filename_collisions_across_splits`)
avec deux images de contenu RÉELLEMENT différent partageant le même nom.
**Changement de comportement assumé** : un dataset `train/good/` +
`train/scratch/` + équivalent sous `test/` — auparavant REJETÉ
("train/ ne doit contenir que des images normales") — est maintenant
accepté comme classification à 2 classes ("good" vs "scratch"). Jugé
strictement meilleur : le dataset est exploité plutôt que perdu, aucun
utilisateur ne préfère un rejet à un résultat exploitable quand les deux
interprétations sont structurellement valides. `test_mvtec_train_with_defect_folder_rejected`
renommé `test_train_with_a_non_good_folder_is_reinterpreted_as_classification`
pour refléter ce changement délibéré.
**Vérifié en conditions réelles** : les 3 fichiers "multiclasse" fournis
par l'utilisateur (`cable_multiclass.zip`, `leather_multiclass.zip`,
`screw_multiclass.zip` — train/test/val, 5 à 8 classes de défauts
nommées, aucun dossier "good"), auparavant tous rejetés, sont désormais
tous correctement détectés comme classification avec le bon compte
d'images par classe et ZÉRO perte de fichier (comptes sur disque
vérifiés égaux aux comptes rapportés).
**Remise en cause si** : un besoin de RESPECTER le split d'origine
(train/test/val figés, pas un re-split aléatoire) est exprimé — chantier
séparé, toucherait `vision_classification_training.py` en profondeur.

### D6A.6 — Terminologie "MVTec AD" : `structure_type` stocké inchangé, seuls les libellés/messages utilisateur renommés

**Question** (D0.3, reportée depuis le Lot 0) : `structure_type ==
"mvtec_ad"` et les messages associés nomment explicitement "MVTec AD"
alors que la structure est générique. Renommer quoi exactement ?
**Retenu** : `structure_type` reste littéralement `"mvtec_ad"` en base
(valeur technique interne, jamais vue par un utilisateur, déjà utilisée
par des enregistrements existants — la renommer exigerait une migration
de données pour zéro bénéfice utilisateur). Seuls les LIBELLÉS et
MESSAGES vus par un utilisateur changent : "MVTec AD" → "Normal /
défaut" (frontend, `STRUCTURE_LABELS` dans `VisionDatasets.tsx` ET
`VisionDatasetPicker.tsx`), messages d'erreur API/worker (`vision_anomalies.py`,
`vision_datasets.py`, `vision_anomaly_worker.py`, `vision_classification_worker.py`),
docstrings de module/classe décrivant le PÉRIMÈTRE de la fonctionnalité.
**Écarté** : les commentaires décrivant des particularités RÉELLES du
vrai jeu de données MVTec AD (ex. "un téléchargement MVTec AD officiel
inclut un dossier ground_truth/", "certaines catégories MVTec AD
officielles ont moins de 10 images") — laissés tels quels, ce sont des
références factuelles exactes justifiant un choix technique, pas un
mauvais étiquetage de LA fonctionnalité elle-même.
**Remise en cause si** : un besoin réel de renommer aussi la valeur
stockée apparaît (ex. rebranding produit) — migration de données
dédiée à ce moment, jamais mélangée à un correctif de libellés.

### D6A.7 — I10 : wizard 4 étapes (pas 5), stepper reconstruit avec les tokens corrigés plutôt que copié tel quel

**Question** : `Training.tsx` sur CETTE branche (basée sur `main`, avant
la branche `lot-2a-design-system` non fusionnée) porte encore le
stepper D'AVANT ses propres correctifs UI trouvés en revue directe
cette session (`overflow-x-auto` + boutons de défilement plutôt que
`flex-wrap`, `bg-white text-white` cassant en thème sombre). Fallait-il
porter fidèlement CE code (bug compris), ou la version déjà corrigée ?
**Retenu** : reconstruit avec le pattern CORRIGÉ (`flex flex-wrap`,
jamais de défilement + flèches ; `bg-card`/`text-primary-foreground`
pour les pastilles, jamais `bg-white`/`text-white`) — les deux tokens
existent déjà sur cette branche (vérifié dans `index.css` avant
utilisation, pas supposé), aucune raison de réintroduire un bug déjà
diagnostiqué et corrigé ailleurs dans une toute NOUVELLE page. `Training.tsx`
lui-même non touché (hors périmètre de ce lot, vit sur une autre
branche) — la convergence se fera à la fusion des branches.
**Retenu aussi** : 4 étapes ("Données & modèle" / "Augmentation" / "Mode
expert" / "Lancement"), pas 5 — Vision n'a pas d'équivalent à l'étape
"Qualité des données" de `Training.tsx` (pas d'EDA sur des images).
L'esprit du pattern (pastilles navigables, mode expert replié par
défaut, récapitulatif avant lancement) est porté fidèlement ; le NOMBRE
d'étapes de `Training.tsx` (5) était descriptif de SES étapes à lui,
jamais une exigence universelle.
**Retenu aussi** : le preset d'augmentation (I9) est pré-rempli avec la
recommandation du dataset choisi (`recommended_augmentation_preset`),
mais seulement tant que l'utilisateur n'a pas lui-même cliqué un preset
(`augmentationTouched`) — change de dataset après coup ne doit jamais
écraser un choix déjà fait explicitement.
**Vérifié** : `tsc -b` (0 erreur), `eslint` (0 erreur, mêmes
avertissements pré-existants sans rapport), `vite build` complet, build
de production réussi. Contrat API vérifié champ par champ contre
`VisionClassificationJobCreate` (backend) — tous les noms de champs du
payload envoyé correspondent exactement.
**Non vérifié visuellement** : aucun outil de capture d'écran
disponible dans cet environnement au moment de ce lot (tenté avec
Playwright, retiré à la demande explicite de l'utilisateur — "je teste
moi-même"). Le rendu réel (espacement, retour à la ligne des cartes de
preset, alignement de la pastille courante) n'a été vérifié que par les
tokens/classes utilisés, jamais par une capture — à confirmer par
l'utilisateur en conditions réelles.
