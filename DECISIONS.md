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

## Lot 2A

### D2.1 — Contraste des tokens sémantiques : vérifié par le calcul, deux bugs réels trouvés en sombre

**Méthode** : conversion OKLCH → sRGB linéaire → luminance relative → ratio
WCAG, implémentée directement (formules de référence, aucune dépendance
ajoutée) plutôt que devinée à l'œil. Script conservé dans le scratchpad de
session, résultats reproduits ci-dessous.

**Trouvé en vérifiant, pas supposé** :
1. `--color-destructive`/`--color-warning`/`--color-success` n'avaient
   JAMAIS été redéfinies pour le mode sombre — le thème sombre réutilisait
   silencieusement les valeurs calibrées pour un fond blanc (L 0.58/0.56/
   0.53), qui tombent à 3.96–4.01:1 sur `--color-card` en sombre (sous le
   seuil AA 4.5:1 texte normal). `--color-primary` avait bien une valeur
   sombre dédiée (0.6) mais elle aussi sous le seuil (4.33:1).
2. **Un même token ne peut pas satisfaire deux rôles à la fois en sombre** :
   "texte de la couleur sur une carte" veut un L PLUS HAUT (≥0.59–0.64
   selon la teinte) ; "texte blanc sur un remplissage plein de cette
   couleur" (ex. `Button variant="destructive"`) veut un L PLUS BAS
   (≤0.54–0.585) pour que le blanc garde 4.5:1. Les deux plages ne se
   recoupent pas.

**Retenu** :
- `--color-primary/destructive/warning/success` (mode sombre) relevés à
  0.62/0.64/0.63/0.60 — texte lisible sur `--color-card` (4.65–4.77:1,
  marge incluse).
- Nouveaux tokens `--color-{nom}-solid` (mode sombre uniquement — en clair,
  ils valent la même chose que le token de base, pas de divergence) :
  0.53/0.555/0.53/0.51 — pour un remplissage plein + texte blanc dessus
  (`Button variant="destructive"` migré vers `bg-destructive-solid`).
  5.07–5.32:1 avec le texte blanc, marge incluse.
- `--color-info`/`--color-info-foreground` ajoutés (4ᵉ sémantique demandée
  par l'audit, hue 230 — distincte de primary/258) : 5.42:1 en clair,
  4.69:1 en sombre (rôle texte) / 4.71:1 (rôle solid + blanc).

**Non fait dans ce lot (2B)** : les usages PAGE existants de `bg-primary
text-white`/`bg-success text-white` (ex. `Training.tsx`, indicateurs
d'étape) continuent d'utiliser le token "texte" (maintenant plus clair en
sombre, donc leur contraste ne s'améliore ni ne se dégrade par rapport à
avant ce lot — toujours non vérifié). Migrer ces usages vers les tokens
`-solid` appropriés est un travail de page, hors périmètre 2A ("aucune
page métier modifiée").
**Remise en cause si** : le Lot 2B révèle d'autres combinaisons
texte/fond non couvertes par cette vérification ponctuelle — étendre le
script de vérification plutôt que revalider à l'œil.

### D2.2 — Périmètre du critère « plus aucune taille arbitraire »

L'audit liste, comme critère de fin du Lot 2A : « plus aucune taille
arbitraire de type `text-[11px]` ne doit subsister ». 19 fichiers de
pages/composants métier utilisent aujourd'hui ce motif (`grep -rn
"text-\["`) — les migrer TOUS entrerait en contradiction directe avec
l'autre exigence du même lot : « aucune page métier modifiée ».
**Retenu** : le critère s'applique aux PRIMITIVES (`components/ui/*`) et à
la page `/design` elle-même — reconstruites dans ce lot sans aucune taille
arbitraire (`text-overline` remplace explicitement le motif `text-[11px]`
partout où il apparaissait dans les primitives touchées : `Table`,
`StatTile`, `Tabs`). Les 19 fichiers métier restants sont balisés pour le
Lot 2B, qui applique le système page par page.
**Remise en cause si** : l'utilisateur, à la revue de `/design`, indique
que le critère visait bien un balayage complet immédiat — dans ce cas
c'est un changement de périmètre du 2A à traiter avant le 2B, pas un
oubli à corriger silencieusement.

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

## Correctif — Routage `/api`, taille d'upload nginx, test de fumée Docker

**Trouvé (retour utilisateur, incident réel — l'application ne fonctionnait
pas du tout une fois déployée)** : aucun router backend n'était préfixé
`/api` (`app.include_router(auth_router)` etc., sans `prefix=`), alors que
`nginx/nginx.conf` ne proxifie QUE `location /api/` vers le backend — tout
le reste (`/auth/login`, `/datasets`, `/training/...`) tombait dans
`try_files ... /index.html` et recevait le HTML de la SPA en 200.
`POST /auth/login` renvoyait donc du HTML, `res.json()` levait une
`SyntaxError` côté frontend, connexion impossible. Symptôme identique en
dev : `vite.config.ts` proxifiait des préfixes par domaine métier
(`/training`, `/clustering`, `/datasets`, `/vision`, ...) qui interceptaient
AUSSI les routes de PAGE de même nom — rafraîchir `/training` en dev
renvoyait le JSON d'erreur du backend, jamais le HTML de la SPA (c'est ce
qui m'empêchait de vérifier les pages du Lot 2A/2B dans un navigateur).

**Retenu — cause racine (pas un contournement)** :
- Tous les routers métier préfixés `/api` dans `api/main.py`
  (`app.include_router(router, prefix="/api")`, se compose avec le préfixe
  propre de chaque router — aucun router modifié individuellement).
- `frontend/src/api/client.ts` : un point de composition UNIQUE
  (`apiUrl(path)`) préfixe `/api` pour les 4 chemins fetch existants
  (`request`, `requestForm`, `uploadFileWithFields`, `exportModel`) ET pour
  le fetch direct de `VisionImage.tsx` — les ~50 chemins d'endpoints dans
  l'objet `api` restent inchangés (`/auth/login`, `/datasets`, ...), jamais
  touchés un par un (risque d'oubli).
- `vite.config.ts` réduit à UNE SEULE entrée de proxy (`/api`) — les 6
  entrées par domaine métier supprimées, cause du symptôme dev (point 3).
- **Pas de `VITE_API_URL` injecté au build** (délibéré, pas un oubli) :
  une fois TOUT préfixé `/api` et nginx proxifiant `/api/` vers le
  backend, un chemin relatif (`BASE_URL=""`) fonctionne nativement en
  same-origin derrière nginx — injecter une URL absolue au build
  coupartirait un artefact Docker unique à une seule URL de déploiement,
  contrairement au principe même d'une image Docker réutilisable.
- **267 chemins littéraux dans 15 fichiers de tests backend** (`client.post("/auth/...")`
  etc.) préfixés `/api` mécaniquement (script Python, remplacement
  `"<prefixe>` → `"/api<prefixe>`, vérifié par diff avant/après) — sans ça,
  le correctif routeur cassait silencieusement toute la suite (525 tests).
  Vérifié sur un sous-ensemble ciblé (auth/body-size-limit/datasets,
  32/32) plutôt que la suite complète, à la demande explicite de
  l'utilisateur («uniquement ce test, refais pas tous les tests»).

**Retenu — `client_max_body_size`** : 20 Mo en dur dans nginx alors que
`max_upload_size_mb=200`/`max_vision_upload_size_mb=500` côté backend —
tout import réel aurait été rejeté en 413 avant même d'atteindre
l'application. `nginx/nginx.conf` → `nginx/templates/default.conf.template`
(templating envsubst officiel de l'image nginx, déclenché automatiquement
au démarrage du conteneur) : `client_max_body_size ${MAX_VISION_UPLOAD_SIZE_MB}M`
lit LA MÊME variable que le backend, ajoutée explicitement à
`backend/.env`/`backend/.env.example` (jusqu'ici seulement un défaut
Python implicite, jamais une ligne visible dans `.env`) — `docker-compose.yml`
donne au service `frontend` `env_file: backend/.env` pour que cette
variable atteigne le conteneur nginx à l'exécution.
**Écarté** : dupliquer un nombre nginx distinct en dur — exactement le
problème signalé (deux limites qui peuvent diverger silencieusement).

**Trouvé en marge, corrigé (bloquant pour le test de fumée)** : `docker
compose up -d --build` (commande documentée dans `README.md`,
`backend/README.md` et l'en-tête de `docker-compose.yml`) résolvait
`${POSTGRES_USER}`/`${POSTGRES_PASSWORD}`/`${POSTGRES_DB}` à des chaînes
VIDES — Docker Compose interpole son propre fichier via un `.env` à la
racine du dépôt, un mécanisme distinct de `env_file: backend/.env` (qui
n'injecte des variables QUE dans les conteneurs qui le déclarent, jamais
dans le texte du fichier compose lui-même). `DATABASE_URL` se résolvait
donc en `postgresql://:@db:5432/` — le backend ne pouvait jamais se
connecter à Postgres en Docker, silencieusement, aucune erreur au
démarrage de `docker compose up`. `.env.example` ajouté à la racine
(mêmes valeurs que `backend/.env.example`, fichiers distincts par
nécessité des deux mécanismes Compose) ; README.md/backend/README.md et
l'en-tête de `docker-compose.yml` mis à jour pour documenter les DEUX
copies `.env.example` → `.env` requises.
**Pourquoi traité ici et pas signalé sans agir** : ce bug bloquait
littéralement le test de fumée demandé (item 4) — `docker compose up`
n'aurait jamais démarré une base fonctionnelle, peu importe la qualité du
reste du correctif. Nécessaire à la livraison, pas une extension de
périmètre choisie librement.

**Retenu — test de fumée (item 4)** : `backend/scripts/smoke_test_docker.py`,
httpx (déjà une dépendance backend, aucune nouvelle dépendance) — attend
`GET /api/health` (`database: up`), vérifie `GET /` sert du HTML (pas une
erreur nginx), puis rejoue inscription → connexion (form-encoded, PAS
JSON — exactement le chemin qui recevait du HTML pendant l'incident) →
liste des datasets, contre `http://localhost` (nginx, port 80), jamais
contre le backend ni `uvicorn --reload` directement. Nouveau job CI
`smoke` (`.github/workflows/ci.yml`), parallèle aux jobs `backend`/
`frontend` existants : construit la stack Docker complète, lance le
script, publie les logs des conteneurs en cas d'échec, arrête toujours la
stack (`if: always()`).
**Pourquoi aucun des 525+42 tests existants ne pouvait attraper ça** :
`tests/*.py` appelle `TestClient(app)` (FastAPI en mémoire, jamais nginx) ;
les tests frontend (Vitest) montent des composants isolés, jamais un
navigateur réel contre un serveur. Les deux contournent PAR CONSTRUCTION
la couche qui a cassé (le routage entre nginx et le backend) — un vert
sur les deux ne prouve rien sur le déploiement réel.
**Remise en cause si** : le job `smoke` s'avère trop lent/flaky en CI
(build Docker complet à chaque run) — envisager de le limiter aux push
sur `main`/PR vers `main` plutôt que toutes branches, si le coût devient
gênant. Pas fait préventivement : pas de signal actuel que c'est un
problème.

### D2.3 — Correctifs Arrêt A (direction visuelle refusée en premier passage)

**Question** : le premier rendu du Lot 2A a été jugé « techniquement
correct mais générique » — 6 corrections demandées avant toute nouvelle
tentative de validation, plus une démonstration sur la page Tableau de
bord (et uniquement celle-ci) avant d'aller plus loin.

**Retenu** :
1. **Palette catégorielle reconstruite** (`theme/charts.ts`) — l'ancienne
   (bleu roi/orange/vert menthe/jaune/rose/vert foncé) confondait les
   séries 3 et 6 (deux verts) et plaçait orange/jaune en teintes
   adjacentes. Nouvelle séquence ancrée sur Okabe-Ito puis affinée par
   recherche gloutonne en OKLab pour maximiser le ΔE minimal entre paires,
   vérifié simultanément en mode clair, sombre et simulation
   deutéranopie : ΔE min. 15,6 (vision normale) / 13,1 (deutéranopie),
   contre 3,8 pour l'ancienne palette. Limitée à 6 séries.
2. **Désaturation des remplissages pleins** — `--color-destructive`,
   `--color-warning`, `--color-success` et leurs variantes `-solid`
   étaient à chroma quasi maximal. Chroma réduit en clair (0.58→0.560,
   0.56→0.545, 0.53→0.510) en conservant les ratios de contraste déjà
   validés (≥4.5:1, carte ET fond — voir script de vérification étendu).
3. **Nesting de conteneurs** — vérifié sur Dashboard.tsx : la page ne
   présentait pas l'anti-motif page→section→carte→sous-carte décrit par
   l'audit (déjà page→Card→liste à filets `divide-y`, sans sous-carte).
   Aucun changement structurel nécessaire ; seuls les deux titres de
   section ad hoc (`text-sm font-medium`) ont été alignés sur la nouvelle
   hiérarchie typographique (`text-subtitle`, point 4).
4. **Hiérarchie de sections** — `PageHeader` migré vers `text-title`,
   `SectionHeader` vers `text-subtitle` (était `text-sm font-medium`,
   `mb-3`→`mb-4`) : les deux niveaux se distinguent maintenant clairement
   en taille/poids/espacement, plus une hiérarchie plate.
5. **StatTile recomposé** — l'ancienne mise en page (icône + bloc
   chiffre/libellé côte à côte + barre colorée en haut) cassait sous
   contrainte réelle (libellé tronqué, delta qui repasse à la ligne).
   Refaite en lignes empilées pleine largeur (icône seule / chiffre
   dominant + delta aligné / libellé complet) ; barre d'accent supprimée,
   l'identité de couleur reste portée par l'icône.
6. **Typographie unifiée** — `--font-sans` défini une fois dans
   `index.css` et appliqué à `body` ; `font-serif` retiré du titre héros
   (`AuthBrandPanel`) et de `PageHeader`, seuls endroits où il subsistait.

**Démonstration** : système appliqué à `frontend/src/pages/Dashboard.tsx`
uniquement (titres de section alignés sur `text-subtitle` ; le reste de
la page héritait déjà des primitives corrigées ci-dessus sans changement
propre à la page). Les 16 autres pages restent inchangées, réservées au
Lot 2B. Gate complet relancé et vert (tsc, lint 0 erreur, 42/42 tests,
build).

**Écarté** : retoucher `Card.tsx` (les 4 variantes existantes,
notamment `outlined`, couvrent déjà le besoin de filets sans ombre —
aucune page de démonstration n'a exposé de cas que la primitive actuelle
ne couvre pas).

**Remise en cause si** : en appliquant le système à une page du Lot 2B,
l'anti-motif page→section→carte→sous-carte apparaît réellement (contrairement
à Dashboard) — alors `Card.tsx` devra être revu à ce moment-là, pas avant.

### D2.4 — Résilience du Dashboard par pilier + couleurs de pilier réelles

**Trouvé (retour utilisateur, après validation D2.3)** : un seul endpoint
en échec (`GET /vision/anomalies/jobs`, 500 — voir la migration de
rattrapage D1.x sur `fix-lot1-migration-safety`) mettait à « — » les
compteurs Supervisé, Non supervisé, Vision ET En cours, alors que les
autres endpoints répondaient 200. Cause : `jobs`/`clusteringJobs`/
`dimensionalityJobs`/`anomalyJobs`/`visionClassificationJobs`/
`visionAnomalyJobs` restaient à `null` indéfiniment sur erreur (seul un
message d'erreur était posé, jamais la liste elle-même) — `allJobsLoaded`
exigeait que LES SIX soient non-null, donc UN SEUL échec permanent
bloquait TOUS les compteurs, pour toujours (pas seulement le temps du
chargement).

**Retenu** :
- Les 6 loaders règlent maintenant leur liste à `[]` sur erreur (plus
  jamais `null` indéfiniment) — un pilier "réglé" (settled) veut dire
  "a une réponse, succès ou échec", distinct de "encore en chargement".
- 3 indicateurs de réglage indépendants (`supervisedSettled`/
  `nonSupervisedSettled`/`visionSettled`) remplacent le seul
  `allJobsLoaded` global pour la tuile "Analyses ML" : chaque colonne du
  split (Supervisé/Non supervisé/Vision) affiche son vrai chiffre dès que
  SON pilier répond, indépendamment des 2 autres.
- "Dernière activité" affiche désormais un message ciblé par pilier
  (`Activité Vision indisponible — <détail>`) au lieu d'un unique bandeau
  d'erreur qui remplaçait toute la liste — et continue d'afficher les
  entrées des piliers qui ONT répondu même si un autre a échoué (le calcul
  `activity` tolérait déjà les sources `null`/vides via `?.forEach` ; seul
  le RENDU bloquait tout derrière un seul état global).
- "En cours" (agrégat trans-pilier, pas de pilier propre) se peuple dès
  que les 6 sources sont réglées — plus jamais bloqué indéfiniment par un
  échec permanent d'une seule d'entre elles.

**Couleurs de pilier** (même retour) : `Datasets en bleu, Membres en
violet` sur les tuiles n'avait aucun sens — aucune de ces données
n'appartient à un pilier. `AccentColor` (`ColorIconBadge.tsx`) gagne une
valeur `"neutral"` (tokens `muted`/`border` du thème, même convention que
`Badge variant="neutral"`) pour ces tuiles trans-piliers (Datasets,
Analyses ML dans son ensemble, En cours, Membres). `Pillar.color`
(`config/pillars.ts`) devient la source UNIQUE de la couleur d'un pilier
(`pillarColor(id)`) — violet/rose/teal pour supervisé/non supervisé/
Vision respectivement, alignées sur l'usage déjà dominant de chaque
pilier sur ses propres pages (`Training.tsx`, `Clustering.tsx`,
`VisionDatasets.tsx`) pour que le Lot 2B n'ait pas à réconcilier une
couleur différente plus tard. `StatTileSplitPart` gagne un `color?`
optionnel pour teinter individuellement chaque colonne du split avec la
couleur de SON pilier.

**Écarté** : recolorer les icônes de `ACTIVITY_KIND_META` (liste
"Dernière activité") sur la couleur du pilier — elles différencient
aujourd'hui le TYPE de job (clustering/réduction/anomalies en 3 teintes
distinctes au sein du même pilier non supervisé), une information plus
fine que l'identité de pilier, non signalée comme "arbitraire" par le
retour utilisateur (contrairement aux tuiles). Non touché.

**Remise en cause si** : le Lot 2B révèle qu'une page a besoin de la
couleur de pilier à un endroit où `pillarColor()` ne suffit pas (ex.
mélange avec un statut) — étendre l'API plutôt que dupliquer la couleur
en dur localement.

## Lot 2B

### D2.5 — Premier incrément : couleurs de pilier + dernières tailles arbitraires

**Contexte** : Lot 2B applique le système du Lot 2A (déjà validé sur
Dashboard) aux 16 autres pages. Travaillé dans un worktree Git séparé
(`../app-analyse-lot2b`) pendant qu'une vérification backend tournait sur
une autre branche, pour ne jamais perturber un `pytest` en cours en
changeant les fichiers sous ses pieds (leçon tirée d'un incident réel
cette même session — un `git stash`/`checkout` avait fait disparaître des
fichiers pendant qu'un test les utilisait encore).

**Retenu — couleurs de pilier partout** : les 11 pages appartenant
explicitement à un pilier (déjà déclaré sans ambiguïté via `<AppShell
pillarId="...">`, un signal existant réutilisé plutôt qu'inventé) migrent
leur `PageHeader` de leur couleur codée en dur vers `pillarColor(id)`.
Plusieurs collisions/incohérences réelles trouvées en croisant la couleur
canonique du Lot 2A (D2.4 : supervisé=violet, non supervisé=rose,
Vision=teal) avec l'usage existant : `TrainingHistory` (amber),
`AnomalyDetection` (amber), `DimensionalityReduction` (blue),
`UnsupervisedHistory` (violet — entrait en collision avec supervisé),
`VisionClassification`/`VisionHistory` (violet — même collision),
`VisionAnomalies` (amber), `Datasets` (teal — entrait en collision avec
Vision). Chacune corrigée vers la couleur réelle de son pilier.
**Écarté** : recolorer aussi les `SectionHeader`/`MetricTile` internes à
chaque page (ex. `Clustering.tsx` a un `SectionHeader color="violet"`
pour "Profils de segments") — même raisonnement que D2.4 : ils
différencient une sous-section, pas l'identité de la page, non signalés
comme un problème.
**Dashboard/Profile/DesignSystem non touchées** : leur couleur (bleu/
violet) n'appartient à aucun pilier par construction (vue d'ensemble
trans-pilier, compte, page de référence hors nav) — laisser une couleur
neutre-ish plutôt que d'en forcer une de pilier serait plus cohérent,
mais aucune de ces 3 pages n'a été signalée comme un problème ; non
touchées pour rester strictement dans le périmètre du retour utilisateur.

**Retenu — dernières tailles arbitraires** : `_schema` ci-dessus n'a rien
à voir, mais même logique que D2.2 : `grep -rn "text-\["` sur tout `src/`
trouvait encore 17 fichiers (Table.tsx/StatTile.tsx, déjà migrés au Lot
2A malgré ce que D2.2 affirmait, se sont révélés propres — la divergence
venait d'avoir grep-é la mauvaise branche, voir note ci-dessous). Tous
migrés vers `text-caption` (texte simple, la majorité des cas) ou
`text-overline` (motif déjà uppercase/tracking-wide/badge sémantique,
ex. `AppShell.tsx` en-tête de section de nav, badges de qualité) —
critère : la présence de `font-semibold`/`uppercase`/`tracking-wide`
redondants dans la classe existante signale un overline, sinon caption.
`text-[9px]`/`text-[10px]` (Heatmap, Avatar) n'ont pas d'équivalent plus
petit que `text-overline`/`text-caption` (11px/12px) dans l'échelle —
légère augmentation de taille assumée plutôt que de garder une valeur
arbitraire.

**Piège rencontré, documenté pour la suite** : grep-er `frontend/src`
depuis la branche `fix-lot1-migration-safety` (qui ne contient AUCUN
changement Lot 2A, basée sur `main`) donnait des résultats trompeurs
(StatTile.tsx y apparaissait encore avec `text-[11px]`, alors qu'il est
déjà propre sur `lot-2a-design-system`). Toujours vérifier sur quelle
branche/worktree une lecture de fichier a réellement lieu avant d'en
tirer une conclusion.

**Reste à faire (Lot 2B, pas dans cet incrément)** : nesting de
conteneurs et hiérarchie de sections page par page (au-delà de ce que
les primitives déjà corrigées appliquent automatiquement), vérification
visuelle par l'utilisateur sur les 16 pages comme cela a été fait pour
Dashboard.

### D2.6 — Audit du nesting de conteneurs : un seul cas réel trouvé, corrigé

**Vérifié, pas supposé** : plutôt que de retravailler visuellement les 16
pages à l'aveugle, audit systématique des 82 usages de `Card` (17
fichiers) pour trouver où l'anti-motif "carte dans une carte" existe
RÉELLEMENT, plutôt que de le supposer partout par précaution.

**Trouvé** : un seul cas, dupliqué à deux endroits. `VisionDatasetPicker`
(`components/vision/VisionDatasetPicker.tsx`), partagé par
`VisionClassification.tsx` et `VisionAnomalies.tsx`, enveloppait sa zone
de dépôt (drag-and-drop ZIP) dans sa propre `<Card>` — et les deux pages
appelantes enveloppent déjà tout leur formulaire de configuration dans
une `<Card>`. Résultat : une carte bordée+ombrée dans une carte
bordée+ombrée, exactement le motif visé par le correctif 3.
**Retenu** : la zone de dépôt devient un `<div>` avec un simple filet
pointillé à faible opacité (`border border-dashed border-border/70`,
sans `shadow-card` ni `bg-card` propres — elle hérite du fond de la Card
parente), conformément au principe "la carte est l'exception qui signale
un regroupement, pas le conteneur par défaut". L'état actif au survol
d'un fichier glissé (`border-primary/60 bg-primary/5`) est conservé
inchangé.
**Écarté** : retravailler `Card.tsx` lui-même ou les 14 autres pages —
l'audit n'y a trouvé AUCUNE carte imbriquée (toutes les cartes sont des
frères dans une grille/pile), confirmant que le système de cartes du
Lot 2A n'a pas de défaut structurel généralisé ; c'était un usage
ponctuel d'un composant partagé, pas un problème de primitive.
**Remise en cause si** : un futur composant partagé réintroduit ce motif
(une Card à l'intérieur d'un composant destiné à être imbriqué dans une
page qui a elle-même une Card) — vérifier au moment de l'écrire, pas
après coup.

### D2.7 — Derniers titres de section sans hiérarchie réelle

**Trouvé** : `grep -rn 'text-sm font-medium text-foreground'` sur
`pages/` et `components/` — au-delà des `<p>` (libellés d'item de liste,
hors périmètre, même raisonnement que pour Dashboard) — a trouvé 4 vrais
titres de section (`<h2>`/`<h3>`) encore au même poids visuel que le
corps de carte qu'ils introduisent, le bug exact corrigé sur Dashboard
par le correctif 4 mais pas propagé au reste du produit : `Profile.tsx`
(section "Équipe", formulaire "Ajouter un membre"), `Training.tsx`
(titre de chaque étape du wizard, `StepContent`), et `Modal.tsx` (titre
de CHAQUE modale de l'app — primitive partagée par `ModelResultModal`,
`EdaModal`, les modales de détail Vision, etc.).
**Retenu** : les 4 migrés vers `text-subtitle`, même traitement que
`SectionHeader`/Dashboard. `Modal.tsx` étant une primitive, ce correctif
bénéficie automatiquement à toutes les modales de l'app sans les
retoucher une par une.
**Écarté** : les usages `<p>` du même motif (libellés d'étape dans
`HelpModal`, nom de classe dans `VisionDatasetExplorer`, "Mode expert"
dans `ExpertModePanel`) — ce sont des libellés d'item, pas des titres de
section, même distinction que pour Dashboard (D2.3 correctif 4).
Gate complet relancé et vert (tsc, lint, 42/42 tests, build).

### D2.8 — Dernier `font-serif` : 4 pages hors `PageHeader`

**Trouvé** : `grep -rn font-serif` sur tout `src/` — le correctif 6
(typographie unique) n'avait retiré `font-serif` que d'`AuthBrandPanel`
et de `PageHeader`. 4 pages qui n'utilisent PAS `PageHeader` (écrans hors
`AppShell` : `Login.tsx`, `Register.tsx`, `Orientation.tsx`,
`ComingSoon.tsx`) avaient leur propre `<h1 className="text-2xl
font-serif">` local, invisible à la recherche initiale limitée aux
composants déjà touchés.
**Retenu** : les 4 migrées vers `text-title` (même token que
`PageHeader`), plus aucune occurrence de `font-serif` dans tout `src/`.
**Vérifié** : `grep -rn font-serif frontend/src` ne retourne plus rien.
Gate complet relancé et vert (tsc, lint, 42/42 tests, build).

### D2.9 — Retour utilisateur en revue directe (branche mélangée par erreur)

**Incident de process, corrigé** : entre la revue Dashboard et ce point,
j'ai basculé le répertoire de travail PARTAGÉ (`E:\mlops\app-analyse`,
celui que le serveur de dev de l'utilisateur sert) vers d'autres branches
(`fix-lot1-migration-safety`, `main`, `fix-api-prefix-routing`) pour des
correctifs sans rapport, sans que l'utilisateur en soit informé. Résultat :
en révisant l'app dans son navigateur, il voyait par moments l'état
PRÉ-Lot 2A (sans les tokens `rounded-card`/`text-overline`/`text-body`/
`text-caption`), a édité 4 fichiers directement dans ce répertoire en
réutilisant ces tokens de bonne foi (cohérents avec ce qu'il avait vu
validé), mais sur la mauvaise branche — ses edits ne "prenaient" pas
visuellement (tokens non définis dans l'`index.css` de cette branche-là).
**Retenu** : les 4 fichiers relus intégralement, la substance de chaque
edit portée sur `lot-2a-design-system` (la seule branche où ces tokens
existent) :
- `Datasets.tsx` — barre colorée pleine largeur retirée de `DatasetCard`
  (`accentBarClass` importé mais le motif n'avait jamais été balayé sur ce
  composant précis lors du correctif StatTile — trouvaille réelle, pas un
  faux positif dû au mélange de branches).
- `ProtectedRoute.tsx` — écran de chargement nu remplacé par une silhouette
  de l'application (`AppSkeleton`), rend l'attente perceptuellement plus
  courte sans changer sa durée réelle (voir D2.10 pour la durée elle-même).
- `Training.tsx` — `StepperNav` : `overflow-x-auto` + flèches de défilement
  remplacés par `flex-wrap` (barre de défilement native + 5ᵉ étape coupée
  constatées à 1568 px) ; `StepPill` : `bg-white`/`text-white` en dur
  remplacés par `bg-card`/`text-primary-foreground` (bug de mode sombre
  réel — pastille blanche vive sur fond sombre pour les étapes non
  atteintes).
- `accentBarClass` (`ColorIconBadge.tsx`) supprimé — plus aucun appelant
  après le retrait ci-dessus (vérifié par recherche globale avant
  suppression).
**Pourquoi ce n'est pas un désaccord sur le contenu des correctifs** :
vérifié directement que `lot-2a-design-system` (avant ce commit) avait
encore la barre `DatasetCard` et n'avait ni `ProtectedRoute` amélioré ni
les deux correctifs `Training.tsx` — les 4 trouvailles de l'utilisateur
sont réelles, indépendamment de la confusion de branche.
**Remise en cause si** : je dois de nouveau travailler sur une branche
différente de celle que l'utilisateur révise activement — utiliser un
`git worktree` séparé (déjà fait une fois ce lot pour Lot 2B) plutôt que
de changer la branche du répertoire partagé, systématiquement à partir de
maintenant.
Gate complet relancé et vert (tsc, lint, 42/42 tests, build).

### D2.10 — Diagnostic des 7–8 s avant le premier rendu (pas corrigé, `AppSkeleton` en corrige la perception)

**Vérifié empiriquement, pas supposé** : backend local redémarré à froid,
`GET /api/health` chronométré. Premier appel après démarrage : ~2 s
d'attente de connexion puis échec (le process importe encore) ; prêt après
~15 s ; une fois chaud, 5 requêtes consécutives entre 7 et 24 ms. La lenteur
n'est donc PAS dans la logique de `/api/auth/me` (décodage JWT + une seule
requête SQL) — elle est entièrement dans l'IMPORT PYTHON initial du
process.
**Cause racine identifiée** : `api/main.py` importe TOUS les routers de
façon inconditionnelle au démarrage, y compris `vision_classification.py`
(`import torch` en tête de fichier) — et par transitivité
`ml_training.py`, `ml_explainability.py`, `ml_registry.py`,
`vision_anomaly_training.py`, `vision_anomaly_registry.py`,
`vision_classification_training.py`, `vision_classification_registry.py`,
`vision_gradcam.py` (torch/lightgbm/catboost/shap/umap). Résultat : même
un endpoint trivial comme `/api/auth/me` attend que l'intégralité de la
pile ML soit chargée en mémoire avant de pouvoir répondre — à CHAQUE
démarrage du process, y compris à chaque redémarrage déclenché par
`uvicorn --reload` en développement actif (très fréquent pendant ce lot).
**Non corrigé dans ce commit** : rendre ces imports paresseux (déplacer
`import torch`/`lightgbm`/`catboost`/`shap`/`umap` du niveau module vers
l'intérieur des fonctions qui les utilisent réellement) touche 8 fichiers
de code ML de production, avec un risque réel d'erreur d'ordre
d'import/typage si fait à la hâte — mérite son propre lot testé
séparément, pas un correctif improvisé en fin de session. `AppSkeleton`
(D2.9) corrige la PERCEPTION de l'attente dans l'intervalle, pas sa durée.
**Remise en cause si** : la latence de démarrage devient bloquante en
production (ex. redémarrages fréquents d'un conteneur, autoscaling) —
alors le lot d'imports paresseux devient prioritaire, pas seulement un
confort de développement.

## Lot 3 — Verdict post-entraînement (Phase 2 de l'audit, correctif I1)

### D3.1 — Portée : règles déterministes sur des données déjà persistées, aucun nouveau calcul ML

**Question** : l'audit (§E.3/§P) identifie « aucune aide à la décision
post-entraînement » comme le plus gros écart entre la promesse produit et
le code, et demande `services/model_verdict.py` + composant
`ModelVerdict`. Comment le construire sans transformer ça en chantier de
recherche ?
**Retenu** : 7 vérifications, toutes des règles déterministes (seuils) sur
des nombres DÉJÀ calculés et persistés par `ml_training.py` — aucun
nouvel entraînement, aucune nouvelle métrique coûteuse :
surapprentissage (`delta_r2`/`delta_accuracy`), fiabilité (largeur de
l'IC bootstrap), choix de métrique (déséquilibre de classes déduit de
`confusion_matrix`, classification uniquement), écart au 2ᵉ candidat
(`selection_score` vs écart-type des `fold_scores`), honnêteté des
probabilités (écart à la diagonale de `calibration_json`, classification
uniquement), utilité de plus de données (plateau de `learning_curve_json`),
couverture CQR (`empirical_coverage` vs `target_coverage`, régression
uniquement). Chaque vérification omise (donnée absente — job antérieur au
correctif, ou non applicable au type de tâche) est simplement absente du
résultat, jamais remplacée par une affirmation inventée.
**Convention reprise de `services/data_quality.py`** (même vocabulaire de
niveau `critique`/`attention`/`info`, mêmes champs `code`/`title`/
`explanation`) — l'utilisateur reconnaît immédiatement le même type de
garde-fou que les avertissements de qualité de données déjà affichés
avant l'entraînement.
**Écarté** : étendre au clustering et à la vision dans ce même lot (la
roadmap de l'audit le mentionne en extension) — le pilier non supervisé a
déjà son propre équivalent (`utils/clusterQuality.ts`), la vision n'a pas
encore de notion de candidats/leaderboard comparable. Fait quand le
besoin se présente, pas par anticipation.

### D3.2 — Donnée manquante trouvée en vérifiant le code réel : pas d'`accuracy_train` en classification

**Trouvé** : `_regression_metrics` calcule `r2_train`/`r2_test`/`delta_r2`
depuis `pred_train`/`pred_test`, mais `_classification_metrics` ne
prenait QUE `y_test`/`pred_test` — `pred_train` était déjà calculé
inconditionnellement dans `train_and_evaluate` (ligne commune aux deux
branches) mais jamais utilisé côté classification. Le tableau de l'audit
(§E.3) suppose "accuracy train/test" disponible pour juger le
surapprentissage — ce n'est vrai qu'en régression.
**Retenu** : `_classification_metrics` prend maintenant `y_train`/
`pred_train` en plus, calcule `accuracy_train`/`delta_accuracy` — même
paire de clés que `r2_train`/`delta_r2` côté régression, aucun calcul
supplémentaire (juste un `accuracy_score` de plus sur des prédictions
déjà calculées). `services/model_verdict.py` en dépend pour juger le
surapprentissage en classification exactement comme en régression, sans
traiter les deux tâches différemment.
**Vérifié** : aucun test n'asserte l'ensemble exact des clés de `metrics`
(`grep metrics\.keys` sans résultat) — ajout sans risque de régression.
Tests classification ciblés de `test_ml_training.py` relancés (3/3).

### D3.3 — Verdict calculé à la volée, jamais persisté

**Question** : stocker le verdict en base (nouvelle colonne
`verdict_json` sur `MLModel`) ou le calculer à chaque lecture ?
**Retenu** : calculé à la volée dans `_to_model_detail()` (routers/
training.py), jamais persisté. Les règles de `model_verdict.py` peuvent
évoluer (seuils affinés, nouvelles vérifications) sans backfill ni
migration — un modèle entraîné avant ce lot affiche immédiatement un
verdict cohérent avec les règles actuelles, pas figé sur ce qui existait
au moment de son entraînement. Coût : une requête `ModelCandidate`
supplémentaire par lecture de modèle (déjà nécessaire pour le
leaderboard, `GET /jobs/{id}/candidates` — même filtre, dupliqué ici
plutôt que partagé, la fonction reste appelable sans devoir aussi
récupérer le leaderboard).
**Remise en cause si** : le calcul devient coûteux (aujourd'hui : une
requête SQL + quelques dizaines de comparaisons de flottants, négligeable)
ou si l'historique des verdicts doit être auditable (aujourd'hui : non
demandé).

### D3.4 — Suppression du texte ad hoc "Interprétation du modèle" qui doublonnait la question du 2ᵉ candidat

**Trouvé, corrigé** : `ModelResultModal.tsx` avait déjà DEUX tentatives ad
hoc de répondre à « ce modèle surapprend-il ? »/« le gagnant est-il
vraiment meilleur ? » — un texte inline sous les métriques de performance
(`delta_r2 < 0.08 ? ... : ...`, régression uniquement, aucune notion de
significativité) et une phrase dans `ModelInterpretation` comparant
`selection_score` du gagnant et du 2ᵉ avec un seuil arbitraire (0,01),
sans jamais utiliser `fold_scores` (déjà disponible, Lot D). Garder les
trois (l'ancien texte, l'ancienne phrase, ET le nouveau `ModelVerdict`)
aurait affiché deux verdicts différents, avec des seuils différents, sur
la même page.
**Retenu** : les deux supprimés. `ModelVerdict`, en tête de page, répond
maintenant aux deux questions avec la vraie donnée disponible
(`delta_r2`/`delta_accuracy` pour l'un, écart-type des `fold_scores` pour
l'autre). `ModelInterpretation` ne garde que ce que `ModelVerdict` ne
couvre pas : l'importance des variables (SHAP).
**Remise en cause si** : un besoin réapparaît de comparer TOUS les
candidats deux à deux (pas seulement le gagnant au 2ᵉ) — hors périmètre
de ce lot, `ModelVerdict` répond à "le gagnant est-il clairement
meilleur", pas à un classement complet qualifié.

### D3.5 — Vérification : sous-ensemble ciblé, pas la suite complète (525 tests)

**Retenu** (demande explicite de l'utilisateur, "évite de lancer tous les
tests, juste la partie concernée") : `test_model_verdict.py` (29/29,
nouveau), sous-ensemble classification de `test_ml_training.py` ciblé par
`-k classification` (3/3, vérifie `accuracy_train`/`delta_accuracy`),
`test_training_api.py`+`test_model_registry.py`+`test_job_comparison.py`
(36/36, les trois fichiers qui exercent `_to_model_detail()`/`GET
/jobs/{id}/model`/`promote_model`, dont la signature a changé). Gate
frontend complet (tsc, lint, 35/35 tests, build) — plus rapide, moins de
surface de risque de régression que le backend.
**Remise en cause si** : une régression apparaît ailleurs dans la suite
complète au prochain lot qui touche `ml_training.py`/`training.py` — la
suite complète devra alors être relancée avant fusion vers `main`.

## Lot 4 — Tenir la charge (Phase 3 de l'audit, correctifs I3, I4, I6 et I7)

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

### D4.9 — I7 : logs JSON + request_id, /metrics Prometheus, Sentry conditionnel

**Question** : I7 (AUDIT_DATALAB_2026-08-16.md §I7) demande une
observabilité minimale — sans elle, un incident en production ne laisse
que des logs texte libre à parcourir à la main, sans moyen de relier les
lignes d'une même requête entre elles ni de mesurer la charge réelle.
**Retenu**, un seul module `api/core/observability.py` pour les 3
briques :
- Logs JSON (`JsonFormatter`) — une ligne par log, `request_id` injecté
  sur CHAQUE ligne via un `ContextVar` + `logging.Filter`
  (`_RequestIdFilter`), `"-"` par défaut hors requête HTTP (démarrage,
  worker RQ). Remplace `logging.basicConfig` (texte libre) dans
  `api/main.py`.
- `RequestIdMiddleware` — lit `X-Request-ID` du client s'il est fourni
  (traçage bout-en-bout derrière un reverse proxy qui le fixe déjà),
  sinon en génère un (`uuid4`) ; toujours renvoyé dans la réponse. Ajouté
  EN DERNIER dans `api/main.py` (`add_middleware`) : le dernier ajouté
  devient le plus externe de la pile Starlette, donc le seul ordre où
  MÊME les logs de `CORSMiddleware`/`MaxJsonBodySizeMiddleware` portent
  le `request_id` de leur requête.
- `PrometheusMiddleware` + `GET /metrics` — compteur de requêtes,
  histogramme de latence, jauge de requêtes en cours, labellisés par le
  GABARIT de route (`request.scope["route"].path`, ex.
  `/datasets/{dataset_id}`), jamais le chemin brut (`/datasets/42`) —
  sans quoi la cardinalité de la métrique grandirait sans borne avec
  l'usage réel. `scope["route"]` n'existe qu'APRÈS résolution du routing
  (accessible seulement après `call_next`) ; absent sur un 404, repli sur
  le chemin brut (cardinalité alors bornée par les tentatives d'un
  attaquant, pas par un usage légitime).
- Sentry — un simple `sentry_sdk.init(...)` conditionnel dans
  `api/main.py`, actif SEULEMENT si `SENTRY_DSN` est défini
  (`Settings.sentry_dsn: Optional[str] = None`) : dégradation honnête,
  aucun changement de comportement en dev/CI sans DSN configuré, jamais
  un crash au démarrage faute de DSN.
**Écarté** : `structlog` ou une lib de logging structuré tierce — un
`logging.Formatter` standard (stdlib) suffit pour un JSON par ligne, pas
besoin d'une dépendance supplémentaire pour ce besoin précis. Le module
s'appelle `observability.py`, jamais `logging.py`, pour éviter toute
ambiguïté de lecture avec le module standard `logging` malgré l'absence
de collision technique réelle (imports absolus Python 3).
**Écarté aussi** : exposer `X-Next-Cursor`/`X-Request-ID` dans
`CORSMiddleware(expose_headers=...)` — `X-Next-Cursor` n'est pas encore
lu côté frontend (P6, pagination UI, différé — voir D4.1) ;
`X-Request-ID` reste utile côté logs serveur même illisible en JS
cross-origin, et le déploiement cible (nginx même origine, voir branche
`fix-api-prefix-routing`) rend la question théorique pour l'instant —
à rouvrir si un usage cross-origin réel apparaît.
**Vérifié, pas supposé** : `prometheus-client`/`sentry-sdk` ajoutés à
`requirements.txt` (0.26.0 / 2.68.0, dernières stables au moment du
lot) — testés en conditions réelles via `TestClient` (démarrage complet
de l'app, `/api/health`, `/metrics`, en-têtes de requête) avant d'écrire
les tests automatisés, pas seulement lus dans la doc de ces libs.
**Remise en cause si** : un besoin de traçage distribué plus riche
apparaît (spans, pas seulement un id de corrélation) — à ce moment,
OpenTelemetry remplacerait probablement ce module fait main plutôt que
de l'étendre indéfiniment.

### D4.10 — Résolution du conflit `Dashboard.tsx` à la fusion vers `main` (tranche la remise en cause de D4.5)

**Contexte** : D4.5 avait explicitement différé la décision « dégradation
par pilier vs endpoint agrégé » au moment de fusionner Lot 4 avec Lot 2A.
C'est ce moment — fusion séquentielle de `lot-4-perf` dans `main` (qui
contient déjà `lot-2a-design-system` et `lot-3-verdict`), conflit réel sur
`frontend/src/pages/Dashboard.tsx` (4 sections divergentes).
**Trouvé en résolvant** : les 6 états `useState` par pilier
(`jobs`/`clusteringJobs`/`datasets`/`members`/...) que la dégradation fine
de Lot 2A (D2.4) manipulait avaient déjà été supprimés par la fusion
automatique (hunk non conflictuel, Lot 2A n'avait touché que le corps des
callbacks, pas leur déclaration) — conserver le code de dégradation par
pilier aurait donc référencé des variables inexistantes, invalidant le
gate frontend (`tsc`) immédiatement. Le choix n'était donc pas seulement
architectural, il était aussi mécanique.
**Retenu** :
- Modèle de chargement Lot 4 conservé tel quel (`GET /dashboard/summary`,
  un seul état `summary`/`summaryError`) — la cause racine de l'incident
  D2.4 (colonnes manquantes en base faisant échouer `_to_summary()`) est
  désormais interceptée en amont, au démarrage, par la vérification de
  schéma D1.8 (`_schema_matches_metadata()`) : elle refuse de démarrer sur
  un schéma drifté plutôt que de le découvrir tardivement via un 500 sur
  un endpoint. Le risque qui justifiait la dégradation fine par pilier est
  donc structurellement réduit, pas seulement accepté par confort.
- **Décision UX de D2.4 conservée par-dessus** : les 4 tuiles agrégées
  restent `color="neutral"`, et les 3 colonnes de la tuile "Analyses ML"
  gardent `pillarColor(id)` (au lieu des couleurs bleu/teal/amber/violet
  arbitraires que `lot-4-perf` avait héritées de la base, antérieure à la
  correction D2.4) — ce point n'a aucun rapport avec la stratégie de
  chargement (1 requête vs 8), donc les deux décisions ne s'opposaient
  pas réellement, seul le code source se chevauchait ligne à ligne.
**Écarté** : réintroduire une dégradation partielle CÔTÉ BACKEND (un
`try/except` par pilier dans `get_dashboard_summary()` qui renverrait un
200 partiel si une seule sous-requête échoue) — option explicitement
envisagée par D4.5. Non retenue ici : changement de comportement non
testé, introduit au milieu d'une résolution de conflit de fusion plutôt
que comme un lot dédié avec ses propres tests.
**Remise en cause si** : un incident réel en production montre qu'une
sous-requête de `get_dashboard_summary()` échoue alors que le reste du
schéma est sain (donc un cas que D1.8 ne couvre pas) — alors la résilience
partielle CÔTÉ BACKEND (200 partiel) redevient la bonne réponse, à traiter
comme son propre correctif testé, pas en fusionnant à nouveau ce commit.
