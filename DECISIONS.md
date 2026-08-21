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

## Lot 5 — Traçabilité (Phase 4 de l'audit, correctifs I2, P1 et P2)

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

### D5.5 — P1 : version + archived + rollback pour `MLModel` uniquement, extension aux 5 autres pillars EXPLICITEMENT différée

**Question** : P1 (AUDIT_DATALAB_2026-08-16.md §P1) demande "`ModelVersion` :
numéro, alias, `archived`, historique de transitions, rollback ; étendre
aux 5 autres types" — une seule ligne, sans colonne "fichiers concernés"
(contrairement aux items "I", plus détaillés). Le Lot 9 (déjà en place)
avait déjà `stage`("staging"/"production")/`promoted_at` + une règle "un
seul modèle production par dataset+cible" + un journal d'audit
("model.promoted") — qu'est-ce qui manque vraiment ?
**Retenu**, pour `MLModel` (pilier supervisé) seulement :
- `version: int`, dénormalisé une fois à la création
  (`services/model_versioning.py::next_version`, numérotation par lignée
  organisation+dataset+cible, jamais recalculée).
- `stage` étendu à une 4ᵉ valeur `"archived"` (retire du registre actif
  SANS supprimer — `GET /models/registry` exclut désormais explicitement
  les modèles archivés, avant : tout `stage` non NULL y apparaissait).
- `GET /jobs/{id}/model/versions` — toute la lignée, la plus récente
  d'abord (retrouver le job_id d'une version antérieure).
- `GET /jobs/{id}/model/history` — lit `AuditLog` (déjà écrit par
  `promote_model` depuis le Lot 9), jamais un second mécanisme de
  journalisation parallèle.
- Rollback : AUCUN endpoint dédié — repromouvoir une version antérieure
  en "production" démet automatiquement la version courante (mécanisme
  de démotion du Lot 9, inchangé) ; documenté comme LE mécanisme de
  rollback dans `promote_model`, jamais dupliqué.
**Écarté** : un `alias` libre façon MLflow ("champion"/"challenger"
personnalisables) — `stage` fonctionne déjà comme 2 alias fixes bien
compris (staging/production) SANS UI de nommage à construire ; un
système d'alias générique demanderait une surface produit (comment
créer/renommer un alias ?) hors du périmètre de ce correctif backend,
et pour laquelle aucun besoin concret n'est exprimé — inventer cette UI
maintenant aurait été concevoir pour un besoin hypothétique.
**Écarté aussi** : une table `ModelVersion` séparée de `MLModel` — le
"numéro de version" n'est qu'un attribut de plus du modèle déjà
existant (comme `stage`), pas une nouvelle ENTITÉ avec un cycle de vie
propre ; une table séparée aurait imposé une jointure supplémentaire à
chaque lecture pour un gain nul ici.
**Extension aux 5 autres types (clustering, dimensionnalité, anomalies
tabulaires, classification vision, anomalies vision) : DIFFÉRÉE**,
décision explicite et non un oubli — chacun exigerait sa propre colonne
`version`+`dataset_id` dénormalisé, sa propre migration (avec le même
rétropeuplement soigné que celui testé ici), son propre triplet
d'endpoints. Cinq fois le travail fait ici, pour des pilotes dont
l'usage "plusieurs versions du même problème" est aujourd'hui moins
établi que pour le supervisé (Lot 9 y était déjà rodé). Voir I2 (D5.1)
pour le même raisonnement de scope déjà appliqué dans ce lot.
**Vérifié, pas supposé** : migration (`f983e8e87dcb`) testée contre une
base SQLite PEUPLÉE (pas seulement vide) — plusieurs "problèmes"
distincts (même dataset+cibles différentes, datasets différents+même
nom de cible), vérifié que chaque lignée reçoit bien 1, 2, 3... dans
l'ordre `id` croissant, jamais `created_at` (même leçon que D4.2) —
upgrade → vérification ligne par ligne → downgrade → vérification que
les colonnes disparaissent sans perte des autres lignes → ré-upgrade.
Bug réel trouvé en écrivant les tests : `tests/test_model_registry.py`
avait un helper `_complete_job` qui codait en dur `target_column="cible"`
dans le modèle créé, indépendamment de la vraie cible du job (masqué
avant ce lot par un correctif a posteriori `model.target_column = "x2"`
dans 2 tests) — la nouvelle contrainte UNIQUE l'a fait échouer
immédiatement (`UNIQUE constraint failed`), révélant l'incohérence.
Corrigé à la source (`target_column=job.target_column`) plutôt que
patché une fois de plus.
**Remise en cause si** : un besoin produit réel pour l'extension aux 5
autres types, ou pour un alias nommé librement, est exprimé — traiter
alors comme un correctif séparé, avec son propre "fait quand".

### D5.6 — P2 : hash SHA-256 + détection de doublon, `DatasetVersion` explicitement écarté (aucun re-upload n'existe)

**Question** : P2 (AUDIT_DATALAB_2026-08-16.md §P2) demande "`DatasetVersion`
+ SHA-256 ; le modèle référence une version, pas un id" — une ligne,
sans détail. Que signifie concrètement "versionner" un dataset dans CE
produit ?
**Vérifié, pas supposé** : `grep` sur `api/routers/datasets.py` confirme
qu'il n'existe QU'UN SEUL endpoint mutateur, `POST /datasets` (upload) —
aucun PUT/PATCH pour "remplacer" ou "mettre à jour" le fichier d'un
dataset existant. Documenté ailleurs dans ce projet, de façon répétée et
déjà structurante pour d'autres décisions (D4.6, D4.9 sur `lot-4-perf`,
`get_dataset_eda` sur cette branche) : *"un dataset peut changer de
statut mais son fichier ne change jamais une fois uploadé"*. Chaque ligne
`Dataset` est donc DÉJÀ, par construction, une version immuable unique —
il n'existe aujourd'hui aucune action utilisateur qui produirait une
"version 2" d'un dataset existant.
**Retenu** : `content_hash` (SHA-256, calculé à l'upload depuis les
octets déjà en mémoire — aucune relecture disque) + `duplicate_of_dataset_id`
(renseigné si un dataset de LA MÊME organisation partage déjà ce hash,
jamais bloquant — l'upload aboutit toujours, purement informatif, même
philosophie "informer, ne jamais bloquer" que `services/data_quality.py`).
Donne un usage réel et immédiat au hash (repérer un ré-upload
accidentel du même fichier, vérifier l'intégrité d'un fichier stocké)
sans rien inventer au-delà de ce que permet le produit actuel.
**Écarté** : scinder `Dataset` en `Dataset` (entité logique) +
`DatasetVersion` (fichier uploadé versionné), avec migration de TOUTES
les FK `dataset_id` existantes (`TrainingJob`, `ClusteringJob`,
`DimensionalityJob`, `AnomalyJob`, `MLModel` — ce dernier tout juste
ajouté par P1, D5.5) vers `dataset_version_id`. Sans fonctionnalité de
remplacement de fichier, cette scission n'aurait AUCUN effet
observable : une seule version aurait jamais existé pour chaque
dataset, la FK aurait pointé exactement vers la même donnée qu'aujourd'hui,
juste à travers une table intermédiaire de plus. "Le modèle référence
une version, pas un id" est déjà vrai EN PRATIQUE (chaque `Dataset` est
sa propre version unique et immuable) — le formaliser en une seconde
table maintenant, avant qu'un vrai besoin de remplacement existe, aurait
été concevoir pour un besoin hypothétique, contre les principes de ce
projet et contre le choix déjà fait en D5.5 pour l'extension P1 aux 5
autres pilotes.
**Vérifié aussi** : migration (`d6231c7548cf`) — colonnes nullables,
AUCUN rétropeuplement (contrairement à P1/D5.5) : recalculer le hash
exigerait de relire chaque fichier sur disque depuis la migration elle-même,
chemin non garanti identique entre environnements (dev/CI/prod) et
potentiellement lent sur de gros fichiers — hors périmètre d'une
migration de schéma. Dégradation honnête : `content_hash IS NULL` pour
tout dataset antérieur à ce lot, pour toujours, comme `stage`/
`promoted_at` sur `ml_models` avant ce lot. upgrade → downgrade →
ré-upgrade testés avant ce commit.
**Remise en cause si** : une fonctionnalité de remplacement/mise à jour
d'un dataset existant est demandée — c'est EXACTEMENT le moment où
`DatasetVersion` (et la migration de `dataset_id` vers
`dataset_version_id` sur les 5 tables qui le référencent) devient
nécessaire, pas avant.

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

### D6A.8 — Régression réelle trouvée à la fusion vers `main` : `uploadFiles()` sans préfixe `/api`

**Trouvé en résolvant le conflit `frontend/src/api/client.ts`** : la
fusion automatique de `lot-6a-vision-wizard` dans `main` n'a signalé
AUCUN conflit sur ce fichier pour la fonction `uploadFiles<T>()` (Lot 6A,
upload multi-fichiers pour l'import de dossier) — parce que cette
fonction est une pure addition, absente de `main` avant ce lot, donc rien
à réconcilier du point de vue de `git merge`. Mais son corps appelait
`fetch(\`${BASE_URL}${path}\`, ...)`, jamais `fetch(apiUrl(path), ...)`
comme les 4 autres fonctions de fetch du fichier — parce que la branche
`lot-6a-vision-wizard` a été créée AVANT le correctif `fix-api-prefix-routing`
et n'a jamais vu `apiUrl()` exister. Résultat : `api.visionDatasets.upload()`
et `.uploadFolder()` auraient silencieusement appelé `/vision/datasets` au
lieu de `/api/vision/datasets` une fois fusionnées — exactement la classe de
bug que l'utilisateur avait explicitement demandé de surveiller à chaque
fusion pour ce fichier précis, mais sous une forme qu'un simple `git diff`
des conflits n'aurait jamais révélée (pas de marqueur de conflit à lire —
il fallait relire tout le fichier fusionné, pas seulement les hunks
marqués `UU`).
**Retenu** : `uploadFiles()` migrée vers `fetch(apiUrl(path), ...)`, comme
les 4 autres. Aucun test frontend n'aurait attrapé ça (les tests Vitest de
ce fichier ne montent pas de vrai `fetch` réseau) — trouvé uniquement en
relisant le fichier fusionné ligne par ligne avant de committer, pas par
un gate automatisé.
**Remise en cause si** : d'autres branches en attente de fusion (aucune
identifiée à ce jour) ajoutent elles aussi un appel `fetch()` direct —
relire systématiquement `client.ts` en ENTIER après toute fusion future,
pas seulement les sections marquées en conflit par Git.

### D6A.9 — 16G : ROC/PR binaire ET multiclasse, jamais de branche `task_type` — réutilisation complète d'`EvaluationCharts.tsx`

**Constat** : le pilier tabulaire (`ml_training.py`) calcule déjà
ROC/PR (binaire et multiclasse OvR via `label_binarize`) et les
affiche via `EvaluationCharts.tsx`, qui consomme un objet
`{confusion_matrix, class_names, roc_curves, pr_curves}` typé
`ClassificationEvaluation`. Plutôt que d'écrire un second calcul ou un
second composant pour Vision, la forme de sortie de
`_compute_roc_pr_curves()` (nouveau, dans
`vision_classification_training.py`) a été calquée EXACTEMENT sur
celle du pilier tabulaire (mêmes clés, mêmes types :
`Record<string, RocCurve | PrCurve>`), avec le même
`_downsample_curve()` (max 100 points) recopié tel quel — pas
factorisé dans un module partagé, car les deux pipelines
(tabulaire vs vision) n'ont aucune autre dépendance croisée
aujourd'hui et une factorisation prématurée créerait un couplage non
demandé.
**Retenu** : binaire → une seule courbe (`class_names[1]`, la classe
positive), `all_proba[:, 1]`. Multiclasse → une courbe par classe
(One-vs-Rest) via `label_binarize`. `test_roc_auc` global :
`roc_auc_score(..., multi_class="ovr", average="weighted")` en
multiclasse, capturé dans un `try/except ValueError` (dégradation
honnête → `None`, jamais de crash si une classe est absente du jeu de
test) ; produit direct de `roc_auc_score` en binaire.
**Retenu aussi** : `VisionClassification.tsx` remplace son ancien
`<Heatmap>` de matrice de confusion par `<EvaluationCharts
taskType="classification" evaluation={{...}} />` — récupère
gratuitement les courbes ROC/PR en plus de la matrice, aucun nouveau
code de graphique écrit côté Vision.
**Vérifié** : nouvelle migration Alembic (`77a16b5c0e66`, 3 colonnes
nullables sur `vision_classification_models`) testée
upgrade→downgrade→upgrade sur la base de dev réelle. Tests ciblés :
binaire (1 courbe), multiclasse (1 courbe par classe) — tous verts.
Pas de lancement de la suite complète (682 tests) pendant ce lot, sur
consigne explicite de l'utilisateur — seuls les fichiers
concernés ont tourné.

### D6A.10 — EDA d'images : métadonnées capturées gratuitement pendant la validation existante, jamais un second passage disque

**Constat** : le legacy chargeait les pixels une seconde fois pour
calculer des statistiques d'image. `_validate_and_copy_images` ouvre
déjà chaque image avec PIL (`Image.open()` + `.load()`) pour la
validation d'intégrité — largeur/hauteur/format/mode sont disponibles
sur cet objet déjà ouvert, sans coût I/O supplémentaire.
**Retenu** : `_ValidImage` étendu avec `width/height/format/mode`,
capturés au même endroit que la validation. `_compute_image_eda()`
calcule ensuite `resolution_buckets` (bornes 128/224/512/1024, choisies
pour englober les tailles d'entrée réelles des backbones enregistrés —
p.ex. 224 = ResNet/MobileNet standard), `width/height {min,max,mean}`,
`format_distribution`, `color_mode_distribution` — uniquement sur les
images CONSERVÉES après dédoublonnage (`kept_images`), jamais sur les
doublons exclus, pour que la distribution affichée corresponde
exactement au dataset qui sera réellement entraîné.
**Retenu aussi** : exposé via le `validation_report_json` déjà
existant (clé `image_eda` ajoutée), pas de nouvel endpoint — le
frontend (`VisionDatasetExplorer.tsx`) l'affiche via un nouveau
`ImageEdaSummary` + `DistributionBars` (histogramme CSS pur, sans
Recharts, cohérent avec le fait qu'il s'agit de distributions
catégorielles simples et non de séries continues).
**Vérifié** : dataset vide → dégradation honnête (dict vide, jamais de
division par zéro) ; dédoublonnage → images exclues absentes de l'EDA ;
labels de buckets contigus. Tests service + test API bout-en-bout
(upload réel → `image_eda` présent dans la réponse) tous verts.

### D6A.11 — Sélection d'exemples représentatifs : round-robin par groupe, remplace le slicing naïf `[:12]`

**Problème du slicing naïf** : `incorrect_examples[:12]` favorise
systématiquement les classes qui apparaissent en premier dans l'ordre
d'itération du DataLoader — sur un dataset déséquilibré, les 12
exemples affichés pouvaient tous provenir d'une seule paire
(vrai, prédit), rendant la revue d'erreurs inutile pour les autres
classes.
**Retenu** : `_representative_sample(items, group_key, limit)` — regroupe
par clé (paire `(vrai, prédit)` pour les erreurs, `vrai` seul pour les
exemples corrects), puis prend l'index 0 de chaque groupe, puis l'index
1, etc. (round-robin), jusqu'à `limit`. Généralisable à n'importe quel
nombre de groupes, dégrade proprement si un groupe s'épuise avant les
autres (passe simplement au suivant, jamais d'erreur).
**Vérifié** : tests dédiés — répartition round-robin confirmée sur
groupes de tailles inégales, épuisement gracieux d'un petit groupe.

### D6A.12 — P11 (registre de modèles Vision) explicitement reporté, hors périmètre de cette session

**Décision utilisateur explicite** : "P11 ... registre peut attendre."
Une première ébauche de code de registre avait été commencée par
erreur plus tôt dans le chantier 6A (avant clarification du périmètre
exact 6A/6B), puis retirée proprement (migration vide supprimée, aucun
code orphelin laissé dans l'arbre). P11 reste donc entièrement à faire
et appartient à la clôture réelle du Lot 6A, pas à ce lot ni au Lot 6B
(ML non supervisé) qui suit.
**Remise en cause si** : une prochaine session Vision doit traiter le
stage/version/promotion/export de modèles — repartir de zéro sur ce
point, aucun code partiel n'existe à réutiliser.

## Lot 6B — ML non supervisé (§F.2 : contrôle qualité, transparence, stabilité, assignation)

### D6B.1 — Contrôle qualité intégré par pure réutilisation de `DataQualityWarnings.tsx`, aucun changement backend

**Constat** : `analyze_data_quality()` supportait déjà `target_column=None`
(détections structurelles uniquement) et `DataQualityWarnings.tsx` acceptait
déjà `targetColumn` en optionnel — seuls les 3 pages non supervisées
(Clustering/DimensionalityReduction/AnomalyDetection) ne le montaient jamais.
**Retenu** : le composant est monté tel quel (sans `targetColumn`) dans les 3
formulaires, avec `selectedFeatures`/`onExcludeColumns` branchés pour
conserver l'action "Exclure" déjà existante côté tabulaire — zéro ligne de
code backend, uniquement du câblage frontend.
**Vérifié** : `tsc -b` (0 erreur), `eslint` (0 erreur), suite Vitest complète
verte.

### D6B.2 — Transparence d'échantillonnage pour le clustering : plafond à 5000 lignes (comme la réduction de dimension), pas 20 000 (comme les anomalies)

**Question** : quel plafond adopter pour `MAX_ROWS_FOR_CLUSTERING`, sachant
que le registre inclut un algorithme O(n²) en mémoire (hiérarchique, linkage
de Ward) ?
**Retenu** : 5000, aligné sur `MAX_ROWS_FOR_EMBEDDING` (réduction de
dimension), pas sur `MAX_ROWS_FOR_ANOMALY` (20 000, Isolation Forest/LOF
restent efficaces à cette échelle). Sans ce plafond, un dataset volumineux
pouvait déclencher un `MemoryError` en cours de job sans avertissement
préalable — code défensif déjà présent côté worker
(`_user_safe_error_message`) mais jamais un garde-fou proactif.
**Retenu aussi** : `model_card["n_samples"]` (nom historique déjà lu par
`Clustering.tsx::totalSamples`) redéfini pour pointer vers les données
RÉELLEMENT clusterisées (`n_samples_used`), pas le total avant échantillon —
sinon la somme des tailles de profils ne correspondrait plus à ce total,
cassant silencieusement le calcul de répartition déjà affiché.
**Vérifié** : tests dédiés (plafond forcé bas via `monkeypatch`), aucune
régression sur les 16 tests déjà existants.

### D6B.3 — Transparence catégorielle du clustering : extension de `categorical_summary` (population_pct + lift), pas un nouveau concept parallèle à `categorical_flags`

**Question** : la détection d'anomalies expose déjà `categorical_flags`
(valeurs RARES par observation) — fallait-il reproduire exactement ce
concept pour le clustering ?
**Écarté** : dupliquer `categorical_flags` tel quel — le clustering décrit
des SEGMENTS (agrégats), pas des observations individuelles ; une "valeur
rare" par observation n'a pas de sens au niveau d'un profil de cluster.
**Retenu** : étendre le `categorical_summary` déjà existant
(`top_category`/`top_pct`) avec `population_pct` (fréquence de cette
catégorie sur l'ENSEMBLE du dataset) et `lift` (sur-représentation dans ce
cluster vs la population, `None` — jamais une division par zéro — si la
catégorie est absente ailleurs) : même esprit que le z-score déjà calculé
pour les variables numériques (`numeric_summary`), qui compare déjà chaque
cluster à la population globale. Cohérence de conception plutôt qu'un
concept importé tel quel d'un autre pilier.
**Retenu aussi (réduction de dimension)** : `categorical_columns`/
`numeric_columns`/`n_categorical_dimensions`/`n_dimensions_after_encoding`
exposés dans le `model_card`, dérivés de `preprocessor.get_feature_names_out()`
déjà calculé pour les loadings — zéro coût de calcul supplémentaire, juste
un comptage des noms préfixés `cat__`.
**Vérifié** : tests dédiés (lift ≈3 sur un cluster pur pesant 1/3 de la
population, comptage exact des dimensions one-hot).

### D6B.4 — Taux de contamination des anomalies : réglable en mode expert, "auto" reste le défaut guidé

**Question** : exposer `contamination` (IsolationForest/LOF) comme paramètre
utilisateur — jusqu'ici codé en dur sur `"auto"` dans
`anomaly_registry.py` ?
**Retenu** : `AnomalyJobCreate.contamination: Optional[float]` (borné
`(0, 0.5]`, validation Pydantic) — `None`/absent conserve le comportement
"auto" par défaut (mode guidé, cohérent avec le reste du produit : rien à
régler tant que l'utilisateur n'en éprouve pas le besoin). Réglable
explicitement via une case à cocher dédiée côté `AnomalyDetection.tsx`
("Régler moi-même la proportion attendue d'anomalies"), jamais affiché par
défaut.
**Vérifié** : le paramètre est bien câblé aux DEUX estimateurs (pas
seulement stocké dans le model_card) — test dédié confirmant qu'une
contamination stricte (1 %) ne peut jamais flagger plus d'observations que
le réglage "auto".

### D6B.5 — Stabilité de k par sous-échantillonnage (ARI), pas une comparaison multi-seed

**Question** : comment mesurer la sensibilité du nombre de clusters retenu
aux données exactement utilisées, sachant que le registre mélange des
algorithmes stochastiques (KMeans) et déterministes (hiérarchique, DBSCAN,
pour lesquels une comparaison multi-seed n'aurait aucun sens) ?
**Écarté** : comparaison multi-seed (méthode usuelle pour KMeans seul) —
non généralisable à hiérarchique/DBSCAN, qui produiraient exactement le
même résultat à chaque seed (déterministes par construction), donnant une
fausse impression de stabilité parfaite.
**Retenu** : sous-échantillonnage (80 % des lignes, sans remise, 5 rounds) —
family-agnostic, méthode établie de la littérature sur la stabilité de
clustering (von Luxburg et al.). Stabilité = ARI moyen entre les étiquettes
obtenues sur les points communs à deux sous-échantillons consécutifs.
Calculée UNIQUEMENT pour la configuration gagnante (pas tous les
candidats, coût maîtrisé), sur un sous-ensemble borné indépendamment de
`MAX_ROWS_FOR_CLUSTERING` (`MAX_ROWS_FOR_STABILITY = 1000`) — l'estimation
n'a pas besoin de la totalité de l'échantillon principal, et l'hiérarchique
(O(n²)) rendrait 5 refits coûteux sur un sous-échantillon déjà grand.
**Retenu aussi** : `None` (jamais un score inventé) si moins de 20 lignes ou
moins de 2 sous-échantillons ajustés avec succès — dégradation honnête,
même principe que le reste du produit.
**Vérifié** : stabilité >0.7 sur 3 groupes très séparés (comportement
attendu, pas une valeur arbitraire) ; `None` confirmé sous le seuil minimal.

### D6B.6 — Assignation de nouvelles observations : exacte pour K-Means, approximations documentées pour hiérarchique/DBSCAN (jamais un "impossible" silencieux)

**Constat** : `AgglomerativeClustering`/`DBSCAN` (sklearn) n'ont PAS de
méthode `.predict()` — ce sont des modèles transductifs, contrairement à
KMeans/MiniBatchKMeans. Le pipeline_bundle clustering n'était de toute
façon jamais exploité après l'entraînement (aucun endpoint de prédiction).
**Retenu** : `services/clustering_inference.py` (nouveau module, séparé de
`ml_inference.py` par le même raisonnement que `clustering_training.py`/
`ml_training.py` — seul `load_bundle`, générique, est réutilisé tel quel) —
3 méthodes selon la famille :

- K-Means/K-Means rapide : `.predict()` natif → `assignment_method: "exact"`.
- Hiérarchique : centroïdes calculés UNE SEULE FOIS à l'entraînement
  (moyenne des points par cluster dans l'espace préprocessé) et persistés
  dans le pipeline_bundle → assignation au centroïde le plus proche
  (`"approximate_centroid"`).
- DBSCAN : points "cœurs" (`core_sample_indices_`, déjà calculés par
  sklearn) persistés → assignation au point cœur le plus proche SI la
  distance est ≤ eps, sinon bruit/atypique (`"approximate_nearest_core"`) —
  même règle que celle appliquée par DBSCAN à ses propres points
  d'entraînement, pas une heuristique inventée.
**Retenu aussi** : `assignment_method` toujours retourné (jamais silencieux
sur la nature exacte/approchée de l'assignation) — y compris
`"unsupported"` pour la rétrocompatibilité par absence : un clustering
entraîné AVANT ce lot n'a pas les clés `centroids`/`core_points` dans son
pipeline_bundle, dégradation honnête plutôt qu'un crash.
**Vérifié** : assignation exacte K-Means vérifiée contre le centre le plus
proche construit ; approximation centroïde hiérarchique testée ; DBSCAN
testé à la fois pour un point proche d'un groupe ET un point isolé (bruit) ;
dégradation "unsupported" testée en retirant artificiellement la clé
`centroids` d'un bundle réel. Endpoint `POST /clustering/jobs/{id}/predict`
testé bout-en-bout (worker réel + appel HTTP), même pattern que
`training.py::predict_with_model` côté supervisé.

### D6B.7 — Vocabulaire de verdict qualité extrait dans `qualityAssessment.ts`, partagé par les 3 piliers non supervisés

**Problème identifié** : `Clustering.tsx` définissait localement
`QUALITY_ACCENT: Record<QualityTone, AccentColor>` — étendre ce pattern à
la réduction de dimension et aux anomalies sans extraction aurait
nécessité soit une réimportation d'un type nommé "cluster*" dans des pages
sans rapport avec le clustering, soit (pire) une redéfinition indépendante
du même vocabulaire tone/label/caveat dans chaque page, avec le risque
réel de divergence de seuils/couleurs signalé par l'audit.
**Retenu** : `frontend/src/utils/qualityAssessment.ts` (nouveau) porte le
type partagé `QualityTone`/`QualityAssessment` et la palette
`QUALITY_TONE_ACCENT` — `clusterQuality.ts` réexporte les types (aucun
import existant cassé), `dimensionalityQuality.ts`/`anomalyQuality.ts`
(nouveaux) les importent directement. Chaque module de verdict garde sa
propre fonction d'évaluation et ses propres seuils (silhouette, ARI de
stabilité, trustworthiness, taux de consensus) — seule la FORME du verdict
et sa palette de couleurs sont partagées, jamais la logique de seuillage
elle-même (métriques non comparables entre piliers).
**Vérifié** : tests unitaires dédiés pour chaque nouvelle fonction
d'évaluation (`assessStabilityQuality`, `assessTrustworthinessQuality`,
`assessConsensusQuality`), suite Vitest complète verte (53/53).

## Lot 7 — Produit et parcours (§J.1, §J.2, §J.3, P5-P9, I11/I12/I14)

### D7.1 — Avertissement de suppression en cascade : décompte chiffré, pas un texte générique

**Constat** : les 4 tables de job (`TrainingJob`/`ClusterModel` via
`ClusteringJob`/`DimensionalityJob`/`AnomalyJob`) référencent déjà
`datasets.id` en `ON DELETE CASCADE` — la suppression était déjà SÛRE côté
base, seul l'utilisateur n'était jamais informé de son ampleur avant de
confirmer (confirmation générique à deux clics, `useConfirmAction`).
**Retenu** : nouvel endpoint `GET /datasets/{id}/usage` (décompte par type
de job, lecture seule) — appelé côté frontend UNIQUEMENT à l'armement de la
confirmation (premier clic sur "Supprimer"), pas au montage de chaque carte
(éviterait une requête par dataset affiché pour une info consultée
seulement en cas de suppression réelle). Le message n'apparaît que si
`total > 0`.
**Vérifié** : décompte exact multi-types testé (entraînement + clustering
sur le même dataset), isolation multi-tenant testée, dataset neuf → décompte
nul.

### D7.2 — Estimation de durée avant lancement : dérivée de l'historique réel, jamais une constante inventée

**Question** : comment estimer une durée d'entraînement sans donnée de
calibration existante, sachant que le principe du produit interdit tout
chiffre inventé (skill senior-ai-saas-engineer, data-science.md) ?
**Écarté** : une formule à coefficients fixes (ex. "X secondes par ligne")
— ce serait exactement le genre de statistique inventée que le reste du
produit refuse systématiquement (SHAP, profils de clusters...).
**Retenu** : `services/duration_estimate.py` calcule un taux
(durée / (lignes × modèles × essais Optuna × folds)) sur les
entraînements RÉELLEMENT terminés de l'organisation (`TrainingJob.started_at`/
`finished_at`, 50 plus récents), médiane des taux (pas la moyenne — un seul
entraînement anormalement long ne doit pas décaler l'estimation), appliqué
aux paramètres du nouveau job. Dégradation honnête (`status: "degraded"`)
en dessous de `MIN_COMPLETED_JOBS_FOR_ESTIMATE = 3` — pas d'organisation
"type" pour combler l'absence d'historique. Affiché uniquement à l'étape
récapitulative de `Training.tsx`, jamais comme une garantie ("repère
indicatif").
**Vérifié** : dégradation honnête sans historique et sous le seuil minimal ;
proportionnalité vérifiée explicitement (doubler `n_models` double
l'estimation, sur le même taux historique) ; un job sans `row_count` connu
ne fausse jamais le taux (ignoré, pas une division par zéro).

### D7.3 — Mémoire du dernier pilier : un raccourci "Reprendre", jamais une redirection automatique

**Question** : "revenir sur `/` force aujourd'hui à rechoisir" — fallait-il
rediriger automatiquement vers le dernier pilier utilisé ?
**Écarté** : redirection automatique — l'écran d'orientation (`Orientation.tsx`)
a pour fonction explicite de faire choisir un OBJECTIF à chaque visite
(voir son propre commentaire de tête) ; une redirection systématique
retirerait ce choix plutôt que de l'assister, et surprendrait un
utilisateur qui revient volontairement sur `/` pour changer d'objectif.
**Retenu** : `frontend/src/utils/lastPillar.ts` (localStorage, dégrade
silencieusement si indisponible) — `AppShell` enregistre le pilier courant
à chaque page qui en a un ; `Orientation.tsx` affiche un lien "Reprendre « nom du pilier »" au-dessus
de la grille des 3 cartes, TOUJOURS visibles, jamais masquées.

### D7.4 — `n_models` par défaut de l'estimation de durée fixé à 4, vérifié dans le registre (pas une supposition)

**Constat** : `services/ml_registry.py::MODEL_REGISTRY` a exactement 4
entrées `is_default=True` ("stratégie produit B — boosters + RandomForest",
commentaire déjà présent dans le registre). La valeur par défaut du
paramètre `n_models` de `GET /training/estimate-duration` (et l'appel
frontend en mode guidé) est donc `4`, vérifiée par lecture directe du
registre — pas une estimation approximative du nombre de modèles comparés
par défaut.

### D7.5 — Annulation de job : un statut `"cancelled"` distinct, jamais confondu avec `"failed"` ni avec une suppression

**Question** : les 6 types de job (`TrainingJob`/`ClusteringJob`/
`DimensionalityJob`/`AnomalyJob`/`VisionClassificationJob`/`VisionAnomalyJob`)
n'avaient qu'un `DELETE /jobs/{id}` (annule le job RQ best-effort PUIS
supprime la ligne) — aucun moyen de stopper un job sans perdre toute trace
qu'il a existé.
**Retenu** : `POST /jobs/{id}/cancel` sur les 6 routers — statut
`"cancelled"` (nouvelle valeur, colonne déjà `String` libre, aucune
migration nécessaire), `error_message = "Annulé par l'utilisateur."`,
`finished_at` renseigné. Rejette avec 409 (`JOB_NON_ANNULABLE`) si le job
n'est plus `queued`/`running`. Le job reste consultable dans l'historique
— `DELETE` reste inchangé, distinct, pour qui veut réellement l'effacer.
**Retenu aussi** : extraction de `services/job_lifecycle.py`
(`try_cancel_rq_job`, `ACTIVE_STATUSES`, `CANCELLED_MESSAGE`) — la logique
d'annulation RQ best-effort était déjà dupliquée à l'identique dans les 6
`DELETE`, ce lot l'aurait sinon fait passer à 12 exemplaires. Extraction
justifiée par CE lot (sert directement le nouvel endpoint), pas un
refactoring opportuniste hors périmètre — les 6 `DELETE` existants
réutilisent la même fonction au passage plutôt que de garder l'ancien code
dupliqué à côté du nouveau.
**Retenu aussi (frontend)** : `"cancelled"` devient sa propre `Phase`
distincte de `"failed"` sur les 6 pages de job — un rendu neutre (icône
`Ban`, tons `muted`) remplace le rendu rouge `AlertCircle`/`destructive`
utilisé pour un vrai échec, pour qu'une annulation délibérée par
l'utilisateur ne se lise jamais comme si quelque chose s'était mal passé.
Bouton "Annuler" ajouté sur chaque carte de progression (clic simple, pas
de confirmation à deux clics — contrairement à la suppression, annuler
laisse une trace consultable, donc moins destructif).
**Vérifié** : 113 tests backend (annulation réussie + statut conservé en
historique, rejet 409 sur job déjà terminé, isolation multi-tenant) sur
les 6 routers ; `tsc -b`/`eslint`/vitest (58/58)/`vite build` verts côté
frontend.

### D7.6 — Relance depuis une configuration existante : reconstruction du corps de création + appel direct de la fonction existante, jamais une copie de sa validation

**Question** : "le geste le plus fréquent en pratique, impossible aujourd'hui — l'utilisateur ressaisit tout." Comment l'implémenter sans dupliquer (et risquer de faire diverger) la validation déjà écrite dans chaque `POST /jobs` ?
**Retenu** : `POST /jobs/{id}/rerun` sur les 6 routers — lit `config_json`
(+ les colonnes dédiées : `feature_columns_json`, `target_column`,
`group_column`, `task_type`, `feature_engineering_json` pour le
supervisé), reconstruit le Pydantic `*JobCreate` d'origine, puis appelle
DIRECTEMENT la fonction Python `create_*_job(body, current_user, db)` —
jamais une requête HTTP interne, jamais une réécriture partielle de sa
validation (dataset toujours prêt, colonnes toujours présentes, quota
non dépassé...). Un job relancé traverse exactement le même chemin
qu'un job créé à la main, donc ne peut jamais diverger avec le temps.
**Deux stratégies de reconstruction selon la forme du `config_json`
existant** :
- Vision (classification/anomalies) : `config_json` reprend déjà TOUS
  les champs de son `*JobCreate` à l'identique (vérifié champ par champ
  contre le schéma avant de choisir cette voie) → dépaquetage direct
  (`VisionClassificationJobCreate(vision_dataset_id=job.vision_dataset_id, **config)`),
  aucune reconstruction manuelle.
- Supervisé/clustering/réduction de dimension/anomalies : `config_json`
  ne contient qu'un SOUS-ENSEMBLE des champs (le reste vit dans des
  colonnes dédiées, ex. `feature_columns_json`) → reconstruction
  explicite champ par champ, jamais un dépaquetage aveugle qui
  échouerait silencieusement ou lèverait une erreur Pydantic peu claire
  sur un champ inconnu.
**Cas particulier anomalies** : `config_json["contamination"]` vaut soit
`"auto"` (chaîne) soit une fraction (nombre) — reconstruit vers `None`
(réglage automatique) dans le premier cas, jamais passé tel quel à
`AnomalyJobCreate.contamination: Optional[float]` qui rejetterait une
chaîne.
**Piège rencontré en testant** : les tests `_create_job()` de chaque
fichier utilisent `with patch(".../analysis_queue") as mock_queue:` en
gestionnaire de contexte scopé au SEUL appel de création initiale — le
job relancé, créé par un DEUXIÈME appel HTTP hors de ce `with`, tentait
donc un vrai `enqueue()` contre Redis (absent en test), échec en
`ConnectionError`. Corrigé en enveloppant aussi l'appel `/rerun` dans le
même mock, dans chacun des 5 fichiers concernés (le test training,
utilisant un `@patch` en décorateur couvrant toute la fonction de test,
n'a pas eu besoin de ce correctif). Pas un bug de production — Redis/RQ
tournent réellement en dehors des tests — mais un piège à connaître pour
tout futur test qui enchaîne deux appels de création dans le même test.
**Vérifié** : nouvelle configuration identique au job d'origine (dataset,
colonnes, hyperparamètres) sur les 6 routers, isolation multi-tenant
testée.
