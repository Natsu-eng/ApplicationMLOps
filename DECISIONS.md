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
