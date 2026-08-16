# recap.md — DataLab Pro : où on en est

> Synthèse lisible de la migration, mise à jour à chaque lot. Le détail
> technique fichier par fichier vit dans [`backend/workflow.md`](backend/workflow.md) ;
> ce document répond juste à la question « qu'est-ce qui marche aujourd'hui,
> et pourquoi on a fait comme ça ».

## Le projet en une phrase

Migrer **DataLab Pro** (outil académique Streamlit d'entraînement de
modèles ML/vision) vers un **SaaS FastAPI + React** multi-utilisateurs pour
bureaux d'études, en s'inspirant de l'architecture de **CIAM**
(`concrete-ai-platform`), déjà en production — lot par lot, chaque lot
livrant quelque chose qui fonctionne réellement, jamais une réécriture
d'un coup.

L'ancienne app Streamlit reste intacte dans le dépôt (`src/`, `ui/`,
`helpers/`...) comme référence pendant le portage — voir
[`docs/legacy/README.md`](docs/legacy/README.md).

---

## Ce qui fonctionne aujourd'hui

### Lot 0 — Squelette

Backend FastAPI et frontend React/TypeScript qui démarrent et se parlent.
Aucune fonctionnalité métier — juste la fondation (config, base de données,
Docker) sur laquelle tout le reste s'appuie.

### Lot 1 — Comptes et organisations

Un bureau d'études s'inscrit et devient propriétaire (`owner`) de son
organisation ; il peut y ajouter des collègues (`member`). Chaque
organisation ne voit **jamais** les données d'une autre — vérifié
explicitement à chaque lot suivant, pas juste supposé.

*Décision produit validée : modèle "Organisation/équipe" plutôt qu'un
compte individuel isolé (comme CIAM) — cohérent avec un usage en équipe.*

### Lot 2 — Données

Upload de fichiers tabulaires (CSV, Excel, Parquet, JSON), catalogage
automatique (nombre de lignes/colonnes, types détectés), aperçu, suppression.
Interface à cartes modernes avec glisser-déposer.

### Lot 3 — Entraînement

Le cœur du produit. Sur un dataset et une colonne cible choisis :

- **3 algorithmes comparés automatiquement** (LightGBM, XGBoost, CatBoost),
  chacun optimisé par recherche d'hyperparamètres (Optuna) — l'utilisateur
  n'a jamais à choisir un algorithme lui-même, le meilleur est sélectionné
  sur un score de validation croisée (jamais sur le score final, pour ne
  pas biaiser le choix).
- **Anti-fuite de données** : si des lignes partagent un même échantillon
  (mesures répétées), une colonne de groupe garantit qu'elles ne se
  retrouvent jamais à la fois dans les données d'entraînement et de test.
- **Explicabilité (SHAP)** : quelles variables pèsent le plus dans les
  décisions du modèle.
- **Fiabilité des prédictions (CQR)** : en régression, une fourchette de
  confiance calibrée, pas juste un chiffre nu.
- Tourne en tâche de fond (file d'attente Redis) avec **progression visible
  en direct** — un entraînement peut prendre plusieurs minutes, il ne
  bloque jamais l'interface.

*Méthodologie reprise d'un notebook de référence partagé par l'utilisateur
(lu intégralement puis supprimé du dépôt une fois sa méthodologie extraite
— voir `backend/workflow.md`).*

### Lot 4a — Prédire avec un modèle entraîné, et gérer l'historique

Jusqu'ici, un modèle entraîné ne servait à rien : impossible de l'utiliser.
Corrigé : un formulaire généré automatiquement (une case par variable)
permet de saisir un nouveau cas et d'obtenir une prédiction immédiate, avec
sa fourchette de confiance ou ses probabilités par classe. Des info-bulles
expliquent en langage clair chaque métrique affichée (R², SHAP, CQR...)
pour un utilisateur qui n'est pas data scientist de métier. La sélection
manuelle des variables d'entraînement (exclure une colonne sans intérêt
prédictif) est aussi devenue accessible depuis le formulaire — et chaque
entraînement peut désormais être supprimé de l'historique (avec annulation
du job en file si besoin).

### Lot 4b — Explorer avant d'entraîner, voir au-delà des chiffres

Deux manques signalés après le Lot 3 : impossible d'explorer un dataset
avant de choisir sa cible et ses variables, et le résultat d'un modèle
n'affichait que des métriques chiffrées, jamais un graphique. Corrigé :

- **Exploration de données (EDA)** accessible depuis "Mes données" —
  statistiques par colonne, matrice de corrélation, valeurs manquantes
  signalées visuellement au-delà de 30 %, histogramme à la demande pour
  n'importe quelle variable.
- **Graphiques d'évaluation** dans le résultat d'un modèle : matrice de
  confusion et courbes ROC/précision-rappel en classification ; nuage
  prédit-vs-réel et résidus en régression — avec les mêmes info-bulles
  pédagogiques qu'au Lot 4a pour rester lisible par un non-expert.
- Bibliothèque de graphiques tranchée à ce lot : **Recharts**, pas Plotly
  (décision déjà actée mais implémentée ici pour la première fois).

*Vérifié bout en bout sur les vrais datasets de l'utilisateur (Iris,
Concrete Compressive Strength) — pas seulement sur données synthétiques.*

### Lot 4c — Ingénierie de variables guidée, sans fuite

Suggestions de variables dérivées (décomposition de date, ratios,
regroupement des modalités rares + encodage de fréquence, imputation
configurable par colonne) proposées automatiquement à partir des garde-fous
déjà détectés sur le dataset, **approuvées explicitement par l'utilisateur**
avant d'entrer dans l'entraînement — jamais appliquées silencieusement. La
transparence va jusqu'au résultat : le modèle affiche quelles
transformations ont réellement été utilisées.

### Lot 5 — Catalogue supervisé élargi, architecture modulable par registre

Jusqu'ici, seuls 3 algorithmes de boosting étaient comparés (décision
volontaire du Lot 3, pour une explicabilité uniforme). Ce lot élargit le
catalogue à **9 modèles sur 3 familles** (arbres/ensembles, régression
linéaire régularisée, distance/noyau) via une **architecture en registre** :
ajouter un modèle au catalogue ne demande plus de toucher le moteur
d'entraînement, seulement de déclarer une nouvelle entrée. Par défaut,
l'outil ne lance que le sous-ensemble le plus robuste et rapide (les 3
boosters + Random Forest) — les modèles plus sensibles ou plus lents (SVM,
KNN, régression linéaire, Naive Bayes) restent disponibles dans le
catalogue, prêts à être activés par un utilisateur avancé dans un lot futur,
sans qu'aucune UI de choix ne soit encore proposée.

Deux angles techniques rouverts pour ce lot, tous deux prouvés fonctionner
correctement au-delà des seuls arbres :

- **L'explicabilité (SHAP)** s'adapte désormais au type de modèle plutôt que
  de supposer un arbre — avec, pour les modèles les plus coûteux à
  expliquer, un calcul borné dans le temps et un message clair plutôt qu'un
  blocage silencieux quand l'explication détaillée n'est pas disponible.
- **La fiabilité des prédictions (CQR)**, déjà indépendante de l'algorithme
  gagnant depuis le Lot 3, continue de fonctionner sans adaptation pour
  n'importe quel nouveau modèle de régression du catalogue.

*Mesuré, pas estimé : le surcoût du catalogue complet par rapport au
sous-ensemble par défaut est d'environ 7 % sur un entraînement réel — les
nouveaux modèles sont bon marché à entraîner, le temps reste dominé par la
recherche d'hyperparamètres des boosters, commune aux deux configurations.*

### Lot déséquilibre — rééquilibrer les classes rares, sans y toucher en silence

Quand une classe est nettement plus rare que les autres (ex. 92 %/8 %), un
modèle peut afficher une bonne exactitude globale tout en ratant
systématiquement la classe rare — souvent la plus importante à détecter
(fraude, panne, défaut...). Ce lot ajoute la possibilité de **donner plus de
poids à la classe rare pendant l'entraînement**, proposée à l'utilisateur
avec une explication claire de l'arbitrage (rappel de la classe rare ↑,
fausses alertes sur la classe majoritaire ↑), jamais activée d'office —
même principe d'approbation que l'ingénierie de variables du Lot 4c.

Techniquement, un seul mécanisme (pondération par échantillon) couvre 8 des
9 modèles du catalogue de façon uniforme, sans code différent par
bibliothèque — seul le plus proche voisin (KNN) n'a structurellement aucune
notion de pondération. Aucune ligne n'est dupliquée ni supprimée : le split
train/test et la validation croisée restent identiques, seule la
pondération vue par le modèle change.

*Vérifié sur un cas construit : activer le rééquilibrage fait passer le
rappel de la classe minoritaire de 12,5 % à 62,5 %, au prix attendu d'un
rappel global un peu plus faible — l'arbitrage même que le message affiché
décrit.*

### Lot D — Leaderboard : voir tous les modèles comparés, pas seulement le gagnant

Jusqu'ici, un entraînement comparait plusieurs modèles en coulisses mais
seul le gagnant était affiché ("LightGBM, accuracy = 0,947") — le travail
de comparaison réel de l'outil restait invisible, tout comme la raison du
choix. Ce lot rend ce travail visible : la fiche de résultat affiche
désormais **tous** les modèles comparés, classés sur la métrique qui a
réellement décidé du gagnant (jamais l'exactitude brute, trompeuse sur un
dataset déséquilibré), avec une phrase en langage clair expliquant
l'écart ("LightGBM retenu : meilleur ROC-AUC en validation croisée, devant
CatBoost de 0,023 points"). En régression, l'erreur en unité réelle (RMSE)
est affichée à côté du R², plus lisible pour un ingénieur BE.

*Corrigé au passage : la carte d'historique d'un entraînement de
classification affichait l'exactitude brute au lieu de la métrique de
sélection — même piège que celui que ce lot cherche à éviter partout
ailleurs, corrigé dans son propre commit.*

*Comparer plusieurs entraînements entre eux (pas seulement les modèles
d'un même entraînement) est volontairement reporté à un lot dédié
(D-bis), pour ne pas bâcler ni l'un ni l'autre.*

### Fix — un entraînement qui échoue ne montre plus jamais de code brut à l'utilisateur

Un entraînement sur un dataset réel (une colonne quasi-identifiant, comme un
numéro de série) avait fait planter le serveur avec un message d'erreur
brut — code technique et chemins de fichiers internes affichés tels quels.
Deux corrections, dont une bien plus large que le seul incident :

- **La vraie cause, corrigée pour de bon, pas juste contournée** : l'outil
  convertissait inutilement les données encodées en un format "plein"
  (dense) avant de les donner aux modèles, alors que ceux-ci savent
  travailler directement avec leur format compact d'origine (sparse) — une
  colonne comme un identifiant, qui reste minuscule sous sa forme compacte,
  explosait en centaines de mégaoctets une fois "aplatie" pour rien. Corrigé
  à la source, dans le moteur d'entraînement : ça ne dépend plus du dataset
  utilisé, ni de la colonne en cause.
- **Un entraînement qui échoue quand même** (mémoire insuffisante sur un
  très gros dataset, par exemple) affiche désormais un message clair et une
  piste d'action ("essayez de retirer cette colonne, ou de réduire la
  taille du jeu de données") — jamais plus de code brut. Le détail
  technique complet reste dans les journaux du serveur, consultables par
  l'équipe si besoin.

*Le garde-fou qui repère déjà les colonnes ressemblant à un identifiant
(Lot B) existait avant ce correctif et aurait dû alerter — il reste en
place, enrichi d'un détail chiffré, mais n'est plus la seule protection :
la vraie cause est désormais réglée à la racine.*

### Lot E1-ter — Refonte structurelle des pages Dashboard / Données / Entraînement / Résultats

Après le socle visuel (palette claire, navigation par piliers, logo),
la STRUCTURE des 4 pages métier a été revue. Le dashboard ne montrait que
la gestion d'équipe : il ouvre maintenant sur une vue d'ensemble de
l'activité (datasets, entraînements récents, statuts). La page Données
avait sa zone d'upload qui prenait la moitié de l'écran et repoussait les
datasets hors-vue — inversé : bande d'upload compacte, grille dense en
dessous. L'entraînement est devenu un **pipeline guidé en 5 étapes
numérotées** (données → contrôle qualité → améliorations automatiques →
réglages avancés repliés → lancer), sur une **page dédiée** : on configure,
on lance, et le résultat s'affiche en place sur la même page — l'historique
complet des entraînements passés se consulte désormais depuis le tableau de
bord. La page Résultats gagne un bloc "Interprétation du modèle" en langage
clair (pourquoi ce modèle a gagné, ce que dit l'analyse des variables) et
des courbes ROC/précision-rappel multiclasses enfin lisibles (isolation
d'une classe au clic ou au survol).

**Bug de crédibilité corrigé en premier** : le graphe "Variance entre les
découpages de validation croisée" affichait des valeurs absurdes (jusqu'à
9 chiffres) au lieu d'un score entre 0 et 1. Ce n'était pas un bug de
lecture de données — le R² par fold de validation croisée est
mathématiquement non borné en dessous, et s'effondre légitimement quand un
petit découpage a une cible presque constante. Corrigé côté affichage
(les valeurs sont maintenant bornées pour la lecture, sans jamais toucher
au calcul qui sert réellement à choisir le modèle), avec un test qui vérifie
que ce que voit l'utilisateur reste toujours entre 0 et 1.

*Rendu visuel non vérifié en conditions réelles par ce lot — aucun outil de
navigateur disponible dans cet environnement de travail ; une revue visuelle
page par page reste à faire.*

### Lot E2 — Mode guidé / mode expert

Le Lot 5 avait construit un catalogue de 9 modèles et des paramètres
(nombre de blocs de validation croisée, graine aléatoire, niveau de
confiance des intervalles) déjà fonctionnels côté moteur, mais jamais
accessibles depuis l'écran d'entraînement — soit non exposés en interface,
soit tapés côté client sans jamais être envoyés au serveur. Ce lot les
rend pilotables, sans jamais imposer cette complexité à l'utilisateur
courant :

- **Mode guidé (toujours le défaut)** : rien ne change. Un clic sur
  "Lancer" utilise exactement les mêmes réglages qu'avant ce lot.
- **Mode expert (activation volontaire, repliée)** : un interrupteur dans
  l'étape "Réglages avancés" du pipeline guidé déplie des manettes
  supplémentaires, chacune expliquée en langage clair (pas de jargon nu) :
  - **Modèles comparés** — cases à cocher sur le catalogue complet des 9
    modèles (regroupés par famille), sous-ensemble par défaut pré-coché ;
    les modèles plus lents (SVM, KNN) sont signalés par un avertissement.
  - **Recherche d'hyperparamètres** (nombre d'essais Optuna) — déplacée
    du formulaire guidé vers le mode expert, où elle a plus sa place.
  - **Validation croisée** (nombre de blocs), **graine aléatoire**
    (reproductibilité) et **confiance des intervalles** (régression) —
    branchés sur des paramètres du moteur qui existaient déjà mais
    n'étaient jusqu'ici jamais transmis par le formulaire.
  - **Rééquilibrage des classes** — la suggestion automatique existante
    reste en mode guidé ; le mode expert permet en plus de la forcer ou de
    l'annuler manuellement.

Chaque manette experte démarre à la valeur par défaut du mode guidé :
activer le mode expert sans rien changer produit exactement le même
entraînement — vérifié par un test dédié, côté frontend (construction du
payload) et côté backend (le serveur retombe sur son sous-ensemble par
défaut dès que rien n'est envoyé).

*Rendu visuel non vérifié en conditions réelles par ce lot, comme pour
E1-ter — aucun outil de navigateur disponible dans cet environnement.*

### Lot Explicabilité globale — comprendre le modèle au-delà des chiffres

Le SHAP du Lot 5 disait déjà quelles variables comptaient en moyenne
(barres), jamais dans quel sens ni pour quel cas précis. Ce lot ajoute trois
angles complémentaires sur le modèle retenu : un **nuage de points (beeswarm)**
qui montre, observation par observation, si une variable pousse la
prédiction vers le haut ou vers le bas ; une **importance par permutation**,
mesure indépendante du SHAP qui vient recouper ses conclusions ; une
**courbe de calibration** (le modèle est-il "sûr à raison" quand il annonce
une probabilité ?) ; et une **courbe d'apprentissage** (le modèle
bénéficierait-il de plus de données ?). Chaque graphique est accompagné
d'une phrase d'interprétation en langage clair, jamais un graphe brut sans
explication, et dégrade proprement (message clair) plutôt que de planter
quand le calcul n'est pas disponible pour un modèle donné.

### Refonte UI : design system moderne

Le premier système visuel (dégradé teal, Lot E1/E1-bis) a été remplacé par
une refonte complète calquée sur une maquette de référence moderne : palette
de tokens sémantiques cohérente (le bleu de marque devient l'accent
principal partout, pas seulement sur le logo), sidebar fixe groupée par
pilier métier, pipeline d'entraînement en wizard horizontal à étapes
visibles, cartes réorganisées en grille plutôt qu'empilées sur les pages
Résultats et Exploration de données. Plusieurs bugs réels d'affichage
corrigés au passage (build cassé par un commentaire CSS mal formé, bouton
tronqué sur une carte trop étroite, nuage de points faussé par une valeur
manquante).

*Les deux lots ci-dessus ont été livrés par une session parallèle sur ce
même dépôt (voir `backend/workflow.md` pour le détail fichier par fichier,
reconstitué a posteriori à partir du contenu réel des commits).*

### Audit backend + Lot Nettoyage guidé des variables, refonte Résultats/Datasets

Un audit expert du backend (lecture seule, rapport validé avant tout code)
a confirmé un pipeline ML supervisé déjà solide (anti-fuite bout en bout,
sélection sur la validation croisée, explicabilité multi-famille) et
identifié une lacune concrète, signalée par l'utilisateur : les garde-fous
détectaient déjà les colonnes sans valeur prédictive (identifiants,
constantes) avec une recommandation textuelle, mais rien ne permettait de
les exclure en un clic — il fallait lire l'alerte puis décocher la colonne
à la main, ailleurs dans le formulaire. Corrigé, avec deux détections
supplémentaires dans le même esprit :

- **Colonnes dupliquées** (contenu strictement identique sous un autre nom)
  et **variables numériques mal typées en texte** (virgule décimale,
  séparateur de milliers — invisibles jusqu'ici, traitées à tort comme des
  catégories) viennent compléter les détections déjà en place (constantes,
  identifiants).
- **Action "Exclure" directement sur l'alerte**, plus un bouton "Tout
  exclure" — approuver une suggestion retire vraiment la colonne du
  formulaire, sans aller-retour manuel. La conversion numérique suit le
  même principe d'approbation explicite que l'ingénierie de variables du
  Lot 4c.
- **Analyse de qualité utilisable dès l'exploration d'un dataset**, avant
  même de choisir une cible pour un entraînement (auparavant réservée au
  formulaire d'entraînement).

*Ce même passage a aussi fait évoluer le design des deux écrans identifiés
comme les moins aboutis malgré un contenu déjà riche : la page d'exploration
de données (9 analyses jusque-là empilées verticalement, sans hiérarchie)
et la page de résultat d'un entraînement (10+ sections dans la même
situation) sont désormais organisées en onglets thématiques, avec des
en-têtes de section identifiables (icône colorée) plutôt qu'une liste de
libellés gris uniformes ; la sidebar est passée d'un blanc quasi invisible
à un fond bleu de marque assombri, pour une identité visuelle plus nette.
Rendu réel non vérifié dans un navigateur (aucun outil disponible dans cet
environnement) — revue visuelle à faire.*

### Refonte visuelle globale, corrigée en direct sur retour utilisateur

Le fond des pages (hors sidebar) restait quasi blanc malgré le token dédié
(jamais réellement appliqué depuis la refonte sidebar) — retinté en
bleu-gris visible, cohérent avec la sidebar. Nouveaux interrupteurs et
onglets au style "contrôle segmenté" (fond neutre, pastille active en
relief). Palette de graphiques revalidée pour l'accessibilité daltonienne
avec l'outil dédié (un ancien réglage échouait un test de séparation des
couleurs, jamais vérifié avant). Une première version colorait les cartes
Dataset/Dashboard par statut plutôt que par identité — corrigée en direct
sur capture d'écran fournie par l'utilisateur : en usage réel, la
quasi-totalité des éléments partagent le même statut au même moment, ce qui
rendait les grilles monochromes plutôt que vivantes. La table de résumé de
l'exploration de données a aussi été retravaillée (badges de type colorés,
grands nombres lisibles avec séparateurs de milliers).

### Lot Explicabilité locale — pourquoi CETTE prédiction précise

L'explicabilité SHAP existante ne répondait qu'à "quelles variables comptent
en moyenne pour ce modèle" — jamais "pourquoi ce cas précis a reçu cette
prédiction", la question la plus naturelle juste après avoir testé une
prédiction. Le formulaire de prédiction affiche désormais un graphique en
barres divergentes montrant, variable par variable, ce qui pousse la
prédiction vers le haut ou vers le bas pour l'observation saisie.

*Bug réel trouvé en testant sur un modèle réel* : pour une classification à
deux classes, la bibliothèque d'explicabilité peut renvoyer un calcul pour
une seule des deux classes, quelle que soit celle réellement prédite —
sans correction, expliquer un cas prédit dans l'autre classe aurait montré
l'inverse de la réalité (une variable en faveur de la prédiction affichée
comme y étant opposée). Détecté en vérifiant que la somme des effets
affichés retombe bien sur la probabilité réellement annoncée par le modèle
— pas supposé correct — puis corrigé et verrouillé par un test qui
vérifie les deux classes, pas seulement celle qui fonctionnait déjà par
hasard.

### Lot D-bis — comparer plusieurs entraînements entre eux

Le Lot D rendait déjà visible la comparaison des modèles au sein d'UN
entraînement (leaderboard). Ce lot ajoute la comparaison ENTRE
entraînements : une nouvelle page "Historique" liste tous les
entraînements passés (jusqu'ici seuls les 5 plus récents étaient visibles
sur le tableau de bord), avec une sélection multiple pour comparer
métriques et réglages côte à côte — les réglages qui diffèrent réellement
entre les entraînements sélectionnés sont surlignés automatiquement.
*Corrige au passage* un lien du tableau de bord ("Voir tout") qui menait
vers le formulaire de lancement d'un entraînement plutôt que vers un
historique — l'historique n'avait tout simplement pas d'écran dédié
jusqu'ici.

### Lot 9 — registre de modèles versionné

L'artefact d'un modèle entraîné existait déjà, mais rien ne distinguait
"un modèle entraîné parmi d'autres" de "LE modèle sur lequel on peut
compter pour ce problème", et rien ne permettait de le récupérer hors de
la plateforme. Depuis la page Résultats, un modèle peut désormais être
marqué "en validation" ou "en production" — un seul modèle en production à
la fois par dataset et cible, promouvoir un nouveau modèle démet
automatiquement l'ancien (jamais supprimé, juste plus la référence
actuelle). Un bouton "Exporter l'artefact" télécharge le modèle complet
pour un usage hors de la plateforme. La page Historique affiche désormais
un panneau "Registre de modèles" listant tout ce qui a été promu.

*Export ONNX volontairement écarté pour l'instant* : le pipeline peut
inclure une transformation maison (regroupement de catégories rares) sans
équivalent standard côté ONNX — mieux vaut un export qui fonctionne
vraiment (le format Python natif) qu'un export qui promet l'interopérabilité
et échoue silencieusement sur certains cas.

*Traçabilité complétée au même moment* : chaque modèle entraîné enregistre
désormais les versions exactes des librairies ML utilisées
(scikit-learn, LightGBM, XGBoost...) — un modèle promu en production doit
pouvoir être audité et reproduit, pas seulement réutilisé.

### Lot 10 — durcissement SaaS (portée technique)

Deux garde-fous, sans toucher aux questions commerciales (plans
tarifaires, facturation — hors périmètre technique) :

- **Journal d'audit** : qui a supprimé ce dataset, ajouté ce membre, promu
  ce modèle, et quand — consultable par le propriétaire de l'organisation
  depuis le tableau de bord.
- **Quota d'entraînements concurrents** par organisation (3 par défaut) :
  un seul worker traite les entraînements de toutes les organisations à la
  fois — sans limite, une organisation qui lance beaucoup d'entraînements
  d'affilée pourrait faire attendre toutes les autres. Un entraînement
  terminé ou en échec libère immédiatement sa place.

### Lot Audit + durcissement — corriger avant de continuer, pas empiler dessus

Un audit expert complet (lecture seule, backend + frontend + code legacy) a
d'abord établi un état des lieux honnête : le ML supervisé était solide,
mais un nombre limité de points concrets restaient à corriger avant
d'ajouter un nouveau pilier par-dessus — pas un blocage total, un point
d'arrêt ciblé.

**Corrigé côté fiabilité** : un job d'entraînement dont le worker meurt en
cours de route ne bloque plus indéfiniment le quota de l'organisation
(nouveau watchdog) ; le serveur refuse désormais de démarrer en production
avec la clé de sécurité par défaut du dépôt (avant : un simple
avertissement journalisé) ; les tentatives de connexion échouées sont
limitées par IP (brute force rendu impraticable) ; une classe rare qui finit
entièrement dans les données de test après un split anti-fuite par groupe
produit maintenant un message diagnosticable plutôt qu'une erreur technique
brute. *Un correctif a nécessité deux passes : la première interceptait
toutes les erreurs "RuntimeError" pour leur donner un message plus clair,
mais ça incluait aussi de vraies erreurs techniques de bibliothèque — la
suite de tests existante l'a détecté immédiatement, corrigé avec un type
d'erreur dédié plutôt qu'une catégorie trop large.*

**Refonte visuelle guidée par des retours directs sur captures d'écran**,
en plusieurs allers-retours : le fond des pages est passé d'un bleu-gris
terne à un quasi-blanc (les cartes se détachent maintenant par l'ombre) ;
chaque page a un en-tête avec une icône en dégradé de marque, plus un titre
flottant sans ancrage ; les cartes de résultats et d'exploration portent
une couleur assortie à leur section, sur tous les onglets ; un vrai premier
onboarding du produit remplace la barre de recherche qui ne faisait rien.
*Un vrai bug de crédibilité trouvé au passage : le graphe de variance entre
les découpages de validation croisée affichait des valeurs aberrantes — pas
retrouvé rapidement, retiré plutôt que laissé cassé, sur décision
explicite.*

**Deux trous UX refermés**, l'un comme l'autre déjà identifiés mais jamais
traités : rafraîchir la page pendant qu'un entraînement tourne ne renvoie
plus silencieusement au formulaire de configuration (persistance de
session) ; ouvrir un résultat ou l'exploration d'un dataset se reflète
maintenant dans l'URL — partageable, et ça survit à un rafraîchissement.

*Reporté consciemment, pas oublié* : découper les deux plus gros composants
frontend en sous-parties plus petites, et mettre en place des tests de
composants React (aucune infrastructure de test de ce type dans le dépôt
aujourd'hui) — des chantiers plus lourds, pour un lot dédié. Détail complet
dans [`AUDIT_ROADMAP.md`](AUDIT_ROADMAP.md) et
[`backend/workflow.md`](backend/workflow.md).

### Lot 11+12 — ML non supervisé : clustering et profils de segments

Premier module du deuxième pilier de l'app (`ML non supervisé`, jusqu'ici
"Bientôt disponible"). Module **séparé** de l'entraînement supervisé — pas
une extension : le clustering n'a pas de cible, les hypothèses du moteur
supervisé (score de sélection, encodage de `y`) n'auraient pas de sens ici.
Même méthodologie que le reste du projet : registre d'algorithmes (comme le
catalogue de 9 modèles du Lot 5), tâche de fond RQ, watchdog et quota
partagés avec le supervisé (un seul worker physique traite les deux).

- **Comparaison automatique de plusieurs configurations** — K-Means,
  clustering hiérarchique et DBSCAN comparés sur plusieurs nombres de
  groupes (2 à 8) à chaque lancement, classés sur le score de silhouette
  (jamais un seul essai lancé à l'aveugle) — même esprit que le leaderboard
  du Lot D côté supervisé. K-Means rapide (gros volumes) disponible en mode
  expert.
- **Profils de segments automatiques** — chaque groupe découvert est décrit
  par sa taille, ses statistiques et surtout ce qui le distingue le plus du
  reste (pas seulement ses propres moyennes) : jamais un texte inventé,
  uniquement des statistiques réellement calculées.
- **DBSCAN sans réglage à deviner** : le rayon de voisinage (`eps`) est
  résolu automatiquement à partir de la distribution réelle des distances
  entre points du dataset, pas une valeur générique à l'aveugle.
- Vérifié sur un cas construit (3 groupes séparés, avec une variable
  catégorielle parfaitement corrélée) : les trois algorithmes retrouvent
  les 3 groupes attendus, silhouette proche de 0,9, chaque segment
  correctement dominé par sa vraie catégorie d'origine — pas seulement "ça
  ne plante pas".

*Volontairement pas dans ce lot* (livré ensuite, voir section suivante) :
détection d'anomalies tabulaire et réduction de dimension (PCA/t-SNE/UMAP)
— signalé honnêtement dans la page elle-même plutôt que promis sans être
livré.

### Lot 13+14 — Réduction de dimension et détection d'anomalies : le pilier non supervisé au complet

Les deux modules manquants du pilier "ML non supervisé" annoncés
honnêtement par la page Clustering ("arrive bientôt") — chacun un module
backend séparé, même principe déjà appliqué au clustering (Lot 11+12) :
aucune notion de cible partagée avec le supervisé, aucun code partagé
au-delà des utilitaires déjà génériques (préprocesseur, échantillonnage).

**Bug réel trouvé en testant en direct, pas en théorie** : en ouvrant la
page Clustering dans un vrai navigateur, une erreur 404 est apparue au
lancement d'un clustering — le proxy de développement Vite (qui redirige
les appels API du frontend vers le backend) listait `/api`, `/auth`,
`/datasets`, `/training`, mais avait été oublié pour `/clustering` au
Lot 11+12. Corrigé, et les préfixes `/dimensionality`/`/anomalies` de ce
lot ajoutés par avance pour ne pas reproduire le même oubli.

**Réduction de dimension (Lot 13)** — PCA, t-SNE et UMAP :

- UMAP inclus dès ce lot (décision explicite, malgré le risque
  d'installation sous Windows) — `umap-learn==0.5.6` épinglé volontairement
  (pas la dernière version) : les versions récentes exigent
  scikit-learn≥1.6, ce qui aurait forcé une mise à niveau majeure du
  scikit-learn du projet (1.3.2) et un risque de régression sur tout le
  pipeline supervisé/clustering déjà testé. Vérifié par un calcul réel
  avant d'écrire le reste du lot.
- Pas de "leaderboard" façon clustering : PCA, t-SNE et UMAP n'ont pas de
  métrique de qualité commune permettant d'élire un gagnant. La PCA est
  **toujours** calculée en plus de la méthode choisie (variance expliquée,
  variables les plus contributives) et sert de repère de fidélité
  (`trustworthiness`, calculée aussi pour la méthode principale) — jamais
  un texte inventé sur la qualité d'une projection.
- **Rigueur d'affichage corrigée sur retour direct** : la variance
  expliquée n'est réellement définie que pour la PCA — elle n'est plus
  affichée comme une métrique de t-SNE/UMAP, mais isolée dans un bloc
  "Référence PCA" clairement séparé (calculée en plus, à titre de repère).
  Le tableau des variables contributives (PC1/PC2) n'a de sens que pour une
  projection PCA — masqué pour t-SNE/UMAP, dont les axes ne s'interprètent
  pas de cette façon. "Fidélité de la projection" renommée "Conservation
  des voisinages" (nom qui correspond réellement à ce que mesure
  `trustworthiness` : dans quelle mesure les voisinages proches sur la
  projection l'étaient déjà dans les données d'origine).
- Note explicative obligatoire sur chaque résultat t-SNE/UMAP : ces
  méthodes préservent les voisinages locaux mais pas les distances
  globales — jamais présentées comme une carte fidèle des distances
  réelles (skill senior-ai-saas-engineer, data-science.md).
- Nuage de points interactif (Recharts, première utilisation d'un
  `ScatterChart` dans le projet) avec coloration au choix par n'importe
  quelle variable analysée (recalcul instantané, endpoint séparé et léger —
  pas de nouveau calcul coûteux pour changer la couleur).
- Lien croisé depuis un résultat de clustering ("Visualiser en 2D") —
  dataset et variables pré-remplis via l'URL, **aucun couplage backend**
  (pas de transmission des labels de cluster).

**Détection d'anomalies (Lot 14)** — Isolation Forest et LOF :

- Les deux algorithmes tournent **toujours ensemble** (jamais un seul essai
  à l'aveugle, même principe que le reste du produit) : sans vérité terrain
  disponible, impossible d'élire un "gagnant" comme au clustering. Un score
  de consensus continu (moyenne des rangs de chaque méthode — leurs scores
  bruts ne sont pas sur la même échelle) et un niveau de confiance
  (confirmée par les deux méthodes / une seule / aucune) recoupent les deux
  résultats sans jamais inventer de nombre.
- Chaque observation classée porte une explication réelle : les variables
  qui s'écartent le plus de la population (écart-type), et les valeurs
  catégorielles rares — jamais un texte généré sans base statistique.
- **Correction d'un oubli du Lot 11+12 trouvé en construisant ce lot** : le
  quota de jobs concurrents partagés supervisé/clustering ne comptait en
  réalité que dans un sens — créer un entraînement supervisé ignorait les
  clusterings déjà actifs. Extrait en helper commun
  (`services/job_quota.py`), réutilisé par les 4 types de job désormais
  (supervisé, clustering, réduction de dimension, anomalies).

**Vérifié** : 64 nouveaux tests (34 réduction de dimension, 30 anomalies —
registre, moteur sur des cas construits avec structure connue, worker,
API), suite de régression complète du pilier non supervisé + supervisé
rejouée après chaque lot (99 tests, aucune casse), `tsc`/`vite build`/
`npm run lint`/`vitest` verts à chaque étape.

---

## Robustesse — pas juste "ça marche chez moi"

- **Tests automatisés** (`backend/tests/`, pytest) : 174 tests qui restent
  dans le dépôt et couvrent l'isolation entre organisations, les
  permissions, l'entraînement réel (pas mocké, y compris sur les 9 modèles
  du catalogue Lot 5), la prédiction, la suppression, l'exploration de
  données (EDA), les données d'évaluation, et la cohérence du leaderboard
  (Lot D : gagnant ↔ candidat, classement sur le score de sélection).
- **Bugs réels trouvés et corrigés en usage réel**, pas en théorie :
  - SHAP change de format de sortie en classification multiclasse selon la
    version installée — trouvé en testant sur un vrai dataset Iris, corrigé,
    couvert par un test de non-régression.
  - RQ (la file de tâches) est conçu autour de primitives Unix absentes de
    Windows (`os.fork`, `signal.SIGALRM`) — corrigé une fois pour toutes
    avec un point d'entrée qui fonctionne pareil sur Windows et sur
    Linux/Docker.
  - Le conteneur Redis de développement ne redémarrait pas automatiquement
    avec Docker Desktop — reconfiguré pour survivre aux redémarrages.
  - Suppression d'un entraînement **terminé** (avec modèle associé)
    impossible sur PostgreSQL (500 systématique) — l'ORM tentait de mettre
    à `NULL` une colonne `NOT NULL` avant de laisser la base gérer la
    cascade ; le job sans modèle (encore en file) se supprimait bien, d'où
    la confusion initiale. Corrigé + couvert par un test de régression qui
    insère un vrai modèle avant suppression (voir `backend/workflow.md`).
- **Migrations de schéma idempotentes** (façon CIAM) plutôt qu'exiger une
  base vierge à chaque évolution du modèle de données.
- **Isolation vérifiée à chaque lot**, pas supposée : chaque nouvelle
  ressource (datasets, entraînements) est testée pour confirmer qu'une
  organisation ne voit jamais les données d'une autre.

---

## Décisions produit prises en cours de route

| Sujet | Décision | Pourquoi |
| --- | --- | --- |
| Multi-utilisateurs | Organisation/équipe, pas compte individuel isolé | Usage en équipe dans un bureau d'études |
| File de tâches | RQ + Redis | CIAM n'en a pas besoin (tâches courtes) ; un entraînement ML, si |
| Positionnement | Généraliste multi-secteurs | Pas de verrouillage sur un métier particulier dès le départ |
| Catalogue Lot 3 | 3 algos de boosting seulement au lancement | Permet SHAP + CQR de qualité uniforme le temps de livrer l'architecture par registre (élargi à 9 modèles au Lot 5) |
| Sélection par défaut Lot 5 | Seuls boosters + RandomForest tournent automatiquement | Modèles plus lents/sensibles (SVM, KNN...) réservés au mode expert (Lot E2), pour garder un temps d'entraînement raisonnable par défaut |
| Rééquilibrage des classes | Pondération (`class_weight`/`sample_weight`), pas de rééchantillonnage (SMOTE) | Le rééchantillonnage synthétique est sensible aux fuites (doit être fait DANS les folds) — réservé à un lot expert dédié ; la pondération est sans risque et couvre déjà la majorité des besoins |
| Graphiques | Recharts, pas Plotly | Plus léger, thémable à notre design, déjà éprouvé par CIAM |
| Progression | Polling REST, pas WebSocket | Plus simple à fiabiliser pour ce volume d'événements |

---

## Ce qui manque encore (feuille de route)

Identifié explicitement en testant le produit, pas oublié :

| Lot | Contenu |
| --- | --- |
| **15** | Vision par ordinateur — découpé en 4 sous-lots (voir `AUDIT_ROADMAP.md` section H). **Backend + frontend livrés** (2026-08-15/16, A infra dataset, B classification/transfer learning, C anomalies visuelles MVTec AD, D Grad-CAM) — 9 bugs critiques du legacy corrigés pendant le portage du sous-lot C, pas après. Testé avec l'app réelle (serveurs + navigateur) : plusieurs bugs réels trouvés et corrigés (structure MVTec AD officielle avec dossier catégorie englobant + `ground_truth/`, proxy Vite `/vision` manquant, endpoint de service d'image manquant, chemins Windows non portables). 106 tests verts. Détection d'objets/annotation assistée hors périmètre, reportées à un Lot 16+ non cadré. |

*Durcissement SaaS commercial restant (plans tarifaires, facturation, quota
de stockage) : hors périmètre technique, décision produit à cadrer
séparément. SMOTE avancé : hors périmètre pour l'instant.*

---

## Démarrer en local

Voir [`README.md`](README.md) pour les commandes complètes. En résumé :
backend (`uvicorn`), worker (`python -m workers.run_worker`), Redis (Docker),
frontend (`npm run dev`) — les quatre doivent tourner pour un entraînement
de bout en bout ; les trois premiers suffisent pour tout le reste.
