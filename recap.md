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
| **D-bis** (prochain) | Comparaison inter-jobs — tri par score, diff de config entre entraînements (reporté du Lot D pour ne pas le bâcler) |
| **6-8** | Vision par ordinateur / détection d'anomalies (l'autre grand pilier de l'app historique, pas encore porté) |
| **9** | Registre de modèles versionné (l'artefact existe déjà, pas encore le versioning/export) |
| **10** | Durcissement SaaS : audit, quotas, facturation — prêt pour un client pilote |

*Clustering (non supervisé) et SMOTE avancé : hors périmètre pour l'instant,
non planifiés dans les lots ci-dessus.*

---

## Démarrer en local

Voir [`README.md`](README.md) pour les commandes complètes. En résumé :
backend (`uvicorn`), worker (`python -m workers.run_worker`), Redis (Docker),
frontend (`npm run dev`) — les quatre doivent tourner pour un entraînement
de bout en bout ; les trois premiers suffisent pour tout le reste.
