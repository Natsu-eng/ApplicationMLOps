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

---

## Robustesse — pas juste "ça marche chez moi"

- **Tests automatisés** (`backend/tests/`, pytest) : 146 tests qui restent
  dans le dépôt et couvrent l'isolation entre organisations, les
  permissions, l'entraînement réel (pas mocké, y compris sur les 9 modèles
  du catalogue Lot 5), la prédiction, la suppression, l'exploration de
  données (EDA) et les données d'évaluation.
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
| Sélection par défaut Lot 5 | Seuls boosters + RandomForest tournent automatiquement | Modèles plus lents/sensibles (SVM, KNN...) réservés à un mode expert futur (Lot E), pour garder un temps d'entraînement raisonnable par défaut |
| Graphiques | Recharts, pas Plotly | Plus léger, thémable à notre design, déjà éprouvé par CIAM |
| Progression | Polling REST, pas WebSocket | Plus simple à fiabiliser pour ce volume d'événements |

---

## Ce qui manque encore (feuille de route)

Identifié explicitement en testant le produit, pas oublié :

| Lot | Contenu |
| --- | --- |
| **E** (prochain) | Mode expert — exposer le choix d'activer les modèles hors sous-ensemble par défaut (ExtraTrees, linéaire, SVM, KNN, Naive Bayes) |
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
