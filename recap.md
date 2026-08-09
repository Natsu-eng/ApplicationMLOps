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

---

## Robustesse — pas juste "ça marche chez moi"

- **Tests automatisés** (`backend/tests/`, pytest) : 26 tests qui restent
  dans le dépôt et couvrent l'isolation entre organisations, les
  permissions, l'entraînement réel (pas mocké), la prédiction, la
  suppression.
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
| Catalogue Lot 3 | 3 algos de boosting seulement (pas RF/SVM/linéaire...) | Permet SHAP + CQR de qualité uniforme ; le catalogue large arrive au Lot 5 |
| Graphiques | Recharts, pas Plotly | Plus léger, thémable à notre design, déjà éprouvé par CIAM |
| Progression | Polling REST, pas WebSocket | Plus simple à fiabiliser pour ce volume d'événements |

---

## Ce qui manque encore (feuille de route)

Identifié explicitement en testant le produit, pas oublié :

| Lot | Contenu |
| --- | --- |
| **4b** (en cours) | Exploration de données (EDA) avant l'entraînement — distributions, corrélations, valeurs manquantes ; et graphiques d'évaluation (matrice de confusion, ROC/PR, résidus) |
| **4c** | Ingénierie de variables — créer des variables dérivées (ratios, transformations) avant l'entraînement |
| **5** | Catalogue ML complet (RandomForest, régression linéaire/logistique, SVM, KNN, Naive Bayes, + SMOTE, + clustering) comparé automatiquement |
| **6-8** | Vision par ordinateur / détection d'anomalies (l'autre grand pilier de l'app historique, pas encore porté) |
| **9** | Registre de modèles versionné (l'artefact existe déjà, pas encore le versioning/export) |
| **10** | Durcissement SaaS : audit, quotas, facturation — prêt pour un client pilote |

---

## Démarrer en local

Voir [`README.md`](README.md) pour les commandes complètes. En résumé :
backend (`uvicorn`), worker (`python -m workers.run_worker`), Redis (Docker),
frontend (`npm run dev`) — les quatre doivent tourner pour un entraînement
de bout en bout ; les trois premiers suffisent pour tout le reste.
