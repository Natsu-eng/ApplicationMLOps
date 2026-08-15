# AUDIT_PILIER2_ET_REFONTE_UX.md — Pilier non supervisé + refonte visuelle (2026-08-15)

> Audit lecture seule, aucun fichier de code modifié. Complète `AUDIT_ROADMAP.md`
> (daté du 14/15 août, qui couvrait surtout le ML supervisé + un premier passage
> UX) sur deux points qu'il ne couvrait pas encore en détail : le code réel des
> Lots 11-14 (clustering, réduction de dimension, détection d'anomalies) livrés
> après/pendant cet audit, et une proposition concrète de nouvelle maquette.
> Méthode : deux audits délégués et croisés (backend non supervisé, état des
> lieux frontend page par page) + vérification directe de `index.css`,
> `PageHeader.tsx`, `ColorIconBadge.tsx`, `AppShell.tsx`, `HelpModal.tsx`.
> `git log`/`git status` vérifiés en préalable — le dépôt est propre, HEAD =
> `fab4cd6` (Lot 14), cohérent avec `recap.md`/`workflow.md`.

---

## A. Audit backend — pilier ML non supervisé (Lots 11-14)

### Points forts vérifiés (par lecture de code, pas par confiance dans la doc)

| Module | Point vérifié | Preuve |
|---|---|---|
| Clustering | Silhouette/Davies-Bouldin/Calinski-Harabasz calculées en excluant le bruit DBSCAN | `clustering_training.py::_compute_cluster_metrics` L81-109 |
| Clustering | `eps` DBSCAN résolu depuis la distribution réelle des distances (percentile 90 de la k-distance), jamais une constante | `resolve_dbscan_eps` L125-135 |
| Clustering | Sélection du k par comparaison de plusieurs configurations (2 à 8) × plusieurs algorithmes | `CANDIDATE_K_VALUES`, `train_and_evaluate_clustering` L192-256 |
| Clustering | Profils de segments = uniquement des statistiques calculées, jamais un texte halluciné | `_build_cluster_profiles` L112-162 |
| Réduction de dimension | Variance expliquée/loadings réservés à PCA, jamais présentés pour t-SNE/UMAP | `dimensionality_training.py` L118-139 |
| Réduction de dimension | `trustworthiness` bornée [0,1], testée pour les 3 algorithmes | `test_trustworthiness_always_in_unit_interval` |
| Réduction de dimension | Note de non-fidélité des distances renvoyée par l'API (pas seulement supposée côté frontend) | `DimensionalityResultOut.distance_fidelity_note` |
| Anomalies | Isolation Forest + LOF toujours exécutés ensemble, aucun chemin ne permet un seul | `train_and_evaluate_anomalies` L147-156 |
| Anomalies | Score de consensus = moyenne des **rangs percentiles**, pas une moyenne brute sur échelles différentes | `anomaly_training.py` L159-164 |
| Anomalies | Explication par variable réellement calculée (z-score, rareté catégorielle) | `_build_numeric_deviations`/`_build_categorical_flags` L89-109 |
| Infra partagée | `job_quota.py` compte bien les 4 types de job (Training/Clustering/Dimensionality/Anomaly) dans les 4 routers | grep confirmé, testé par `test_quota_shared_across_all_four_job_types` |
| Infra partagée | Les 4 workers suivent un pattern quasi identique (session DB dédiée, `progress_updated_at`, distinction erreur métier/technique) | comparaison directe des 4 fichiers |

**Conclusion générale** : les 3 nouveaux modules respectent la méthodologie annoncée dans `recap.md` — aucune fuite trouvée, aucune régression sur le pattern "module séparé, jamais une branche de plus dans `ml_training.py`". La cohérence architecturale inter-jobs est réelle, pas seulement documentée.

### Problèmes trouvés

**Important**

- **P1 — Journal d'audit (`log_action`) absent des 3 routers non supervisés.** `training.py` trace `training_job.deleted` à la suppression (L906-910) ; `clustering.py`, `dimensionality.py`, `anomalies.py` ne l'appellent jamais (grep confirmé : `log_action` n'existe que dans `datasets.py`/`auth.py`/`training.py`). **Exemple concret** : un owner ouvre le journal d'audit (Lot 10, "qui a supprimé quoi") pour retrouver qui a supprimé une détection d'anomalies sensible — l'action est invisible, alors que supprimer un entraînement supervisé équivalent serait tracé. Le Lot 10 n'a jamais été étendu aux 3 nouveaux types de job.
- **P2 — Sélection du "gagnant" clustering ignore le taux de bruit DBSCAN.** `clustering_training.py` L242-256 trie uniquement sur la silhouette (`valid.sort(key=lambda c: c["silhouette"], reverse=True)`), le `noise_ratio` n'intervient jamais comme filtre. **Exemple concret** : un DBSCAN qui classe 95 % des points en bruit et ne structure que 5 % du dataset en 2-3 clusters très compacts peut afficher une silhouette de 0.9 (calculée sur les seuls points core) et battre un K-Means qui structure honnêtement 100 % des données avec une silhouette de 0.5 — le "gagnant" persisté serait alors quasi inutilisable en pratique. Aucun test ne couvre ce cas (tests existants = blobs bien séparés, 0 % de bruit).

**Amélioration**

- **P3** — `job_watchdog.py` : commentaires/`TypeVar` encore rédigés pour 2 types de job sur 4 (fonctionne à l'exécution car Python n'impose pas les bornes du `TypeVar`, mais aucun test n'exerce la réconciliation d'un `ClusteringJob`/`DimensionalityJob`/`AnomalyJob` orphelin — seul `TrainingJob` l'est).
- **P4** — Champs `score_isolation_forest`/`score_lof` (API + DB) contiennent en réalité des **rangs percentiles**, pas les scores bruts des modèles — nom trompeur pour un futur consommateur de l'API qui lirait uniquement le schéma OpenAPI.
- **P5** — `AlgorithmCatalogEntry`/`AlgorithmCatalogResponse` dupliqués à l'identique entre `clustering.py` et `dimensionality.py`, sans la justification écrite qui existe pour les autres duplications assumées du projet (ex. `_user_safe_error_message` dans les 4 workers, justifiée en commentaire).

---

## B. État des lieux frontend — vérifié page par page

### Constat central : D1 (tokens sémantiques) est réellement corrigé aujourd'hui

Comptage par grep sur tout `src/` : les classes Tailwind en dur (`slate-`/`rose-`/`emerald-`...) ne subsistent que dans **5 fichiers**, contre une utilisation systématique des tokens sémantiques ailleurs (Training.tsx 54 occurrences de tokens, Clustering.tsx 42, Dashboard.tsx 40...). `ColorIconBadge.tsx` (20 occurrences) n'est pas un résidu — c'est un système d'accent multicolore délibéré et documenté. Les H-items du roadmap (H4, H9, H10, H14, H15, H16, H18, H20) sont **tous confirmés vrais** par lecture directe (`npm run lint` rejoué : 0 erreur/11 avertissements, identique au chiffre documenté).

**Le vrai problème n'est donc plus la dette de tokens — c'est un choix de design déjà assumé dans le code lui-même.**

### La cause du "tout est blanc" est documentée dans `index.css` — c'est une décision, pas un oubli

```css
/* index.css:46-55 */
/* REVU (retour utilisateur direct sur captures, 2026-08-14) : la version
   bleu-gris appuyée précédente (0.94 de luminosité) lisait comme
   "fade"/terne — retour à un quasi-blanc (0.98)... la séparation
   canevas/carte se fait désormais par l'OMBRE plutôt que par un écart
   de couleur de fond */
--color-background: oklch(0.98 0.006 258);   /* fond app */
--color-card: oklch(1 0 0);                   /* carte = blanc pur */
```

Autrement dit : un aller-retour précédent est déjà passé d'un fond teinté à un fond quasi-blanc, sur retour utilisateur — et c'est justement ce résultat qui fatigue les yeux aujourd'hui. Seules zones de couleur réelles hors sidebar : le bandeau `PageHeader` (lavis ~5-10 %) et les liserés de carte. `bg-white` en dur subsiste ponctuellement dans 11 endroits (EdaModal, ExpertModePanel, FeatureEngineeringSuggestions, Switch, AuthBrandPanel...).

### Le logo : confirmé bricolé, pas conçu comme une icône

Il n'existe **aucun asset icône dédié**. Le seul fichier (`public/logo.png`, image composite 1254×1254 contenant le pictogramme + le wordmark + la baseline "Explorer·Analyser·Prédire") est recadré à la volée par deux mécaniques CSS différentes selon l'endroit :

```css
/* index.css:138-145 — écran auth */
.auth-logo-crop {
  width: 168px; height: 128px;
  background-size: 224px 224px;
  background-position: -25px -7px;   /* recadrage magique par coordonnées */
}
```
```tsx
{/* AppShell.tsx:37-43 — sidebar, mécanique DIFFÉRENTE pour le même usage */}
<img src="/logo.png" className="h-[49px] w-[49px] max-w-none -translate-x-[5.5px] -translate-y-[1.5px]" />
```

Deux techniques de recadrage différentes pour extraire la même icône de la même image, à coups de coordonnées en pixels ajustées à la main — c'est exactement le symptôme d'un logo qui n'a jamais existé comme icône autonome. Le résultat visuel (icône légèrement floue/décentrée selon le zoom navigateur, absente de `favicon`) confirme le ressenti utilisateur ("le logo aussi pas bien fait").

### Divergences trouvées (dette réelle, pas de la théorie)

| # | Fichier | Problème |
|---|---|---|
| 1 | `pages/Training.tsx:64,138-139` | Confirmation de suppression réimplémentée en state local, alors que `useConfirmAction` prétend (JSDoc) avoir été extrait de ce même fichier |
| 2 | `pages/ComingSoon.tsx:31-53` | Code mort : texte marketing "à venir" pour le pilier non supervisé, actif depuis le Lot 11, cette branche n'est plus jamais routée |
| 3 | `EdaModal.tsx:517`, `FeatureEngineeringSuggestions.tsx:193` | `<select>` bruts avec `bg-white` en dur au lieu du composant `Select` |
| 4 | `Clustering.tsx:457`, `DimensionalityReduction.tsx:249,465,476`, `AnomalyDetection.tsx:447` | `.catch(() => setXxx([]))` silencieux — exactement le pattern D3 dénoncé ailleurs, reproduit sur le code le plus récent |
| 5 | `TrainingHistory.tsx:262-265` | Seul résidu de couleur en dur (`amber-*`) hors composants déjà identifiés |
| 6 | Aucune route `/profile` ni `/settings` | Gestion d'équipe encore mélangée au Dashboard (`Dashboard.tsx:366-462`) — H19 toujours reporté |
| 7 | `HelpModal.tsx` | Ne documente que le parcours ML supervisé en 5 étapes — **aucune mention du pilier non supervisé**, actif depuis le Lot 11 |

---

## C. Recommandations de différenciation produit

Positionnement : bureaux d'études (pas des équipes MLOps internes), utilisateurs non-data-scientists **et** experts qui veulent creuser. Comparé aux plateformes généralistes (DataRobot, Azure ML, Vertex AI, H2O.ai — conçues pour des équipes data internes) :

1. **Rapport livrable client, pas seulement un dashboard.** Un bureau d'études doit remettre un document à SON client, pas juste consulter un écran. Aucune plateforme généraliste ne fait ça bien — elles exportent des artefacts techniques (modèle, notebook), jamais un document de synthèse présentable. *Exemple concret* : un bouton "Générer le rapport" sur `ModelResultView`/résultat de clustering/anomalies, qui assemble automatiquement ce qui existe déjà (phrase de sélection du leaderboard, profils de segments, variables SHAP dominantes, graphiques déjà calculés) en un PDF avec en-tête du bureau d'études — zéro nouveau calcul, uniquement de la mise en forme de données déjà produites.
2. **Traçabilité méthodologique comme argument de confiance visible, pas juste un fait interne.** Le projet a une vraie discipline (jamais de texte inventé sur un cluster, score de consensus jamais un "gagnant" arbitraire en anomalies, note de non-fidélité systématique sur t-SNE/UMAP) — mais rien ne le dit à l'écran. *Exemple concret* : un badge discret "Analyse basée uniquement sur vos données — aucun texte généré par IA" sur les blocs d'interprétation (profils de cluster, explication SHAP, résumé de leaderboard), qui devient un argument commercial différenciant face à des outils low-code qui habillent des statistiques de texte IA non vérifiable.
3. **Lecture croisée entre piliers, qu'aucun concurrent généraliste ne propose.** Le lien croisé clustering → réduction de dimension existe déjà (Lot 13, query params, sans couplage backend). *Exemple concret à étendre* : depuis un résultat d'anomalies, un lien "Voir ces observations dans le contexte des clusters" pré-remplit la page Clustering sur le même dataset — sans coupler les backends (même principe déjà validé), avec un intérêt réel pour un ingénieur BE qui veut savoir si les anomalies détectées appartiennent à un segment particulier plutôt qu'un point isolé.
4. **Étiquette client/mission légère sur les datasets, sans construire un Workspace complet.** H22 (Workspace/Projet) est à raison non traité — trop lourd pour le besoin actuel. Mais un bureau d'études gère plusieurs missions en parallèle dans la même organisation. *Exemple concret* : un simple champ `client_label` optionnel sur `Dataset` (une chaîne, pas une nouvelle entité), affiché en badge sur les cartes dataset et filtrable dans "Mes données" — répond à 80 % du besoin de cloisonnement visuel sans la complexité RBAC d'un vrai Workspace.
5. **Mode "explication" permanent plutôt que des tooltips isolés.** Les info-bulles existantes (R², SHAP, CQR...) sont un bon début mais restent dispersées. *Exemple concret* : un glossaire accessible depuis la page Aide, qui liste tous les termes déjà expliqués en tooltip à travers l'app en un seul endroit consultable — zéro nouveau contenu à rédiger, seulement une réorganisation des textes déjà écrits dans `Tooltip`/`LabelWithHelp`.

---

## D. Proposition de nouvelle maquette

**Principe** : réutiliser le design system existant (tokens OKLCH, `PageHeader`, `ColorIconBadge`, `Table`, `Card`, `Badge`...) — pas le réinventer. Les changements ci-dessous touchent des **valeurs de tokens** et des **compositions de pages**, jamais une nouvelle bibliothèque de composants.

### D1 — Palette : sortir du "tout blanc" sans repeindre l'app

Le problème n'est pas le choix du bleu de marque (cohérent, bien exécuté sur la sidebar) — c'est l'absence de **second niveau de profondeur** entre le fond quasi-blanc et les cartes blanches. Proposition, dans `index.css`, en gardant la même teinte (258) partout pour rester cohérent avec la marque :

| Token | Valeur actuelle | Valeur proposée | Effet |
|---|---|---|---|
| `--color-background` | `oklch(0.98 0.006 258)` (quasi-blanc) | `oklch(0.955 0.012 258)` | Fond de page légèrement plus présent — la carte blanche se détache par la couleur **et** l'ombre, pas l'ombre seule |
| `--color-card` | `oklch(1 0 0)` (blanc pur) | inchangé | La carte reste la valeur la plus claire de l'écran — c'est ce contraste qui doit porter la hiérarchie |
| *(nouveau)* `--color-canvas-alt` | — | `oklch(0.92 0.02 258)` | Fond de section secondaire (ex. zone de filtres, aside), un cran sous le canvas principal |

Ce n'est **pas** un retour à la version "bleu-gris terne" déjà essayée et rejetée (0.94 sans chroma suffisante, lue comme fade) — la nuance proposée garde une chroma légèrement supérieure (0.012 vs 0.006) pour rester perceptible comme "teinté" plutôt que "gris terne", tout en restant nettement en dessous de la carte. À valider sur capture réelle avant d'aller plus loin, comme les itérations précédentes du projet.

**Mode sombre en option** (pas par défaut) : les tokens sont déjà en OKLCH, prêts pour une bascule — *exemple concret* : un interrupteur dans la future page Profil (`prefers-color-scheme` + override utilisateur persisté), qui redéfinit uniquement `--color-background`/`--color-card`/`--color-foreground`/`--color-border` en gardant `--color-primary`/`--color-sidebar` identiques. Un utilisateur qui passe des heures sur des tableaux de résultats (le cas d'usage explicitement cité comme fatigant) choisit lui-même. Pas un chantier du même lot que D1 — à cadrer séparément une fois D1 validé.

### D2 — Logo : remplacer le recadrage bricolé par une vraie icône

*Exemple concret* : extraire une fois pour toutes un SVG carré (viewBox 64×64 ou 128×128) du seul pictogramme "D", sans le wordmark ni la baseline — utilisable tel quel en `<img src="/icon.svg">` partout (sidebar, favicon, écran auth), sans `background-position`/`translate` en pixels magiques. Le wordmark ("DataLab Pro") redevient du texte HTML à côté (déjà le cas dans `AppShell.tsx:44-47` et probablement `AuthBrandPanel`), jamais réintégré dans l'image. Bénéfice direct : un vrai favicon net (actuellement absent/dérivé du même hack), et un seul asset à maintenir au lieu de deux mécaniques de recadrage divergentes.

### D3 — Nouvelle page Profil & Paramètres (`/profile`)

Actuellement absente (confirmé : aucune route dans `App.tsx`) — le profil personnel et l'administration d'équipe sont mélangés sur le Dashboard (`Dashboard.tsx:366-462`), ce qui correspond exactement au H19 déjà identifié comme reporté dans `AUDIT_ROADMAP.md`.

Structure proposée, en réutilisant `PageHeader` + `Tabs` (déjà utilisé dans `EdaModal`/`ModelResultModal`) :

```
PageHeader (icône User, couleur "blue")
  Tabs : [Profil] [Organisation & équipe] [Préférences]

  Profil            → nom, email, changement de mot de passe
                      (déplace ce qui existe déjà dans Dashboard,
                      pas de nouveau champ)
  Organisation      → membres de l'équipe, ajout de membre, rôle
                      (déplace le bloc équipe hors du Dashboard —
                      libère le Dashboard pour être 100% activité ML)
  Préférences       → bascule thème clair/sombre (D1), rien d'autre
                      pour ce lot — pas de sur-ingénierie
```

Lien d'accès : avatar en pied de sidebar (`AppShell.tsx:114-130`, actuellement juste nom + bouton déconnexion) devient cliquable vers `/profile`, pattern standard SaaS.

### D4 — Page Aide : couvrir le pilier non supervisé

`HelpModal.tsx` documente aujourd'hui uniquement les 5 étapes du ML supervisé — silence total sur clustering/réduction de dimension/anomalies, actifs depuis le Lot 11. *Exemple concret de correctif* : passer `HelpModal` d'une grille unique à deux sections consécutives (même composant `Card`/`ColorIconBadge`, aucun nouveau composant) :

```
"Le parcours du ML supervisé" — les 5 étapes existantes, inchangées
"Le parcours du ML non supervisé" — nouvelle section, 3 cartes :
  1. Découvrir des groupes (Clustering) — dataset → segments → profils
  2. Visualiser en 2D (Réduction de dimension) — PCA/t-SNE/UMAP,
     rappel explicite que ce ne sont pas des distances réelles
  3. Repérer les atypiques (Détection d'anomalies) — score de
     consensus, jamais un seul algorithme
```

### D5 — Dashboard : séparer activité ML et administration (H19, concrétisé par D3)

Une fois D3 livré, le bloc "Équipe — DataLab Pro" (visible dans la capture fournie, sous les listes d'activité) quitte le Dashboard. Le Dashboard redevient uniquement : tuiles statistiques (inchangées), derniers entraînements/datasets (inchangés) — page plus courte, plus rapide à scanner, cohérent avec "Dashboard = activité, Profil = administration".

### D6 — Densité des pages de résultats (constat existant D8/H12, non rouvert ici)

Non repris en détail dans ce document — le roadmap l'a déjà correctement priorisé en "reporté, chantier dédié" (H12). Signalé seulement pour mémoire : toute nouvelle maquette de `ModelResultModal`/`Training.tsx` devrait inclure ce découpage en sous-composants **au moment** de la refonte visuelle, pour ne pas retoucher deux fois le même fichier à quelques semaines d'intervalle.

---

## E. Roadmap priorisée

### 🔴 Critique

*Aucun point critique trouvé dans le pilier non supervisé ni dans l'état des lieux frontend — le socle est sain.* Les points ci-dessous sont classés Important/Amélioration en conséquence, pas de faux Critique pour gonfler la liste.

### 🟠 Important

- [ ] **P1** — Étendre `log_action` (journal d'audit) aux suppressions des 3 types de job non supervisés, même pattern que `training.py:906-910`.
- [ ] **P2** — Exclure ou pénaliser les configurations DBSCAN à `noise_ratio` élevé (ex. > 50 %) du classement clustering, pas seulement les afficher a posteriori — éviter d'élire un "gagnant" dégénéré sur 5 % du dataset.
- [ ] **D4** — Mettre à jour `HelpModal.tsx` pour couvrir le pilier non supervisé (section 2, voir D4) — écart visible immédiatement par tout utilisateur qui ouvre l'Aide depuis Clustering/DimensionalityReduction/AnomalyDetection.
- [ ] **D3** — Nouvelle page `/profile` (Profil, Organisation & équipe, Préférences) — sépare Dashboard et administration (H19).
- [ ] **D1** — Ajustement de `--color-background` (voir tableau D1) — à valider sur capture réelle avant généralisation, comme les itérations précédentes du fond de page.
- [ ] **D2** — Icône logo dédiée (SVG autonome) — supprime les deux mécaniques de recadrage divergentes, corrige l'absence de favicon net.

### 🟢 Amélioration

- [ ] **P3** — Ajouter un test de réconciliation watchdog pour au moins un des 3 nouveaux types de job (`ClusteringJob` par exemple), pas seulement `TrainingJob`.
- [ ] **P4** — Renommer ou documenter clairement `score_isolation_forest`/`score_lof` comme rangs percentiles, pas scores bruts.
- [ ] **P5** — Factoriser `AlgorithmCatalogEntry`/`AlgorithmCatalogResponse` (dupliqués entre clustering/dimensionality) dans un schéma partagé.
- [ ] Divergences frontend #1 à #5 (tableau section B) — corrections ponctuelles, faible risque, à regrouper dans un petit lot de nettoyage.
- [ ] **C4** — Champ `client_label` optionnel sur `Dataset` (différenciation, section C).
- [ ] **C1/C2/C3** — Rapport PDF exportable, badge de confiance méthodologique, lien croisé anomalies → clustering (différenciation produit, section C) — chacun un lot dédié, pas à bâcler ensemble.
- [ ] **D5/D6** — Dashboard allégé une fois D3 livré ; découpage `ModelResultModal`/`Training.tsx` à faire **au moment** de la prochaine refonte visuelle de ces deux écrans, pas séparément.

---

## F. Recommandation finale

Le pilier non supervisé est arrivé au niveau de rigueur du pilier supervisé — aucun problème critique, deux trous de gouvernance ciblés (P1, P2) faciles à corriger sans toucher à l'architecture. **Le vrai chantier attendu par l'utilisateur est D (refonte visuelle)**, pas un audit correctif de plus : le design system est déjà solide (tokens respectés partout, composants réutilisables riches) — ce qui manque, ce n'est pas de la dette à rattraper, c'est un choix de densité de couleur à assumer (D1), un asset logo à refaire proprement (D2), et deux pages à ajouter (D3 Profil, D4 Aide mise à jour).

Séquencement proposé pour l'implémentation à venir (une fois validé) :
1. D2 (logo) — isolé, aucun risque de régression ailleurs.
2. D1 (palette) — un seul fichier (`index.css`), itératif sur capture comme les fois précédentes.
3. D4 (Aide) — un seul composant, aucun nouveau backend.
4. D3 (page Profil) — nouvelle route + déplacement du bloc équipe existant.
5. P1/P2 (backend, Important) — en parallèle, indépendant du frontend.
