# Rapport final — Refonte visuelle DataLab Pro (Lots 1 à 11)

Ce rapport clôt la mission de refonte visuelle en 11 lots. Il résume ce qui a
été livré, ce qui a été trouvé et corrigé en cours de route, et — point le
plus important pour la transparence — ce qui a été **délibérément laissé de
côté** parce que ce n'était pas un simple reskin mais une fonctionnalité
backend qui n'existe pas encore. Chaque affirmation ci-dessous est vérifiable
dans `_design/JOURNAL.md`, qui contient le détail lot par lot avec les vraies
sorties de commande.

## Méthodologie appliquée

- Un git branch par lot (`ui/1-fondations` → `ui/11-verification-finale`),
  fusionné dans `main` après une porte de qualité non négociable : `tsc`/
  `build` propres, `lint` propre, grep de couleurs en dur propre, les 5
  thèmes vérifiés par de vraies captures Playwright, navigation clavier +
  axe-core complètes sans violation sérieuse/critique.
- Zéro donnée inventée : chaque écart entre une maquette et l'implémentation
  a été vérifié contre le vrai backend (`services/engine.py`, les modèles
  SQLAlchemy, les routes réelles) avant de décider de construire, ou de
  documenter l'écart comme hors périmètre.
- Aucune pause pour validation intermédiaire, sauf les 3 déclencheurs
  explicites de la mission (migration destructive, nouvelle dépendance
  &gt;100 Ko, contradiction de spec non résoluble) — jamais atteints.

## Bilan par lot

1. **Fondations** — 5 thèmes calculés (`_design/tune.py`, contraste WCAG réel
   par construction), nettoyage des couleurs en dur, `Modal.tsx` corrigé
   (piégeait le focus clavier derrière la barre latérale).
2. **Bibliothèque de composants** — `Badge`, `Card`, `Table`, `Select`,
   `ColorIconBadge` etc. extraits et harmonisés.
3. **Graphiques** — palette de séries calculée (`--s1…--s6`, distinguable en
   deutéranopie simulée), `charts.ts`/`Heatmap.tsx` migrés.
4. **Entrer dans le produit** — Login/Register/Onboarding sur le nouveau
   matériau visuel.
5. **Données et qualité** — `EdaModal`, avertissements de qualité de données,
   bug clavier corrigé (référait au correctif de la décision 13).
6. **Supervisé** — Entraînement/Progression/Verdict/Leaderboard : preuves
   réelles du Verdict enfin affichées (`claim.details`, jamais exposé avant),
   comparaison de modèles, journal de progression en direct (SSE).
7. **Non supervisé** — Clustering/Anomalies/Projection : exploration de
   seuil réelle sur l'histogramme de scores, qualité/stabilité du clustering
   expliquées en langage clair.
8. **Vision** — légende Grad-CAM en dégradé fixe (`jet`) documentée comme
   exception délibérée (jamais un token de thème, car la carte de chaleur
   sous-jacente est coloriée serveur avec cette même palette fixe).
9. **Exploiter et tracer** — carte « Environnement d'entraînement »
   (versions des bibliothèques ML au moment de l'entraînement, déjà calculée
   côté serveur, jamais affichée avant).
10. **Aide** — page `/aide` complète (guide par pilier en onglets repliables,
    FAQ/glossaire en accordéon, vrai formulaire de signalement), export de
    modèle + configuration JSON étendu aux 5 domaines non supervisés/vision,
    menu du tableau de bord corrigé (`createPortal`), et un bug critique
    d'inscription/connexion trouvé et corrigé le jour même.
11. **Vérification finale** — voir ci-dessous.

## Lot 11 — ce qui a réellement été vérifié

- **20 écrans × 5 thèmes** capturés (`frontend/scripts/lot11-all-screens.mjs`)
  avec axe-core (`wcag2a`/`wcag2aa`/`wcag21aa`) sur chacun : **0 violation
  sérieuse/critique**, sur les 100 combinaisons, après correctifs (voir
  bugs ci-dessous).
- **Navigation clavier** sur les 20 écrans (`lot11-keyboard-all-screens.mjs`) :
  25 pressions de Tab par écran, comparaison de l'élément focus par identité
  DOM réelle (pas une comparaison de chaîne, qui donnait de faux positifs à
  la première tentative) — **aucun focus perdu, aucune boucle bloquée**, sur
  les 20 écrans.
- **Responsive** à 1280/1440/1920px sur les 20 écrans
  (`lot11-responsive.mjs`, détection de débordement horizontal réel via
  `scrollWidth`) : 1 bug réel trouvé et corrigé (voir ci-dessous), 0 après
  correctif.
- **Contraste programmatique en CI** (`frontend/scripts/check-contrast.mjs`,
  ajouté à `.github/workflows/ci.yml`) : calcule le vrai contraste WCAG à
  partir des couleurs et opacités extraites des fichiers source (pas de
  valeurs recopiées à la main), pour les combinaisons `text-muted-foreground`
  sur `accentSurfaceClass` et `Badge` (warning/danger) sur surface nue, sur
  les 5 thèmes. Portée assumée : ce script couvre le cas à un seul lavis ; le
  cas plus rare de double lavis (composant coloré niché dans un conteneur
  déjà teinté) reste couvert par les scripts axe-core Playwright, pas par ce
  script rapide sans navigateur.

### Bugs réels trouvés et corrigés pendant ce lot

1. **Contraste texte insuffisant, 11 violations axe réelles au départ** — sur
   les vues de résultat retabulées au Lot 10 (Clustering, Anomalies,
   Réduction de dimension, Vision × 2, Verdict d'entraînement), en ivoire et
   porcelaine (les 2 thèmes clairs, calibrés au minimum légal 4,52-4,55:1).
   Deux causes racines réelles, distinctes :
   - Les valeurs numériques des tuiles de métrique (`text-xl font-semibold`)
     n'atteignaient PAS le seuil `bold` (600 &lt; 700) requis par WCAG pour
     bénéficier de l'exception « texte large » (3:1 au lieu de 4,5:1) —
     corrigé en passant à `font-bold` sur les 6 occurrences réelles
     (Clustering, Anomalies, Réduction de dimension, Vision × 2,
     `ModelResultModal`).
   - Les couleurs catégorielles de série (`--s1…--s4`, alias `accent-1…4`)
     étaient réutilisées comme texte de badge à 12-14px (jamais assez grand
     pour l'exception « texte large », quel que soit le poids) et l'opacité
     de fond des cartes teintées (`accentSurfaceClass`, `Badge` warning/
     danger) était calibrée pour un seul lavis mais retombait sous le seuil
     une fois composée avec un second fond déjà teinté (ligne de tableau
     mise en évidence, carte de teinte imbriquée) — mesuré réellement par
     axe-core à 4,35-4,49:1 selon les cas. Corrigé en réduisant l'opacité
     (`/4` → `/2` ou `/3` selon les jetons, `ColorIconBadge.tsx`/`Badge.tsx`)
     et en repassant les libellés de badge de qualité/stabilité en
     `text-foreground` neutre (l'identité de couleur restant portée par
     l'icône adjacente). **Résultat final : 0 violation restante** sur les 5
     thèmes après correctif, revérifié par axe-core en conditions réelles à
     chaque itération (pas seulement calculé à la main).
2. **2 vrais bugs d'accessibilité structurelle**, trouvés par le passage
   systématique sur les 20 écrans (jamais testés ensemble avant) :
   - 5 `&lt;select&gt;` de filtre sans nom accessible (`AllHistory.tsx` ×4,
     `TrainingHistory.tsx` ×1, plus 3 de plus trouvés par inspection dans
     `EdaModal.tsx` pendant l'investigation, bien que non capturés par le
     scan automatique puisque c'est une modale) — corrigé par
     `aria-label` sur chacun.
   - 1 lien inline distinguable UNIQUEMENT par la couleur
     (`Training.tsx`, « tableau de bord ») — corrigé en ajoutant
     `underline underline-offset-2`, motif déjà établi ailleurs
     (`Dashboard.tsx`, `Aide.tsx`).
3. **1 vrai bug de mise en page responsive** — `historique` et
   `training-history` débordaient horizontalement à 1280/1440px. Diagnostic
   en 3 temps, le premier plus long que prévu : la piste initiale (la grille
   de filtres à 5 colonnes ne tiendrait pas dans l'espace disponible) était
   plausible mais fausse — vérifié en mesurant directement chaque élément
   via `getBoundingClientRect()` dans un vrai navigateur plutôt que de
   raisonner à l'aveugle, ce qui a montré que `document.body.scrollWidth`
   restait strictement identique avant/après avoir changé le point de
   rupture de la grille, prouvant que la grille n'était pas la cause. Deux
   ajouts de `min-w-0` mal placés (sur `&lt;main&gt;`, puis sur `Select.tsx`)
   n'ont eu aucun effet mesurable non plus, pour la même raison. La vraie
   cause : le conteneur flex racine d'`AppShell.tsx`
   (`flex min-h-screen flex-1 flex-col lg:pl-[292px]`) est le véritable
   élément flex dont le `min-width` par défaut (`auto`) laissait son
   contenu le pousser au-delà du viewport, alors que la barre latérale
   fixe réserve son espace par un simple `padding-left` (elle est en
   `position: fixed`, donc hors du flux flex) — un tableau large en
   dessous forçait ce conteneur, pas `&lt;main&gt;`, à s'élargir. Corrigé en
   ajoutant `min-w-0` à ce seul conteneur ; revérifié directement
   (`document.body.scrollWidth === document.documentElement.clientWidth`
   exactement) avant de relancer la vérification complète des 60
   combinaisons écran × largeur : **0 débordement restant**.

### Nettoyage du dépôt (hors refonte visuelle, demandé explicitement)

144 fichiers suivis par git ont été supprimés : l'intégralité de l'ancienne
application Streamlit/computer-vision qui précédait la migration vers
DataLab Pro (`helpers/`, `monitoring/`, `notebooks/`, `orchestrators/`,
`pipeline_visio/`, `scripts/` et `tests/` à la racine, `src/`, `ui/`,
`utils/`, plus `diagnostic_pipeline.py`, `logging_patch.py`,
`test_augmentation_fix.py`, `requirements.txt` et `pytest.ini` à la racine).
Vérifié avant suppression : aucune référence depuis `backend/`, `frontend/`,
`docker-compose.yml`, les deux `Dockerfile`, ni `.github/workflows/ci.yml`.
Un environnement virtuel Python égaré (`env/`, déjà exclu par
`.gitignore`) a aussi été supprimé. `_design/` et tous les fichiers `.md` du
dépôt ont été conservés tels quels.

## Ce qui a été délibérément laissé de côté (honnêteté du périmètre)

Chaque point ci-dessous a été vérifié contre le vrai backend avant d'être
classé hors périmètre — jamais une supposition. Référence à la décision
numérotée dans `_design/JOURNAL.md` entre parenthèses.

- **Détections statistiques « valeurs aberrantes à 3σ » et « comparaison R²
  par groupe »** (Qualité des données) — nouveaux contrôles qui n'existent
  pas dans `data_quality.py`, un travail de science des données, pas de
  refonte visuelle (décision 39).
- **Carte 2D des groupes, export CSV étiqueté, renommage** (Clustering) —
  aucune coordonnée 2D par observation calculée par le job de clustering
  lui-même (c'est une capacité séparée, la réduction de dimension) ; aucune
  route d'export existante (décision 50).
- **Tableau détaillé PC1-PC4, détection de capteurs redondants, cerclage des
  points isolés** (Réduction de dimension) — la PCA de référence ne calcule
  que 2 composantes ; aucune donnée de corrélation inter-capteurs ni de
  score d'isolement par point n'existe côté backend (décision 51).
- **Export de lignes/ré-analyse en chaîne** (Projection, Anomalies) —
  aucune route backend correspondante (décision 52).
- **Indices « vu de X à Y » par variable et alerte d'extrapolation**
  (Prédire) — le schéma de formulaire de prédiction ne porte aucune borne
  observée à l'entraînement (décision 57).
- **Écran « Registre » complet** (historique multi-versions, alerte de
  dérive, journal de décisions en langage libre) — trois chantiers
  fonctionnels distincts, chacun plus gros que l'ensemble d'un lot visuel
  (décision 56).
- **Rééquilibrage des classes : SMOTE en option** — le pondérage de classes
  actuel est un défaut sûr et déjà correct (tous les modèles du portefeuille
  le supportent nativement, zéro risque de fuite) ; SMOTE exigerait SMOTENC
  strictement replié dans chaque pli de validation croisée pour éviter une
  fuite de données — un vrai chantier de « mode expert », pas un
  interrupteur de fin de lot (décision 66).
- **Prévisualisation avant/après de l'augmentation** (ML et Vision) —
  aucun échantillon avant/après n'est retourné par le backend actuel ;
  exigerait un nouvel endpoint appliquant la transformation à quelques
  lignes/images réelles (décision 67).
- **Légende de couleur fixe du Grad-CAM** — ce n'est PAS un écart laissé de
  côté mais une exception délibérée et documentée : la carte de chaleur
  affichée est coloriée côté serveur avec une palette fixe, indépendante du
  thème ; remplacer la légende par des jetons de thème la rendrait fausse
  par rapport à l'image réellement affichée (décision 53).

## Résultats de vérification finaux (commandes réelles)

- **Frontend** — `npx tsc --noEmit` : 0 erreur. `npm run build` : succès,
  `1 143,86 Ko` JS (gzip 319,65 Ko) / `79,97 Ko` CSS (gzip 12,62 Ko).
  `npm run lint` : `0 erreur, 18 avertissements` (ligne de base identique
  depuis le Lot 1, tous pré-existants). `npm run test` (Vitest) :
  **64/64 tests passent** (11 fichiers). `node scripts/check-contrast.mjs` :
  toutes les combinaisons vérifiées ≥ 4,5:1 sur les 5 thèmes.
- **Backend** — aucun fichier applicatif backend modifié par ce lot (seul
  `.github/workflows/ci.yml` a changé, pour ajouter l'étape de contraste
  programmatique côté frontend). Une exécution locale de `pytest` a été
  lancée par honnêteté de vérification, mais arrêtée à 27 % de progression :
  Docker Desktop n'est pas démarré sur cette machine, donc le Redis local
  (`datalab_redis`) est injoignable, et les tests touchant le rate-limiting
  ralentissent fortement en tentant de s'y connecter (fail-open réel mais
  lent) — un vrai gap d'environnement local, pas un bug de code, et sans
  rapport avec les changements de ce lot. La CI réelle (`.github/workflows/
  ci.yml`) provisionne son propre conteneur Redis pour le job backend et
  n'est pas affectée par ce gap local.
- **Accessibilité** — axe-core sur 20 écrans × 5 thèmes : **0 violation
  sérieuse/critique**. Navigation clavier sur 20 écrans : **0 focus perdu**.
- **Responsive** — 1280/1440/1920px sur 20 écrans : **0 débordement
  horizontal** après correctif.
- **Couleurs en dur** — grep : 2 occurrences Vision (exception documentée,
  Lot 8) + 4 dans un fichier de test (assertions, pas du rendu) — ligne de
  base inchangée depuis le Lot 1.

## Recommandations pour la suite (non traitées dans cette mission)

1. Avant tout déploiement en production : implémenter le journal de
   décisions et les alertes de dérive du Registre (décision 56) si le
   produit doit réellement servir de plateforme MLOps de confiance.
2. Évaluer sérieusement SMOTENC en mode expert (décision 66) si les
   utilisateurs traitent régulièrement des jeux de données très déséquilibrés
   au-delà de ce que le pondérage de classes gère bien.
3. Ajouter un endpoint de prévisualisation avant/après pour l'augmentation
   (décision 67) — la demande utilisateur est réelle et légitime.
4. Le double-lavis de couleur (badge niché dans un conteneur déjà teinté)
   reste un point d'attention : `check-contrast.mjs` ne le couvre pas
   encore programmatiquement — un futur renforcement pourrait modéliser la
   composition à deux niveaux directement dans ce script.
