# DataLab — spécification d'interface

Ce document est la **source de vérité** de la refonte visuelle. En cas de
désaccord entre ce document, une maquette et le code existant :
**maquette > spécification > code existant.**

Les maquettes de référence sont dans `_design/apercu/*.html` (ouvrables
directement dans un navigateur, une page par écran) et `_design/maquettes/*.dc.html`
(mêmes écrans, format de la toile de conception).

---

## 1. Jetons de couleur

Tout est dans `_design/themes.css`. **Aucune valeur de couleur en dur n'est
tolérée nulle part** — ni `#fff`, ni `bg-white`, ni `text-slate-400`, ni
`oklch(...)` écrit dans un composant. Un composant lit un jeton, point.

| Jeton | Rôle | Ne jamais l'utiliser pour |
|---|---|---|
| `--canvas` | fond de page | une carte |
| `--surface` | carte posée sur le fond | le fond de page |
| `--raised` | carte survolée, menu déroulant, popover | du texte |
| `--text` | texte principal | un fond |
| `--text-muted` | texte secondaire, légendes, en-têtes de colonne | du texte porteur d'une décision |
| `--border` | séparateur décoratif | le contour d'un champ ou d'un focus |
| `--border-strong` | champ, contour actionnable, anneau de focus (≥3:1) | un séparateur discret |
| `--accent` | action principale, sélection, lien | une surface de fond large |
| `--on-accent` | texte posé **sur** l'accent | autre chose |
| `--success` `--warning` `--danger` `--info` | états | de la décoration |
| `--s1`…`--s6` | séries de graphiques | de l'interface |

**Règle du sens, pas de la couleur.** Aucune information ne doit être portée
par la seule couleur (WCAG 1.4.1). Un point rouge s'accompagne d'un pictogramme
et d'un mot. Une série de courbes s'accompagne d'étiquettes posées sur la
courbe, pas seulement d'une légende.

## 2. Les 5 thèmes

`graphite` (défaut) · `ivoire` · `minuit` · `ardoise` · `porcelaine`.

Ils partagent **exactement la même structure de jetons**. Changer de thème
n'est qu'un attribut : `document.documentElement.dataset.theme = "minuit"`.
Deux thèmes sont clairs (`ivoire`, `porcelaine`) : tout composant qui suppose
un fond sombre est un bug.

Chaque valeur a été calculée par `_design/tune.py` :
- toute couleur de texte atteint **≥ 4,5:1 sur les trois fonds** du thème ;
- `--border-strong` atteint **≥ 3:1 sur surface** ;
- les 6 séries sont séparées de **ΔE2000 ≥ 21,5** entre paires et **≥ 18,6 en
  deutéranopie simulée**, grâce à un escalier de clarté délibéré.

Ajouter une couleur = relancer `tune.py`. Pas de couleur choisie à l'œil.

## 3. Typographie

| Rôle | Police | Taille / graisse | Interlettrage |
|---|---|---|---|
| Titre de page (`h1`) | Bricolage Grotesque | 26–29 px / 600 | −0,03em |
| Titre de section (`h2`) | Bricolage Grotesque | 17–19 px / 600 | −0,02em |
| Titre de carte (`h3`) | Bricolage Grotesque | 15 px / 600 | −0,015em |
| Texte courant | IBM Plex Sans | 13,5–14,5 px / 400 | 0 |
| Légende, aide | IBM Plex Sans | 12,5 px / 400, `--text-muted` | 0 |
| Surtitre (`.ov`) | IBM Plex Sans | 10,5 px / 600 MAJUSCULES | 0,11em |
| **Tout chiffre** | IBM Plex Mono | selon contexte, `font-variant-numeric: tabular-nums` | −0,01em |

**Les chiffres sont toujours en chasse fixe tabulaire.** Une colonne de scores
qui ne s'aligne pas verticalement est un défaut, pas un détail.

Charger les polices en local (`woff2` dans `public/fonts/`), pas depuis Google
Fonts : la plateforme doit fonctionner sur un réseau fermé. Prévoir une pile de
repli (`ui-sans-serif, system-ui`) et `font-display: swap`.

## 4. Formes et profondeur

- Rayons : `--radius-1: 10px` (bouton, champ, puce) · `--radius-2: 16px`
  (carte) · `--radius-3: 22px` (grand panneau, rail).
- Hauteur des contrôles : bouton 36 px (compact 31–32 px), champ 38 px.
- Une carte au repos : `background: var(--surface)`, `1px solid var(--border)`,
  `--radius-2`, `--highlight` en ombre interne.
- Une carte survolée : bordure teintée d'accent à 38 %, `translateY(-2px)`,
  `--shadow-2`. **La transition dure 150 ms**, et rien ne bouge sous
  `prefers-reduced-motion: reduce`.
- Verre (`.glass`) : réservé aux éléments **flottant au-dessus** du contenu —
  rail, barre haute, panneau d'estimation. Un flou n'a de sens que s'il a
  quelque chose à flouter : les écrans posent deux halos radiaux et une grille
  très faible derrière. Sans halo, pas de verre.

## 5. Anatomie applicative

- **Rail d'icônes** flottant, 60 px, `position: absolute; left:18px; top:18px;
  bottom:18px`, en verre, `--radius-3`. 5 entrées + aide + avatar en bas.
- **Barre haute** en verre, 56 px, `left:96px; right:22px; top:22px`, contient
  le fil d'Ariane à gauche et les actions à droite.
- **Contenu** : `left:96px; right:22px; top:96px; bottom:22px`.

## 6. États obligatoires de chaque composant

Aucun composant n'est considéré comme terminé s'il lui manque un de ces états :

1. **repos**
2. **survol** (souris)
3. **focus clavier** — anneau visible `2px solid var(--accent)` + `2px` de
   décalage, jamais `outline: none` sans remplacement
4. **actif / sélectionné**
5. **désactivé** — opacité 42 %, `cursor: not-allowed`, `aria-disabled`
6. **chargement** — silhouette (skeleton) qui reprend la forme réelle du
   contenu, jamais un « Chargement… » nu
7. **vide** — que faire, pas seulement « aucune donnée »
8. **erreur** — ce qui s'est passé, et l'action de sortie

## 7. Règles de fond, non négociables

Ces règles sont ce qui distingue la plateforme d'une démonstration. Elles sont
déjà appliquées dans les maquettes ; les reproduire est une exigence, pas une
option.

1. **La phrase avant le chiffre.** Tout écran de résultat ouvre sur une phrase
   en français qui répond à « est-ce utilisable ? », puis descend vers les
   métriques.
2. **La réserve est aussi visible que la réussite.** Si le modèle échoue
   quelque part, cela apparaît dans le bloc de verdict, pas trois onglets plus
   loin.
3. **Un avertissement dit quoi faire.** Chaque alerte propose au moins une
   action, et « ne rien changer » en est toujours une.
4. **Aucune destruction sans accord.** Valeurs extrêmes, colonnes suspectes,
   lignes en double : la plateforme signale et propose, l'utilisateur trance.
5. **Le coût du réglage est chiffré.** Déplacer un seuil affiche ce qu'on gagne
   *et* ce qu'on perd, en unités métier.
6. **Le vocabulaire est celui du métier.** « Le modèle se trompe en moyenne de
   2,8 MPa », pas « RMSE = 2.81 ». Le terme technique vient en second, en
   légende.
7. **Tout chiffre est traçable.** Depuis n'importe quel résultat, on atteint en
   un clic la fiche qui dit avec quelles données, quelle graine et quelles
   versions il a été obtenu.
