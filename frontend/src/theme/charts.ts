/** Thème Recharts centralisé — source unique pour tous les graphes de
 * l'app (EDA, évaluation de modèle, boxplots). Avant ce fichier, chaque
 * composant redéfinissait ses propres hex (souvent identiques, parfois
 * copiés-collés) : changer une couleur de grille ou de tooltip demandait de
 * toucher N fichiers. Ici, une seule fois.
 *
 * Lot 3 (refonte visuelle, Graphiques) — toutes les valeurs ci-dessous sont
 * désormais des jetons de thème (`var(--...)`), jamais un hex figé : les
 * couleurs de série sont `--s1…--s6` (calculées par `_design/tune.py`,
 * séparées par un écart perceptuel minimal y compris en deutéranopie
 * simulée), le chrome (grille/tick/tooltip) suit `--border`/`--text-muted`/
 * `--popover`. Les CSS custom properties sont valables comme attributs de
 * présentation SVG (`stroke`, `fill`) dans tous les navigateurs modernes —
 * aucune lecture JS de couleur calculée n'est nécessaire pour CETTE partie
 * (voir plus bas pour `beeswarmColor`, seule exception qui a vraiment besoin
 * d'un RGB numérique interpolable).
 *
 * Avant ce lot (Lot 1, décision journalisée) : ce fichier gardait
 * volontairement ses hex figés, le nettoyage étant explicitement reporté
 * ici — voir _design/JOURNAL.md, décision 1. */

// Chrome commun (grille, axes, tooltip) — partagé par tous les graphes
export const CHART_GRID_STROKE = "var(--border)";
export const CHART_TICK_COLOR = "var(--text-muted)";
// Avant : deux valeurs distinctes clair/sombre pour un "tick plus marqué".
// Les jetons de thème résolvent déjà cette distinction (--text est toujours
// le texte le plus contrasté du thème actif, quel qu'il soit) — un seul nom
// suffit désormais, plus de bascule manuelle clair/sombre à maintenir.
export const CHART_TICK_COLOR_MUTED = "var(--text)";
export const CHART_REFERENCE_STROKE = "var(--border-strong)";

// Hauteurs normalisées (Lot 2A, AUDIT_DATALAB_2026-08-16.md §J.4) — avant
// ce lot, 180/200/220/240/360 coexistaient sans raison d'écarter l'un ou
// l'autre. Trois tailles, usage prescrit : SM pour un graphe secondaire
// dans une grille dense (EDA), MD pour un graphe principal de carte, LG
// pour un graphe qui porte seul tout le contenu d'une page (projection 2D).
export const CHART_HEIGHT_SM = 200;
export const CHART_HEIGHT_MD = 260;
export const CHART_HEIGHT_LG = 360;

export const CHART_TICK_STYLE = { fill: CHART_TICK_COLOR, fontSize: 11 };
export const CHART_TICK_STYLE_SM = { fill: CHART_TICK_COLOR, fontSize: 10 };
export const CHART_TICK_STYLE_MUTED = { fill: CHART_TICK_COLOR_MUTED, fontSize: 11 };

export const CHART_TOOLTIP_STYLE = {
  contentStyle: {
    backgroundColor: "var(--popover)",
    border: `1px solid var(--border)`,
    borderRadius: 8,
    fontSize: 12,
    color: "var(--text)",
  },
  labelStyle: { color: "var(--text-muted)" },
  itemStyle: { color: "var(--text)" },
};

// Palette de séries — les 6 couleurs de thème (--s1…--s6), calculées une
// fois par _design/tune.py pour rester séparées d'un écart perceptuel
// minimal (ΔE2000) simultanément en vision normale ET en deutéranopie
// simulée — jamais choisies à l'œil, jamais redérivées ici. Remplace
// l'ancienne palette Okabe-Ito figée (un seul jeu de couleurs pour les 5
// thèmes, donc incohérente avec l'accent de chaque thème).
export const CHART_SERIES_COLORS = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)", "var(--s5)", "var(--s6)"];

// Couleurs sémantiques par usage — alignées sur les positions de
// CHART_SERIES_COLORS (jamais une teinte hors palette, pour rester
// cohérent si un graphe combine une couleur "sémantique" et la palette de
// séries dans la même figure).
export const CHART_COLOR_PRIMARY = "var(--s1)"; // série principale (prédit vs réel, distributions, corrélations)
export const CHART_COLOR_SECONDARY = "var(--s2)"; // série de contraste (résidus, valeurs manquantes)
export const CHART_COLOR_TERTIARY = "var(--s5)"; // distribution de la cible
export const CHART_COLOR_WARNING = "var(--s3)"; // anomalies/outliers

// Échelle divergente du beeswarm SHAP (Lot Explicabilité globale) — colore
// chaque point par la valeur (normalisée par variable) de la feature, PAS
// par la valeur SHAP (déjà portée par la position x). Convention SHAP
// standard : bleu = valeur basse, rouge = valeur haute — permet de lire
// d'un coup d'œil si "haut" ou "bas" pousse la prédiction dans un sens.
// Ancrée sur --info (bas) et --destructive (haut) de chaque thème plutôt
// que deux hex figés (Lot 1 → Lot 3, _design/JOURNAL.md).
export const CHART_BEESWARM_LOW_VAR = "--info";
export const CHART_BEESWARM_HIGH_VAR = "--danger";

let _cachedBeeswarmAnchors: { low: [number, number, number]; high: [number, number, number]; theme: string } | null = null;

// Résolution en 2 étapes : (1) un élément réel du DOM + getComputedStyle
// résout tout `var(--x)` selon la cascade réelle (jetons de thème posés sur
// [data-theme]) — un <canvas> ne sait PAS lire les custom properties CSS ;
// (2) le résultat (parfois déjà "rgb(...)", parfois encore une fonction
// couleur récente type oklch()/color-mix() selon le moteur) est repassé
// dans un contexte 2D dont fillStyle normalise TOUJOURS vers des octets
// RGB concrets, quel que soit l'espace colorimétrique — fiabilité plus
// importante ici qu'une lecture directe dont le format sérialisé varie
// selon le navigateur.
let _probeEl: HTMLSpanElement | null = null;
let _probeCtx: CanvasRenderingContext2D | null = null;
function parseRgb(cssColor: string): [number, number, number] {
  if (!_probeEl) {
    _probeEl = document.createElement("span");
    _probeEl.style.display = "none";
    document.body.appendChild(_probeEl);
  }
  _probeEl.style.color = cssColor;
  const resolved = getComputedStyle(_probeEl).color;

  if (!_probeCtx) {
    const canvas = document.createElement("canvas");
    canvas.width = 1;
    canvas.height = 1;
    _probeCtx = canvas.getContext("2d");
  }
  if (!_probeCtx) return [128, 128, 128];
  _probeCtx.fillStyle = resolved;
  _probeCtx.fillRect(0, 0, 1, 1);
  const [r, g, b] = _probeCtx.getImageData(0, 0, 1, 1).data;
  return [r, g, b];
}

/** Résout les 2 ancres du dégradé beeswarm en RGB numérique (nécessaire pour
 * interpoler — un `var()` ne peut pas être moyenné côté CSS pour N valeurs
 * continues). Mis en cache par thème actif : ne relit le DOM qu'au premier
 * appel après un changement de thème, jamais par point tracé. */
function beeswarmAnchors(): { low: [number, number, number]; high: [number, number, number] } {
  const theme = document.documentElement.getAttribute("data-theme") ?? "graphite";
  if (_cachedBeeswarmAnchors && _cachedBeeswarmAnchors.theme === theme) return _cachedBeeswarmAnchors;
  const anchors = {
    low: parseRgb(`var(${CHART_BEESWARM_LOW_VAR})`),
    high: parseRgb(`var(${CHART_BEESWARM_HIGH_VAR})`),
    theme,
  };
  _cachedBeeswarmAnchors = anchors;
  return anchors;
}

/** Interpolation RGB pure — aucune dépendance au DOM, testable telle
 * quelle en environnement Node (contrairement à `beeswarmColor`, qui a
 * besoin d'un navigateur pour résoudre les ancres de couleur du thème
 * actif). `t` normalisé dans [0, 1], borné en dehors de cette plage. */
export function lerpRgb(low: [number, number, number], high: [number, number, number], t: number): string {
  const clamped = Math.min(1, Math.max(0, t));
  const lerp = (a: number, b: number) => Math.round(a + (b - a) * clamped);
  return `rgb(${lerp(low[0], high[0])}, ${lerp(low[1], high[1])}, ${lerp(low[2], high[2])})`;
}

/** Interpole entre l'ancre "basse" et l'ancre "haute" du thème actif — `t`
 * normalisé dans [0, 1] (valeur de la feature ramenée à son propre min/max).
 * Nécessite un navigateur (résout `--info`/`--danger` via le DOM) — pour
 * tester la seule logique d'interpolation sans navigateur, voir `lerpRgb`. */
export function beeswarmColor(t: number): string {
  const { low, high } = beeswarmAnchors();
  return lerpRgb(low, high, t);
}

// ── Heatmap (corrélations, matrice de confusion) ────────────────────────────
//
// Rampe SÉQUENTIELLE (une seule teinte, clair→foncé — comptages, matrice de
// confusion) et rampe DIVERGENTE (deux teintes + point neutre gris — signe
// d'une corrélation). `color-mix()` calcule chaque palier à la volée à
// partir de --info (séquentiel + branche positive) et --danger (branche
// négative), mélangés vers --surface — jamais une rampe hex figée qui ne
// s'adapterait à aucun thème alternatif (Lot 1 → Lot 3, _design/JOURNAL.md).
// 7 paliers, buckets discrets (pas un dégradé continu) : un vrai palier par
// intensité se distingue mieux qu'une simple variation de transparence.
const HEATMAP_STEP_COUNT = 7;
function sequentialStep(index: number): string {
  const pct = Math.round(((index + 1) / HEATMAP_STEP_COUNT) * 88 + 8); // 8%..96%
  return `color-mix(in oklch, var(--info) ${pct}%, var(--surface))`;
}
function divergingPositiveStep(index: number): string {
  const pct = Math.round(((index + 1) / HEATMAP_STEP_COUNT) * 82 + 10);
  return `color-mix(in oklch, var(--info) ${pct}%, var(--surface))`;
}
function divergingNegativeStep(index: number): string {
  const pct = Math.round(((index + 1) / HEATMAP_STEP_COUNT) * 82 + 10);
  return `color-mix(in oklch, var(--danger) ${pct}%, var(--surface))`;
}
export const HEATMAP_SEQUENTIAL_STEPS = Array.from({ length: HEATMAP_STEP_COUNT }, (_, i) => sequentialStep(i));
export const HEATMAP_DIVERGING_POSITIVE_STEPS = Array.from({ length: HEATMAP_STEP_COUNT }, (_, i) => divergingPositiveStep(i));
export const HEATMAP_DIVERGING_NEGATIVE_STEPS = Array.from({ length: HEATMAP_STEP_COUNT }, (_, i) => divergingNegativeStep(i));
export const HEATMAP_NEUTRAL_STEP = "var(--muted)"; // point neutre (corrélation ≈ 0) — jamais une teinte, pour ne pas lire comme "un peu positif/négatif"
export const HEATMAP_MISSING_FILL = "var(--muted)"; // cellule sans valeur

/** Choisit le nombre de paliers de HEATMAP_*_STEPS le plus proche d'une
 * intensité normalisée [0, 1] — quantification discrète plutôt qu'une
 * interpolation continue, pour que chaque palier reste identifiable. */
export function heatmapStepIndex(intensity: number, stepCount: number): number {
  const clamped = Math.min(1, Math.max(0, intensity));
  return Math.min(stepCount - 1, Math.floor(clamped * stepCount));
}

/** Luminance relative WCAG (0 = noir, 1 = blanc) d'un RGB résolu — sert
 * UNIQUEMENT à choisir noir/blanc pour le texte d'une cellule de heatmap,
 * dont le fond composite (`color-mix`) n'est ni --canvas ni --surface ni
 * --raised : aucun jeton "-foreground" n'est calculé pour lui (piège déjà
 * rencontré sur les badges au Lot 2, _design/JOURNAL.md — un jeton conçu
 * pour UN fond précis ne garantit rien sur un composite différent). Un
 * vrai calcul de contraste ici est plus sûr qu'un jeton emprunté. */
function relativeLuminance([r, g, b]: [number, number, number]): number {
  const toLinear = (c: number) => {
    const s = c / 255;
    return s <= 0.03928 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
  };
  const [rl, gl, bl] = [toLinear(r), toLinear(g), toLinear(b)];
  return 0.2126 * rl + 0.7152 * gl + 0.0722 * bl;
}

/** Contraste WCAG entre deux luminances relatives (formule officielle :
 * (L_claire + 0,05) / (L_sombre + 0,05)). */
function contrastRatio(l1: number, l2: number): number {
  const [lighter, darker] = l1 >= l2 ? [l1, l2] : [l2, l1];
  return (lighter + 0.05) / (darker + 0.05);
}

/** Encre de texte lisible sur une cellule de heatmap — calculée en comparant
 * le VRAI contraste noir-sur-fond et blanc-sur-fond de ce palier précis
 * (pas un seuil de luminance approximatif : un premier essai à "luminance >
 * 0,45 → noir, sinon blanc" choisissait encore du blanc à un contraste de
 * 2,3:1 sur un fond `color-mix(in oklch, var(--info) 96%, var(--surface))`
 * mesurant 0,40 de luminance — sous le seuil 0,45 mais où le noir donnait
 * déjà 9:1. La luminance relative WCAG n'est pas linéaire perceptuellement :
 * comparer les deux contrastes réels plutôt que deviner un seuil est le
 * seul calcul qui ne se trompe pas, trouvé par axe-core, _design/
 * JOURNAL.md Lot 3). */
export function heatmapTextColor(cssBackground: string): string {
  const bgLuminance = relativeLuminance(parseRgb(cssBackground));
  const blackContrast = contrastRatio(0, bgLuminance);
  const whiteContrast = contrastRatio(1, bgLuminance);
  return blackContrast >= whiteContrast ? "black" : "white";
}

// Boxplot (composant custom BoxPlot.tsx, pas de primitive Recharts native)
export const BOXPLOT_FILL = "color-mix(in oklch, var(--s1) 12%, transparent)";
export const BOXPLOT_STROKE = CHART_COLOR_PRIMARY;
export const BOXPLOT_MEDIAN_STROKE = CHART_COLOR_SECONDARY;
export const BOXPLOT_WHISKER_STROKE = CHART_TICK_COLOR;
export const BOXPLOT_OUTLIER_FILL = CHART_COLOR_WARNING;

// Tooltip du boxplot — composant custom (pas de RechartsTooltip standard),
// mêmes couleurs de fond/bordure que CHART_TOOLTIP_STYLE mais structure CSS différente
export const BOXPLOT_TOOLTIP_CONTENT_STYLE = {
  backgroundColor: "var(--popover)",
  border: `1px solid var(--border)`,
  borderRadius: 8,
  padding: "8px 10px",
  fontSize: 12,
};
export const BOXPLOT_TOOLTIP_TITLE_COLOR = "var(--text)";
export const BOXPLOT_TOOLTIP_TEXT_COLOR = "var(--text-muted)";
