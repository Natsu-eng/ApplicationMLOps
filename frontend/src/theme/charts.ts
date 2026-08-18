/** Thème Recharts centralisé — source unique pour tous les graphes de
 * l'app (EDA, évaluation de modèle, boxplots). Avant ce fichier, chaque
 * composant redéfinissait ses propres hex (souvent identiques, parfois
 * copiés-collés) : changer une couleur de grille ou de tooltip demandait de
 * toucher N fichiers. Ici, une seule fois.
 *
 * E1-bis (thème clair) : les couleurs de CHROME (grille/tooltip) sont
 * simplement inversées en miroir. Les couleurs de DONNÉES (séries,
 * boxplot, primary/secondary/tertiary/warning) suivent une règle
 * systématique : toute nuance -400 (choisie pour ressortir sur fond
 * quasi-noir) devient -600 (plus saturée, lisible sur blanc — les -400
 * tombent sous le seuil de contraste sur fond clair, notamment le jaune).
 * Un cas mérite une note à part : CHART_TICK_COLOR_MUTED servait à faire
 * ressortir DAVANTAGE un libellé que le tick standard — sur fond sombre ça
 * voulait dire "plus clair" (slate-400 vs slate-500), sur fond clair ça
 * veut dire l'inverse : "plus foncé" (slate-700 vs slate-500). Inverser
 * bêtement la valeur aurait produit l'effet contraire de celui voulu.
 *
 * Refonte UI : la couleur primaire de série passe du teal au bleu de marque
 * (#1d4ed8, même token que --color-primary/index.css et le dégradé du
 * logo) — un seul accent dans toute l'app plutôt que teal (composants) et
 * bleu (marque) en concurrence. Le teal ne disparaît pas : il devient la
 * couleur tertiaire (cible/distribution), toujours présent mais plus la
 * série n°1. */

// Chrome commun (grille, axes, tooltip) — partagé par tous les graphes
export const CHART_GRID_STROKE = "#e2e8f0"; // slate-200
export const CHART_TICK_COLOR = "#64748b"; // slate-500 — contraste correct sur blanc comme sur fond sombre
export const CHART_TICK_COLOR_MUTED = "#334155"; // slate-700 — plus foncé que le tick standard (voir note ci-dessus)
export const CHART_REFERENCE_STROKE = "#94a3b8"; // slate-400

// Hauteurs normalisées (Lot 2A, AUDIT_DATALAB_2026-08-16.md §J.4) — avant
// ce lot, 180/200/220/240/360 coexistaient sans raison d'écarter l'un ou
// l'autre. Trois tailles, usage prescrit : SM pour un graphe secondaire
// dans une grille dense (EDA), MD pour un graphe principal de carte, LG
// pour un graphe qui porte seul tout le contenu d'une page (projection 2D).
// Nouveau code seulement — appliquer aux graphes existants est le travail
// du Lot 2B, pas de celui-ci (aucune page métier modifiée ici).
export const CHART_HEIGHT_SM = 200;
export const CHART_HEIGHT_MD = 260;
export const CHART_HEIGHT_LG = 360;

export const CHART_TICK_STYLE = { fill: CHART_TICK_COLOR, fontSize: 11 };
export const CHART_TICK_STYLE_SM = { fill: CHART_TICK_COLOR, fontSize: 10 };
export const CHART_TICK_STYLE_MUTED = { fill: CHART_TICK_COLOR_MUTED, fontSize: 11 };

export const CHART_TOOLTIP_STYLE = {
  contentStyle: {
    backgroundColor: "#ffffff",
    border: `1px solid ${CHART_GRID_STROKE}`,
    borderRadius: 8,
    fontSize: 12,
  },
  labelStyle: { color: "#334155" }, // slate-700
};

// Palette de séries — 6 maximum (retour utilisateur, Lot 2A correctif 1) :
// l'ordre précédent (bleu roi/orange/vert menthe/jaune/rose/vert foncé)
// n'était pas une VRAIE séquence — deux verts se confondaient (slots 3 et
// 6) et l'orange/le jaune étaient voisins. Reconstruite depuis zéro :
// ancrée sur 3 teintes Okabe-Ito (« Color Universal Design », référence
// établie pour la distinguabilité en daltonisme — bleu/ambre/vermillon),
// complétée par recherche gloutonne sur les 3 emplacements restants,
// maximisant l'écart perceptuel MINIMAL (distance euclidienne dans OKLab,
// métrique quasi perceptuellement uniforme) simultanément en vision
// normale ET sous simulation de deutéranopie (matrice Machado/Oliveira/
// Fernandes 2009, appliquée en sRGB linéaire) — jamais un seul des deux
// critères optimisé au détriment de l'autre. Écart minimal obtenu sur les
// 15 paires : 15.6 ΔE en vision normale, 13.1 ΔE en deutéranopie simulée
// (contre un écart de 3.8 pour l'ancienne palette). Méthode et calcul
// complets : DECISIONS.md, Lot 2A.
//
// Compromis assumé : les slots 2 (ambre) et 4 (azur) sont volontairement
// clairs (nécessaire à leur séparation perceptuelle des slots voisins) —
// contraste plus faible qu'idéal sur carte blanche pour un TRAIT fin
// (2.3:1 et 1.8:1 respectivement, mesuré). Acceptable pour des éléments
// graphiques (lignes/points, jamais du texte, qui a ses propres tokens
// validés à 4.5:1) combinés à la légende isolable déjà en place
// (IsolatableLegend) — mais à garder à l'œil si un usage futur les
// affiche en traits très fins sans légende adjacente.
export const CHART_SERIES_COLORS = ["#2d6fcd", "#e69f00", "#d45e00", "#5ecdf6", "#23735c", "#37a0a6"];

// Couleurs sémantiques par usage — alignées sur les positions validées de
// CHART_SERIES_COLORS (jamais une teinte hors palette, pour rester
// cohérent si un graphe combine une couleur "sémantique" et la palette de
// séries dans la même figure).
export const CHART_COLOR_PRIMARY = "#2d6fcd"; // bleu de marque (slot 1) — série principale (prédit vs réel, distributions, corrélations)
export const CHART_COLOR_SECONDARY = "#e69f00"; // ambre (slot 2) — série de contraste (résidus, valeurs manquantes)
export const CHART_COLOR_TERTIARY = "#23735c"; // vert foncé bleuté (slot 5) — distribution de la cible
export const CHART_COLOR_WARNING = "#d45e00"; // vermillon (slot 3) — anomalies/outliers

// Échelle divergente du beeswarm SHAP (Lot Explicabilité globale) — colore
// chaque point par la valeur (normalisée par variable) de la feature, PAS
// par la valeur SHAP (déjà portée par la position x). Convention SHAP
// standard : bleu = valeur basse, rouge = valeur haute — permet de lire
// d'un coup d'œil si "haut" ou "bas" pousse la prédiction dans un sens.
export const CHART_BEESWARM_LOW = "#2563eb"; // blue-600
export const CHART_BEESWARM_HIGH = "#dc2626"; // red-600

/** Interpole entre CHART_BEESWARM_LOW et CHART_BEESWARM_HIGH — `t` normalisé
 * dans [0, 1] (valeur de la feature ramenée à son propre min/max). */
export function beeswarmColor(t: number): string {
  const clamped = Math.min(1, Math.max(0, t));
  const lerp = (a: number, b: number) => Math.round(a + (b - a) * clamped);
  const [r1, g1, b1] = [0x25, 0x63, 0xeb];
  const [r2, g2, b2] = [0xdc, 0x26, 0x26];
  return `rgb(${lerp(r1, r2)}, ${lerp(g1, g2)}, ${lerp(b1, b2)})`;
}

// ── Heatmap (corrélations, matrice de confusion) ────────────────────────────
//
// Rampe SÉQUENTIELLE (une seule teinte, clair→foncé — comptages, matrice de
// confusion) et rampe DIVERGENTE (deux teintes + point neutre gris — signe
// d'une corrélation) : reprennent la méthode et les teintes de référence du
// skill dataviz (bleu séquentiel, paire divergente bleu↔rouge), plutôt que
// les anciens dégradés d'opacité ad hoc (rgba(29,78,216, alpha) codés en dur
// dans Heatmap.tsx, jamais revalidés). Étapes discrètes (buckets), pas un
// dégradé continu par opacité — un vrai palier par intensité se distingue
// mieux qu'une simple variation de transparence sur une petite cellule.
export const HEATMAP_SEQUENTIAL_STEPS = ["#eaf1fc", "#cde2fb", "#9ec5f4", "#5598e7", "#256abf", "#184f95", "#0d366b"];
export const HEATMAP_DIVERGING_POSITIVE_STEPS = ["#eaf1fc", "#cde2fb", "#9ec5f4", "#5598e7", "#256abf", "#184f95", "#0d366b"];
export const HEATMAP_DIVERGING_NEGATIVE_STEPS = ["#fbeae9", "#f7d2cf", "#eea19b", "#e2685e", "#c73f34", "#a12f26", "#7a231c"];
export const HEATMAP_NEUTRAL_STEP = "#f0efec"; // point neutre gris (corrélation ≈ 0) — jamais une teinte, pour ne pas lire comme "un peu positif/négatif"
export const HEATMAP_MISSING_FILL = "#f1f5f9"; // slate-100 — cellule sans valeur

/** Choisit le nombre de paliers de HEATMAP_*_STEPS le plus proche d'une
 * intensité normalisée [0, 1] — quantification discrète plutôt qu'une
 * interpolation continue, pour que chaque palier reste identifiable. */
export function heatmapStepIndex(intensity: number, stepCount: number): number {
  const clamped = Math.min(1, Math.max(0, intensity));
  return Math.min(stepCount - 1, Math.floor(clamped * stepCount));
}

/** Encre de texte lisible sur une cellule de heatmap — blanc à partir du
 * palier le plus soutenu (index ≥ 4 sur les rampes à 7 paliers ci-dessus),
 * sombre en-dessous. Calculé une fois ici plutôt que deviné composant par
 * composant. */
export function heatmapTextColor(stepIndex: number): string {
  return stepIndex >= 4 ? "#ffffff" : "#0f172a"; // slate-900
}

// Boxplot (composant custom BoxPlot.tsx, pas de primitive Recharts native)
export const BOXPLOT_FILL = "rgba(45, 111, 205, 0.12)"; // CHART_COLOR_PRIMARY translucide
export const BOXPLOT_STROKE = CHART_COLOR_PRIMARY;
export const BOXPLOT_MEDIAN_STROKE = CHART_COLOR_SECONDARY;
export const BOXPLOT_WHISKER_STROKE = CHART_TICK_COLOR;
export const BOXPLOT_OUTLIER_FILL = CHART_COLOR_WARNING;

// Tooltip du boxplot — composant custom (pas de RechartsTooltip standard),
// mêmes couleurs de fond/bordure que CHART_TOOLTIP_STYLE mais structure CSS différente
export const BOXPLOT_TOOLTIP_CONTENT_STYLE = {
  backgroundColor: "#ffffff",
  border: `1px solid ${CHART_GRID_STROKE}`,
  borderRadius: 8,
  padding: "8px 10px",
  fontSize: 12,
};
export const BOXPLOT_TOOLTIP_TITLE_COLOR = "#0f172a"; // slate-900
export const BOXPLOT_TOOLTIP_TEXT_COLOR = "#64748b"; // slate-500
