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
 * bêtement la valeur aurait produit l'effet contraire de celui voulu. */

// Chrome commun (grille, axes, tooltip) — partagé par tous les graphes
export const CHART_GRID_STROKE = "#e2e8f0"; // slate-200
export const CHART_TICK_COLOR = "#64748b"; // slate-500 — contraste correct sur blanc comme sur fond sombre
export const CHART_TICK_COLOR_MUTED = "#334155"; // slate-700 — plus foncé que le tick standard (voir note ci-dessus)
export const CHART_REFERENCE_STROKE = "#94a3b8"; // slate-400

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

// Palette de séries — jusqu'à 6 classes lisibles (courbes ROC/PR multi-classes).
// Nuances -600 (règle -400→-600, voir note en tête de fichier) : teal, pink, yellow, indigo, emerald, orange.
export const CHART_SERIES_COLORS = ["#0d9488", "#db2777", "#ca8a04", "#4f46e5", "#059669", "#ea580c"];

// Couleurs sémantiques par usage — cohérentes avec CHART_SERIES_COLORS
export const CHART_COLOR_PRIMARY = "#0d9488"; // teal-600 — série principale (prédit vs réel, distributions, corrélations)
export const CHART_COLOR_SECONDARY = "#db2777"; // pink-600 — série de contraste (résidus, valeurs manquantes)
export const CHART_COLOR_TERTIARY = "#4f46e5"; // indigo-600 — distribution de la cible
export const CHART_COLOR_WARNING = "#d97706"; // amber-600 — anomalies/outliers

// Boxplot (composant custom BoxPlot.tsx, pas de primitive Recharts native)
export const BOXPLOT_FILL = "rgba(13, 148, 136, 0.15)"; // teal-600 translucide
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
