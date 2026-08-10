/** Thème Recharts centralisé — source unique pour tous les graphes de
 * l'app (EDA, évaluation de modèle, boxplots). Avant ce fichier, chaque
 * composant redéfinissait ses propres hex (souvent identiques, parfois
 * copiés-collés) : changer une couleur de grille ou de tooltip demandait de
 * toucher N fichiers. Ici, une seule fois.
 *
 * Ce lot est une EXTRACTION mécanique des valeurs déjà en place — le rendu
 * de chaque graphe est inchangé, seule la source du style change. */

// Chrome commun (grille, axes, tooltip) — partagé par tous les graphes
export const CHART_GRID_STROKE = "#1e293b";
export const CHART_TICK_COLOR = "#64748b";
export const CHART_TICK_COLOR_MUTED = "#94a3b8"; // libellés de catégorie (ex : barres horizontales)
export const CHART_REFERENCE_STROKE = "#475569";

export const CHART_TICK_STYLE = { fill: CHART_TICK_COLOR, fontSize: 11 };
export const CHART_TICK_STYLE_SM = { fill: CHART_TICK_COLOR, fontSize: 10 };
export const CHART_TICK_STYLE_MUTED = { fill: CHART_TICK_COLOR_MUTED, fontSize: 11 };

export const CHART_TOOLTIP_STYLE = {
  contentStyle: {
    backgroundColor: "#0f172a",
    border: `1px solid ${CHART_GRID_STROKE}`,
    borderRadius: 8,
    fontSize: 12,
  },
  labelStyle: { color: "#cbd5e1" },
};

// Palette de séries — jusqu'à 6 classes lisibles (courbes ROC/PR multi-classes)
export const CHART_SERIES_COLORS = ["#2dd4bf", "#f472b6", "#facc15", "#818cf8", "#34d399", "#fb923c"];

// Couleurs sémantiques par usage — cohérentes avec CHART_SERIES_COLORS
export const CHART_COLOR_PRIMARY = "#2dd4bf"; // teal — série principale (prédit vs réel, distributions, corrélations)
export const CHART_COLOR_SECONDARY = "#f472b6"; // rose — série de contraste (résidus, valeurs manquantes)
export const CHART_COLOR_TERTIARY = "#818cf8"; // indigo — distribution de la cible
export const CHART_COLOR_WARNING = "#fbbf24"; // amber — anomalies/outliers

// Boxplot (composant custom BoxPlot.tsx, pas de primitive Recharts native)
export const BOXPLOT_FILL = "rgba(45, 212, 191, 0.22)";
export const BOXPLOT_STROKE = CHART_COLOR_PRIMARY;
export const BOXPLOT_MEDIAN_STROKE = CHART_COLOR_SECONDARY;
export const BOXPLOT_WHISKER_STROKE = CHART_TICK_COLOR;
export const BOXPLOT_OUTLIER_FILL = CHART_COLOR_WARNING;

// Tooltip du boxplot — composant custom (pas de RechartsTooltip standard),
// mêmes couleurs de fond/bordure que CHART_TOOLTIP_STYLE mais structure CSS différente
export const BOXPLOT_TOOLTIP_CONTENT_STYLE = {
  backgroundColor: "#0f172a",
  border: `1px solid ${CHART_GRID_STROKE}`,
  borderRadius: 8,
  padding: "8px 10px",
  fontSize: 12,
};
export const BOXPLOT_TOOLTIP_TITLE_COLOR = "#e2e8f0";
export const BOXPLOT_TOOLTIP_TEXT_COLOR = "#94a3b8";
