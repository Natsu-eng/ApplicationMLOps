import { CartesianGrid, ErrorBar, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip as RechartsTooltip, XAxis, YAxis } from "recharts";
import { CHART_COLOR_PRIMARY, CHART_GRID_STROKE, CHART_REFERENCE_STROKE, CHART_TICK_STYLE, CHART_TOOLTIP_STYLE } from "../../theme/charts";
import { ChartFallbackTable, ChartFrame } from "./ChartFrame";

export interface ScatterPredVsRealPoint {
  actual: number;
  predicted: number;
  /** Demi-largeur de l'intervalle de prédiction (ex. CQR) — optionnel. */
  intervalHalfWidth?: number;
}

/** Prédit vs réel — un point par observation du test. Diagonale pointillée
 * = prédiction parfaite ; les barres verticales (si fournies) montrent
 * l'intervalle de prédiction, pas seulement le point central. */
export function ScatterPredVsReal({ points, unit = "" }: { points: ScatterPredVsRealPoint[]; unit?: string }) {
  const hasIntervals = points.some((p) => p.intervalHalfWidth !== undefined);
  const values = points.flatMap((p) => [p.actual, p.predicted]);
  const bounds: [number, number] = values.length > 0 ? [Math.min(...values), Math.max(...values)] : [0, 1];
  const meanAbsError =
    points.length > 0 ? points.reduce((sum, p) => sum + Math.abs(p.actual - p.predicted), 0) / points.length : 0;

  return (
    <ChartFrame
      title="Prédit vs réel — chaque point est une observation du jeu de test"
      reading="Plus un point est proche de la diagonale pointillée, plus la prédiction est fidèle. Un nuage qui s'écarte régulièrement d'un côté indique un biais systématique, pas seulement du bruit."
      ariaLabel={`Nuage de points prédit contre réel sur ${points.length} observations. Écart moyen absolu de ${meanAbsError.toFixed(2)}${unit ? " " + unit : ""} à la diagonale idéale.`}
      fallbackTable={
        <ChartFallbackTable
          columns={["Réel", "Prédit", "Écart"]}
          rows={points.slice(0, 50).map((p) => [Number(p.actual.toFixed(3)), Number(p.predicted.toFixed(3)), Number((p.predicted - p.actual).toFixed(3))])}
        />
      }
    >
      <ResponsiveContainer width="100%" height={260}>
        <ScatterChart margin={{ left: 0, right: 12, bottom: 8 }} accessibilityLayer={false}>
          <CartesianGrid stroke={CHART_GRID_STROKE} />
          <XAxis type="number" dataKey="actual" domain={bounds} tick={CHART_TICK_STYLE} label={{ value: `Valeur réelle${unit ? ` (${unit})` : ""}`, position: "insideBottom", offset: -5, ...CHART_TICK_STYLE }} />
          <YAxis type="number" dataKey="predicted" domain={bounds} tick={CHART_TICK_STYLE} label={{ value: "Valeur prédite", angle: -90, position: "insideLeft", ...CHART_TICK_STYLE }} />
          <RechartsTooltip {...CHART_TOOLTIP_STYLE} formatter={(v) => Number(v).toFixed(3)} />
          <ReferenceLine segment={[{ x: bounds[0], y: bounds[0] }, { x: bounds[1], y: bounds[1] }]} stroke={CHART_REFERENCE_STROKE} strokeDasharray="4 4" />
          <Scatter data={points} fill={CHART_COLOR_PRIMARY} fillOpacity={0.6} isAnimationActive={false}>
            {hasIntervals && <ErrorBar dataKey="intervalHalfWidth" stroke={CHART_COLOR_PRIMARY} strokeOpacity={0.35} direction="y" />}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}
