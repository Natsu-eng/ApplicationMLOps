import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip as RechartsTooltip, XAxis, YAxis } from "recharts";
import { CHART_GRID_STROKE, CHART_REFERENCE_STROKE, CHART_SERIES_COLORS, CHART_TICK_STYLE, CHART_TOOLTIP_STYLE } from "../../theme/charts";
import { IsolatableLegend, useSeriesIsolation } from "../ui/ChartLegend";
import { ChartFallbackTable, ChartFrame } from "./ChartFrame";

export interface CalibrationSeries {
  name: string;
  points: { predicted: number; observed: number }[];
}

/** Courbe de calibration — la probabilité prédite moyenne par tranche
 * contre la fréquence réellement observée. Un modèle bien calibré colle à
 * la diagonale ; au-dessus, il sous-estime le risque, en dessous il le
 * surestime. */
export function CalibrationCurve({ series }: { series: CalibrationSeries[] }) {
  const isolation = useSeriesIsolation();
  const manyClasses = series.length > 3;

  return (
    <ChartFrame
      title="Calibration — probabilité prédite vs fréquence réelle observée"
      reading="Un point sur la diagonale pointillée signifie une probabilité fiable. Au-dessus, le modèle sous-estime le risque pour cette tranche ; en dessous, il le surestime."
      ariaLabel={`Courbe de calibration pour ${series.length} classe(s), comparant la probabilité prédite à la fréquence observée.`}
      fallbackTable={
        <ChartFallbackTable
          columns={["Classe", "Probabilité prédite", "Fréquence observée"]}
          rows={series.flatMap((s) => s.points.map((p) => [s.name, Number(p.predicted.toFixed(3)), Number(p.observed.toFixed(3))]))}
        />
      }
    >
      {manyClasses && <p className="text-caption text-muted-foreground mb-1">Survolez ou cliquez une classe dans la légende pour l'isoler.</p>}
      <ResponsiveContainer width="100%" height={240}>
        <LineChart margin={{ left: 0, right: 12 }} accessibilityLayer={false}>
          <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID_STROKE} />
          <XAxis type="number" dataKey="x" domain={[0, 1]} tick={CHART_TICK_STYLE} label={{ value: "Probabilité prédite", position: "insideBottom", offset: -5, ...CHART_TICK_STYLE }} />
          <YAxis type="number" domain={[0, 1]} tick={CHART_TICK_STYLE} label={{ value: "Fréquence observée", angle: -90, position: "insideLeft", ...CHART_TICK_STYLE }} />
          <RechartsTooltip {...CHART_TOOLTIP_STYLE} formatter={(v) => Number(v).toFixed(3)} />
          <IsolatableLegend isolation={isolation} />
          <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={CHART_REFERENCE_STROKE} strokeDasharray="4 4" />
          {series.map((s, i) => (
            <Line
              key={s.name}
              data={s.points.map((p) => ({ x: p.predicted, y: p.observed }))}
              dataKey="y"
              name={s.name}
              stroke={CHART_SERIES_COLORS[i % CHART_SERIES_COLORS.length]}
              dot={{ r: 3 }}
              strokeWidth={isolation.hovered === s.name ? 3 : 2}
              strokeOpacity={isolation.hidden.has(s.name) ? 0 : isolation.hovered && isolation.hovered !== s.name ? 0.15 : 1}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}
