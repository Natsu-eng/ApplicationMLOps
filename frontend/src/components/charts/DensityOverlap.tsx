import { Area, AreaChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip as RechartsTooltip, XAxis, YAxis } from "recharts";
import { CHART_COLOR_PRIMARY, CHART_COLOR_WARNING, CHART_GRID_STROKE, CHART_TICK_STYLE, CHART_TOOLTIP_STYLE } from "../../theme/charts";
import { ChartFallbackTable, ChartFrame } from "./ChartFrame";

interface DensityPoint {
  x: number;
  normal: number;
  anomaly: number;
}

/** Recouvrement de densités — deux populations (ex. scores "normal" vs
 * "anomalie") sur le même axe, seuil de décision matérialisé par une ligne
 * verticale. La zone où les deux courbes se chevauchent est la zone
 * d'ambiguïté réelle du modèle — pas un défaut de réglage, une limite de
 * séparabilité des données elles-mêmes. */
export function DensityOverlap({
  points,
  threshold,
  normalLabel = "Normal",
  anomalyLabel = "Anomalie",
}: {
  points: DensityPoint[];
  threshold: number;
  normalLabel?: string;
  anomalyLabel?: string;
}) {
  return (
    <ChartFrame
      title={`Recouvrement des distributions — ${normalLabel.toLowerCase()} vs ${anomalyLabel.toLowerCase()}`}
      reading={`La zone où les deux courbes se chevauchent est l'ambiguïté réelle du modèle sur ces données — un seuil ne peut pas la faire disparaître, seulement déplacer où les erreurs se produisent. La ligne verticale est le seuil actuel.`}
      ariaLabel={`Distributions de ${normalLabel.toLowerCase()} et ${anomalyLabel.toLowerCase()}, seuil de décision à ${threshold.toFixed(3)}.`}
      fallbackTable={<ChartFallbackTable columns={["Score", normalLabel, anomalyLabel]} rows={points.map((p) => [Number(p.x.toFixed(3)), Number(p.normal.toFixed(4)), Number(p.anomaly.toFixed(4))])} />}
    >
      <ResponsiveContainer width="100%" height={240}>
        <AreaChart data={points} margin={{ left: 0, right: 12 }} accessibilityLayer={false}>
          <CartesianGrid stroke={CHART_GRID_STROKE} />
          <XAxis type="number" dataKey="x" tick={CHART_TICK_STYLE} label={{ value: "Score", position: "insideBottom", offset: -5, ...CHART_TICK_STYLE }} />
          <YAxis type="number" tick={CHART_TICK_STYLE} label={{ value: "Densité", angle: -90, position: "insideLeft", ...CHART_TICK_STYLE }} />
          <RechartsTooltip {...CHART_TOOLTIP_STYLE} formatter={(v) => Number(v).toFixed(4)} />
          <ReferenceLine x={threshold} stroke="var(--border-strong)" strokeWidth={2} label={{ value: "Seuil", position: "top", ...CHART_TICK_STYLE }} />
          <Area type="monotone" dataKey="normal" name={normalLabel} stroke={CHART_COLOR_PRIMARY} fill={CHART_COLOR_PRIMARY} fillOpacity={0.25} strokeWidth={2} isAnimationActive={false} />
          <Area type="monotone" dataKey="anomaly" name={anomalyLabel} stroke={CHART_COLOR_WARNING} fill={CHART_COLOR_WARNING} fillOpacity={0.25} strokeWidth={2} isAnimationActive={false} />
        </AreaChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}
