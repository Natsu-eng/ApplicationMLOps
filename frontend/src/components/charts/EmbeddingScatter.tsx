import { CartesianGrid, ResponsiveContainer, Scatter, ScatterChart, Tooltip as RechartsTooltip, XAxis, YAxis, ZAxis } from "recharts";
import { CHART_GRID_STROKE, CHART_SERIES_COLORS, CHART_TICK_STYLE, CHART_TOOLTIP_STYLE } from "../../theme/charts";
import { ChartFallbackTable, ChartFrame } from "./ChartFrame";

interface EmbeddingPoint {
  x: number;
  y: number;
  group: string;
  label?: string;
}

/** Projection 2D (UMAP/t-SNE/PCA) — colorée par groupe. Avertissement de
 * fond n°3 du pilier non supervisé : une DISTANCE sur cette projection ne
 * se lit pas de la même façon qu'une distance réelle entre deux amas
 * éloignés — l'algorithme préserve les voisinages locaux, pas les échelles
 * globales. Le rappeler dans la légende plutôt que le laisser deviner. */
export function EmbeddingScatter({ points, groups }: { points: EmbeddingPoint[]; groups: string[] }) {
  return (
    <ChartFrame
      title="Projection 2D — voisinages préservés, pas les distances globales"
      reading="Des points proches se ressemblaient dans l'espace d'origine. En revanche, la distance entre deux AMAS éloignés sur cette carte ne se compare pas directement — l'algorithme ne préserve que les voisinages locaux, jamais l'échelle globale."
      ariaLabel={`Projection 2D de ${points.length} points répartis sur ${groups.length} groupe(s).`}
      fallbackTable={<ChartFallbackTable columns={["Groupe", "X", "Y"]} rows={points.slice(0, 100).map((p) => [p.group, Number(p.x.toFixed(3)), Number(p.y.toFixed(3))])} />}
    >
      <ResponsiveContainer width="100%" height={320}>
        <ScatterChart margin={{ left: 0, right: 12, bottom: 8 }} accessibilityLayer={false}>
          <CartesianGrid stroke={CHART_GRID_STROKE} />
          <XAxis type="number" dataKey="x" tick={CHART_TICK_STYLE} name="x" />
          <YAxis type="number" dataKey="y" tick={CHART_TICK_STYLE} name="y" />
          <ZAxis range={[40, 40]} />
          <RechartsTooltip {...CHART_TOOLTIP_STYLE} formatter={(v) => Number(v).toFixed(2)} />
          {groups.map((g, i) => (
            <Scatter key={g} name={g} data={points.filter((p) => p.group === g)} fill={CHART_SERIES_COLORS[i % CHART_SERIES_COLORS.length]} fillOpacity={0.7} isAnimationActive={false} />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}
