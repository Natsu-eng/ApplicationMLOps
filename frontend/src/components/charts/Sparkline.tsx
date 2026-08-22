import { Line, LineChart, ResponsiveContainer } from "recharts";
import { CHART_COLOR_PRIMARY } from "../../theme/charts";

/** Mini-tendance inline — pensée pour une tuile de statistique (StatTile),
 * pas un graphique de section à part entière : pas d'axes, pas de grille,
 * pas de titre/légende de lecture propres (`ChartFrame` serait disproportionné
 * ici). L'alternative textuelle est un unique `aria-label` résumant la
 * tendance en une phrase — porté par le conteneur, le SVG reste
 * `aria-hidden`. */
export function Sparkline({ values, trendLabel, width = 72, height = 24 }: { values: number[]; trendLabel: string; width?: number; height?: number }) {
  const data = values.map((v, i) => ({ i, v }));
  const color = values.length >= 2 && values[values.length - 1] < values[0] ? "var(--danger)" : CHART_COLOR_PRIMARY;

  return (
    <span role="img" aria-label={trendLabel} className="inline-block" style={{ width, height }}>
      <span aria-hidden="true">
        <ResponsiveContainer width="100%" height={height}>
          <LineChart data={data} margin={{ top: 2, right: 2, bottom: 2, left: 2 }} accessibilityLayer={false}>
            <Line type="monotone" dataKey="v" stroke={color} strokeWidth={1.5} dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </span>
    </span>
  );
}
