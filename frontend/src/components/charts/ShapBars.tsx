import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip as RechartsTooltip, XAxis, YAxis } from "recharts";
import { CHART_BEESWARM_HIGH_VAR, CHART_BEESWARM_LOW_VAR, CHART_GRID_STROKE, CHART_TICK_STYLE, CHART_TOOLTIP_STYLE } from "../../theme/charts";
import { ChartFallbackTable, ChartFrame } from "./ChartFrame";

export interface ShapBarDatum {
  feature: string;
  contribution: number;
}

/** Contributions locales (SHAP) — divergent, centré sur zéro. Convention
 * standard : une barre qui pousse vers la droite (rouge) augmente la
 * prédiction pour CE cas précis, une barre vers la gauche (bleu) la
 * diminue — jamais l'importance moyenne (ça, c'est un graphe global
 * séparé), toujours propre à cette observation. */
export function ShapBars({ data }: { data: ShapBarDatum[] }) {
  const sorted = [...data].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  const top = data.length > 0 ? sorted[0] : null;

  return (
    <ChartFrame
      title="Pourquoi cette prédiction — contribution de chaque variable"
      reading="Chaque barre part de zéro. Vers la droite (rouge), la variable pousse la prédiction vers le haut pour ce cas précis ; vers la gauche (bleu), elle la pousse vers le bas. Longueur = force de l'effet, pas seulement son sens."
      ariaLabel={
        top
          ? `Contributions locales par variable. La plus forte est « ${top.feature} », qui ${top.contribution >= 0 ? "augmente" : "diminue"} la prédiction de ${Math.abs(top.contribution).toFixed(3)}.`
          : "Aucune contribution locale disponible."
      }
      fallbackTable={<ChartFallbackTable columns={["Variable", "Contribution"]} rows={sorted.map((d) => [d.feature, Number(d.contribution.toFixed(4))])} />}
    >
      <ResponsiveContainer width="100%" height={Math.max(160, sorted.length * 28)}>
        <BarChart data={sorted} layout="vertical" margin={{ left: 8, right: 24 }} accessibilityLayer={false}>
          <CartesianGrid stroke={CHART_GRID_STROKE} horizontal={false} />
          <XAxis type="number" tick={CHART_TICK_STYLE} />
          <YAxis type="category" dataKey="feature" width={120} tick={CHART_TICK_STYLE} />
          <RechartsTooltip {...CHART_TOOLTIP_STYLE} formatter={(v) => Number(v).toFixed(4)} />
          <ReferenceLine x={0} stroke="var(--border-strong)" />
          <Bar dataKey="contribution" isAnimationActive={false} radius={3}>
            {sorted.map((d, i) => (
              <Cell key={i} fill={d.contribution >= 0 ? `var(${CHART_BEESWARM_HIGH_VAR})` : `var(${CHART_BEESWARM_LOW_VAR})`} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}
