import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip as RechartsTooltip, XAxis, YAxis } from "recharts";
import { CHART_BEESWARM_HIGH_VAR, CHART_BEESWARM_LOW_VAR, CHART_GRID_STROKE, CHART_TICK_STYLE, CHART_TOOLTIP_STYLE } from "../../theme/charts";
import { ChartFallbackTable, ChartFrame } from "./ChartFrame";

interface WaterfallStep {
  label: string;
  /** Effet de cette étape (peut être négatif) — la 1ʳᵉ étape est la valeur
   * de base (moyenne du jeu d'entraînement), les suivantes les
   * contributions successives. */
  value: number;
}

/** Cascade locale — part de la moyenne du jeu d'entraînement, arrive à LA
 * prédiction de ce cas précis, une contribution à la fois. Chaque barre est
 * dessinée comme un segment flottant (base invisible + segment visible) :
 * la somme des étapes retombe exactement sur la barre finale, à l'arrondi
 * près affiché (SPEC-UI.md, Lot 9). */
export function WaterfallLocal({ baseValue, steps, finalLabel = "Prédiction" }: { baseValue: number; steps: WaterfallStep[]; finalLabel?: string }) {
  let running = baseValue;
  const bars = [
    { label: "Base (moyenne)", invisibleBase: 0, visible: baseValue, isTotal: true, raw: baseValue },
    ...steps.map((s) => {
      const from = running;
      running += s.value;
      const lo = Math.min(from, running);
      const hi = Math.max(from, running);
      return { label: s.label, invisibleBase: lo, visible: hi - lo, isTotal: false, raw: s.value };
    }),
    { label: finalLabel, invisibleBase: 0, visible: running, isTotal: true, raw: running },
  ];
  const sumCheck = baseValue + steps.reduce((s, x) => s + x.value, 0);

  return (
    <ChartFrame
      title="Cascade locale — de la moyenne du jeu à cette prédiction"
      reading="Chaque barre flottante est l'effet d'une variable, dans l'ordre. La somme de la base et de toutes les contributions retombe exactement sur la barre finale, à l'arrondi près affiché."
      ariaLabel={`Cascade partant d'une base de ${baseValue.toFixed(3)} et arrivant à ${running.toFixed(3)} après ${steps.length} contributions.`}
      fallbackTable={<ChartFallbackTable columns={["Étape", "Effet"]} rows={bars.map((b) => [b.label, Number(b.raw.toFixed(4))])} />}
    >
      <ResponsiveContainer width="100%" height={Math.max(180, bars.length * 30)}>
        <BarChart data={bars} layout="vertical" margin={{ left: 8, right: 40 }} accessibilityLayer={false}>
          <CartesianGrid stroke={CHART_GRID_STROKE} horizontal={false} />
          <XAxis type="number" tick={CHART_TICK_STYLE} />
          <YAxis type="category" dataKey="label" width={130} tick={CHART_TICK_STYLE} />
          <RechartsTooltip {...CHART_TOOLTIP_STYLE} formatter={(_v, _name, entry) => (entry.payload as (typeof bars)[number]).raw.toFixed(3)} />
          <Bar dataKey="invisibleBase" stackId="w" fill="transparent" isAnimationActive={false} />
          <Bar dataKey="visible" stackId="w" isAnimationActive={false} radius={3}>
            {bars.map((b, i) => (
              <Cell key={i} fill={b.isTotal ? "var(--text-muted)" : b.raw >= 0 ? `var(${CHART_BEESWARM_HIGH_VAR})` : `var(${CHART_BEESWARM_LOW_VAR})`} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <p className="text-caption text-muted-foreground mt-1 num">
        Vérification : {baseValue.toFixed(3)} + Σ contributions = {sumCheck.toFixed(3)}
      </p>
    </ChartFrame>
  );
}
