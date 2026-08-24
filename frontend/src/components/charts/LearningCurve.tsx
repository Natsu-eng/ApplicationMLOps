import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip as RechartsTooltip, XAxis, YAxis } from "recharts";
import { CHART_COLOR_PRIMARY, CHART_COLOR_SECONDARY, CHART_GRID_STROKE, CHART_TICK_STYLE, CHART_TOOLTIP_STYLE } from "../../theme/charts";
import { ChartFallbackTable, ChartFrame } from "./ChartFrame";

export interface LearningCurvePoint {
  size: number;
  train: number;
  validation: number;
}

/** Courbe d'apprentissage — score selon la taille du jeu d'entraînement.
 * Diagnostic de sur/sous-apprentissage : les 2 courbes qui se rejoignent en
 * restant basses = sous-apprentissage (modèle trop simple) ; un grand écart
 * qui persiste = sur-apprentissage (le modèle mémorise plutôt qu'il
 * généralise). */
export function LearningCurve({ points, metricLabel }: { points: LearningCurvePoint[]; metricLabel: string }) {
  const last = points[points.length - 1];
  const gap = last ? Math.abs(last.train - last.validation) : 0;

  return (
    <ChartFrame
      title="Courbe d'apprentissage — score selon la taille du jeu d'entraînement"
      reading="Les deux courbes qui restent basses ensemble indiquent un modèle trop simple (sous-apprentissage). Un écart qui persiste ou grandit indique que le modèle mémorise plutôt qu'il généralise (sur-apprentissage)."
      ariaLabel={`Courbe d'apprentissage, ${metricLabel}. Écart final entre entraînement et validation de ${gap.toFixed(3)}.`}
      fallbackTable={
        <ChartFallbackTable
          columns={["Taille", "Entraînement", "Validation"]}
          rows={points.map((p) => [p.size, Number(p.train.toFixed(3)), Number(p.validation.toFixed(3))])}
        />
      }
    >
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={points} margin={{ left: 0, right: 12 }} accessibilityLayer={false}>
          <CartesianGrid stroke={CHART_GRID_STROKE} />
          <XAxis type="number" dataKey="size" tick={CHART_TICK_STYLE} label={{ value: "Taille du jeu d'entraînement", position: "insideBottom", offset: -5, ...CHART_TICK_STYLE }} />
          <YAxis type="number" tick={CHART_TICK_STYLE} label={{ value: metricLabel, angle: -90, position: "insideLeft", ...CHART_TICK_STYLE }} />
          <RechartsTooltip {...CHART_TOOLTIP_STYLE} formatter={(v) => Number(v).toFixed(3)} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Line dataKey="train" name="Score sur l'entraînement" stroke={CHART_COLOR_PRIMARY} dot={{ r: 3 }} strokeWidth={2} isAnimationActive={false} />
          <Line dataKey="validation" name="Score en validation croisée" stroke={CHART_COLOR_SECONDARY} dot={{ r: 3 }} strokeWidth={2} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}
