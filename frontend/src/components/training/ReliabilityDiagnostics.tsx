import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Calibration, LearningCurveData } from "../../api/client";
import {
  CHART_COLOR_PRIMARY,
  CHART_COLOR_SECONDARY,
  CHART_GRID_STROKE,
  CHART_REFERENCE_STROKE,
  CHART_SERIES_COLORS,
  CHART_TICK_STYLE,
  CHART_TOOLTIP_STYLE,
} from "../../theme/charts";
import { IsolatableLegend, useSeriesIsolation } from "./EvaluationCharts";

/** Courbe de calibration / reliability diagram (Lot Explicabilité globale)
 * — compare la probabilité prédite moyenne par tranche à la fréquence
 * réelle observée. Même motif d'isolation par classe que les courbes ROC/PR
 * (Lot E1-ter) en multiclasse. */
export function CalibrationChart({ calibration }: { calibration: Calibration }) {
  const classNames = Object.keys(calibration);
  const isolation = useSeriesIsolation();
  if (classNames.length === 0) return null;

  return (
    <>
      {classNames.length > 3 && (
        <p className="text-[11px] text-slate-400 mb-1">Survolez ou cliquez une classe dans la légende pour l'isoler.</p>
      )}
      <ResponsiveContainer width="100%" height={240}>
        <LineChart margin={{ left: 0, right: 12 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID_STROKE} />
          <XAxis
            type="number"
            dataKey="x"
            domain={[0, 1]}
            tick={CHART_TICK_STYLE}
            label={{ value: "Probabilité prédite", position: "insideBottom", offset: -5, ...CHART_TICK_STYLE }}
          />
          <YAxis
            type="number"
            domain={[0, 1]}
            tick={CHART_TICK_STYLE}
            label={{ value: "Fréquence réelle observée", angle: -90, position: "insideLeft", ...CHART_TICK_STYLE }}
          />
          <RechartsTooltip {...CHART_TOOLTIP_STYLE} formatter={(v) => Number(v).toFixed(3)} />
          <IsolatableLegend isolation={isolation} />
          <ReferenceLine
            segment={[
              { x: 0, y: 0 },
              { x: 1, y: 1 },
            ]}
            stroke={CHART_REFERENCE_STROKE}
            strokeDasharray="4 4"
          />
          {classNames.map((name, i) => (
            <Line
              key={name}
              data={calibration[name].mean_predicted.map((x, j) => ({ x, y: calibration[name].fraction_positive[j] }))}
              dataKey="y"
              name={name}
              stroke={CHART_SERIES_COLORS[i % CHART_SERIES_COLORS.length]}
              dot={{ r: 3 }}
              strokeWidth={isolation.hovered === name ? 3 : 2}
              strokeOpacity={
                isolation.hidden.has(name) ? 0 : isolation.hovered && isolation.hovered !== name ? 0.15 : 1
              }
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </>
  );
}

/** Courbe d'apprentissage (Lot Explicabilité globale) — score en validation
 * croisée (même CV que la sélection du modèle) selon la taille du train :
 * diagnostic de sur/sous-apprentissage complémentaire à l'écart train/test
 * déjà affiché dans les métriques. */
export function LearningCurveChart({ data }: { data: LearningCurveData }) {
  if (data.train_sizes.length === 0) return null;
  const points = data.train_sizes.map((size, i) => ({
    size,
    train: data.train_scores_mean[i],
    val: data.val_scores_mean[i],
  }));

  return (
    <ResponsiveContainer width="100%" height={240}>
      <LineChart data={points} margin={{ left: 0, right: 12 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID_STROKE} />
        <XAxis
          type="number"
          dataKey="size"
          tick={CHART_TICK_STYLE}
          label={{ value: "Taille du jeu d'entraînement", position: "insideBottom", offset: -5, ...CHART_TICK_STYLE }}
        />
        <YAxis
          type="number"
          tick={CHART_TICK_STYLE}
          label={{ value: data.metric_label, angle: -90, position: "insideLeft", ...CHART_TICK_STYLE }}
        />
        <RechartsTooltip {...CHART_TOOLTIP_STYLE} formatter={(v) => Number(v).toFixed(3)} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <Line
          dataKey="train"
          name="Score sur l'entraînement"
          stroke={CHART_COLOR_PRIMARY}
          dot={{ r: 3 }}
          strokeWidth={2}
          isAnimationActive={false}
        />
        <Line
          dataKey="val"
          name="Score en validation croisée"
          stroke={CHART_COLOR_SECONDARY}
          dot={{ r: 3 }}
          strokeWidth={2}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
