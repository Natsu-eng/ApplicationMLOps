import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ModelEvaluation, TaskType } from "../../api/client";
import { Heatmap } from "../ui/Heatmap";
import { LabelWithHelp } from "../ui/Tooltip";

const CHART_TOOLTIP_STYLE = {
  contentStyle: {
    backgroundColor: "#0f172a",
    border: "1px solid #1e293b",
    borderRadius: 8,
    fontSize: 12,
  },
  labelStyle: { color: "#cbd5e1" },
};

// Palette discrète pour distinguer les courbes par classe (jusqu'à 6 classes lisibles)
const SERIES_COLORS = ["#2dd4bf", "#f472b6", "#facc15", "#818cf8", "#34d399", "#fb923c"];

export default function EvaluationCharts({
  taskType,
  evaluation,
}: {
  taskType: TaskType;
  evaluation: ModelEvaluation;
}) {
  if (taskType === "classification" && evaluation.confusion_matrix) {
    return <ClassificationCharts evaluation={evaluation} />;
  }
  if (taskType === "regression" && evaluation.actual) {
    return <RegressionCharts evaluation={evaluation} />;
  }
  return null;
}

function ClassificationCharts({ evaluation }: { evaluation: ModelEvaluation }) {
  const classNames = evaluation.class_names ?? [];
  const rocData = buildCurveSeries(evaluation.roc_curves, "fpr", "tpr");
  const prData = buildCurveSeries(evaluation.pr_curves, "recall", "precision");

  return (
    <>
      <section>
        <p className="text-xs uppercase tracking-wide text-slate-500 mb-2">
          <LabelWithHelp
            label="Matrice de confusion"
            help="Chaque ligne = la vraie classe, chaque colonne = la classe prédite. La diagonale montre les prédictions correctes — tout ce qui est hors diagonale est une erreur."
          />
        </p>
        <Heatmap
          xLabels={classNames}
          yLabels={classNames}
          matrix={evaluation.confusion_matrix ?? []}
          variant="sequential"
        />
      </section>

      {rocData.length > 0 && (
        <section>
          <p className="text-xs uppercase tracking-wide text-slate-500 mb-2">
            <LabelWithHelp
              label="Courbe ROC"
              help="Plus la courbe se rapproche du coin supérieur gauche, mieux le modèle distingue les classes. La diagonale grise correspond au hasard."
            />
          </p>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart margin={{ left: 0, right: 12 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis
                type="number"
                dataKey="x"
                domain={[0, 1]}
                tick={{ fill: "#64748b", fontSize: 11 }}
                label={{ value: "Taux de faux positifs", position: "insideBottom", offset: -5, fill: "#64748b", fontSize: 11 }}
              />
              <YAxis
                type="number"
                domain={[0, 1]}
                tick={{ fill: "#64748b", fontSize: 11 }}
                label={{ value: "Taux de vrais positifs", angle: -90, position: "insideLeft", fill: "#64748b", fontSize: 11 }}
              />
              <RechartsTooltip {...CHART_TOOLTIP_STYLE} formatter={(v) => Number(v).toFixed(3)} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#475569" strokeDasharray="4 4" />
              {rocData.map((series, i) => (
                <Line
                  key={series.name}
                  data={series.points}
                  dataKey="y"
                  name={series.name}
                  stroke={SERIES_COLORS[i % SERIES_COLORS.length]}
                  dot={false}
                  strokeWidth={2}
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </section>
      )}

      {prData.length > 0 && (
        <section>
          <p className="text-xs uppercase tracking-wide text-slate-500 mb-2">
            <LabelWithHelp
              label="Courbe précision-rappel"
              help="Utile en complément de la ROC quand les classes sont déséquilibrées — plus la courbe reste haute, mieux le modèle équilibre précision et rappel."
            />
          </p>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart margin={{ left: 0, right: 12 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis
                type="number"
                dataKey="x"
                domain={[0, 1]}
                tick={{ fill: "#64748b", fontSize: 11 }}
                label={{ value: "Rappel", position: "insideBottom", offset: -5, fill: "#64748b", fontSize: 11 }}
              />
              <YAxis
                type="number"
                domain={[0, 1]}
                tick={{ fill: "#64748b", fontSize: 11 }}
                label={{ value: "Précision", angle: -90, position: "insideLeft", fill: "#64748b", fontSize: 11 }}
              />
              <RechartsTooltip {...CHART_TOOLTIP_STYLE} formatter={(v) => Number(v).toFixed(3)} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              {prData.map((series, i) => (
                <Line
                  key={series.name}
                  data={series.points}
                  dataKey="y"
                  name={series.name}
                  stroke={SERIES_COLORS[i % SERIES_COLORS.length]}
                  dot={false}
                  strokeWidth={2}
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </section>
      )}
    </>
  );
}

function RegressionCharts({ evaluation }: { evaluation: ModelEvaluation }) {
  const actual = evaluation.actual ?? [];
  const predicted = evaluation.predicted ?? [];
  const residuals = evaluation.residuals ?? [];

  const scatterData = actual.map((a, i) => ({ actual: a, predicted: predicted[i] }));
  const residualData = predicted.map((p, i) => ({ predicted: p, residual: residuals[i] }));
  const bounds = actual.length > 0 ? [Math.min(...actual, ...predicted), Math.max(...actual, ...predicted)] : [0, 1];

  return (
    <>
      <section>
        <p className="text-xs uppercase tracking-wide text-slate-500 mb-2">
          <LabelWithHelp
            label="Prédit vs réel"
            help="Chaque point est une observation du test. Plus les points sont proches de la diagonale, plus la prédiction est fidèle à la valeur réelle."
          />
        </p>
        <ResponsiveContainer width="100%" height={240}>
          <ScatterChart margin={{ left: 0, right: 12, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis
              type="number"
              dataKey="actual"
              domain={bounds}
              tick={{ fill: "#64748b", fontSize: 11 }}
              label={{ value: "Valeur réelle", position: "insideBottom", offset: -5, fill: "#64748b", fontSize: 11 }}
            />
            <YAxis
              type="number"
              dataKey="predicted"
              domain={bounds}
              tick={{ fill: "#64748b", fontSize: 11 }}
              label={{ value: "Valeur prédite", angle: -90, position: "insideLeft", fill: "#64748b", fontSize: 11 }}
            />
            <RechartsTooltip {...CHART_TOOLTIP_STYLE} />
            <ReferenceLine
              segment={[{ x: bounds[0], y: bounds[0] }, { x: bounds[1], y: bounds[1] }]}
              stroke="#475569"
              strokeDasharray="4 4"
            />
            <Scatter data={scatterData} fill="#2dd4bf" fillOpacity={0.6} isAnimationActive={false} />
          </ScatterChart>
        </ResponsiveContainer>
      </section>

      <section>
        <p className="text-xs uppercase tracking-wide text-slate-500 mb-2">
          <LabelWithHelp
            label="Résidus"
            help="Écart entre valeur réelle et prédite, par niveau de prédiction. Un nuage sans forme particulière autour de zéro est bon signe ; un motif (entonnoir, courbe) indique que le modèle se trompe différemment selon la zone."
          />
        </p>
        <ResponsiveContainer width="100%" height={220}>
          <ScatterChart margin={{ left: 0, right: 12, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis
              type="number"
              dataKey="predicted"
              tick={{ fill: "#64748b", fontSize: 11 }}
              label={{ value: "Valeur prédite", position: "insideBottom", offset: -5, fill: "#64748b", fontSize: 11 }}
            />
            <YAxis
              type="number"
              dataKey="residual"
              tick={{ fill: "#64748b", fontSize: 11 }}
              label={{ value: "Résidu", angle: -90, position: "insideLeft", fill: "#64748b", fontSize: 11 }}
            />
            <RechartsTooltip {...CHART_TOOLTIP_STYLE} />
            <ReferenceLine y={0} stroke="#475569" strokeDasharray="4 4" />
            <Scatter data={residualData} fill="#f472b6" fillOpacity={0.6} isAnimationActive={false} />
          </ScatterChart>
        </ResponsiveContainer>
      </section>
    </>
  );
}

function buildCurveSeries(
  curves: Record<string, { fpr?: number[]; tpr?: number[]; precision?: number[]; recall?: number[] }> | undefined,
  xKey: "fpr" | "recall",
  yKey: "tpr" | "precision",
): { name: string; points: { x: number; y: number }[] }[] {
  if (!curves) return [];
  return Object.entries(curves).map(([name, curve]) => ({
    name,
    points: (curve[xKey] ?? []).map((x, i) => ({ x, y: (curve[yKey] ?? [])[i] })),
  }));
}
