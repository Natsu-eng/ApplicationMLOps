import { Activity, Crosshair, Grid3x3, Target as TargetIcon, Waves } from "lucide-react";
import {
  CartesianGrid,
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
import { Card } from "../ui/Card";
import { accentSurfaceClass } from "../ui/ColorIconBadge";
import { Heatmap } from "../ui/Heatmap";
import { SectionHeader } from "../ui/SectionHeader";
import {
  CHART_COLOR_PRIMARY,
  CHART_COLOR_SECONDARY,
  CHART_GRID_STROKE,
  CHART_REFERENCE_STROKE,
  CHART_SERIES_COLORS,
  CHART_TICK_STYLE,
  CHART_TOOLTIP_STYLE,
} from "../../theme/charts";
// Lot 2A — IsolatableLegend/useSeriesIsolation vivent maintenant dans
// components/ui/ChartLegend.tsx (réutilisables par n'importe quel graphe
// multi-séries) ; réexportés ici pour ne casser aucun import existant
// (ReliabilityDiagnostics.tsx importe ces deux noms depuis ce fichier).
import { IsolatableLegend, useSeriesIsolation } from "../ui/ChartLegend";

export { IsolatableLegend, useSeriesIsolation };

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
  const rocIsolation = useSeriesIsolation();
  const prIsolation = useSeriesIsolation();
  const manyClasses = classNames.length > 3;

  return (
    <>
      <Card className={`p-4 ${accentSurfaceClass("blue")}`}>
        <SectionHeader
          icon={Grid3x3}
          color="blue"
          label="Matrice de confusion"
          help="Chaque ligne = la vraie classe, chaque colonne = la classe prédite. La diagonale montre les prédictions correctes — tout ce qui est hors diagonale est une erreur."
        />
        <Heatmap
          xLabels={classNames}
          yLabels={classNames}
          matrix={evaluation.confusion_matrix ?? []}
          variant="sequential"
        />
      </Card>

      {rocData.length > 0 && (
        <Card className={`p-4 ${accentSurfaceClass("teal")}`}>
          <SectionHeader
            icon={Activity}
            color="teal"
            label="Courbe ROC"
            help="Plus la courbe se rapproche du coin supérieur gauche, mieux le modèle distingue les classes. La diagonale grise correspond au hasard."
          />
          {manyClasses && (
            <p className="text-caption text-muted-foreground mb-1">
              Survolez ou cliquez une classe dans la légende pour l'isoler.
            </p>
          )}
          <ResponsiveContainer width="100%" height={240}>
            <LineChart margin={{ left: 0, right: 12 }}>
              <CartesianGrid stroke={CHART_GRID_STROKE} />
              <XAxis
                type="number"
                dataKey="x"
                domain={[0, 1]}
                tick={CHART_TICK_STYLE}
                label={{ value: "Taux de faux positifs", position: "insideBottom", offset: -5, ...CHART_TICK_STYLE }}
              />
              <YAxis
                type="number"
                domain={[0, 1]}
                tick={CHART_TICK_STYLE}
                label={{ value: "Taux de vrais positifs", angle: -90, position: "insideLeft", ...CHART_TICK_STYLE }}
              />
              <RechartsTooltip {...CHART_TOOLTIP_STYLE} formatter={(v) => Number(v).toFixed(3)} />
              <IsolatableLegend isolation={rocIsolation} />
              <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={CHART_REFERENCE_STROKE} strokeDasharray="4 4" />
              {rocData.map((series, i) => (
                <Line
                  key={series.name}
                  data={series.points}
                  dataKey="y"
                  name={series.name}
                  stroke={CHART_SERIES_COLORS[i % CHART_SERIES_COLORS.length]}
                  dot={false}
                  strokeWidth={rocIsolation.hovered === series.name ? 3 : 2}
                  strokeOpacity={
                    rocIsolation.hidden.has(series.name)
                      ? 0
                      : rocIsolation.hovered && rocIsolation.hovered !== series.name
                        ? 0.15
                        : 1
                  }
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </Card>
      )}

      {prData.length > 0 && (
        <Card className={`p-4 ${accentSurfaceClass("violet")}`}>
          <SectionHeader
            icon={TargetIcon}
            color="violet"
            label="Courbe précision-rappel"
            help="Utile en complément de la ROC quand les classes sont déséquilibrées — plus la courbe reste haute, mieux le modèle équilibre précision et rappel."
          />
          {manyClasses && (
            <p className="text-caption text-muted-foreground mb-1">
              Survolez ou cliquez une classe dans la légende pour l'isoler.
            </p>
          )}
          <ResponsiveContainer width="100%" height={240}>
            <LineChart margin={{ left: 0, right: 12 }}>
              <CartesianGrid stroke={CHART_GRID_STROKE} />
              <XAxis
                type="number"
                dataKey="x"
                domain={[0, 1]}
                tick={CHART_TICK_STYLE}
                label={{ value: "Rappel", position: "insideBottom", offset: -5, ...CHART_TICK_STYLE }}
              />
              <YAxis
                type="number"
                domain={[0, 1]}
                tick={CHART_TICK_STYLE}
                label={{ value: "Précision", angle: -90, position: "insideLeft", ...CHART_TICK_STYLE }}
              />
              <RechartsTooltip {...CHART_TOOLTIP_STYLE} formatter={(v) => Number(v).toFixed(3)} />
              <IsolatableLegend isolation={prIsolation} />
              {prData.map((series, i) => (
                <Line
                  key={series.name}
                  data={series.points}
                  dataKey="y"
                  name={series.name}
                  stroke={CHART_SERIES_COLORS[i % CHART_SERIES_COLORS.length]}
                  dot={false}
                  strokeWidth={prIsolation.hovered === series.name ? 3 : 2}
                  strokeOpacity={
                    prIsolation.hidden.has(series.name)
                      ? 0
                      : prIsolation.hovered && prIsolation.hovered !== series.name
                        ? 0.15
                        : 1
                  }
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </Card>
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
      <Card className={`p-4 ${accentSurfaceClass("blue")}`}>
        <SectionHeader
          icon={Crosshair}
          color="blue"
          label="Prédit vs réel"
          help="Chaque point est une observation du test. Plus les points sont proches de la diagonale, plus la prédiction est fidèle à la valeur réelle."
        />
        <ResponsiveContainer width="100%" height={240}>
          <ScatterChart margin={{ left: 0, right: 12, bottom: 8 }}>
            <CartesianGrid stroke={CHART_GRID_STROKE} />
            <XAxis
              type="number"
              dataKey="actual"
              domain={bounds}
              tick={CHART_TICK_STYLE}
              label={{ value: "Valeur réelle", position: "insideBottom", offset: -5, ...CHART_TICK_STYLE }}
            />
            <YAxis
              type="number"
              dataKey="predicted"
              domain={bounds}
              tick={CHART_TICK_STYLE}
              label={{ value: "Valeur prédite", angle: -90, position: "insideLeft", ...CHART_TICK_STYLE }}
            />
            <RechartsTooltip {...CHART_TOOLTIP_STYLE} />
            <ReferenceLine
              segment={[{ x: bounds[0], y: bounds[0] }, { x: bounds[1], y: bounds[1] }]}
              stroke={CHART_REFERENCE_STROKE}
              strokeDasharray="4 4"
            />
            <Scatter data={scatterData} fill={CHART_COLOR_PRIMARY} fillOpacity={0.6} isAnimationActive={false} />
          </ScatterChart>
        </ResponsiveContainer>
      </Card>

      <Card className={`p-4 ${accentSurfaceClass("amber")}`}>
        <SectionHeader
          icon={Waves}
          color="amber"
          label="Résidus"
          help="Écart entre valeur réelle et prédite, par niveau de prédiction. Un nuage sans forme particulière autour de zéro est bon signe ; un motif (entonnoir, courbe) indique que le modèle se trompe différemment selon la zone."
        />
        <ResponsiveContainer width="100%" height={220}>
          <ScatterChart margin={{ left: 0, right: 12, bottom: 8 }}>
            <CartesianGrid stroke={CHART_GRID_STROKE} />
            <XAxis
              type="number"
              dataKey="predicted"
              tick={CHART_TICK_STYLE}
              label={{ value: "Valeur prédite", position: "insideBottom", offset: -5, ...CHART_TICK_STYLE }}
            />
            <YAxis
              type="number"
              dataKey="residual"
              tick={CHART_TICK_STYLE}
              label={{ value: "Résidu", angle: -90, position: "insideLeft", ...CHART_TICK_STYLE }}
            />
            <RechartsTooltip {...CHART_TOOLTIP_STYLE} />
            <ReferenceLine y={0} stroke={CHART_REFERENCE_STROKE} strokeDasharray="4 4" />
            <Scatter data={residualData} fill={CHART_COLOR_SECONDARY} fillOpacity={0.6} isAnimationActive={false} />
          </ScatterChart>
        </ResponsiveContainer>
      </Card>
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
