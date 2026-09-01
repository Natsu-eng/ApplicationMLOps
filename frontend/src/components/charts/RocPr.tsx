import { useState } from "react";
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip as RechartsTooltip, XAxis, YAxis } from "recharts";
import { Segmented } from "../ui/Segmented";
import { CHART_GRID_STROKE, CHART_REFERENCE_STROKE, CHART_SERIES_COLORS, CHART_TICK_STYLE, CHART_TOOLTIP_STYLE } from "../../theme/charts";
import { IsolatableLegend, useSeriesIsolation } from "../ui/ChartLegend";
import { ChartFallbackTable, ChartFrame } from "./ChartFrame";

interface RocPrSeries {
  name: string;
  roc: { fpr: number; tpr: number }[];
  pr: { recall: number; precision: number }[];
}

/** ROC et précision-rappel — une bascule, pas deux graphiques séparés à
 * faire défiler : la ROC dit à quel point le modèle sépare les classes en
 * général, la PR est plus parlante quand les classes sont déséquilibrées
 * (le cas fréquent en contrôle qualité — peu de défauts parmi beaucoup de
 * pièces conformes). */
export function RocPr({ series }: { series: RocPrSeries[] }) {
  const [mode, setMode] = useState<"roc" | "pr">("roc");
  const isolation = useSeriesIsolation();
  const manyClasses = series.length > 3;

  const rocDomainLabel = mode === "roc" ? { x: "Taux de faux positifs", y: "Taux de vrais positifs" } : { x: "Rappel", y: "Précision" };

  return (
    <ChartFrame
      title={mode === "roc" ? "Courbe ROC — capacité à séparer les classes" : "Courbe précision-rappel — fiabilité sur la classe rare"}
      reading={
        mode === "roc"
          ? "Plus la courbe se rapproche du coin supérieur gauche, mieux le modèle distingue les classes. La diagonale grise correspond au hasard."
          : "Plus la courbe reste haute sur toute la plage de rappel, mieux le modèle équilibre précision et rappel — plus parlant que la ROC quand une classe est rare."
      }
      ariaLabel={`${mode === "roc" ? "Courbe ROC" : "Courbe précision-rappel"} pour ${series.length} classe(s).`}
      fallbackTable={
        <ChartFallbackTable
          columns={mode === "roc" ? ["Classe", "Taux faux positifs", "Taux vrais positifs"] : ["Classe", "Rappel", "Précision"]}
          rows={series.flatMap((s) =>
            mode === "roc"
              ? s.roc.map((p) => [s.name, Number(p.fpr.toFixed(3)), Number(p.tpr.toFixed(3))])
              : s.pr.map((p) => [s.name, Number(p.recall.toFixed(3)), Number(p.precision.toFixed(3))])
          )}
        />
      }
      controls={<Segmented aria-label="Courbe affichée" value={mode} onChange={setMode} options={[{ id: "roc", label: "ROC" }, { id: "pr", label: "Précision-rappel" }]} />}
    >
      {manyClasses && <p className="text-caption text-muted-foreground mb-1">Survolez ou cliquez une classe dans la légende pour l'isoler.</p>}
      <ResponsiveContainer width="100%" height={240}>
        <LineChart margin={{ left: 0, right: 12 }} accessibilityLayer={false}>
          <CartesianGrid stroke={CHART_GRID_STROKE} />
          <XAxis type="number" dataKey="x" domain={[0, 1]} tick={CHART_TICK_STYLE} label={{ value: rocDomainLabel.x, position: "insideBottom", offset: -5, ...CHART_TICK_STYLE }} />
          <YAxis type="number" domain={[0, 1]} tick={CHART_TICK_STYLE} label={{ value: rocDomainLabel.y, angle: -90, position: "insideLeft", ...CHART_TICK_STYLE }} />
          <RechartsTooltip {...CHART_TOOLTIP_STYLE} formatter={(v) => Number(v).toFixed(3)} />
          <IsolatableLegend isolation={isolation} />
          {mode === "roc" && <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={CHART_REFERENCE_STROKE} strokeDasharray="4 4" />}
          {series.map((s, i) => (
            <Line
              key={s.name}
              data={mode === "roc" ? s.roc.map((p) => ({ x: p.fpr, y: p.tpr })) : s.pr.map((p) => ({ x: p.recall, y: p.precision }))}
              dataKey="y"
              name={s.name}
              stroke={CHART_SERIES_COLORS[i % CHART_SERIES_COLORS.length]}
              dot={false}
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
