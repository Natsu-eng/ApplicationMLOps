import {
  CartesianGrid,
  ComposedChart,
  Customized,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
  Bar,
} from "recharts";

/** Boxplot (boîte à moustaches) réutilisable, en Recharts PUR — jamais
 * Plotly, jamais de SVG hors-Recharts. Recharts n'a pas de primitive
 * boxplot native : on dessine la boîte/moustaches/médiane/outliers à la
 * main via `Customized`, qui donne accès aux échelles réelles des axes
 * (xAxisMap/yAxisMap) — plus robuste qu'une reconstruction approximative à
 * partir de la géométrie d'un `Bar` flottant. Une barre invisible
 * (dataKey="range") sert uniquement de zone de survol pour le Tooltip
 * (Recharts n'active le survol que sur une vraie série).
 *
 * Réutilisé pour les outliers IQR et le boxplot-par-classe de l'EDA (Lot
 * B), et prévu pour resservir aux graphiques post-entraînement (résidus,
 * distributions — Lot C). */

export interface BoxPlotDatum {
  name: string;
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
  outliers?: number[];
}

const BOX_FILL = "rgba(45, 212, 191, 0.22)"; // teal-400 translucide
const BOX_STROKE = "#2dd4bf";
const MEDIAN_STROKE = "#f472b6"; // rose-400 — contraste net avec la boîte teal
const WHISKER_STROKE = "#64748b"; // slate-500
const OUTLIER_FILL = "#fbbf24"; // amber-400 — attire l'œil sans alarmer (pas rouge)

function BoxPlotLayer({ data }: { data: BoxPlotDatum[] }) {
  return function renderLayer(props: Record<string, unknown>) {
    const xAxisMap = props.xAxisMap as Record<string, any> | undefined;
    const yAxisMap = props.yAxisMap as Record<string, any> | undefined;
    const xAxis = xAxisMap ? Object.values(xAxisMap)[0] : undefined;
    const yAxis = yAxisMap ? Object.values(yAxisMap)[0] : undefined;
    if (!xAxis || !yAxis) return null;

    const xScale = xAxis.scale;
    const yScale = yAxis.scale;
    const bandwidth = typeof xScale.bandwidth === "function" ? xScale.bandwidth() : 40;
    const boxWidth = Math.min(bandwidth * 0.5, 44);

    return (
      <g>
        {data.map((d) => {
          const bandStart = xScale(d.name);
          if (bandStart === undefined) return null;
          const centerX = bandStart + bandwidth / 2;
          const boxTop = yScale(d.q3);
          const boxBottom = yScale(d.q1);
          const height = Math.max(Math.abs(boxBottom - boxTop), 1);
          const y = Math.min(boxTop, boxBottom);
          const medianY = yScale(d.median);
          const minY = yScale(d.min);
          const maxY = yScale(d.max);
          const capHalf = boxWidth / 2;

          return (
            <g key={d.name}>
              <line x1={centerX} x2={centerX} y1={y} y2={maxY} stroke={WHISKER_STROKE} strokeWidth={1} />
              <line x1={centerX} x2={centerX} y1={y + height} y2={minY} stroke={WHISKER_STROKE} strokeWidth={1} />
              <line
                x1={centerX - capHalf} x2={centerX + capHalf} y1={maxY} y2={maxY}
                stroke={WHISKER_STROKE} strokeWidth={1}
              />
              <line
                x1={centerX - capHalf} x2={centerX + capHalf} y1={minY} y2={minY}
                stroke={WHISKER_STROKE} strokeWidth={1}
              />
              <rect
                x={centerX - boxWidth / 2} y={y} width={boxWidth} height={height}
                fill={BOX_FILL} stroke={BOX_STROKE} strokeWidth={1.5} rx={2}
              />
              <line
                x1={centerX - boxWidth / 2} x2={centerX + boxWidth / 2} y1={medianY} y2={medianY}
                stroke={MEDIAN_STROKE} strokeWidth={2}
              />
              {(d.outliers ?? []).map((value, i) => (
                <circle key={i} cx={centerX} cy={yScale(value)} r={2.5} fill={OUTLIER_FILL} fillOpacity={0.8} />
              ))}
            </g>
          );
        })}
      </g>
    );
  };
}

function BoxPlotTooltip({ active, payload }: { active?: boolean; payload?: { payload: BoxPlotDatum }[] }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  const outlierCount = d.outliers?.length ?? 0;
  return (
    <div
      style={{
        backgroundColor: "#0f172a",
        border: "1px solid #1e293b",
        borderRadius: 8,
        padding: "8px 10px",
        fontSize: 12,
      }}
    >
      <p style={{ color: "#e2e8f0", marginBottom: 4, fontWeight: 500 }}>{d.name}</p>
      <p style={{ color: "#94a3b8" }}>Max : {d.max.toFixed(2)}</p>
      <p style={{ color: "#94a3b8" }}>Q3 : {d.q3.toFixed(2)}</p>
      <p style={{ color: "#f472b6" }}>Médiane : {d.median.toFixed(2)}</p>
      <p style={{ color: "#94a3b8" }}>Q1 : {d.q1.toFixed(2)}</p>
      <p style={{ color: "#94a3b8" }}>Min : {d.min.toFixed(2)}</p>
      {outlierCount > 0 && (
        <p style={{ color: "#fbbf24" }}>
          {outlierCount} valeur{outlierCount > 1 ? "s" : ""} atypique{outlierCount > 1 ? "s" : ""}
        </p>
      )}
    </div>
  );
}

export function BoxPlotChart({
  data,
  height = 240,
  valueLabel,
}: {
  data: BoxPlotDatum[];
  height?: number;
  valueLabel?: string;
}) {
  if (data.length === 0) return null;

  const allValues = data.flatMap((d) => [d.min, d.max, ...(d.outliers ?? [])]);
  const lo = Math.min(...allValues);
  const hi = Math.max(...allValues);
  const margin = (hi - lo) * 0.08 || 1;
  const chartData = data.map((d) => ({ ...d, range: [d.min, d.max] as [number, number] }));

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={chartData} margin={{ left: 0, right: 12, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
        <XAxis dataKey="name" tick={{ fill: "#64748b", fontSize: 11 }} />
        <YAxis
          domain={[lo - margin, hi + margin]}
          tick={{ fill: "#64748b", fontSize: 11 }}
          label={
            valueLabel
              ? { value: valueLabel, angle: -90, position: "insideLeft", fill: "#64748b", fontSize: 11 }
              : undefined
          }
        />
        <RechartsTooltip content={<BoxPlotTooltip />} />
        {/* Zone de survol invisible — Recharts n'active le Tooltip que sur une vraie série */}
        <Bar dataKey="range" fill="transparent" isAnimationActive={false} />
        <Customized component={BoxPlotLayer({ data: chartData })} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
