import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ErrorBar,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { PermutationImportanceFeature, ShapBeeswarm, ShapBeeswarmPoint } from "../../api/client";
import {
  CHART_BEESWARM_HIGH_VAR,
  CHART_BEESWARM_LOW_VAR,
  CHART_COLOR_PRIMARY,
  CHART_GRID_STROKE,
  CHART_REFERENCE_STROKE,
  CHART_TICK_STYLE,
  CHART_TOOLTIP_STYLE,
  beeswarmColor,
} from "../../theme/charts";

/** Jitter déterministe (pas de dépendance beeswarm dédiée, juste Recharts) —
 * petit bruit pseudo-aléatoire seedé par l'index du point, reproductible
 * (jamais Math.random) pour un rendu stable d'un rafraîchissement à l'autre. */
function seededJitter(seed: number): number {
  const x = Math.sin(seed * 12.9898) * 43758.5453;
  return (x - Math.floor(x) - 0.5) * 0.8; // dans [-0.4, 0.4]
}

interface BeeswarmRow {
  x: number;
  y: number;
  color: string;
  feature: string;
  feature_value: number;
}

function buildRows(points: ShapBeeswarmPoint[], featureOrder: string[]): BeeswarmRow[] {
  const rows: BeeswarmRow[] = [];
  featureOrder.forEach((feature, rowIndex) => {
    const featurePoints = points.filter((p) => p.feature === feature);
    const values = featurePoints.map((p) => p.feature_value);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = max - min || 1;
    featurePoints.forEach((p, i) => {
      const t = (p.feature_value - min) / span;
      rows.push({
        x: p.shap_value,
        y: rowIndex + seededJitter(rowIndex * 1000 + i),
        color: beeswarmColor(t),
        feature: p.feature,
        feature_value: p.feature_value,
      });
    });
  });
  return rows;
}

function BeeswarmTooltip({ active, payload }: { active?: boolean; payload?: { payload: BeeswarmRow }[] }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div style={CHART_TOOLTIP_STYLE.contentStyle} className="px-3 py-2">
      <p style={CHART_TOOLTIP_STYLE.labelStyle} className="font-medium mb-1">
        {p.feature}
      </p>
      <p className="text-muted-foreground">
        Valeur SHAP : <span className="tabular-nums">{p.x.toFixed(3)}</span>
      </p>
      <p className="text-muted-foreground">
        Valeur de la variable : <span className="tabular-nums">{p.feature_value.toFixed(2)}</span>
      </p>
    </div>
  );
}

/** Beeswarm SHAP (Lot Explicabilité globale) — une ligne par variable, un
 * point par observation du test : la position horizontale (signée) donne la
 * direction et l'ampleur de l'effet, la couleur donne si la variable avait
 * une valeur basse (bleu) ou haute (rouge) pour cette observation — ce que
 * le barres (moyenne des |valeurs|) ne montre pas. */
export function ShapBeeswarmChart({ beeswarm }: { beeswarm: ShapBeeswarm }) {
  const classNames = Object.keys(beeswarm);
  const [selected, setSelected] = useState(classNames[0] ?? "");
  const points = beeswarm[selected] ?? beeswarm[classNames[0]] ?? [];

  const featureOrder = useMemo(() => {
    const seen: string[] = [];
    for (const p of points) {
      if (!seen.includes(p.feature)) seen.push(p.feature);
    }
    return seen;
  }, [points]);

  const rows = useMemo(() => buildRows(points, featureOrder), [points, featureOrder]);

  if (points.length === 0 || featureOrder.length === 0) return null;

  return (
    <div>
      {classNames.length > 1 && (
        <div className="flex gap-1.5 mb-2 flex-wrap">
          {classNames.map((name) => (
            <button
              key={name}
              type="button"
              onClick={() => setSelected(name)}
              className={`text-xs px-2.5 py-1 rounded-full border transition-colors ${
                name === (selected || classNames[0])
                  ? "border-primary bg-primary/10 text-primary"
                  : "border-border text-muted-foreground hover:border-input"
              }`}
            >
              {name}
            </button>
          ))}
        </div>
      )}
      <ResponsiveContainer width="100%" height={Math.max(180, featureOrder.length * 34)}>
        <ScatterChart margin={{ left: 8, right: 12, top: 8, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID_STROKE} horizontal={false} />
          <XAxis
            type="number"
            dataKey="x"
            tick={CHART_TICK_STYLE}
            label={{
              value: "Valeur SHAP (impact sur la prédiction)",
              position: "insideBottom",
              offset: -5,
              ...CHART_TICK_STYLE,
            }}
          />
          <YAxis
            type="number"
            dataKey="y"
            domain={[-0.5, featureOrder.length - 0.5]}
            ticks={featureOrder.map((_, i) => i)}
            tickFormatter={(v) => featureOrder[Math.round(Number(v))] ?? ""}
            tick={CHART_TICK_STYLE}
            width={140}
          />
          <ReferenceLine x={0} stroke={CHART_REFERENCE_STROKE} strokeDasharray="4 4" />
          <RechartsTooltip content={<BeeswarmTooltip />} cursor={{ strokeDasharray: "3 3" }} />
          <Scatter
            data={rows}
            isAnimationActive={false}
            shape={(props: { cx?: number; cy?: number; payload?: BeeswarmRow }) => (
              <circle cx={props.cx} cy={props.cy} r={3} fill={props.payload?.color} fillOpacity={0.75} />
            )}
          />
        </ScatterChart>
      </ResponsiveContainer>
      <div className="flex items-center gap-2 mt-2 text-caption text-muted-foreground">
        <span>Valeur de la variable :</span>
        <span
          className="inline-block w-16 h-2 rounded-full"
          style={{ background: `linear-gradient(to right, var(${CHART_BEESWARM_LOW_VAR}), var(${CHART_BEESWARM_HIGH_VAR}))` }}
        />
        <span>basse → haute</span>
      </div>
    </div>
  );
}

/** Importance par permutation (Lot Explicabilité globale) — chute du score
 * du modèle quand une variable est mélangée aléatoirement dans le test,
 * moyenne ± écart-type sur plusieurs répétitions (barre d'erreur). Mesure
 * indépendante du SHAP, sert à le recouper. */
export function PermutationImportanceChart({ features }: { features: PermutationImportanceFeature[] }) {
  if (features.length === 0) return null;
  const data = [...features.slice(0, 10)].reverse(); // Recharts vertical bar : bas → haut

  return (
    <ResponsiveContainer width="100%" height={Math.max(160, data.length * 32)}>
      <BarChart data={data} layout="vertical" margin={{ left: 8, right: 24, top: 8, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID_STROKE} horizontal={false} />
        <XAxis
          type="number"
          tick={CHART_TICK_STYLE}
          label={{
            value: "Chute de score quand la variable est mélangée",
            position: "insideBottom",
            offset: -5,
            ...CHART_TICK_STYLE,
          }}
        />
        <YAxis type="category" dataKey="feature" tick={CHART_TICK_STYLE} width={140} />
        <ReferenceLine x={0} stroke={CHART_REFERENCE_STROKE} strokeDasharray="4 4" />
        <RechartsTooltip {...CHART_TOOLTIP_STYLE} formatter={(v) => Number(v).toFixed(4)} />
        <Bar dataKey="importance_mean" fill={CHART_COLOR_PRIMARY} radius={[0, 4, 4, 0]} isAnimationActive={false}>
          <ErrorBar dataKey="importance_std" width={4} strokeWidth={1.5} stroke={CHART_REFERENCE_STROKE} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
