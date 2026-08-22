import { RadialBar, RadialBarChart, ResponsiveContainer } from "recharts";

/** Jauge radiale — une valeur 0–100 (ou bornes personnalisées), un arc coloré
 * par seuil (succès/attention/danger). Comme `Sparkline`, pensée pour un
 * usage compact (carte de verdict, tuile) : l'alternative textuelle est le
 * chiffre lui-même, déjà affiché en `.num` au centre — jamais caché,
 * contrairement au SVG de l'arc qui est purement décoratif. */
export function Gauge({
  value,
  max = 100,
  label,
  size = 96,
  thresholds = { warning: 60, danger: 40 },
}: {
  value: number;
  max?: number;
  label?: string;
  size?: number;
  /** Sous ce seuil = danger, sous le suivant = attention, au-dessus = succès. */
  thresholds?: { warning: number; danger: number };
}) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100));
  const color = pct < thresholds.danger ? "var(--danger)" : pct < thresholds.warning ? "var(--warning)" : "var(--success)";
  const data = [{ value: pct, fill: color }];

  return (
    <div className="relative inline-flex flex-col items-center" style={{ width: size }}>
      <div aria-hidden="true" style={{ width: size, height: size }}>
        <ResponsiveContainer width="100%" height="100%">
          <RadialBarChart
            data={data}
            innerRadius="70%"
            outerRadius="100%"
            startAngle={90}
            endAngle={-270}
            barSize={size * 0.14}
            accessibilityLayer={false}
          >
            <RadialBar dataKey="value" background={{ fill: "var(--muted)" }} cornerRadius={99} isAnimationActive={false} />
          </RadialBarChart>
        </ResponsiveContainer>
      </div>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="num text-h3 text-foreground" aria-label={`${label ? label + " : " : ""}${Math.round(pct)} sur ${max === 100 ? 100 : max}`}>
          {Math.round(pct)}
          {max === 100 ? "%" : ""}
        </span>
      </div>
      {label && <p className="text-caption text-muted-foreground mt-1 text-center">{label}</p>}
    </div>
  );
}
