import { Fragment } from "react";
import {
  HEATMAP_DIVERGING_NEGATIVE_STEPS,
  HEATMAP_DIVERGING_POSITIVE_STEPS,
  HEATMAP_MISSING_FILL,
  HEATMAP_NEUTRAL_STEP,
  HEATMAP_SEQUENTIAL_STEPS,
  heatmapStepIndex,
  heatmapTextColor,
} from "../../theme/charts";

/** Heatmap générique en grille CSS — pas de dépendance graphique dédiée pour
 * ça : donne un contrôle total sur le style (cohérent avec le reste du
 * système de design) pour un composant qui n'a que deux vrais usages
 * (corrélations, matrice de confusion). Couleurs par paliers discrets
 * (`theme/charts.ts`, méthode skill dataviz), pas un dégradé d'opacité ad
 * hoc — chaque palier reste identifiable, et l'encre du texte s'adapte au
 * palier (blanc sur les teintes soutenues) plutôt qu'un slate-800 fixe qui
 * devenait peu lisible aux intensités élevées. */

interface HeatmapProps {
  xLabels: string[];
  yLabels: string[];
  matrix: (number | null)[][];
  /** "diverging" : centré sur 0, bleu (positif) ↔ rouge (négatif), point
   * neutre gris (corrélations). "sequential" : 0→max, dégradé bleu clair→
   * foncé (comptages, ex. matrice de confusion). */
  variant?: "diverging" | "sequential";
  formatValue?: (value: number) => string;
}

function cellStyle(value: number | null, variant: "diverging" | "sequential", max: number): { background: string; color: string } {
  if (value === null) return { background: HEATMAP_MISSING_FILL, color: "#94a3b8" };
  if (variant === "diverging") {
    if (Math.abs(value) < 0.02) return { background: HEATMAP_NEUTRAL_STEP, color: "#0f172a" };
    const steps = value >= 0 ? HEATMAP_DIVERGING_POSITIVE_STEPS : HEATMAP_DIVERGING_NEGATIVE_STEPS;
    const idx = heatmapStepIndex(Math.abs(value), steps.length);
    return { background: steps[idx], color: heatmapTextColor(idx) };
  }
  const intensity = max > 0 ? value / max : 0;
  const idx = heatmapStepIndex(intensity, HEATMAP_SEQUENTIAL_STEPS.length);
  return { background: HEATMAP_SEQUENTIAL_STEPS[idx], color: heatmapTextColor(idx) };
}

export function Heatmap({ xLabels, yLabels, matrix, variant = "sequential", formatValue }: HeatmapProps) {
  const max = variant === "sequential" ? Math.max(1, ...matrix.flat().map((v) => v ?? 0)) : 1;
  const format = formatValue ?? ((v: number) => (Number.isInteger(v) ? String(v) : v.toFixed(2)));

  // Cellules compactes (32px, pas 48+) + libellés de colonnes pivotés dès
  // qu'il y en a plus de 6 — au-delà, des libellés horizontaux tronqués
  // devenaient illisibles sur une matrice large (ex. 8-10 variables
  // numériques), signalé en usage réel.
  const rotateLabels = xLabels.length > 6;

  return (
    <div className="overflow-x-auto rounded-lg border border-border">
      <div
        className="inline-grid gap-px bg-border p-px"
        style={{ gridTemplateColumns: `auto repeat(${xLabels.length}, minmax(34px, 1fr))` }}
      >
        <div className="bg-card" />
        {xLabels.map((label) => (
          <div
            key={label}
            className={`bg-card text-[9px] text-muted-foreground px-0.5 pb-1 truncate ${
              rotateLabels ? "[writing-mode:vertical-rl] rotate-180 text-left max-h-20" : "text-center"
            }`}
            title={label}
          >
            {label}
          </div>
        ))}
        {yLabels.map((yLabel, i) => (
          <Fragment key={yLabel}>
            <div className="bg-card text-[9px] text-muted-foreground pr-1.5 flex items-center justify-end truncate max-w-24" title={yLabel}>
              {yLabel}
            </div>
            {matrix[i].map((value, j) => {
              const style = cellStyle(value, variant, max);
              return (
                <div
                  key={`${yLabel}-${xLabels[j]}`}
                  className="aspect-square min-h-8 flex items-center justify-center text-[10px] font-medium tabular-nums transition-colors"
                  style={{ backgroundColor: style.background, color: style.color }}
                  title={`${yLabel} × ${xLabels[j]} : ${value ?? "—"}`}
                >
                  {value !== null ? format(value) : "—"}
                </div>
              );
            })}
          </Fragment>
        ))}
      </div>
    </div>
  );
}
