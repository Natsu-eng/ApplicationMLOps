import { Fragment } from "react";

/** Heatmap générique en grille CSS — pas de dépendance graphique dédiée pour
 * ça : donne un contrôle total sur le style (cohérent avec le reste du
 * système de design) pour un composant qui n'a que deux vrais usages
 * (corrélations, matrice de confusion). */

interface HeatmapProps {
  xLabels: string[];
  yLabels: string[];
  matrix: (number | null)[][];
  /** "diverging" : centré sur 0, bleu↔rose (corrélations). "sequential" :
   * 0→max, dégradé bleu (comptages, ex. matrice de confusion). */
  variant?: "diverging" | "sequential";
  formatValue?: (value: number) => string;
}

function colorFor(value: number | null, variant: "diverging" | "sequential", max: number): string {
  if (value === null) return "rgba(100, 116, 139, 0.12)"; // slate-500 translucide — valeur manquante
  if (variant === "diverging") {
    // value ∈ [-1, 1] : rose doux pour le négatif, bleu de marque pour le positif — intensité
    // plafonnée (jamais > ~0.65) pour que le texte de cellule (slate-800) reste
    // lisible même à intensité maximale, contrairement au thème sombre d'origine.
    const intensity = Math.min(Math.abs(value), 1);
    return value >= 0
      ? `rgba(29, 78, 216, ${0.1 + intensity * 0.55})` // blue-700 (CHART_COLOR_PRIMARY)
      : `rgba(219, 39, 119, ${0.1 + intensity * 0.45})`; // pink-600, discret
  }
  const intensity = max > 0 ? Math.min(value / max, 1) : 0;
  return `rgba(29, 78, 216, ${0.1 + intensity * 0.55})`; // blue-700 (CHART_COLOR_PRIMARY)
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
    <div className="overflow-x-auto rounded-lg border border-slate-100">
      <div
        className="inline-grid gap-px bg-slate-100 p-px"
        style={{ gridTemplateColumns: `auto repeat(${xLabels.length}, minmax(34px, 1fr))` }}
      >
        <div className="bg-white" />
        {xLabels.map((label) => (
          <div
            key={label}
            className={`bg-white text-[9px] text-slate-500 px-0.5 pb-1 truncate ${
              rotateLabels ? "[writing-mode:vertical-rl] rotate-180 text-left max-h-20" : "text-center"
            }`}
            title={label}
          >
            {label}
          </div>
        ))}
        {yLabels.map((yLabel, i) => (
          <Fragment key={yLabel}>
            <div className="bg-white text-[9px] text-slate-500 pr-1.5 flex items-center justify-end truncate max-w-24" title={yLabel}>
              {yLabel}
            </div>
            {matrix[i].map((value, j) => (
              <div
                key={`${yLabel}-${xLabels[j]}`}
                className="aspect-square min-h-8 flex items-center justify-center text-[10px] text-slate-800 tabular-nums"
                style={{ backgroundColor: colorFor(value, variant, max) }}
                title={`${yLabel} × ${xLabels[j]} : ${value ?? "—"}`}
              >
                {value !== null ? format(value) : "—"}
              </div>
            ))}
          </Fragment>
        ))}
      </div>
    </div>
  );
}
