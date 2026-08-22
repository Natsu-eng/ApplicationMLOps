/** Barre de progression déterminée — jauge d'entraînement, import de
 * fichier... `label` optionnel affiché au-dessus, valeur en chiffre `.num`
 * (chasse fixe, SPEC-UI.md §3) alignée à droite. */
export function ProgressBar({
  value,
  max = 100,
  label,
  size = "md",
  className = "",
}: {
  value: number;
  max?: number;
  label?: string;
  size?: "sm" | "md";
  className?: string;
}) {
  const pct = Math.min(100, Math.max(0, max > 0 ? (value / max) * 100 : 0));
  const height = size === "sm" ? "h-1.5" : "h-2";

  return (
    <div className={className}>
      {label && (
        <div className="flex items-center justify-between text-caption text-muted-foreground mb-1.5">
          <span>{label}</span>
          <span className="num">{Math.round(pct)}%</span>
        </div>
      )}
      <div
        role="progressbar"
        aria-valuenow={Math.round(pct)}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={label ?? "Progression"}
        className={`w-full ${height} rounded-full bg-muted overflow-hidden`}
      >
        <div
          className="h-full rounded-full bg-primary transition-[width] duration-200 motion-reduce:transition-none"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

/** Variante indéterminée — durée inconnue (ex. file d'attente avant qu'un
 * job démarre). Bande qui glisse, jamais un pourcentage inventé. */
export function IndeterminateProgressBar({ label, size = "md", className = "" }: { label?: string; size?: "sm" | "md"; className?: string }) {
  const height = size === "sm" ? "h-1.5" : "h-2";
  return (
    <div className={className}>
      {label && <p className="text-caption text-muted-foreground mb-1.5">{label}</p>}
      <div
        role="progressbar"
        aria-label={label ?? "Progression"}
        className={`w-full ${height} rounded-full bg-muted overflow-hidden relative`}
      >
        <div className="absolute inset-y-0 w-1/3 rounded-full bg-primary animate-[indeterminate-progress_1.4s_ease-in-out_infinite] motion-reduce:animate-none motion-reduce:w-full motion-reduce:opacity-40" />
      </div>
    </div>
  );
}
