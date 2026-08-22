import type { LucideIcon } from "lucide-react";

export interface SegmentedOption<T extends string> {
  id: T;
  label: string;
  icon?: LucideIcon;
  disabled?: boolean;
}

/** Contrôle segmenté — bascule entre quelques modes mutuellement exclusifs
 * (ex. vue "Tableau" / "Graphique"), jamais de la navigation entre pages
 * (ça, c'est `Tabs`, `role="tablist"`). Ici `role="radiogroup"` : un seul
 * choix actif à la fois, comme un groupe de boutons radio. */
export function Segmented<T extends string>({
  options,
  value,
  onChange,
  size = "md",
  "aria-label": ariaLabel,
}: {
  options: SegmentedOption<T>[];
  value: T;
  onChange: (id: T) => void;
  size?: "sm" | "md";
  "aria-label": string;
}) {
  const padY = size === "sm" ? "py-1" : "py-1.5";
  const textSize = size === "sm" ? "text-caption" : "text-body";

  return (
    <div role="radiogroup" aria-label={ariaLabel} className="inline-flex items-center gap-1 rounded-control bg-muted p-1">
      {options.map((opt) => {
        const active = value === opt.id;
        const Icon = opt.icon;
        return (
          <button
            key={opt.id}
            type="button"
            role="radio"
            aria-checked={active}
            disabled={opt.disabled}
            onClick={() => onChange(opt.id)}
            className={`flex-shrink-0 flex items-center gap-1.5 rounded-control px-3 ${padY} ${textSize} font-medium whitespace-nowrap transition-colors duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-1 focus-visible:ring-[var(--accent)] disabled:opacity-[.42] disabled:cursor-not-allowed ${
              active ? "bg-card text-primary shadow-control" : "text-muted-foreground hover:text-foreground"
            }`}
          >
            {Icon && <Icon size={14} aria-hidden="true" />}
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}
