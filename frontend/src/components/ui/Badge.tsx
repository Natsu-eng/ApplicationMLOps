import type { ReactNode } from "react";

type Variant = "neutral" | "accent" | "success" | "warning" | "danger" | "primary";

// Opacité de fond volontairement faible (/4, pas /10) : --success/--warning/
// --destructive/--primary ne sont garantis ≥4,5:1 QUE sur les 3 fonds neutres
// du thème (canvas/surface/raised, voir themes.css) — un lavis de ces teintes
// composite un 4ᵉ fond légèrement différent, sous le seuil en ivoire/
// porcelaine (les 2 thèmes au minimum de contraste le plus serré, 4.52/
// 4.55:1) quand le badge est en plus posé sur une ligne de tableau
// elle-même teintée (double lavis composé) — /6 puis /5 encore
// insuffisants (4.47:1 mesuré), /4 revérifié par axe-core dans ce cas
// précis, voir _design/JOURNAL.md Lot 2.
const VARIANT_CLASSES: Record<Variant, string> = {
  neutral: "bg-muted text-muted-foreground border border-border",
  accent: "bg-accent text-accent-foreground border border-accent",
  primary: "bg-primary/4 text-primary border border-primary/20",
  success: "bg-success/4 text-success border border-success/20",
  warning: "bg-warning/4 text-warning border border-warning/20",
  danger: "bg-destructive/4 text-destructive border border-destructive/20",
};

const DOT_CLASSES: Record<Variant, string> = {
  neutral: "bg-muted-foreground",
  accent: "bg-accent-foreground",
  primary: "bg-primary",
  success: "bg-success",
  warning: "bg-warning",
  danger: "bg-destructive",
};

export function Badge({
  variant = "neutral",
  dot = false,
  pulse = false,
  children,
}: {
  variant?: Variant;
  /** Puce colorée avant le texte — statuts (Terminé/En cours/Échec...). */
  dot?: boolean;
  /** Puce clignotante — réservé aux statuts réellement en cours. */
  pulse?: boolean;
  children: ReactNode;
}) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 whitespace-nowrap text-overline font-medium px-2 py-0.5 rounded-full ${VARIANT_CLASSES[variant]}`}
    >
      {dot && (
        <span className={`size-1.5 rounded-full ${DOT_CLASSES[variant]} ${pulse ? "animate-pulse" : ""}`} />
      )}
      {children}
    </span>
  );
}
