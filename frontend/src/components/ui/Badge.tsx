import type { ReactNode } from "react";

type Variant = "neutral" | "accent" | "success" | "warning" | "danger" | "primary";

const VARIANT_CLASSES: Record<Variant, string> = {
  neutral: "bg-muted text-muted-foreground border border-border",
  accent: "bg-accent text-accent-foreground border border-accent",
  primary: "bg-primary/10 text-primary border border-primary/20",
  success: "bg-success/10 text-success border border-success/20",
  warning: "bg-warning/10 text-warning border border-warning/20",
  danger: "bg-destructive/10 text-destructive border border-destructive/20",
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
