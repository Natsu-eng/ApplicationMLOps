import type { HTMLAttributes, ReactNode } from "react";
import type { Density } from "../../theme/density";
import { CARD_PADDING } from "../../theme/density";

type Variant = "elevated" | "outlined" | "interactive" | "stat";

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode;
  /** @deprecated utiliser `variant="interactive"` — conservé pour ne pas
   * casser les appels existants (Lot 2A ne modifie aucune page métier). */
  interactive?: boolean;
  variant?: Variant;
  density?: Density;
  header?: ReactNode;
  footer?: ReactNode;
  /** Carte désactivée (SPEC-UI.md §6.5) — 42% d'opacité, `cursor-not-allowed`,
   * `aria-disabled`. Reste affichée (contrairement à ne pas la rendre du
   * tout) : l'utilisateur voit CE QUI existe mais n'est pas actionnable. */
  disabled?: boolean;
}

const VARIANT_CLASSES: Record<Variant, string> = {
  // elevated : carte "flottante" par défaut — bordure fine + ombre discrète.
  elevated: "border border-border/70 bg-card shadow-card",
  // outlined : pas d'ombre du tout, seulement la bordure — pour empiler
  // plusieurs cartes sans que l'ombre de l'une n'empiète sur la suivante
  // (listes denses).
  outlined: "border border-border bg-card",
  interactive:
    "border border-border/70 bg-card shadow-card transition-all duration-200 hover:border-primary/30 hover:shadow-overlay hover:-translate-y-0.5 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-[var(--accent)]",
  // stat : fond légèrement teinté (accent) plutôt que blanc — se distingue
  // d'une carte de contenu au premier coup d'œil (StatTile).
  stat: "border border-border/70 bg-card shadow-card",
};

/** Carte de base du système de design (Lot 2A) — 4 variantes + slots
 * `header`/`footer` normalisés (AUDIT_DATALAB_2026-08-16.md, §J.4) :
 * chaque page réinventait jusqu'ici son propre en-tête de carte. */
export function Card({
  children,
  interactive = false,
  variant,
  density = "default",
  header,
  footer,
  disabled = false,
  className = "",
  ...rest
}: CardProps) {
  const resolvedVariant = variant ?? (interactive ? "interactive" : "elevated");
  const padding = CARD_PADDING[density];
  const disabledClass = disabled ? "opacity-[.42] cursor-not-allowed pointer-events-none" : "";

  if (!header && !footer) {
    return (
      <div
        className={`rounded-card ${VARIANT_CLASSES[resolvedVariant]} ${padding} ${disabledClass} ${className}`}
        aria-disabled={disabled || undefined}
        {...rest}
      >
        {children}
      </div>
    );
  }

  return (
    <div
      className={`rounded-card overflow-hidden ${VARIANT_CLASSES[resolvedVariant]} ${disabledClass} ${className}`}
      aria-disabled={disabled || undefined}
      {...rest}
    >
      {header && <div className={`${padding} border-b border-border/60`}>{header}</div>}
      <div className={padding}>{children}</div>
      {footer && <div className={`${padding} border-t border-border/60`}>{footer}</div>}
    </div>
  );
}
