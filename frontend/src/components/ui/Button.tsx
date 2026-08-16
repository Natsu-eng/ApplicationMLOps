import type { ButtonHTMLAttributes } from "react";
import { Loader2 } from "lucide-react";

type Variant = "primary" | "secondary" | "ghost" | "danger" | "destructive" | "link";
type Size = "sm" | "md";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  /** État de chargement intégré (Lot 2A) — remplace le motif répété
   * `isSubmitting ? "Lancement…" : "Lancer"` : le bouton affiche un
   * spinner et se désactive lui-même, `children` reste le libellé normal
   * (jamais reformulé au chargement — cohérent avec le principe "jamais de
   * texte inventé" du reste du projet). */
  loading?: boolean;
}

const VARIANT_CLASSES: Record<Variant, string> = {
  // Dégradé de marque (pas un bleu plat) — même traitement que les CTA de
  // l'auth (.bg-brand-gradient) : cohérence visuelle bout en bout sur le
  // bouton d'action principal, qu'il s'agisse de se connecter ou de lancer
  // un entraînement.
  primary: "bg-brand-gradient hover:brightness-110 text-white shadow-control shadow-primary/25 transition-[filter]",
  secondary: "bg-secondary hover:bg-muted text-secondary-foreground border border-border",
  ghost: "bg-transparent hover:bg-muted text-muted-foreground border border-transparent",
  danger: "bg-destructive/10 hover:bg-destructive/15 text-destructive border border-destructive/20",
  // "destructive" : action irréversible engagée directement (pas juste un
  // libellé teinté de rouge comme "danger", qui reste une action secondaire
  // discrète) — fond plein, même poids visuel que "primary" mais en rouge.
  // bg-destructive-solid (pas bg-destructive) : seul le token "solid"
  // garantit 4.5:1 pour le texte blanc dessus en mode sombre — voir
  // index.css et DECISIONS.md.
  destructive: "bg-destructive-solid hover:brightness-110 text-destructive-foreground shadow-control transition-[filter]",
  link: "bg-transparent text-primary hover:underline underline-offset-4 p-0 h-auto",
};

const SIZE_CLASSES: Record<Size, string> = {
  sm: "text-caption px-2.5 py-1.5 rounded-control",
  md: "text-body px-4 py-2 rounded-control",
};

export function Button({
  variant = "primary",
  size = "md",
  loading = false,
  disabled,
  className = "",
  children,
  ...rest
}: ButtonProps) {
  const sizeClass = variant === "link" ? "" : SIZE_CLASSES[size];
  return (
    <button
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      className={`font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed inline-flex items-center justify-center gap-1.5 ${VARIANT_CLASSES[variant]} ${sizeClass} ${className}`}
      {...rest}
    >
      {loading && <Loader2 size={14} className="animate-spin" aria-hidden="true" />}
      {children}
    </button>
  );
}
