import type { ButtonHTMLAttributes } from "react";

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "sm" | "md";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
}

const VARIANT_CLASSES: Record<Variant, string> = {
  // Dégradé de marque (pas un bleu plat) — même traitement que les CTA de
  // l'auth (.bg-brand-gradient) : cohérence visuelle bout en bout sur le
  // bouton d'action principal, qu'il s'agisse de se connecter ou de lancer
  // un entraînement.
  primary: "bg-brand-gradient hover:brightness-110 text-white shadow-sm shadow-primary/25 transition-[filter]",
  secondary: "bg-secondary hover:bg-muted text-secondary-foreground border border-border",
  ghost: "bg-transparent hover:bg-muted text-muted-foreground border border-border",
  danger: "bg-destructive/10 hover:bg-destructive/15 text-destructive border border-destructive/20",
};

const SIZE_CLASSES: Record<Size, string> = {
  sm: "text-xs px-2.5 py-1.5 rounded-lg",
  md: "text-sm px-4 py-2 rounded-xl",
};

export function Button({
  variant = "primary",
  size = "md",
  className = "",
  ...rest
}: ButtonProps) {
  return (
    <button
      className={`font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed inline-flex items-center justify-center gap-1.5 ${VARIANT_CLASSES[variant]} ${SIZE_CLASSES[size]} ${className}`}
      {...rest}
    />
  );
}
