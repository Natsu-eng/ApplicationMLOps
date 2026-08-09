import type { ButtonHTMLAttributes } from "react";

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "sm" | "md";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
}

const VARIANT_CLASSES: Record<Variant, string> = {
  primary: "bg-teal-500 hover:bg-teal-400 text-slate-950 shadow-sm shadow-teal-500/20",
  secondary: "bg-slate-800 hover:bg-slate-700 text-slate-100 border border-slate-700",
  ghost: "bg-transparent hover:bg-slate-800/60 text-slate-300 border border-slate-800",
  danger: "bg-rose-500/10 hover:bg-rose-500/20 text-rose-300 border border-rose-500/30",
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
