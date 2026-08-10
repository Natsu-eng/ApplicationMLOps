import type { ButtonHTMLAttributes } from "react";

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "sm" | "md";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
}

const VARIANT_CLASSES: Record<Variant, string> = {
  primary: "bg-teal-600 hover:bg-teal-700 text-white shadow-sm shadow-teal-600/20",
  secondary: "bg-slate-100 hover:bg-slate-200 text-slate-900 border border-slate-200",
  ghost: "bg-transparent hover:bg-slate-100 text-slate-600 border border-slate-200",
  danger: "bg-rose-50 hover:bg-rose-100 text-rose-700 border border-rose-200",
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
