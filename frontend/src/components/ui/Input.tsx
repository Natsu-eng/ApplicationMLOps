import type { InputHTMLAttributes } from "react";

/** Champ de saisie standard du système de design — même style partout (formulaires auth, équipe...). */
export function Input({ className = "", ...rest }: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      className={`w-full rounded-lg border border-input bg-card px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring/40 focus:border-ring/50 transition-colors disabled:opacity-60 disabled:cursor-not-allowed ${className}`}
      {...rest}
    />
  );
}
