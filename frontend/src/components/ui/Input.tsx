import type { InputHTMLAttributes } from "react";

/** Champ de saisie standard du système de design — même style partout (formulaires auth, équipe...). */
export function Input({ className = "", ...rest }: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      className={`w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-teal-500/40 focus:border-teal-500/50 transition-colors ${className}`}
      {...rest}
    />
  );
}
