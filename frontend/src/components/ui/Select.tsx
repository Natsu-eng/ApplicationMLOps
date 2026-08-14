import type { SelectHTMLAttributes } from "react";

/** Liste déroulante standard du système de design — même style que `Input`.
 * Extrait de 6+ occurrences de la même classe recopiée telle quelle
 * (Training.tsx, EdaModal.tsx...) — AUDIT_ROADMAP.md, H14. */
export function Select({ className = "", ...rest }: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      className={`w-full rounded-lg border border-input bg-card px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring/40 focus:border-ring/50 transition-colors disabled:opacity-60 disabled:cursor-not-allowed ${className}`}
      {...rest}
    />
  );
}
