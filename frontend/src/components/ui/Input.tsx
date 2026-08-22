import type { InputHTMLAttributes } from "react";

/** Champ de saisie standard du système de design — même style partout
 * (formulaires auth, équipe...). Bordure rouge automatique quand
 * `aria-invalid="true"` (posé par `Field.tsx`) — jusqu'ici seul le message
 * d'erreur en dessous changeait, le champ lui-même restait visuellement
 * "normal" (SPEC-UI.md §6.8 : l'erreur doit être visible sur le contrôle
 * lui-même, pas seulement dans le texte qui l'accompagne). */
export function Input({ className = "", ...rest }: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      className={`w-full rounded-lg border border-input bg-card px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring/40 focus:border-ring/50 transition-colors disabled:opacity-[.42] disabled:cursor-not-allowed aria-[invalid=true]:border-destructive aria-[invalid=true]:focus:ring-destructive/30 ${className}`}
      {...rest}
    />
  );
}
