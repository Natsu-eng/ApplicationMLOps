import type { HTMLAttributes, ReactNode } from "react";

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode;
  /** Légère élévation au survol — pour les cartes cliquables (ex : une carte dataset). */
  interactive?: boolean;
}

/** Carte de base du système de design — verre dépoli sur le fond dégradé du corps de page. */
export function Card({ children, interactive = false, className = "", ...rest }: CardProps) {
  return (
    <div
      className={`rounded-2xl border border-slate-200 bg-white shadow-sm ${
        interactive
          ? "transition-all duration-200 hover:border-slate-300 hover:shadow-md hover:-translate-y-0.5"
          : ""
      } ${className}`}
      {...rest}
    >
      {children}
    </div>
  );
}
