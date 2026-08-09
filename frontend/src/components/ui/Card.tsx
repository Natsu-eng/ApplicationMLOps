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
      className={`rounded-2xl border border-slate-800/80 bg-slate-900/70 backdrop-blur-sm shadow-lg shadow-black/20 ${
        interactive
          ? "transition-all duration-200 hover:border-slate-700 hover:shadow-xl hover:shadow-black/30 hover:-translate-y-0.5"
          : ""
      } ${className}`}
      {...rest}
    >
      {children}
    </div>
  );
}
