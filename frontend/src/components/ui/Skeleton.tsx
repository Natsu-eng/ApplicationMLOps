import type { HTMLAttributes, ReactNode } from "react";

/** Bloc de silhouette élémentaire — reprend la FORME du contenu réel (passer
 * `className` avec la largeur/hauteur exactes de ce qu'il remplace), jamais
 * un simple "Chargement…" (SPEC-UI.md §6.6). `motion-reduce:animate-none` :
 * l'animation de balayage n'est jamais indispensable à la compréhension.
 * `aria-hidden` — le statut de chargement est porté par `SkeletonGroup`
 * (`role="status"`) qui doit envelopper un ensemble de blocs, jamais par
 * chaque bloc individuellement (un lecteur d'écran n'a pas à énumérer 5
 * rectangles gris). */
export function Skeleton({ className = "", ...rest }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      aria-hidden="true"
      className={`rounded bg-muted animate-pulse motion-reduce:animate-none ${className}`}
      {...rest}
    />
  );
}

/** Conteneur accessible pour un groupe de `Skeleton` — `role="status"` porté
 * une seule fois ici, pas sur chaque bloc. `label` est lu par un lecteur
 * d'écran mais jamais affiché (le rendu visuel reste les blocs eux-mêmes). */
export function SkeletonGroup({
  children,
  label = "Chargement en cours",
  className = "",
}: {
  children: ReactNode;
  label?: string;
  className?: string;
}) {
  return (
    <div role="status" aria-label={label} className={className}>
      {children}
    </div>
  );
}
