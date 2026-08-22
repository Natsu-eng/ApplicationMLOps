import type { ReactNode } from "react";
import type { LucideIcon } from "lucide-react";
import { BarChart3 } from "lucide-react";

/** État vide / squelette de graphique (Lot 2A, AUDIT_DATALAB_2026-08-16.md
 * §J.4) — jusqu'ici chaque graphe affichait silencieusement une zone vide
 * (pas de donnée) ou clignotait d'un coup au chargement (pas de squelette),
 * jamais un état intermédiaire lisible. `height` doit correspondre à la
 * hauteur du `ResponsiveContainer` réel qu'il remplace, pour ne pas faire
 * sauter la mise en page à l'arrivée des données. */
export function ChartSkeleton({ height = 240 }: { height?: number }) {
  return (
    <div
      className="rounded-card bg-muted/40 animate-pulse motion-reduce:animate-none flex items-center justify-center"
      style={{ height }}
      role="status"
      aria-label="Chargement du graphique"
    >
      <BarChart3 size={24} className="text-muted-foreground/40" aria-hidden="true" />
    </div>
  );
}

export function EmptyChartState({
  height = 240,
  icon: Icon = BarChart3,
  message = "Aucune donnée à afficher.",
  action,
}: {
  height?: number;
  icon?: LucideIcon;
  message?: string;
  /** Ce qu'il y a à faire (SPEC-UI.md §6.7 : « vide » n'est jamais un
   * constat seul) — ex. un bouton "Importer un jeu de données". Optionnel :
   * certains graphes vides n'ont vraiment aucune action possible (ex. une
   * corrélation qui ne peut exister qu'après un premier entraînement). */
  action?: ReactNode;
}) {
  return (
    <div
      className="rounded-card border border-dashed border-border/70 flex flex-col items-center justify-center gap-2 text-center px-4"
      style={{ height }}
    >
      <Icon size={20} className="text-muted-foreground/50" aria-hidden="true" />
      <p className="text-caption text-muted-foreground">{message}</p>
      {action && <div className="mt-1">{action}</div>}
    </div>
  );
}
