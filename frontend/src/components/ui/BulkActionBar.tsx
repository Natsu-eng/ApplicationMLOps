import type { ReactNode } from "react";
import { X } from "lucide-react";

/** Barre d'actions groupées — ancrée en bas de l'écran, centrée
 * horizontalement, apparaît uniquement quand au moins une ligne est
 * sélectionnée (`Table.tsx`, prop `selectable`). Motif SaaS standard
 * (Gmail/Notion/Linear) construit une seule fois ici plutôt que
 * dupliqué à chaque page qui a besoin d'une action groupée — jusqu'ici
 * la sélection multiple de `Table.tsx` n'était câblée qu'à une action de
 * comparaison (`TrainingHistory.tsx`), jamais à une action destructive.
 *
 * `position: fixed` (pas `sticky`) : la barre doit rester visible même si
 * le tableau qui la déclenche est lui-même dans un conteneur au défilement
 * propre (`Table.tsx` a son propre `overflow-y-auto` interne) — `sticky`
 * s'ancrerait au mauvais ancêtre de défilement dans ce cas. */
export function BulkActionBar({
  count,
  onClear,
  children,
}: {
  count: number;
  onClear: () => void;
  children: ReactNode;
}) {
  if (count === 0) return null;
  return (
    <div
      role="toolbar"
      aria-label={`${count} ligne${count > 1 ? "s" : ""} sélectionnée${count > 1 ? "s" : ""}`}
      className="bulk-bar-enter fixed bottom-5 left-1/2 z-50 flex items-center gap-3 rounded-full border border-border bg-popover px-4 py-2.5 shadow-overlay"
    >
      <span className="text-sm font-medium text-foreground whitespace-nowrap">
        {count} sélectionné{count > 1 ? "s" : ""}
      </span>
      <div className="h-4 w-px bg-border" aria-hidden="true" />
      <div className="flex items-center gap-2">{children}</div>
      <button
        type="button"
        onClick={onClear}
        aria-label="Désélectionner tout"
        className="flex-shrink-0 text-muted-foreground hover:text-foreground transition-colors rounded-control focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)]"
      >
        <X size={16} />
      </button>
    </div>
  );
}
