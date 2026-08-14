import { useState } from "react";

/** Motif de confirmation à deux clics (pas de `window.confirm` natif, pour
 * rester cohérent avec le reste de l'app) — extrait de deux implémentations
 * indépendantes (Dashboard.tsx, Training.tsx) — AUDIT_ROADMAP.md, H14.
 *
 * Premier clic sur un item : arme la confirmation. Deuxième clic sur LE
 * MÊME item : exécute `action`. Un clic sur un autre item, ou le curseur qui
 * quitte la zone (`onMouseLeave` recommandé côté appelant), désarme.
 *
 * @example
 * const confirm = useConfirmAction<number>();
 * <button
 *   onClick={() => confirm.trigger(job.id, () => handleDelete(job.id))}
 *   onMouseLeave={confirm.reset}
 * >
 *   {confirm.isPending(job.id) ? "Confirmer ?" : "Supprimer"}
 * </button>
 */
export function useConfirmAction<Id>() {
  const [pendingId, setPendingId] = useState<Id | null>(null);

  function isPending(id: Id): boolean {
    return pendingId === id;
  }

  function trigger(id: Id, action: () => void): void {
    if (pendingId !== id) {
      setPendingId(id);
      return;
    }
    setPendingId(null);
    action();
  }

  function reset(): void {
    setPendingId(null);
  }

  return { isPending, trigger, reset };
}
