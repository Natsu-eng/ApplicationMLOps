/** Densité partagée — Lot 2A (AUDIT_DATALAB_2026-08-16.md, §J.4). Trois
 * réglages, consommés par `Table`/`Card` : ce produit affiche beaucoup de
 * chiffres, `compact` doit rester réellement lisible, pas un simple
 * "moins d'air" qui écrase le contenu. Un seul point de vérité pour les
 * classes de remplissage plutôt qu'un padding réinventé par composant. */
export type Density = "compact" | "default" | "spacious";

export const DENSITY_LABELS: Record<Density, string> = {
  compact: "Compacte",
  default: "Standard",
  spacious: "Spacieuse",
};

/** Remplissage horizontal/vertical d'une cellule de tableau. */
export const TABLE_CELL_PADDING: Record<Density, string> = {
  compact: "px-2.5 py-1.5",
  default: "px-3 py-2.5",
  spacious: "px-4 py-3.5",
};

/** Remplissage interne d'une carte. */
export const CARD_PADDING: Record<Density, string> = {
  compact: "p-3",
  default: "p-5",
  spacious: "p-7",
};

/** Écart vertical entre le contenu d'une carte (titre, corps, pied). */
export const CARD_GAP: Record<Density, string> = {
  compact: "gap-2",
  default: "gap-3",
  spacious: "gap-4",
};
