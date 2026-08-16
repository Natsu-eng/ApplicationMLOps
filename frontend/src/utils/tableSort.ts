/** Logique de tri du composant `Table` (Lot 2A) — extraite en fonction pure
 * testable, même motif que `trainingPayload.ts` (ce dépôt n'a pas
 * d'infrastructure de test de composants React, voir la mémoire de
 * session : la logique non triviale d'un composant s'extrait en fonction
 * pure plutôt que de rester non testée). */
export type SortDirection = "asc" | "desc";

export interface SortState {
  key: string;
  direction: SortDirection;
}

/** Prochain état de tri après un clic sur l'en-tête `key` — cycle
 * asc → desc → aucun tri (jamais bloqué sur un sens une fois activé). */
export function nextSortState(current: SortState | null, key: string): SortState | null {
  if (!current || current.key !== key) return { key, direction: "asc" };
  if (current.direction === "asc") return { key, direction: "desc" };
  return null;
}

/** Trie `rows` selon `sort`, en utilisant `getValue` pour extraire la
 * valeur comparable de chaque ligne. `null`/`undefined` sont toujours
 * envoyés en fin de liste, quel que soit le sens — une valeur absente
 * n'est ni "la plus petite" ni "la plus grande", juste non ordonnable. */
export function sortRows<T>(rows: T[], sort: SortState | null, getValue: (row: T) => string | number | null): T[] {
  if (!sort) return rows;
  const withValue = rows.map((row) => ({ row, value: getValue(row) }));
  withValue.sort((a, b) => {
    if (a.value === null && b.value === null) return 0;
    if (a.value === null) return 1;
    if (b.value === null) return -1;
    if (a.value < b.value) return sort.direction === "asc" ? -1 : 1;
    if (a.value > b.value) return sort.direction === "asc" ? 1 : -1;
    return 0;
  });
  return withValue.map((w) => w.row);
}
