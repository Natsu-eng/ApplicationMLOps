/** Binning en quantiles côté client — utilisé pour colorer un nuage de
 * points (réduction de dimension, Lot 13) par une variable numérique : la
 * palette catégorielle du design system n'a de sens que sur un nombre fini
 * de groupes, pas sur un continuum. Fonctions pures, testées
 * indépendamment de la page qui les utilise. */

/** Bornes de quantiles (interpolation linéaire, même méthode que
 * `numpy.percentile` par défaut) — `bins + 1` valeurs, min et max inclus.
 * Dédoublonne les bornes identiques (beaucoup de valeurs égales) pour ne
 * jamais produire de bucket vide. */
export function computeQuantileEdges(values: number[], bins: number): number[] {
  if (values.length === 0 || bins < 1) return [];
  const sorted = [...values].sort((a, b) => a - b);
  const edges: number[] = [];
  for (let i = 0; i <= bins; i++) {
    const pos = (i / bins) * (sorted.length - 1);
    const lower = Math.floor(pos);
    const upper = Math.ceil(pos);
    const frac = pos - lower;
    edges.push(sorted[lower] + (sorted[upper] - sorted[lower]) * frac);
  }
  return Array.from(new Set(edges));
}

/** Index du bucket (0-indexé) contenant `value` — dernière tranche
 * inclusive des deux côtés pour couvrir le maximum exact. */
export function binIndexForValue(value: number, edges: number[]): number {
  if (edges.length < 2) return 0;
  for (let i = 0; i < edges.length - 1; i++) {
    if (value <= edges[i + 1] || i === edges.length - 2) return i;
  }
  return edges.length - 2;
}

export function formatBinLabel(edges: number[], index: number): string {
  if (index < 0 || index + 1 >= edges.length) return "—";
  return `${edges[index].toFixed(2)} – ${edges[index + 1].toFixed(2)}`;
}
