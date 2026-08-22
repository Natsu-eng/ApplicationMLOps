import { Fragment, useMemo, useState, type ReactNode } from "react";
import { ArrowDown, ArrowUp, ArrowUpDown, ChevronLeft, ChevronRight } from "lucide-react";
import type { Density } from "../../theme/density";
import { TABLE_CELL_PADDING } from "../../theme/density";
import { nextSortState, sortRows, type SortState } from "../../utils/tableSort";

export interface TableColumn<T> {
  key: string;
  header: string;
  align?: "left" | "right" | "center";
  /** Rendu personnalisé — par défaut, affiche `String(row[key])`. */
  render?: (row: T) => ReactNode;
  className?: string;
  /** Active le tri sur cette colonne (Lot 2A) — nécessite `sortValue` si la
   * valeur à comparer diffère de `row[key]` brut (ex. rendu formaté). */
  sortable?: boolean;
  sortValue?: (row: T) => string | number | null;
  /** Colonne figée à gauche au défilement horizontal (Lot 2A) — une seule
   * par tableau en pratique (la première, identifiante), jamais plusieurs :
   * pas de garde-fou technique, juste une convention d'usage. */
  sticky?: boolean;
}

const ALIGN_CLASSES: Record<NonNullable<TableColumn<unknown>["align"]>, string> = {
  left: "text-left",
  right: "text-right",
  center: "text-center",
};

function defaultSortValue<T>(row: T, key: string): string | number | null {
  const value = (row as Record<string, unknown>)[key];
  if (value === null || value === undefined) return null;
  if (typeof value === "number") return value;
  return String(value);
}

/** Tableau générique du système de design (Lot 2A) — tri, pagination,
 * sélection, colonne figée, état vide avec action, skeleton de
 * chargement, 3 densités. Reconstruit depuis une version sans aucune de
 * ces capacités (AUDIT_DATALAB_2026-08-16.md, §J.4) — le composant à plus
 * fort effet de levier du lot : il débloque à lui seul Dashboard et les
 * pages d'historique.
 *
 * Tri et pagination sont NON CONTRÔLÉS par défaut (état interne) — le cas
 * d'usage dominant de ce produit est une liste déjà entièrement chargée
 * côté client (pas de pagination serveur avant le Lot 4). `sort`/
 * `onSortChange` restent disponibles pour un usage contrôlé futur. */
export function Table<T>({
  columns,
  rows,
  rowKey,
  caption,
  emptyMessage = "Aucune donnée à afficher.",
  emptyAction,
  highlightRow,
  loading = false,
  error,
  density = "default",
  pageSize,
  selectable = false,
  selectedKeys,
  onSelectionChange,
  renderExpanded,
}: {
  columns: TableColumn<T>[];
  rows: T[];
  rowKey: (row: T) => string | number;
  /** Résumé accessible du tableau (`<caption>`, visuellement masqué) — ce
   * que ce tableau montre, pour un lecteur d'écran qui n'a pas le contexte
   * visuel de la page. Optionnel pour rester compatible avec les appels
   * existants (Lot 2A ne modifie aucune page métier) — mais un tableau
   * réellement intégré à une page devrait toujours en fournir un
   * spécifique, `emptyMessage` seul n'est pas un résumé de contenu. */
  caption?: string;
  emptyMessage?: string;
  emptyAction?: ReactNode;
  /** Ligne mise en avant — ex. le candidat gagnant, un consensus fort. */
  highlightRow?: (row: T) => boolean;
  /** Affiche un squelette de N lignes au lieu des données — chargement. */
  loading?: boolean;
  /** État d'erreur (SPEC-UI.md §6.8) — remplace tout le tableau par le
   * message et l'action de sortie fournis, jamais un tableau vide muet. */
  error?: { message: string; action?: ReactNode };
  density?: Density;
  /** Active la pagination côté client au-delà de ce nombre de lignes. */
  pageSize?: number;
  selectable?: boolean;
  selectedKeys?: Set<string | number>;
  onSelectionChange?: (keys: Set<string | number>) => void;
  /** Ligne repliable — présent = une colonne chevron s'ajoute à gauche,
   * `renderExpanded(row)` est affiché sur une ligne pleine largeur quand
   * elle est dépliée. */
  renderExpanded?: (row: T) => ReactNode;
}) {
  const [sort, setSort] = useState<SortState | null>(null);
  const [page, setPage] = useState(1);
  const [expandedKeys, setExpandedKeys] = useState<Set<string | number>>(new Set());

  function toggleExpanded(key: string | number) {
    setExpandedKeys((current) => {
      const next = new Set(current);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  const sortedRows = useMemo(() => {
    if (!sort) return rows;
    const column = columns.find((c) => c.key === sort.key);
    if (!column) return rows;
    const getValue = column.sortValue ?? ((row: T) => defaultSortValue(row, sort.key));
    return sortRows(rows, sort, getValue);
  }, [rows, sort, columns]);

  const totalPages = pageSize ? Math.max(1, Math.ceil(sortedRows.length / pageSize)) : 1;
  const currentPage = Math.min(page, totalPages);
  const pagedRows = pageSize ? sortedRows.slice((currentPage - 1) * pageSize, currentPage * pageSize) : sortedRows;

  function toggleSort(key: string) {
    setPage(1);
    setSort((current) => nextSortState(current, key));
  }

  function toggleRow(key: string | number) {
    if (!selectedKeys || !onSelectionChange) return;
    const next = new Set(selectedKeys);
    if (next.has(key)) next.delete(key);
    else next.add(key);
    onSelectionChange(next);
  }

  function toggleAllOnPage() {
    if (!selectedKeys || !onSelectionChange) return;
    const pageKeys = pagedRows.map(rowKey);
    const allSelected = pageKeys.every((k) => selectedKeys.has(k));
    const next = new Set(selectedKeys);
    if (allSelected) pageKeys.forEach((k) => next.delete(k));
    else pageKeys.forEach((k) => next.add(k));
    onSelectionChange(next);
  }

  const cellPadding = TABLE_CELL_PADDING[density];
  const allOnPageSelected = selectable && selectedKeys && pagedRows.length > 0 && pagedRows.every((r) => selectedKeys.has(rowKey(r)));

  if (error) {
    return (
      <div className="rounded-card border border-destructive/25 bg-destructive/8 py-10 px-4 text-center">
        <p className="text-body text-destructive">{error.message}</p>
        {error.action && <div className="mt-3">{error.action}</div>}
      </div>
    );
  }

  if (loading) {
    return (
      <div role="status" aria-label={`${caption ?? "Tableau"} — chargement en cours`} className="overflow-x-auto rounded-card border border-border/70">
        <table className="w-full text-body border-collapse" aria-hidden="true">
          <tbody>
            {Array.from({ length: 5 }).map((_, i) => (
              <tr key={i} className="border-b border-foreground/[0.05] last:border-0">
                {columns.map((col) => (
                  <td key={col.key} className={cellPadding}>
                    <div
                      className="h-4 rounded bg-muted animate-pulse motion-reduce:animate-none"
                      style={{ width: `${55 + ((i * 13 + col.key.length * 7) % 35)}%` }}
                    />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  if (rows.length === 0) {
    return (
      <div className="rounded-card border border-dashed border-border/70 py-10 px-4 text-center">
        <p className="text-body text-muted-foreground">{emptyMessage}</p>
        {emptyAction && <div className="mt-3">{emptyAction}</div>}
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <div className="overflow-x-auto rounded-card border border-border/70 max-h-[32rem] overflow-y-auto">
        <table className="w-full text-body border-collapse">
          <caption className="sr-only">{caption ?? "Tableau"}</caption>
          <thead className="sticky top-0 z-20 bg-muted/95 backdrop-blur-sm text-muted-foreground text-overline uppercase">
            <tr>
              {renderExpanded && <th scope="col" className={`${cellPadding} w-8`} aria-hidden="true" />}
              {selectable && (
                <th scope="col" className={`${cellPadding} w-8`}>
                  <input
                    type="checkbox"
                    aria-label="Sélectionner toutes les lignes de cette page"
                    checked={allOnPageSelected}
                    onChange={toggleAllOnPage}
                    className="rounded border-input"
                  />
                </th>
              )}
              {columns.map((col) => {
                const isSorted = sort?.key === col.key;
                const stickyClass = col.sticky ? "sticky left-0 z-30 bg-muted/95" : "";
                return (
                  <th
                    key={col.key}
                    scope="col"
                    aria-sort={isSorted ? (sort!.direction === "asc" ? "ascending" : "descending") : col.sortable ? "none" : undefined}
                    className={`${cellPadding} font-medium whitespace-nowrap ${ALIGN_CLASSES[col.align ?? "left"]} ${stickyClass} ${col.className ?? ""}`}
                  >
                    {col.sortable ? (
                      <button
                        type="button"
                        onClick={() => toggleSort(col.key)}
                        className="inline-flex items-center gap-1 hover:text-foreground transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)] rounded"
                      >
                        {col.header}
                        {isSorted ? (
                          sort!.direction === "asc" ? <ArrowUp size={12} /> : <ArrowDown size={12} />
                        ) : (
                          <ArrowUpDown size={12} className="opacity-40" />
                        )}
                      </button>
                    ) : (
                      col.header
                    )}
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody className="divide-y divide-foreground/[0.05]">
            {pagedRows.map((row) => {
              const key = rowKey(row);
              const highlighted = highlightRow?.(row) ?? false;
              const rowSelected = selectedKeys?.has(key) ?? false;
              const expanded = expandedKeys.has(key);
              // Sélectionnée : liseré d'accent de 3px (inset, jamais une bordure
              // complète qui décalerait la ligne) + lavis léger — jamais un simple
              // fond plein qui se confondrait avec le survol (SPEC-UI.md §2 :
              // "la couleur porte le sens, jamais seule").
              const rowBg = highlighted ? "bg-primary/5" : rowSelected ? "bg-primary/[0.08]" : undefined;
              const rowStyle = rowSelected ? { boxShadow: "inset 3px 0 0 var(--accent)" } : undefined;
              return (
                <Fragment key={key}>
                  <tr
                    className={`${rowBg ?? ""} hover:bg-foreground/[0.035] transition-colors`}
                    style={rowStyle}
                  >
                    {renderExpanded && (
                      <td className={cellPadding}>
                        <button
                          type="button"
                          onClick={() => toggleExpanded(key)}
                          aria-expanded={expanded}
                          aria-label={expanded ? "Replier la ligne" : "Déplier la ligne"}
                          className="text-muted-foreground hover:text-foreground transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)] rounded"
                        >
                          <ChevronRight size={14} className={`transition-transform ${expanded ? "rotate-90" : ""}`} />
                        </button>
                      </td>
                    )}
                    {selectable && (
                      <td className={cellPadding}>
                        <input
                          type="checkbox"
                          aria-label="Sélectionner cette ligne"
                          checked={rowSelected}
                          onChange={() => toggleRow(key)}
                          className="rounded border-input"
                        />
                      </td>
                    )}
                    {columns.map((col) => {
                      const stickyClass = col.sticky ? `sticky left-0 z-10 ${rowBg ?? "bg-card"}` : "";
                      const isRight = (col.align ?? "left") === "right";
                      return (
                        <td
                          key={col.key}
                          className={`${cellPadding} text-foreground/90 whitespace-nowrap ${ALIGN_CLASSES[col.align ?? "left"]} ${
                            isRight ? "num" : ""
                          } ${stickyClass} ${col.className ?? ""}`}
                        >
                          {col.render ? col.render(row) : String((row as Record<string, unknown>)[col.key] ?? "—")}
                        </td>
                      );
                    })}
                  </tr>
                  {renderExpanded && expanded && (
                    <tr>
                      <td colSpan={columns.length + 1 + (selectable ? 1 : 0)} className={`${cellPadding} bg-muted/30`}>
                        {renderExpanded(row)}
                      </td>
                    </tr>
                  )}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>

      {pageSize && totalPages > 1 && (
        <div className="flex items-center justify-between text-caption text-muted-foreground px-1">
          <span>
            Page {currentPage} sur {totalPages} — {sortedRows.length} ligne{sortedRows.length > 1 ? "s" : ""}
          </span>
          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={currentPage <= 1}
              aria-label="Page précédente"
              className="rounded-control p-1 border border-border disabled:opacity-[.42] disabled:cursor-not-allowed hover:bg-muted transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)]"
            >
              <ChevronLeft size={14} />
            </button>
            <button
              type="button"
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={currentPage >= totalPages}
              aria-label="Page suivante"
              className="rounded-control p-1 border border-border disabled:opacity-[.42] disabled:cursor-not-allowed hover:bg-muted transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)]"
            >
              <ChevronRight size={14} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
