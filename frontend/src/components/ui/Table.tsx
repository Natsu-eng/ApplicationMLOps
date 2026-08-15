import type { ReactNode } from "react";

export interface TableColumn<T> {
  key: string;
  header: string;
  align?: "left" | "right" | "center";
  /** Rendu personnalisé — par défaut, affiche `String(row[key])`. */
  render?: (row: T) => ReactNode;
  className?: string;
}

const ALIGN_CLASSES: Record<NonNullable<TableColumn<unknown>["align"]>, string> = {
  left: "text-left",
  right: "text-right",
  center: "text-center",
};

/** Tableau générique du design system — réutilisé partout où l'app doit
 * comparer plusieurs lignes structurées (candidats de clustering, loadings
 * PCA, observations atypiques...). Défile horizontalement plutôt que de
 * compresser les colonnes sur petit écran (voir frontend-ui-ux.md du skill,
 * section Responsive). Tokens sémantiques uniquement. */
export function Table<T>({
  columns,
  rows,
  rowKey,
  emptyMessage = "Aucune donnée à afficher.",
  highlightRow,
}: {
  columns: TableColumn<T>[];
  rows: T[];
  rowKey: (row: T) => string | number;
  emptyMessage?: string;
  /** Ligne mise en avant — ex. le candidat gagnant, un consensus fort. */
  highlightRow?: (row: T) => boolean;
}) {
  if (rows.length === 0) {
    return <p className="text-sm text-muted-foreground text-center py-6">{emptyMessage}</p>;
  }

  return (
    <div className="overflow-x-auto rounded-xl border border-border/70">
      <table className="w-full text-sm border-collapse">
        <thead className="bg-muted/40 text-muted-foreground text-[11px] uppercase tracking-wide">
          <tr>
            {columns.map((col) => (
              <th
                key={col.key}
                scope="col"
                className={`px-3 py-2 font-medium whitespace-nowrap ${ALIGN_CLASSES[col.align ?? "left"]} ${col.className ?? ""}`}
              >
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-border/60">
          {rows.map((row) => {
            const highlighted = highlightRow?.(row) ?? false;
            return (
              <tr key={rowKey(row)} className={highlighted ? "bg-primary/5" : undefined}>
                {columns.map((col) => (
                  <td
                    key={col.key}
                    className={`px-3 py-2 text-foreground/90 whitespace-nowrap ${ALIGN_CLASSES[col.align ?? "left"]} ${col.className ?? ""}`}
                  >
                    {col.render ? col.render(row) : String((row as Record<string, unknown>)[col.key] ?? "—")}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
