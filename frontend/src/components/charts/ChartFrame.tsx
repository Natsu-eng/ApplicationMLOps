import { useState, type ReactNode } from "react";

/** Coquille commune à tous les graphiques enveloppés (Lot 3) — impose la
 * structure demandée par la mission pour CHAQUE graphique :
 *   1. un titre en une phrase (`title`) ;
 *   2. une légende qui dit COMMENT le lire, pas seulement ce qu'il montre
 *      (`reading`) ;
 *   3. une alternative textuelle pour lecteur d'écran (`ariaLabel`,
 *      décrivant la TENDANCE — jamais juste "graphique") ;
 *   4. un tableau de données accessible en repli (`fallbackTable`).
 *
 * Le SVG Recharts est posé `aria-hidden="true"` : le rendre lisible au
 * clavier point par point serait un chantier à part entière (Recharts ne
 * fournit pas de navigation clavier native sur les points/segments) — la
 * phrase `ariaLabel` + le tableau en repli portent l'équivalent accessible
 * complet, plutôt qu'un graphique à moitié navigable qui donnerait une
 * fausse impression d'accessibilité. */
export function ChartFrame({
  title,
  reading,
  ariaLabel,
  fallbackTable,
  controls,
  children,
}: {
  title: string;
  reading: string;
  ariaLabel: string;
  fallbackTable?: ReactNode;
  /** Contrôles RÉELLEMENT interactifs à côté du graphique (ex. bascule
   * ROC/PR de `RocPr`) — rendus normalement, JAMAIS dans le conteneur
   * `aria-hidden` de `children` : un bouton caché du lecteur d'écran mais
   * toujours atteignable au clavier est une violation axe-core
   * (`aria-hidden-focus`), trouvée et corrigée au Lot 3, voir
   * _design/JOURNAL.md. */
  controls?: ReactNode;
  /** Le graphique Recharts lui-même — purement décoratif, `aria-hidden`. */
  children: ReactNode;
}) {
  const [showTable, setShowTable] = useState(false);

  return (
    <div>
      <p className="text-body font-medium text-foreground">{title}</p>
      <p className="text-caption text-muted-foreground mt-0.5 mb-2 leading-relaxed">{reading}</p>
      <p className="sr-only">{ariaLabel}</p>
      {controls && <div className="mb-2">{controls}</div>}
      <div aria-hidden="true">{children}</div>
      {fallbackTable && (
        <div className="mt-2">
          <button
            type="button"
            onClick={() => setShowTable((v) => !v)}
            className="text-caption text-primary hover:underline underline-offset-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)] rounded"
          >
            {showTable ? "Masquer le tableau de données" : "Voir les données en tableau"}
          </button>
          {showTable && <div className="mt-2 overflow-x-auto">{fallbackTable}</div>}
        </div>
      )}
    </div>
  );
}

/** Petit tableau de repli générique — colonnes + lignes déjà formatées en
 * chaînes par l'appelant (chaque graphique connaît le bon format/l'unité
 * métier de ses propres données, `ChartFrame` reste agnostique).
 *
 * Retour utilisateur direct : la première version (`border-foreground/
 * [0.05]`, quasi invisible) rendait les lignes illisibles à l'œil sans
 * survoler chaque cellule — séparateurs remontés à `border` (jeton de
 * thème, même contraste que le `Table.tsx` principal), zébrage léger une
 * ligne sur deux et surlignage au survol pour suivre une ligne sans
 * confusion, conteneur arrondi avec cadre pour détacher le tableau du
 * texte environnant plutôt que de flotter à même le fond. */
export function ChartFallbackTable({ columns, rows }: { columns: string[]; rows: (string | number)[][] }) {
  return (
    <div className="overflow-hidden rounded-lg border border-border">
      <table className="w-full text-caption border-collapse">
        <thead>
          <tr className="bg-muted border-b border-border">
            {columns.map((c) => (
              <th key={c} scope="col" className="text-left font-medium text-muted-foreground text-overline uppercase py-2 px-3 whitespace-nowrap">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {rows.map((row, i) => (
            <tr key={i} className={`transition-colors hover:bg-muted/60 ${i % 2 === 1 ? "bg-muted/25" : ""}`}>
              {row.map((cell, j) => (
                <td key={j} className={`py-2 px-3 whitespace-nowrap ${typeof cell === "number" ? "num text-right" : "text-foreground/90"}`}>
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
