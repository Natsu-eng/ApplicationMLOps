import { useState } from "react";
import { Legend } from "recharts";

/** Isolation d'une série de courbe par la légende — nécessaire dès qu'il y
 * a plus de 2-3 séries (ROC/PR multiclasse, beeswarm...) : les courbes
 * s'emmêlent sinon. Clic = masquer/réafficher une série, survol = mettre
 * les autres en retrait. Un hook par graphe (deux graphes voisins restent
 * indépendants : isoler une classe sur l'un n'affecte pas l'autre).
 *
 * Extrait de `components/training/EvaluationCharts.tsx` (Lot 2A,
 * AUDIT_DATALAB_2026-08-16.md §J.4) pour être réutilisable par n'importe
 * quel graphe multi-séries, pas seulement l'évaluation de modèle —
 * `EvaluationCharts.tsx` réexporte ces deux noms pour ne casser aucun
 * import existant (`ReliabilityDiagnostics.tsx`). */
export function useSeriesIsolation() {
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [hovered, setHovered] = useState<string | null>(null);

  function toggle(name: string) {
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }

  return { hidden, hovered, setHovered, toggle };
}

export function IsolatableLegend({ isolation }: { isolation: ReturnType<typeof useSeriesIsolation> }) {
  return (
    <Legend
      wrapperStyle={{ fontSize: 11, cursor: "pointer" }}
      onClick={(entry) => isolation.toggle(String(entry.value))}
      onMouseEnter={(entry) => isolation.setHovered(String(entry.value))}
      onMouseLeave={() => isolation.setHovered(null)}
      formatter={(value) => (
        <span
          style={{
            opacity: isolation.hidden.has(String(value)) ? 0.35 : 1,
            textDecoration: isolation.hidden.has(String(value)) ? "line-through" : "none",
          }}
        >
          {value}
        </span>
      )}
    />
  );
}
