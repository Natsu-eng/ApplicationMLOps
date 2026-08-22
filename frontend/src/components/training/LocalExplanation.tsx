import type { LocalExplanation as LocalExplanationData } from "../../api/client";
import { LabelWithHelp } from "../ui/Tooltip";
import { CHART_BEESWARM_HIGH_VAR, CHART_BEESWARM_LOW_VAR } from "../../theme/charts";

const CHART_BEESWARM_HIGH = `var(${CHART_BEESWARM_HIGH_VAR})`;
const CHART_BEESWARM_LOW = `var(${CHART_BEESWARM_LOW_VAR})`;

/** Explicabilité LOCALE (Lot Explicabilité locale) — pourquoi CETTE
 * prédiction précise, contribution par variable, complémentaire à
 * l'explicabilité globale (importance moyenne, page Résultats) qui ne dit
 * rien sur un cas particulier. Convention couleur reprise du beeswarm SHAP
 * déjà utilisé ailleurs dans l'app : rouge = pousse la prédiction vers le
 * haut, bleu = vers le bas — jamais une nouvelle convention concurrente. */
export function LocalExplanationPanel({ explanation }: { explanation: LocalExplanationData | null | undefined }) {
  if (!explanation) return null;

  if (explanation.status === "degraded" || explanation.base_value === null || explanation.base_value === undefined) {
    return (
      <p className="text-xs text-muted-foreground italic mt-2">
        {explanation.message ?? "Explication détaillée non disponible pour cette prédiction."}
      </p>
    );
  }

  const rows = [
    ...explanation.contributions.map((c) => ({ label: c.feature, contribution: c.contribution })),
    ...(explanation.other_contribution && Math.abs(explanation.other_contribution) > 1e-9
      ? [{ label: "Autres variables", contribution: explanation.other_contribution }]
      : []),
  ];
  if (rows.length === 0) return null;

  const maxAbs = Math.max(...rows.map((r) => Math.abs(r.contribution)), 1e-9);
  const finalValue = explanation.base_value + rows.reduce((sum, r) => sum + r.contribution, 0);

  return (
    <div className="mt-3 pt-3 border-t border-primary/10">
      <p className="text-xs text-muted-foreground mb-2">
        <LabelWithHelp
          label="Pourquoi cette prédiction ?"
          help="Point de départ (moyenne du modèle sur l'ensemble d'entraînement), puis l'effet de chaque variable pour CE cas précis — en rouge ce qui pousse la prédiction vers le haut, en bleu ce qui la pousse vers le bas. Contrairement à l'importance moyenne (page Résultats), c'est propre à cette observation."
        />
      </p>
      <div className="space-y-1.5">
        {rows.map((r) => {
          const widthPct = (Math.abs(r.contribution) / maxAbs) * 100;
          const positive = r.contribution >= 0;
          return (
            <div key={r.label} className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground w-32 truncate flex-shrink-0" title={r.label}>
                {r.label}
              </span>
              <div className="flex-1 h-4 flex items-center">
                <div className="relative w-full h-1.5 rounded-full bg-muted">
                  <div
                    className="absolute top-0 h-full rounded-full"
                    style={{
                      width: `${widthPct / 2}%`,
                      left: positive ? "50%" : `${50 - widthPct / 2}%`,
                      backgroundColor: positive ? CHART_BEESWARM_HIGH : CHART_BEESWARM_LOW,
                    }}
                  />
                  <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-px h-3 bg-border" />
                </div>
              </div>
              <span
                className="text-xs w-16 text-right tabular-nums flex-shrink-0"
                style={{ color: positive ? CHART_BEESWARM_HIGH : CHART_BEESWARM_LOW }}
              >
                {positive ? "+" : ""}
                {r.contribution.toFixed(3)}
              </span>
            </div>
          );
        })}
      </div>
      <p className="text-caption text-muted-foreground mt-2 tabular-nums">
        Base du modèle : {explanation.base_value.toFixed(3)} → total après effet des variables :{" "}
        {finalValue.toFixed(3)}
      </p>
    </div>
  );
}
