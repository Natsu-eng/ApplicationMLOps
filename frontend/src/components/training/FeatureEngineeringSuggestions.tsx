import { useEffect, useMemo, useState } from "react";
import { ChevronDown, Sparkles, Wand2 } from "lucide-react";
import {
  ApiError,
  api,
  type FeatureEngineeringSpec,
  type FeatureEngineeringSuggestion,
} from "../../api/client";
import { Badge } from "../ui/Badge";

/** Ingénierie de variables guidée (Lot 4c) — suggestions en langage clair,
 * reliées aux garde-fous Lot B quand pertinent. L'utilisateur APPROUVE une
 * suggestion (case à cocher), il ne configure jamais une transformation
 * technique — seule exception : le choix de stratégie d'imputation, qui
 * reste un simple menu déroulant (médiane par défaut). */

const STRATEGY_LABEL: Record<string, string> = {
  mean: "Moyenne",
  median: "Médiane",
  most_frequent: "Valeur la plus fréquente",
  constant: "Valeur constante",
};

export function FeatureEngineeringSuggestions({
  datasetId,
  targetColumn,
  groupColumn,
  onChange,
}: {
  datasetId: number;
  targetColumn: string;
  groupColumn?: string;
  onChange: (payload: { upstream: FeatureEngineeringSpec["upstream"]; pipeline: FeatureEngineeringSpec["pipeline"] } | null) => void;
}) {
  const [suggestions, setSuggestions] = useState<FeatureEngineeringSuggestion[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [expanded, setExpanded] = useState<Set<number>>(new Set());
  const [approved, setApproved] = useState<Set<number>>(new Set());
  const [strategies, setStrategies] = useState<Record<number, string>>({});
  const [fillValues, setFillValues] = useState<Record<number, string>>({});

  useEffect(() => {
    if (!targetColumn) {
      setSuggestions(null);
      return;
    }
    setLoading(true);
    setError(null);
    setExpanded(new Set());
    setApproved(new Set());
    setStrategies({});
    setFillValues({});
    api.datasets
      .featureEngineeringSuggestions(datasetId, targetColumn, groupColumn)
      // "exclusion_variable" (Lot Nettoyage guidé des variables) est déjà
      // proposée, en plus utilement (bouton direct par colonne, sans étape
      // supplémentaire), dans le panneau "Qualité des données" juste avant
      // (voir DataQualityWarnings.tsx) — l'afficher aussi ici dupliquerait la
      // recommandation ET serait un cul-de-sac : sa `transformation`
      // (`exclude_column`) n'est volontairement pas une entrée de pipeline
      // (voir la note dans `services/feature_engineering.py`), donc l'approuver
      // ici ne ferait rien.
      .then((data) => setSuggestions(data.suggestions.filter((s) => s.code !== "exclusion_variable")))
      .catch((err) => setError(err instanceof ApiError ? err.message : "Suggestions indisponibles"))
      .finally(() => setLoading(false));
  }, [datasetId, targetColumn, groupColumn]);

  // Recalcule le payload (upstream/pipeline) à chaque changement d'approbation
  // ou de stratégie choisie, et le remonte au formulaire parent.
  useEffect(() => {
    if (!suggestions || approved.size === 0) {
      onChange(null);
      return;
    }
    const upstream: FeatureEngineeringSpec["upstream"] = [];
    const frequencyEncoding: string[] = [];
    const imputation: Record<string, { strategy: string; fill_value?: unknown }> = {};

    approved.forEach((index) => {
      const suggestion = suggestions[index];
      const t = suggestion.transformation;
      if (t.type === "datetime_decompose" || t.type === "ratio" || t.type === "numeric_coerce") {
        upstream.push(t);
      } else if (t.type === "frequency_encoding") {
        frequencyEncoding.push(String(t.column));
      } else if (t.type === "imputation") {
        const strategy = strategies[index] ?? suggestion.choice?.default ?? "median";
        const entry: { strategy: string; fill_value?: unknown } = { strategy };
        if (strategy === "constant") entry.fill_value = fillValues[index] ?? "";
        imputation[String(t.column)] = entry;
      }
    });

    onChange({
      upstream,
      pipeline: {
        ...(frequencyEncoding.length ? { frequency_encoding: frequencyEncoding } : {}),
        ...(Object.keys(imputation).length ? { imputation } : {}),
      },
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [suggestions, approved, strategies, fillValues]);

  const approvedCount = useMemo(() => approved.size, [approved]);

  if (!targetColumn) return null;

  function toggleExpanded(index: number) {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  }

  function toggleApproved(index: number) {
    setApproved((prev) => {
      const next = new Set(prev);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  }

  return (
    <div className="space-y-2">
      <p className="text-xs uppercase tracking-wide text-muted-foreground flex items-center gap-1.5">
        <Wand2 size={12} className="text-primary" />
        Ingénierie de variables suggérée
        {approvedCount > 0 && (
          <span className="text-muted-foreground">
            ({approvedCount} approuvée{approvedCount > 1 ? "s" : ""})
          </span>
        )}
      </p>

      {loading && <p className="text-xs text-muted-foreground">Analyse des transformations possibles…</p>}
      {error && <p className="text-xs text-destructive">{error}</p>}

      {!loading && !error && suggestions && suggestions.length === 0 && (
        <p className="text-xs text-muted-foreground">Aucune suggestion pour ce dataset et cette cible.</p>
      )}

      {!loading &&
        !error &&
        suggestions &&
        suggestions.map((suggestion, index) => {
          const isOpen = expanded.has(index);
          const isApproved = approved.has(index);
          const isImputation = suggestion.choice?.type === "imputation_strategy";
          const currentStrategy = strategies[index] ?? suggestion.choice?.default ?? "";
          return (
            <div
              key={`${suggestion.code}-${suggestion.columns.join(",")}-${index}`}
              className={`rounded-lg border px-3 py-2.5 bg-muted ${
                isApproved ? "border-primary/40" : "border-border"
              }`}
            >
              <div className="flex items-start gap-2">
                <input
                  type="checkbox"
                  className="accent-primary mt-1"
                  checked={isApproved}
                  onChange={() => toggleApproved(index)}
                  aria-label={`Approuver : ${suggestion.title}`}
                />
                <button
                  type="button"
                  onClick={() => toggleExpanded(index)}
                  className="flex-1 flex items-start gap-2 text-left min-w-0"
                >
                  <Sparkles size={14} className="flex-shrink-0 mt-0.5 text-primary" />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2 flex-wrap">
                      {suggestion.based_on_warning && <Badge variant="accent">Garde-fou</Badge>}
                      <p className="text-sm text-foreground font-medium">{suggestion.title}</p>
                    </div>
                  </div>
                  <ChevronDown
                    size={14}
                    className={`flex-shrink-0 text-muted-foreground transition-transform mt-1 ${isOpen ? "rotate-180" : ""}`}
                  />
                </button>
              </div>

              {isOpen && <p className="text-xs text-muted-foreground mt-2 pl-6">{suggestion.explanation}</p>}
              <p className="text-xs text-muted-foreground mt-2 pl-6">{suggestion.action}</p>

              {isApproved && isImputation && suggestion.choice && (
                <div className="pl-6 mt-2 flex items-center gap-2">
                  <select
                    value={currentStrategy}
                    onChange={(e) => setStrategies((prev) => ({ ...prev, [index]: e.target.value }))}
                    className="rounded-md border border-input bg-card px-2 py-1 text-xs text-foreground focus:outline-none focus:ring-2 focus:ring-primary/40"
                  >
                    {suggestion.choice.options.map((opt) => (
                      <option key={opt} value={opt}>
                        {STRATEGY_LABEL[opt] ?? opt}
                      </option>
                    ))}
                  </select>
                  {currentStrategy === "constant" && (
                    <input
                      type="text"
                      placeholder="Valeur"
                      value={fillValues[index] ?? ""}
                      onChange={(e) => setFillValues((prev) => ({ ...prev, [index]: e.target.value }))}
                      className="w-24 rounded-md border border-input bg-white px-2 py-1 text-xs text-foreground focus:outline-none focus:ring-2 focus:ring-primary/40"
                    />
                  )}
                </div>
              )}
            </div>
          );
        })}
    </div>
  );
}
