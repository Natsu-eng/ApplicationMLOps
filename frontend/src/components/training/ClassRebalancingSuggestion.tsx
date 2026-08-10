import { useEffect, useState } from "react";
import { ChevronDown, Scale } from "lucide-react";
import { ApiError, api, type DataWarning } from "../../api/client";
import { Badge } from "../ui/Badge";

/** Rééquilibrage des classes (lot déséquilibre) — même pattern d'approbation
 * que l'ingénierie de variables (Lot 4c) : détecté par le garde-fou Lot B
 * (`desequilibre_classes`), expliqué en langage clair, jamais appliqué sans
 * case cochée. Contrairement à `feature_engineering`, ce choix ne modifie
 * que la pondération vue par le modèle pendant l'entraînement — il n'a
 * aucun effet à rejouer à l'inférence, d'où un composant et un champ API
 * séparés plutôt qu'une extension de `FeatureEngineeringSuggestions`. */

export function ClassRebalancingSuggestion({
  datasetId,
  targetColumn,
  groupColumn,
  onChange,
}: {
  datasetId: number;
  targetColumn: string;
  groupColumn?: string;
  onChange: (approved: boolean) => void;
}) {
  const [warning, setWarning] = useState<DataWarning | null>(null);
  const [loading, setLoading] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [approved, setApproved] = useState(false);

  useEffect(() => {
    setApproved(false);
    setExpanded(false);
    if (!targetColumn) {
      setWarning(null);
      return;
    }
    setLoading(true);
    api.datasets
      .qualityCheck(datasetId, targetColumn, groupColumn)
      .then((data) => setWarning(data.warnings.find((w) => w.code === "desequilibre_classes") ?? null))
      .catch(() => setWarning(null)) // discret — le garde-fou est déjà affiché par DataQualityWarnings en cas d'échec
      .finally(() => setLoading(false));
  }, [datasetId, targetColumn, groupColumn]);

  useEffect(() => {
    onChange(approved);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [approved]);

  if (!targetColumn || loading || !warning) return null;

  const ratio = (warning.details as { ratio?: number } | null)?.ratio;
  const explanation = ratio
    ? `Dans ce dataset, la classe la plus fréquente apparaît environ ${Math.round(ratio)} fois plus souvent ` +
      "que la plus rare. Sans rééquilibrage, le modèle a tendance à privilégier la classe majoritaire : bonne " +
      "exactitude globale, mais risque de rater une bonne partie des cas de la classe rare — potentiellement " +
      "les plus importants à détecter. Avec rééquilibrage, le modèle accorde plus de poids aux exemples rares " +
      "pendant son apprentissage : il les détecte mieux, au prix de plus de fausses alertes sur la classe majoritaire."
    : warning.explanation;

  return (
    <div className="space-y-2">
      <p className="text-xs uppercase tracking-wide text-slate-500 flex items-center gap-1.5">
        <Scale size={12} className="text-teal-400" />
        Rééquilibrage des classes suggéré
      </p>

      <div
        className={`rounded-lg border px-3 py-2.5 bg-slate-900/60 ${
          approved ? "border-teal-500/40" : "border-slate-800"
        }`}
      >
        <div className="flex items-start gap-2">
          <input
            type="checkbox"
            className="accent-teal-500 mt-1"
            checked={approved}
            onChange={() => setApproved((v) => !v)}
            aria-label="Approuver le rééquilibrage des classes"
          />
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            className="flex-1 flex items-start gap-2 text-left min-w-0"
          >
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2 flex-wrap">
                <Badge variant="accent">Garde-fou</Badge>
                <p className="text-sm text-slate-200 font-medium">
                  Rééquilibrer l'importance des classes pendant l'entraînement
                </p>
              </div>
            </div>
            <ChevronDown
              size={14}
              className={`flex-shrink-0 text-slate-500 transition-transform mt-1 ${expanded ? "rotate-180" : ""}`}
            />
          </button>
        </div>

        {expanded && <p className="text-xs text-slate-400 mt-2 pl-6">{explanation}</p>}

        <p className="text-xs text-slate-300 mt-2 pl-6">
          Cochez pour donner plus de poids à la classe rare pendant l'entraînement — aucune ligne n'est
          dupliquée ni supprimée.
        </p>
      </div>
    </div>
  );
}
