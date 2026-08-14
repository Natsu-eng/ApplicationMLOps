import { useState, type FormEvent } from "react";
import { Sparkles, Wand2 } from "lucide-react";
import { ApiError, api, type FeatureSchemaEntry, type PredictionResult, type TaskType } from "../../api/client";
import { Button } from "../ui/Button";
import { Card } from "../ui/Card";
import { Input } from "../ui/Input";
import { LabelWithHelp } from "../ui/Tooltip";
import { LocalExplanationPanel } from "./LocalExplanation";
import { formatMetricValue } from "../../utils/format";

function isNumericDtype(dtype: string): boolean {
  return dtype.startsWith("int") || dtype.startsWith("float");
}

/** Formulaire de prédiction — généré dynamiquement à partir du schéma des
 * variables d'entraînement (feature_schema), pour ne jamais désynchroniser
 * le formulaire du modèle réellement entraîné. */
export default function PredictionForm({
  jobId,
  taskType,
  featureSchema,
}: {
  jobId: number;
  taskType: TaskType;
  featureSchema: FeatureSchemaEntry[];
}) {
  const [values, setValues] = useState<Record<string, string>>({});
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setResult(null);
    setIsSubmitting(true);
    try {
      const data: Record<string, unknown> = {};
      for (const field of featureSchema) {
        const raw = values[field.name] ?? "";
        data[field.name] = isNumericDtype(field.dtype) ? Number(raw) : raw;
      }
      setResult(await api.training.predict(jobId, data));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Prédiction impossible");
    } finally {
      setIsSubmitting(false);
    }
  }

  if (featureSchema.length === 0) return null;

  return (
    <Card className="p-5">
      <p className="text-xs uppercase tracking-wide text-slate-500 mb-2 flex items-center gap-1.5">
        <Wand2 size={12} className="text-primary" />
        <LabelWithHelp
          label="Tester une prédiction"
          help="Saisissez des valeurs pour les variables utilisées à l'entraînement — le modèle calcule sa prédiction sur ce cas précis, en temps réel."
        />
      </p>

      <form onSubmit={handleSubmit} className="grid sm:grid-cols-3 gap-3 items-start mb-3">
        {featureSchema.map((field) => (
          <div key={field.name}>
            <label className="block text-xs text-slate-600 mb-1 truncate" title={field.name}>
              {field.name}
            </label>
            <Input
              type={isNumericDtype(field.dtype) ? "number" : "text"}
              step="any"
              required
              value={values[field.name] ?? ""}
              onChange={(e) => setValues((prev) => ({ ...prev, [field.name]: e.target.value }))}
            />
          </div>
        ))}
        <Button type="submit" disabled={isSubmitting} className="sm:col-span-3">
          {isSubmitting ? "Calcul…" : "Prédire"}
        </Button>
      </form>

      {error && <p className="text-sm text-rose-600">{error}</p>}

      {result && (
        <div className="rounded-xl border border-primary/20 bg-primary/10 px-4 py-3">
          <div className="flex items-center gap-2 mb-1">
            <Sparkles size={14} className="text-primary" />
            <p className="text-sm text-slate-800">
              {taskType === "regression" ? (
                <>
                  Valeur prédite :{" "}
                  <span className="font-semibold tabular-nums">
                    {formatMetricValue(result.prediction as number)}
                  </span>
                </>
              ) : (
                <>
                  Classe prédite : <span className="font-semibold">{result.prediction}</span>
                </>
              )}
            </p>
          </div>

          {result.interval && (
            <p className="text-xs text-slate-500 tabular-nums">
              Intervalle de confiance à {Math.round(result.interval.confidence * 100)} % : entre{" "}
              {formatMetricValue(result.interval.low)} et {formatMetricValue(result.interval.high)}
            </p>
          )}

          {result.probabilities && (
            <div className="mt-2 space-y-1.5">
              {Object.entries(result.probabilities)
                .sort(([, a], [, b]) => b - a)
                .map(([label, proba]) => (
                  <div key={label} className="flex items-center gap-2">
                    <span className="text-xs text-slate-500 w-24 truncate">{label}</span>
                    <div className="flex-1 h-1.5 rounded-full bg-slate-100 overflow-hidden">
                      <div
                        className="h-full rounded-full bg-primary"
                        style={{ width: `${proba * 100}%` }}
                      />
                    </div>
                    <span className="text-xs text-slate-400 w-10 text-right tabular-nums">
                      {(proba * 100).toFixed(0)} %
                    </span>
                  </div>
                ))}
            </div>
          )}

          <LocalExplanationPanel explanation={result.explanation} />
        </div>
      )}
    </Card>
  );
}
