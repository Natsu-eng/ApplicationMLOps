import { useCallback, useEffect, useState, type FormEvent } from "react";
import { History, Sparkles, Wand2 } from "lucide-react";
import {
  ApiError,
  api,
  type FeatureSchemaEntry,
  type PredictionHistoryEntry,
  type PredictionResult,
  type TaskType,
} from "../../api/client";
import { Button } from "../ui/Button";
import { Card } from "../ui/Card";
import { accentSurfaceClass } from "../ui/ColorIconBadge";
import { Input } from "../ui/Input";
import { SectionHeader } from "../ui/SectionHeader";
import { Table, type TableColumn } from "../ui/Table";
import { LocalExplanationPanel } from "./LocalExplanation";
import { formatDateTime, formatMetricValue } from "../../utils/format";

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
  const [history, setHistory] = useState<PredictionHistoryEntry[] | null>(null);
  const [showHistory, setShowHistory] = useState(false);

  // Chargé à la demande (repli), pas au montage — un modèle sans jamais
  // avoir servi de prédiction n'a pas besoin de cet appel (Lot lignage des
  // prédictions, voir GET /training/jobs/{id}/predictions, backend Phase 3).
  const loadHistory = useCallback(async () => {
    try {
      const { entries } = await api.training.predictions(jobId);
      setHistory(entries);
    } catch {
      setHistory([]);
    }
  }, [jobId]);

  useEffect(() => {
    if (showHistory && history === null) loadHistory();
  }, [showHistory, history, loadHistory]);

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
      // Rafraîchit l'historique s'il est déjà ouvert — sinon `loadHistory`
      // se déclenchera de toute façon à la prochaine ouverture (toujours à
      // jour, jamais besoin de forcer l'ouverture pour ça).
      if (showHistory) loadHistory();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Prédiction impossible");
    } finally {
      setIsSubmitting(false);
    }
  }

  const historyColumns: TableColumn<PredictionHistoryEntry>[] = [
    {
      key: "created_at",
      header: "Date",
      render: (e) => formatDateTime(e.created_at),
      className: "text-muted-foreground",
    },
    {
      key: "prediction",
      header: taskType === "regression" ? "Valeur prédite" : "Classe prédite",
      render: (e) => (typeof e.prediction === "number" ? formatMetricValue(e.prediction) : String(e.prediction)),
    },
    {
      key: "model_version",
      header: "Version",
      align: "right",
      render: (e) => `v${e.model_version}`,
      className: "text-muted-foreground",
    },
    {
      key: "requested_by",
      header: "Demandé par",
      render: (e) => e.requested_by ?? "—",
      className: "text-muted-foreground",
    },
  ];

  if (featureSchema.length === 0) return null;

  return (
    <Card className={`p-5 ${accentSurfaceClass("amber")}`}>
      <SectionHeader
        icon={Wand2}
        color="amber"
        label="Tester une prédiction"
        help="Saisissez des valeurs pour les variables utilisées à l'entraînement — le modèle calcule sa prédiction sur ce cas précis, en temps réel."
      />

      <form onSubmit={handleSubmit} className="grid sm:grid-cols-3 gap-3 items-start mb-3">
        {featureSchema.map((field) => (
          <div key={field.name}>
            <label
              htmlFor={`predict-field-${field.name}`}
              className="block text-xs text-muted-foreground mb-1 truncate"
              title={field.name}
            >
              {field.name}
            </label>
            <Input
              id={`predict-field-${field.name}`}
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

      {error && <p className="text-sm text-destructive">{error}</p>}

      {result && (
        <div className="rounded-xl border border-primary/20 bg-primary/10 px-4 py-3">
          <div className="flex items-center gap-2 mb-1">
            <Sparkles size={14} className="text-primary" />
            <p className="text-sm text-foreground">
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
            <p className="text-xs text-muted-foreground tabular-nums">
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
                    <span className="text-xs text-muted-foreground w-24 truncate">{label}</span>
                    <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
                      <div
                        className="h-full rounded-full bg-primary"
                        style={{ width: `${proba * 100}%` }}
                      />
                    </div>
                    <span className="text-xs text-muted-foreground w-10 text-right tabular-nums">
                      {(proba * 100).toFixed(0)} %
                    </span>
                  </div>
                ))}
            </div>
          )}

          <LocalExplanationPanel explanation={result.explanation} />
        </div>
      )}

      <div className="mt-4 pt-4 border-t border-border/60">
        <button
          type="button"
          onClick={() => setShowHistory((v) => !v)}
          className="inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
        >
          <History size={13} aria-hidden="true" />
          {showHistory ? "Masquer l'historique des prédictions" : "Voir l'historique des prédictions"}
        </button>

        {showHistory && (
          <div className="mt-3">
            <Table
              columns={historyColumns}
              rows={history ?? []}
              rowKey={(e) => e.id}
              loading={history === null}
              pageSize={10}
              caption="Historique des prédictions demandées sur ce modèle"
              emptyMessage="Aucune prédiction demandée sur ce modèle pour l'instant."
            />
          </div>
        )}
      </div>
    </Card>
  );
}
