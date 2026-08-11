import { useEffect, useState } from "react";
import { AlertTriangle, ChevronDown, Info, ShieldAlert, ShieldCheck } from "lucide-react";
import { ApiError, api, type DataWarning } from "../../api/client";
import { Badge } from "../ui/Badge";

/** Garde-fous de données (Lot B) — avertissements affichés au moment du
 * choix dataset+cible, AVANT le lancement de l'entraînement. Toujours
 * informatif, jamais bloquant : même une fuite critique n'empêche pas de
 * lancer l'entraînement, elle est juste signalée clairement (on propose,
 * on n'impose pas). */

const LEVEL_CONFIG: Record<
  DataWarning["level"],
  { badge: "danger" | "warning" | "accent"; icon: typeof ShieldAlert; border: string; iconColor: string }
> = {
  critique: { badge: "danger", icon: ShieldAlert, border: "border-rose-200", iconColor: "text-rose-600" },
  attention: { badge: "warning", icon: AlertTriangle, border: "border-amber-200", iconColor: "text-amber-600" },
  info: { badge: "accent", icon: Info, border: "border-primary/20", iconColor: "text-primary" },
};

const LEVEL_LABEL: Record<DataWarning["level"], string> = {
  critique: "Critique",
  attention: "Attention",
  info: "Info",
};

export function DataQualityWarnings({
  datasetId,
  targetColumn,
  groupColumn,
}: {
  datasetId: number;
  targetColumn: string;
  groupColumn?: string;
}) {
  const [warnings, setWarnings] = useState<DataWarning[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [expanded, setExpanded] = useState<Set<number>>(new Set());

  useEffect(() => {
    if (!targetColumn) {
      setWarnings(null);
      return;
    }
    setLoading(true);
    setError(null);
    setExpanded(new Set());
    api.datasets
      .qualityCheck(datasetId, targetColumn, groupColumn)
      .then((data) => setWarnings(data.warnings))
      .catch((err) => setError(err instanceof ApiError ? err.message : "Analyse des données indisponible"))
      .finally(() => setLoading(false));
  }, [datasetId, targetColumn, groupColumn]);

  if (!targetColumn) return null;

  function toggle(index: number) {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  }

  return (
    <div className="space-y-2">
      {loading && <p className="text-xs text-slate-500">Analyse des données…</p>}
      {error && <p className="text-xs text-rose-600">{error}</p>}

      {!loading && !error && warnings && warnings.length === 0 && (
        <div className="flex items-center gap-2 text-xs text-emerald-700 bg-emerald-50 border border-emerald-200 rounded-lg px-3 py-2">
          <ShieldCheck size={14} className="flex-shrink-0" />
          Aucune alerte détectée sur ce dataset pour cette cible.
        </div>
      )}

      {!loading &&
        !error &&
        warnings &&
        warnings.length > 0 &&
        warnings.map((warning, index) => {
          const config = LEVEL_CONFIG[warning.level];
          const Icon = config.icon;
          const isOpen = expanded.has(index);
          return (
            <div
              key={`${warning.code}-${warning.columns.join(",")}-${index}`}
              className={`rounded-lg border ${config.border} bg-slate-50 px-3 py-2.5`}
            >
              <button
                type="button"
                onClick={() => toggle(index)}
                className="w-full flex items-start gap-2 text-left"
              >
                <Icon size={15} className={`flex-shrink-0 mt-0.5 ${config.iconColor}`} />
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2 flex-wrap">
                    <Badge variant={config.badge}>{LEVEL_LABEL[warning.level]}</Badge>
                    <p className="text-sm text-slate-800 font-medium">{warning.title}</p>
                  </div>
                </div>
                <ChevronDown
                  size={14}
                  className={`flex-shrink-0 text-slate-400 transition-transform mt-1 ${isOpen ? "rotate-180" : ""}`}
                />
              </button>

              {isOpen && <p className="text-xs text-slate-600 mt-2 pl-6">{warning.explanation}</p>}

              <p className="text-xs text-slate-600 mt-2 pl-6">
                <span className="text-slate-500">Action recommandée : </span>
                {warning.action}
              </p>
            </div>
          );
        })}
    </div>
  );
}
