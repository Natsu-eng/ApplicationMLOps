import { useCallback, useEffect, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { Boxes, History, Sparkles } from "lucide-react";
import { ApiError, api, type VisionAnomalyJobSummary, type VisionClassificationJobSummary } from "../api/client";
import AppShell from "../components/AppShell";
import { pillarColor } from "../config/pillars";
import { Card } from "../components/ui/Card";
import { ColorIconBadge, accentColorForId } from "../components/ui/ColorIconBadge";
import { PageHeader } from "../components/ui/PageHeader";
import { JobStatusBadge } from "../components/ui/StatusBadge";
import { Tabs, type TabItem } from "../components/ui/Tabs";
import { formatDateTime, formatPercent } from "../utils/format";

type ModuleId = "classification" | "anomalies";

const MODULE_TABS: TabItem<ModuleId>[] = [
  { id: "classification", label: "Classification d'images", icon: Boxes },
  { id: "anomalies", label: "Anomalies visuelles", icon: Sparkles },
];

const MODULE_ROUTES: Record<ModuleId, string> = {
  classification: "/vision/classification",
  anomalies: "/vision/anomalies",
};

/** Historique du pilier Vision (Lot 16E) — même rôle et même structure
 * qu'`UnsupervisedHistory.tsx` : VisionClassification.tsx/VisionAnomalies.tsx
 * ne gardent qu'UN résultat actif en session (sessionStorage), perdu dès
 * qu'on relance un calcul — le backend persiste bien chaque job
 * (`GET /vision/classification|anomalies/jobs`), jamais consommé côté UI
 * jusqu'ici. Clic sur une ligne → réouvre le résultat via le deep-link
 * `?job=` déjà supporté par les deux pages de module. */
export default function VisionHistory() {
  const [active, setActive] = useState<ModuleId>("classification");
  const [classificationJobs, setClassificationJobs] = useState<VisionClassificationJobSummary[] | null>(null);
  const [anomalyJobs, setAnomalyJobs] = useState<VisionAnomalyJobSummary[] | null>(null);
  const [classificationError, setClassificationError] = useState<string | null>(null);
  const [anomalyError, setAnomalyError] = useState<string | null>(null);

  const load = useCallback(() => {
    api.visionClassification
      .listJobs()
      .then(setClassificationJobs)
      .catch((err) => setClassificationError(err instanceof ApiError ? err.message : "Impossible de charger l'historique"));
    api.visionAnomalies
      .listJobs()
      .then(setAnomalyJobs)
      .catch((err) => setAnomalyError(err instanceof ApiError ? err.message : "Impossible de charger l'historique"));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const counts: Record<ModuleId, number | null> = {
    classification: classificationJobs?.length ?? null,
    anomalies: anomalyJobs?.length ?? null,
  };

  return (
    <AppShell pillarId="vision">
      <PageHeader
        eyebrow="Vision"
        title="Historique"
        description="Retrouvez vos classifications d'images et détections d'anomalies visuelles passées — même après avoir quitté la page ou relancé une nouvelle analyse."
        icon={History}
        color={pillarColor("vision")}
      />

      <div className="mb-5">
        <Tabs items={MODULE_TABS} active={active} onChange={setActive} />
      </div>

      <Card className="p-5">
        {active === "classification" && (
          <ClassificationHistoryList jobs={classificationJobs} error={classificationError} count={counts.classification} />
        )}
        {active === "anomalies" && (
          <AnomalyHistoryList jobs={anomalyJobs} error={anomalyError} count={counts.anomalies} />
        )}
      </Card>
    </AppShell>
  );
}

function HistoryRow({
  to,
  icon,
  colorSeed,
  primary,
  secondary,
  status,
  right,
}: {
  to: string;
  icon: typeof Boxes;
  colorSeed: number;
  primary: string;
  secondary: string;
  status: string;
  right?: ReactNode;
}) {
  return (
    <Link to={to} className="flex items-center gap-3 py-2.5 px-2 -mx-2 rounded-lg hover:bg-muted/60 transition-colors">
      <ColorIconBadge icon={icon} color={accentColorForId(colorSeed)} size="sm" />
      <div className="min-w-0 flex-1">
        <p className="text-sm text-foreground truncate">{primary}</p>
        <p className="text-xs text-muted-foreground truncate">{secondary}</p>
      </div>
      <div className="flex items-center gap-2 flex-shrink-0">
        {right}
        <JobStatusBadge status={status as VisionClassificationJobSummary["status"]} />
      </div>
    </Link>
  );
}

// Fonction pure (pas un composant JSX appelé en <Placeholder/>) — même
// correctif que UnsupervisedHistory.tsx (un élément JSX est toujours
// "vérité" même s'il rend null, `if (<X/>)` ne filtre jamais rien).
function renderPlaceholder(jobs: unknown[] | null, error: string | null, emptyLabel: string): ReactNode | null {
  if (error) return <p className="text-sm text-destructive py-4">{error}</p>;
  if (jobs === null) return <p className="text-sm text-muted-foreground py-4">Chargement…</p>;
  if (jobs.length === 0) return <p className="text-sm text-muted-foreground py-4">{emptyLabel}</p>;
  return null;
}

function ClassificationHistoryList({
  jobs,
  error,
  count,
}: {
  jobs: VisionClassificationJobSummary[] | null;
  error: string | null;
  count: number | null;
}) {
  const placeholder = renderPlaceholder(jobs, error, "Aucune classification pour l'instant.");
  if (placeholder) return placeholder;
  return (
    <div>
      <p className="text-xs text-muted-foreground mb-2">
        {count} entraînement{count !== 1 ? "s" : ""}
      </p>
      <ul className="divide-y divide-border">
        {jobs!.map((job) => (
          <li key={job.id}>
            <HistoryRow
              to={`${MODULE_ROUTES.classification}?job=${job.id}`}
              icon={Boxes}
              colorSeed={job.id}
              primary={job.vision_dataset_name ?? "Dataset"}
              secondary={`${formatDateTime(job.created_at)} · ${job.backbone_id}${job.created_by ? ` · ${job.created_by}` : ""}`}
              status={job.status}
              right={
                job.test_accuracy !== null ? (
                  <span className="text-xs text-muted-foreground tabular-nums">exactitude = {formatPercent(job.test_accuracy)}</span>
                ) : undefined
              }
            />
          </li>
        ))}
      </ul>
    </div>
  );
}

function AnomalyHistoryList({
  jobs,
  error,
  count,
}: {
  jobs: VisionAnomalyJobSummary[] | null;
  error: string | null;
  count: number | null;
}) {
  const placeholder = renderPlaceholder(jobs, error, "Aucune détection d'anomalies visuelles pour l'instant.");
  if (placeholder) return placeholder;
  return (
    <div>
      <p className="text-xs text-muted-foreground mb-2">
        {count} entraînement{count !== 1 ? "s" : ""}
      </p>
      <ul className="divide-y divide-border">
        {jobs!.map((job) => (
          <li key={job.id}>
            <HistoryRow
              to={`${MODULE_ROUTES.anomalies}?job=${job.id}`}
              icon={Sparkles}
              colorSeed={job.id}
              primary={job.vision_dataset_name ?? "Dataset"}
              secondary={`${formatDateTime(job.created_at)} · ${job.model_id}${job.created_by ? ` · ${job.created_by}` : ""}`}
              status={job.status}
              right={
                job.roc_auc !== null ? (
                  <span className="text-xs text-muted-foreground tabular-nums">AUC = {job.roc_auc.toFixed(3)}</span>
                ) : undefined
              }
            />
          </li>
        ))}
      </ul>
    </div>
  );
}
