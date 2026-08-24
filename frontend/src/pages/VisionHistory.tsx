import { useCallback, useEffect, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { Boxes, History, Sparkles, Trash2 } from "lucide-react";
import { ApiError, api, type VisionAnomalyJobSummary, type VisionClassificationJobSummary } from "../api/client";
import AppShell from "../components/AppShell";
import { pillarColor } from "../config/pillars";
import { BulkActionBar } from "../components/ui/BulkActionBar";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { ColorIconBadge, accentColorForId } from "../components/ui/ColorIconBadge";
import { PageHeader } from "../components/ui/PageHeader";
import { JobStatusBadge } from "../components/ui/StatusBadge";
import { Tabs, type TabItem } from "../components/ui/Tabs";
import { useConfirmAction } from "../hooks/useConfirmAction";
import { useToast } from "../components/ui/Toast";
import { runBulkDelete } from "../utils/bulkDelete";
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

const MODULE_REMOVE: Record<ModuleId, (id: number) => Promise<void>> = {
  classification: api.visionClassification.remove,
  anomalies: api.visionAnomalies.remove,
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
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [bulkDeleting, setBulkDeleting] = useState(false);
  const bulkConfirm = useConfirmAction<"bulk">();
  const toast = useToast();

  function changeModule(next: ModuleId) {
    setActive(next);
    setSelected(new Set());
  }

  function toggleSelect(id: number) {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  async function handleBulkDelete() {
    if (selected.size === 0) return;
    setBulkDeleting(true);
    try {
      const { succeeded, failed } = await runBulkDelete(Array.from(selected), (id) => MODULE_REMOVE[active](id));
      if (failed === 0) {
        toast.push({ variant: "success", title: `${succeeded} élément${succeeded > 1 ? "s" : ""} supprimé${succeeded > 1 ? "s" : ""}` });
      } else {
        toast.push({
          variant: succeeded === 0 ? "danger" : "warning",
          title: succeeded === 0 ? "Échec de la suppression" : "Suppression partielle",
          description: `${succeeded} réussie${succeeded > 1 ? "s" : ""}, ${failed} échouée${failed > 1 ? "s" : ""}.`,
        });
      }
    } finally {
      setBulkDeleting(false);
      setSelected(new Set());
      load();
    }
  }

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
        <Tabs items={MODULE_TABS} active={active} onChange={changeModule} />
      </div>

      <Card className="p-5">
        {active === "classification" && (
          <ClassificationHistoryList
            jobs={classificationJobs}
            error={classificationError}
            count={counts.classification}
            selected={selected}
            onToggleSelect={toggleSelect}
          />
        )}
        {active === "anomalies" && (
          <AnomalyHistoryList jobs={anomalyJobs} error={anomalyError} count={counts.anomalies} selected={selected} onToggleSelect={toggleSelect} />
        )}
      </Card>

      <BulkActionBar count={selected.size} onClear={() => setSelected(new Set())}>
        <Button
          variant="destructive"
          size="sm"
          loading={bulkDeleting}
          onClick={() => bulkConfirm.trigger("bulk", handleBulkDelete)}
          onMouseLeave={bulkConfirm.reset}
        >
          <Trash2 size={14} aria-hidden="true" />
          {bulkConfirm.isPending("bulk") ? "Confirmer la suppression ?" : "Supprimer"}
        </Button>
      </BulkActionBar>
    </AppShell>
  );
}

/** `id`/`selected`/`onToggleSelect` optionnels (Lot bulk-select) — case à
 * cocher en élément FRÈRE du `<Link>`, jamais imbriquée dedans (même
 * raisonnement que `UnsupervisedHistory.tsx::HistoryRow`). */
function HistoryRow({
  to,
  icon,
  colorSeed,
  primary,
  secondary,
  status,
  right,
  id,
  selected,
  onToggleSelect,
}: {
  to: string;
  icon: typeof Boxes;
  colorSeed: number;
  primary: string;
  secondary: string;
  status: string;
  right?: ReactNode;
  id?: number;
  selected?: boolean;
  onToggleSelect?: (id: number) => void;
}) {
  return (
    <div className="flex items-center gap-1 py-2.5 px-2 -mx-2 rounded-lg hover:bg-muted/60 transition-colors">
      {id !== undefined && onToggleSelect && (
        <input
          type="checkbox"
          aria-label={`Sélectionner ${primary}`}
          checked={selected ?? false}
          onChange={() => onToggleSelect(id)}
          className="rounded border-input flex-shrink-0 mr-1.5"
        />
      )}
      <Link to={to} className="flex items-center gap-3 flex-1 min-w-0">
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
    </div>
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

interface SelectionProps {
  selected: Set<number>;
  onToggleSelect: (id: number) => void;
}

function ClassificationHistoryList({
  jobs,
  error,
  count,
  selected,
  onToggleSelect,
}: {
  jobs: VisionClassificationJobSummary[] | null;
  error: string | null;
  count: number | null;
} & SelectionProps) {
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
              id={job.id}
              selected={selected.has(job.id)}
              onToggleSelect={onToggleSelect}
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
  selected,
  onToggleSelect,
}: {
  jobs: VisionAnomalyJobSummary[] | null;
  error: string | null;
  count: number | null;
} & SelectionProps) {
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
              id={job.id}
              selected={selected.has(job.id)}
              onToggleSelect={onToggleSelect}
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
