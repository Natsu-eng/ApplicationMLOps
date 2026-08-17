import { useCallback, useEffect, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { AlertTriangle, History, ScatterChart, Shapes } from "lucide-react";
import {
  ApiError,
  api,
  type AnomalyJobSummary,
  type ClusteringJobSummary,
  type DimensionalityJobSummary,
} from "../api/client";
import AppShell from "../components/AppShell";
import { pillarColor } from "../config/pillars";
import { Card } from "../components/ui/Card";
import { ColorIconBadge, accentColorForId } from "../components/ui/ColorIconBadge";
import { PageHeader } from "../components/ui/PageHeader";
import { JobStatusBadge } from "../components/ui/StatusBadge";
import { Tabs, type TabItem } from "../components/ui/Tabs";
import { formatDateTime, formatPercent } from "../utils/format";

type ModuleId = "clustering" | "dimensionality" | "anomalies";

const MODULE_TABS: TabItem<ModuleId>[] = [
  { id: "clustering", label: "Clustering", icon: Shapes },
  { id: "dimensionality", label: "Réduction de dimension", icon: ScatterChart },
  { id: "anomalies", label: "Détection d'anomalies", icon: AlertTriangle },
];

const MODULE_ROUTES: Record<ModuleId, string> = {
  clustering: "/clustering",
  dimensionality: "/reduction-dimension",
  anomalies: "/anomalies",
};

/** Historique du pilier ML non supervisé — jusqu'ici absent : Clustering.tsx/
 * DimensionalityReduction.tsx/AnomalyDetection.tsx ne gardaient qu'UN seul
 * résultat actif en session (sessionStorage), perdu dès qu'on relançait un
 * calcul ou changeait d'onglet — alors que le backend persiste bien chaque
 * job (`GET /clustering|dimensionality|anomalies/jobs`, jamais consommé côté
 * UI). Même rôle que TrainingHistory.tsx pour le supervisé, mais consolidé
 * en une seule page à onglets (3 modules d'un même pilier, pas 3 concepts
 * d'historique séparés). Clic sur une ligne → réouvre le résultat exact via
 * le deep-link `?job=` déjà ajouté aux 3 pages de module. */
export default function UnsupervisedHistory() {
  const [active, setActive] = useState<ModuleId>("clustering");
  const [clusteringJobs, setClusteringJobs] = useState<ClusteringJobSummary[] | null>(null);
  const [dimensionalityJobs, setDimensionalityJobs] = useState<DimensionalityJobSummary[] | null>(null);
  const [anomalyJobs, setAnomalyJobs] = useState<AnomalyJobSummary[] | null>(null);
  const [clusteringError, setClusteringError] = useState<string | null>(null);
  const [dimensionalityError, setDimensionalityError] = useState<string | null>(null);
  const [anomalyError, setAnomalyError] = useState<string | null>(null);

  const load = useCallback(() => {
    api.clustering
      .listJobs()
      .then(setClusteringJobs)
      .catch((err) => setClusteringError(err instanceof ApiError ? err.message : "Impossible de charger l'historique"));
    api.dimensionality
      .listJobs()
      .then(setDimensionalityJobs)
      .catch((err) => setDimensionalityError(err instanceof ApiError ? err.message : "Impossible de charger l'historique"));
    api.anomalies
      .listJobs()
      .then(setAnomalyJobs)
      .catch((err) => setAnomalyError(err instanceof ApiError ? err.message : "Impossible de charger l'historique"));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const counts: Record<ModuleId, number | null> = {
    clustering: clusteringJobs?.length ?? null,
    dimensionality: dimensionalityJobs?.length ?? null,
    anomalies: anomalyJobs?.length ?? null,
  };

  return (
    <AppShell pillarId="unsupervised">
      <PageHeader
        eyebrow="ML non supervisé"
        title="Historique"
        description="Retrouvez vos clusterings, réductions de dimension et détections d'anomalies passés — même après avoir quitté la page ou relancé une nouvelle analyse."
        icon={History}
        color={pillarColor("unsupervised")}
      />

      <div className="mb-5">
        <Tabs items={MODULE_TABS} active={active} onChange={setActive} />
      </div>

      <Card className="p-5">
        {active === "clustering" && (
          <ClusteringHistoryList jobs={clusteringJobs} error={clusteringError} count={counts.clustering} />
        )}
        {active === "dimensionality" && (
          <DimensionalityHistoryList jobs={dimensionalityJobs} error={dimensionalityError} count={counts.dimensionality} />
        )}
        {active === "anomalies" && (
          <AnomalyHistoryList jobs={anomalyJobs} error={anomalyError} count={counts.anomalies} />
        )}
      </Card>
    </AppShell>
  );
}

/** Ligne générique — même structure visuelle pour les 3 listes, contenu au
 * centre libre (résumé propre à chaque module). */
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
  icon: typeof Shapes;
  colorSeed: number;
  primary: string;
  secondary: string;
  status: string;
  right?: ReactNode;
}) {
  return (
    <Link
      to={to}
      className="flex items-center gap-3 py-2.5 px-2 -mx-2 rounded-lg hover:bg-muted/60 transition-colors"
    >
      <ColorIconBadge icon={icon} color={accentColorForId(colorSeed)} size="sm" />
      <div className="min-w-0 flex-1">
        <p className="text-sm text-foreground truncate">{primary}</p>
        <p className="text-xs text-muted-foreground truncate">{secondary}</p>
      </div>
      <div className="flex items-center gap-2 flex-shrink-0">
        {right}
        <JobStatusBadge status={status as ClusteringJobSummary["status"]} />
      </div>
    </Link>
  );
}

/** Fonction pure (PAS un composant JSX) — appelée directement `placeholder(...)`,
 * jamais `<Placeholder .../>` : un élément JSX est toujours un objet "vérité"
 * même quand le composant qu'il décrit rendra `null`, donc `if (<X/>)` est
 * TOUJOURS vrai et ne laisse jamais passer le contenu réel en dessous — bug
 * réel trouvé en testant (l'historique n'affichait jamais la liste, même
 * avec des jobs chargés), corrigé en appelant une fonction normale. */
function renderPlaceholder(jobs: unknown[] | null, error: string | null, emptyLabel: string): ReactNode | null {
  if (error) return <p className="text-sm text-destructive py-4">{error}</p>;
  if (jobs === null) return <p className="text-sm text-muted-foreground py-4">Chargement…</p>;
  if (jobs.length === 0) return <p className="text-sm text-muted-foreground py-4">{emptyLabel}</p>;
  return null;
}

function ClusteringHistoryList({
  jobs,
  error,
  count,
}: {
  jobs: ClusteringJobSummary[] | null;
  error: string | null;
  count: number | null;
}) {
  const placeholder = renderPlaceholder(jobs, error, "Aucun clustering pour l'instant.");
  if (placeholder) return placeholder;
  return (
    <div>
      <p className="text-xs text-muted-foreground mb-2">
        {count} clustering{count !== 1 ? "s" : ""}
      </p>
      <ul className="divide-y divide-border">
        {jobs!.map((job) => (
          <li key={job.id}>
            <HistoryRow
              to={`${MODULE_ROUTES.clustering}?job=${job.id}`}
              icon={Shapes}
              colorSeed={job.id}
              primary={job.dataset_name ?? "Dataset"}
              secondary={`${formatDateTime(job.created_at)}${job.algorithm ? ` · ${job.algorithm}` : ""}${
                job.n_clusters ? ` · ${job.n_clusters} groupes` : ""
              }`}
              status={job.status}
              right={
                job.silhouette !== null ? (
                  <span className="text-xs text-muted-foreground tabular-nums">silhouette = {job.silhouette.toFixed(3)}</span>
                ) : undefined
              }
            />
          </li>
        ))}
      </ul>
    </div>
  );
}

function DimensionalityHistoryList({
  jobs,
  error,
  count,
}: {
  jobs: DimensionalityJobSummary[] | null;
  error: string | null;
  count: number | null;
}) {
  const placeholder = renderPlaceholder(jobs, error, "Aucune réduction de dimension pour l'instant.");
  if (placeholder) return placeholder;
  return (
    <div>
      <p className="text-xs text-muted-foreground mb-2">
        {count} calcul{count !== 1 ? "s" : ""}
      </p>
      <ul className="divide-y divide-border">
        {jobs!.map((job) => (
          <li key={job.id}>
            <HistoryRow
              to={`${MODULE_ROUTES.dimensionality}?job=${job.id}`}
              icon={ScatterChart}
              colorSeed={job.id}
              primary={job.dataset_name ?? "Dataset"}
              secondary={`${formatDateTime(job.created_at)}${job.algorithm ? ` · ${job.algorithm}` : ""}`}
              status={job.status}
              right={
                job.total_variance_explained !== null ? (
                  <span className="text-xs text-muted-foreground tabular-nums">
                    variance PCA = {formatPercent(job.total_variance_explained)}
                  </span>
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
  jobs: AnomalyJobSummary[] | null;
  error: string | null;
  count: number | null;
}) {
  const placeholder = renderPlaceholder(jobs, error, "Aucune détection d'anomalies pour l'instant.");
  if (placeholder) return placeholder;
  return (
    <div>
      <p className="text-xs text-muted-foreground mb-2">
        {count} détection{count !== 1 ? "s" : ""}
      </p>
      <ul className="divide-y divide-border">
        {jobs!.map((job) => (
          <li key={job.id}>
            <HistoryRow
              to={`${MODULE_ROUTES.anomalies}?job=${job.id}`}
              icon={AlertTriangle}
              colorSeed={job.id}
              primary={job.dataset_name ?? "Dataset"}
              secondary={`${formatDateTime(job.created_at)}${
                job.n_anomalies_consensus !== null ? ` · ${job.n_anomalies_consensus} observations atypiques` : ""
              }`}
              status={job.status}
              right={
                job.anomaly_rate_consensus !== null ? (
                  <span className="text-xs text-muted-foreground tabular-nums">
                    {formatPercent(job.anomaly_rate_consensus)}
                  </span>
                ) : undefined
              }
            />
          </li>
        ))}
      </ul>
    </div>
  );
}
