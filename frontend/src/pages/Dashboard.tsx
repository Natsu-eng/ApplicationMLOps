import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import {
  Activity,
  AlertTriangle,
  BrainCircuit,
  Database,
  FileSpreadsheet,
  LayoutDashboard,
  ScatterChart,
  Shapes,
  Trash2,
  Users,
  type LucideIcon,
} from "lucide-react";
import {
  ApiError,
  api,
  type AnomalyJobSummary,
  type ClusteringJobSummary,
  type DatasetSummary,
  type DimensionalityJobSummary,
  type JobStatus,
  type TrainingJobSummary,
} from "../api/client";
import { useAuth } from "../contexts/AuthContext";
import AppShell from "../components/AppShell";
import { StatTile, StatTileRow } from "../components/dashboard/StatTile";
import ModelResultModal from "../components/training/ModelResultModal";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { ColorIconBadge, accentColorForId, type AccentColor } from "../components/ui/ColorIconBadge";
import { ErrorNote } from "../components/ui/ErrorNote";
import { PageHeader } from "../components/ui/PageHeader";
import { DatasetStatusBadge, JobStatusBadge } from "../components/ui/StatusBadge";
import { useConfirmAction } from "../hooks/useConfirmAction";
import { formatDateTime, formatPercent } from "../utils/format";

const ACTIVE_STATUSES = new Set(["queued", "running"]);

/** Salutation contextuelle à l'heure réelle du navigateur — jamais un texte
 * figé. Trois tranches (retour utilisateur explicite : deux ne suffisaient
 * pas à distinguer le milieu de journée) : "Bonjour" le matin (5h–12h),
 * "Bon après-midi" l'après-midi (12h–18h), "Bonsoir" le soir/la nuit
 * (18h–5h) — convention FR usuelle. */
function greeting(hour: number): string {
  if (hour >= 5 && hour < 12) return "Bonjour";
  if (hour >= 12 && hour < 18) return "Bon après-midi";
  return "Bonsoir";
}

type ActivityKind = "supervised" | "clustering" | "dimensionality" | "anomalies";

interface ActivityItem {
  kind: ActivityKind;
  id: number;
  datasetName: string;
  detailLabel: string;
  createdAt: string;
  status: JobStatus;
  headline: string | null;
  href: string;
  /** Uniquement pour "supervised" — ouvre ModelResultModal en place plutôt
   * que de naviguer, comportement d'origine conservé tel quel. */
  raw?: TrainingJobSummary;
}

const ACTIVITY_KIND_META: Record<ActivityKind, { icon: LucideIcon; color: AccentColor; label: string }> = {
  supervised: { icon: BrainCircuit, color: "violet", label: "Entraînement" },
  clustering: { icon: Shapes, color: "rose", label: "Clustering" },
  dimensionality: { icon: ScatterChart, color: "blue", label: "Réduction de dimension" },
  anomalies: { icon: AlertTriangle, color: "amber", label: "Détection d'anomalies" },
};

/** Page protégée du Lot 1 — vue d'ensemble de l'ACTIVITÉ ML, pas d'un seul
 * pilier. Jusqu'ici "Derniers entraînements" ne montrait que le supervisé
 * (TrainingJob) : un dashboard qui s'annonce général ne peut pas ignorer
 * clustering/réduction de dimension/anomalies — retour utilisateur direct.
 * La gestion d'équipe (profil, membres, journal d'audit) a déménagé sur
 * `/profile` (Lot Profil) : ce n'est pas de l'activité, c'est de
 * l'administration, elle encombrait cette page sans rapport avec son objet.
 * Pilier-agnostique (pas de `pillarId` passé à `AppShell`) — épinglé en haut
 * de la sidebar, accessible depuis n'importe quel pilier sans jamais forcer
 * un changement de contexte de navigation (bug réel corrigé : cliquer
 * "Tableau de bord" depuis le pilier non supervisé bascule sinon la sidebar
 * sur "ML supervisé" par surprise). */
export default function Dashboard() {
  const { user } = useAuth();

  const [members, setMembers] = useState<{ id: number }[] | null>(null);
  const [datasets, setDatasets] = useState<DatasetSummary[] | null>(null);
  const [datasetsError, setDatasetsError] = useState<string | null>(null);
  const [jobs, setJobs] = useState<TrainingJobSummary[] | null>(null);
  const [jobsError, setJobsError] = useState<string | null>(null);
  const [clusteringJobs, setClusteringJobs] = useState<ClusteringJobSummary[] | null>(null);
  const [dimensionalityJobs, setDimensionalityJobs] = useState<DimensionalityJobSummary[] | null>(null);
  const [anomalyJobs, setAnomalyJobs] = useState<AnomalyJobSummary[] | null>(null);
  const [unsupervisedJobsError, setUnsupervisedJobsError] = useState<string | null>(null);
  const [viewingJob, setViewingJob] = useState<TrainingJobSummary | null>(null);
  const confirmDeleteJob = useConfirmAction<number>();

  // Résultat "deep-linkable" (AUDIT_ROADMAP.md, H20/D12) — `?job=<id>`
  // synchronise l'URL avec la modale ouverte, dans les deux sens.
  const [searchParams, setSearchParams] = useSearchParams();

  useEffect(() => {
    const jobId = searchParams.get("job");
    if (!jobId) return;
    api.training
      .getJob(Number(jobId))
      .then(setViewingJob)
      .catch(() => setSearchParams({}, { replace: true }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function openJob(job: TrainingJobSummary) {
    setViewingJob(job);
    setSearchParams({ job: String(job.id) }, { replace: false });
  }

  function closeJob() {
    setViewingJob(null);
    setSearchParams({}, { replace: false });
  }

  const loadMembers = useCallback(async () => {
    try {
      setMembers(await api.team.members());
    } catch {
      // Tuile statistique seulement ici (la gestion d'équipe complète, avec
      // ses propres erreurs distinctes, vit sur /profile) — un échec reste
      // silencieux sur la tuile ("—"), sans bannière dupliquée.
    }
  }, []);

  const loadJobs = useCallback(async () => {
    try {
      setJobs(await api.training.listJobs());
      setJobsError(null);
    } catch (err) {
      // AUDIT_ROADMAP.md, H4/D3 : distinguer "aucun entraînement" (liste
      // vide légitime) d'un échec réseau.
      setJobsError(err instanceof ApiError ? err.message : "Impossible de charger les entraînements");
    }
  }, []);

  const loadUnsupervisedJobs = useCallback(() => {
    api.clustering
      .listJobs()
      .then(setClusteringJobs)
      .catch((err) => setUnsupervisedJobsError(err instanceof ApiError ? err.message : "Impossible de charger l'activité non supervisée"));
    api.dimensionality
      .listJobs()
      .then(setDimensionalityJobs)
      .catch((err) => setUnsupervisedJobsError(err instanceof ApiError ? err.message : "Impossible de charger l'activité non supervisée"));
    api.anomalies
      .listJobs()
      .then(setAnomalyJobs)
      .catch((err) => setUnsupervisedJobsError(err instanceof ApiError ? err.message : "Impossible de charger l'activité non supervisée"));
  }, []);

  const loadDatasets = useCallback(async () => {
    try {
      setDatasets(await api.datasets.list());
      setDatasetsError(null);
    } catch (err) {
      setDatasetsError(err instanceof ApiError ? err.message : "Impossible de charger les datasets");
    }
  }, []);

  useEffect(() => {
    loadMembers();
    loadDatasets();
    loadJobs();
    loadUnsupervisedJobs();
  }, [loadMembers, loadDatasets, loadJobs, loadUnsupervisedJobs]);

  async function handleDeleteJob(id: number) {
    try {
      await api.training.remove(id);
      loadJobs();
    } catch {
      // best-effort — la liste se resynchronisera au prochain chargement
    }
  }

  const allJobsLoaded = jobs !== null && clusteringJobs !== null && dimensionalityJobs !== null && anomalyJobs !== null;
  const unsupervisedJobsCount = (clusteringJobs?.length ?? 0) + (dimensionalityJobs?.length ?? 0) + (anomalyJobs?.length ?? 0);
  const totalJobsCount = (jobs?.length ?? 0) + unsupervisedJobsCount;
  const activeJobsCount = [jobs, clusteringJobs, dimensionalityJobs, anomalyJobs].reduce(
    (sum, list) => sum + (list?.filter((j) => ACTIVE_STATUSES.has(j.status)).length ?? 0),
    0,
  );

  const activity = useMemo<ActivityItem[]>(() => {
    const items: ActivityItem[] = [];
    jobs?.forEach((j) =>
      items.push({
        kind: "supervised",
        id: j.id,
        datasetName: j.dataset_name ?? "Dataset",
        detailLabel: `→ ${j.target_column}`,
        createdAt: j.created_at,
        status: j.status,
        headline:
          j.headline_metric && j.headline_metric.value !== null && j.headline_metric.value !== undefined
            ? `${j.headline_metric.name} = ${j.headline_metric.value.toFixed(3)}`
            : null,
        href: "",
        raw: j,
      }),
    );
    clusteringJobs?.forEach((j) =>
      items.push({
        kind: "clustering",
        id: j.id,
        datasetName: j.dataset_name ?? "Dataset",
        detailLabel: j.n_clusters ? `${j.n_clusters} groupes` : (j.algorithm ?? "Clustering"),
        createdAt: j.created_at,
        status: j.status,
        headline: j.silhouette !== null ? `silhouette = ${j.silhouette.toFixed(3)}` : null,
        href: `/clustering?job=${j.id}`,
      }),
    );
    dimensionalityJobs?.forEach((j) =>
      items.push({
        kind: "dimensionality",
        id: j.id,
        datasetName: j.dataset_name ?? "Dataset",
        detailLabel: j.algorithm ?? "Réduction de dimension",
        createdAt: j.created_at,
        status: j.status,
        headline: j.total_variance_explained !== null ? `variance PCA = ${formatPercent(j.total_variance_explained)}` : null,
        href: `/reduction-dimension?job=${j.id}`,
      }),
    );
    anomalyJobs?.forEach((j) =>
      items.push({
        kind: "anomalies",
        id: j.id,
        datasetName: j.dataset_name ?? "Dataset",
        detailLabel: j.n_anomalies_consensus !== null ? `${j.n_anomalies_consensus} atypiques` : "Détection d'anomalies",
        createdAt: j.created_at,
        status: j.status,
        headline: j.anomaly_rate_consensus !== null ? formatPercent(j.anomaly_rate_consensus) : null,
        href: `/anomalies?job=${j.id}`,
      }),
    );
    items.sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());
    return items.slice(0, 6);
  }, [jobs, clusteringJobs, dimensionalityJobs, anomalyJobs]);

  const recentDatasets = datasets?.slice(0, 5) ?? [];

  if (!user) return null;

  return (
    <AppShell>
      <PageHeader
        eyebrow="Vue d'ensemble"
        title={`${greeting(new Date().getHours())}, ${user.nom.split(" ")[0]}`}
        description={`${user.organization_name} — l'activité récente de votre équipe, tous piliers confondus.`}
        icon={LayoutDashboard}
        color="blue"
        action={
          <Link to="/training">
            <Button>
              <BrainCircuit size={15} />
              Nouvel entraînement
            </Button>
          </Link>
        }
      />

      <StatTileRow wide>
        <StatTile icon={Database} label="Datasets" value={datasets?.length} color="blue" delayMs={0} />
        <StatTile
          icon={Activity}
          label="Analyses ML"
          value={allJobsLoaded ? totalJobsCount : undefined}
          color="teal"
          delayMs={60}
          split={[
            { label: "Supervisé", value: allJobsLoaded ? jobs!.length : undefined },
            { label: "Non supervisé", value: allJobsLoaded ? unsupervisedJobsCount : undefined },
          ]}
        />
        <StatTile
          icon={Activity}
          label="En cours"
          value={allJobsLoaded ? activeJobsCount : undefined}
          color="amber"
          delayMs={120}
        />
        <StatTile icon={Users} label="Membres de l'équipe" value={members?.length} color="violet" delayMs={180} />
      </StatTileRow>

      <div className="grid gap-6 lg:grid-cols-2 mb-10">
        <Card className="p-5">
          <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
            <h2 className="text-sm font-medium text-foreground">Dernière activité</h2>
            <div className="flex items-center gap-3 text-xs">
              <Link to="/training/history" className="text-primary hover:text-primary/80">
                Historique supervisé
              </Link>
              <Link to="/non-supervise/historique" className="text-primary hover:text-primary/80">
                Historique non supervisé
              </Link>
            </div>
          </div>

          {jobsError || unsupervisedJobsError ? (
            <ErrorNote message={jobsError ?? unsupervisedJobsError ?? ""} />
          ) : !allJobsLoaded ? (
            <p className="text-sm text-muted-foreground">Chargement…</p>
          ) : activity.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              Aucune analyse pour l'instant — lancez-en une depuis{" "}
              <Link to="/training" className="text-primary hover:text-primary/80">
                Entraînement
              </Link>{" "}
              ou un module de{" "}
              <Link to="/clustering" className="text-primary hover:text-primary/80">
                ML non supervisé
              </Link>
              .
            </p>
          ) : (
            <ul className="divide-y divide-border">
              {activity.map((item) => {
                const meta = ACTIVITY_KIND_META[item.kind];
                const isCompleted = item.status === "completed";
                const leftContent = (
                  <div className="flex items-center gap-3 min-w-0">
                    <ColorIconBadge icon={meta.icon} color={meta.color} size="sm" />
                    <div className="min-w-0">
                      <p className="text-sm text-foreground/90 truncate">
                        {item.datasetName} <span className="text-muted-foreground">·</span> {item.detailLabel}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {meta.label} · {formatDateTime(item.createdAt)}
                      </p>
                    </div>
                  </div>
                );
                const headlineAndStatus = (
                  <>
                    {isCompleted && item.headline && (
                      <span className="text-xs text-muted-foreground tabular-nums">{item.headline}</span>
                    )}
                    <JobStatusBadge status={item.status} />
                  </>
                );
                const rowContent = (
                  <>
                    {leftContent}
                    <div className="flex items-center gap-2 flex-shrink-0">{headlineAndStatus}</div>
                  </>
                );

                if (item.kind === "supervised") {
                  const pendingDelete = confirmDeleteJob.isPending(item.id);
                  return (
                    <li
                      key={`supervised-${item.id}`}
                      onClick={() => isCompleted && item.raw && openJob(item.raw)}
                      className={`group py-2.5 flex items-center justify-between gap-3 ${
                        isCompleted ? "cursor-pointer hover:bg-muted/50 -mx-1 px-1 rounded-lg transition-colors" : ""
                      }`}
                    >
                      {leftContent}
                      <div className="flex items-center gap-2 flex-shrink-0">
                        {headlineAndStatus}
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            confirmDeleteJob.trigger(item.id, () => handleDeleteJob(item.id));
                          }}
                          onMouseLeave={confirmDeleteJob.reset}
                          aria-label={pendingDelete ? "Confirmer la suppression" : "Supprimer cet entraînement"}
                          title={pendingDelete ? "Cliquer à nouveau pour confirmer" : "Supprimer cet entraînement"}
                          className={`flex-shrink-0 h-7 w-7 flex items-center justify-center rounded-md transition-colors ${
                            pendingDelete
                              ? "text-destructive bg-destructive/15"
                              : "text-muted-foreground/50 hover:text-destructive hover:bg-destructive/10"
                          }`}
                        >
                          <Trash2 size={13} />
                        </button>
                      </div>
                    </li>
                  );
                }

                return (
                  <li key={`${item.kind}-${item.id}`}>
                    {isCompleted ? (
                      <Link
                        to={item.href}
                        className="py-2.5 flex items-center justify-between gap-3 -mx-1 px-1 rounded-lg hover:bg-muted/50 transition-colors"
                      >
                        {rowContent}
                      </Link>
                    ) : (
                      <div className="py-2.5 flex items-center justify-between gap-3">{rowContent}</div>
                    )}
                  </li>
                );
              })}
            </ul>
          )}
        </Card>

        <Card className="p-5">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-medium text-foreground">Derniers datasets</h2>
            <Link to="/datasets" className="text-xs text-primary hover:text-primary/80">
              Voir tout
            </Link>
          </div>

          {datasetsError ? (
            <ErrorNote message={datasetsError} />
          ) : datasets === null ? (
            <p className="text-sm text-muted-foreground">Chargement…</p>
          ) : recentDatasets.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              Aucun dataset pour l'instant — importez-en un depuis{" "}
              <Link to="/datasets" className="text-primary hover:text-primary/80">
                Mes données
              </Link>
              .
            </p>
          ) : (
            <ul className="divide-y divide-border">
              {recentDatasets.map((dataset) => (
                <li key={dataset.id} className="group py-2.5 flex items-center justify-between gap-3 hover:bg-muted/50 -mx-1 px-1 rounded-lg transition-colors">
                  <div className="flex items-center gap-3 min-w-0">
                    <ColorIconBadge icon={FileSpreadsheet} color={accentColorForId(dataset.id)} size="sm" />
                    <div className="min-w-0">
                      <p className="text-sm text-foreground/90 truncate">{dataset.name}</p>
                      <p className="text-xs text-muted-foreground">
                        {dataset.row_count ?? "—"} lignes · {dataset.column_count ?? "—"} colonnes
                      </p>
                    </div>
                  </div>
                  <DatasetStatusBadge status={dataset.status} />
                </li>
              ))}
            </ul>
          )}
        </Card>
      </div>

      {viewingJob && <ModelResultModal job={viewingJob} onClose={closeJob} />}
    </AppShell>
  );
}
