import { useCallback, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Award, BrainCircuit, GitCompareArrows, History, Search, Trash2 } from "lucide-react";
import {
  ApiError,
  api,
  type JobComparisonResponse,
  type JobStatus,
  type ModelRegistryEntry,
  type TrainingJobSummary,
} from "../api/client";
import AppShell from "../components/AppShell";
import { pillarColor } from "../config/pillars";
import { Badge } from "../components/ui/Badge";
import { BulkActionBar } from "../components/ui/BulkActionBar";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { ColorIconBadge, accentColorForId } from "../components/ui/ColorIconBadge";
import { Input } from "../components/ui/Input";
import { PageHeader } from "../components/ui/PageHeader";
import { Select } from "../components/ui/Select";
import { SectionHeader } from "../components/ui/SectionHeader";
import { JobStatusBadge } from "../components/ui/StatusBadge";
import { Table, type TableColumn } from "../components/ui/Table";
import { useConfirmAction } from "../hooks/useConfirmAction";
import { useToast } from "../components/ui/Toast";
import { runBulkDelete } from "../utils/bulkDelete";
import { formatDateTime, formatMetricLabel, formatMetricValue } from "../utils/format";

const STATUS_FILTER_OPTIONS: { value: JobStatus | "all"; label: string }[] = [
  { value: "all", label: "Tous les statuts" },
  { value: "queued", label: "En file" },
  { value: "running", label: "En cours" },
  { value: "completed", label: "Terminé" },
  { value: "failed", label: "Échec" },
  { value: "cancelled", label: "Annulé" },
];

const CONFIG_ROWS: { key: string; label: string }[] = [
  { key: "test_size", label: "Part du jeu de test" },
  { key: "optuna_trials", label: "Essais Optuna" },
  { key: "cv_folds", label: "Blocs de validation croisée" },
  { key: "seed", label: "Graine aléatoire" },
  { key: "cqr_alpha", label: "Alpha CQR (régression)" },
  { key: "class_rebalancing", label: "Rééquilibrage des classes" },
  { key: "model_ids", label: "Modèles comparés (mode expert)" },
];

function formatConfigValue(key: string, value: unknown): string {
  if (value === null || value === undefined) return "Par défaut";
  if (key === "model_ids" && Array.isArray(value)) return value.length > 0 ? value.join(", ") : "Catalogue par défaut";
  if (key === "class_rebalancing") return value ? "Activé" : "Désactivé";
  if (typeof value === "number") return formatMetricValue(value);
  return String(value);
}

/** Historique complet des entraînements — sélection multiple + comparaison
 * inter-jobs (Lot D-bis). Complète le Lot D (leaderboard intra-job, déjà
 * sur la page Résultats) : ici on compare plusieurs ENTRAÎNEMENTS entre
 * eux (config, métriques), pas seulement les modèles d'un même job. Devient
 * aussi la page "Voir tout" du dashboard, qui pointait jusqu'ici vers le
 * formulaire d'entraînement (aucun historique n'y était consultable). */
export default function TrainingHistory() {
  const [jobs, setJobs] = useState<TrainingJobSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [comparison, setComparison] = useState<JobComparisonResponse | null>(null);
  const [comparing, setComparing] = useState(false);
  const [compareError, setCompareError] = useState<string | null>(null);
  const [registry, setRegistry] = useState<ModelRegistryEntry[] | null>(null);
  const [registryError, setRegistryError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState<JobStatus | "all">("all");
  const [bulkDeleting, setBulkDeleting] = useState(false);
  const bulkConfirm = useConfirmAction<"bulk">();
  const toast = useToast();

  const load = useCallback(() => {
    api.training
      .listJobs()
      .then(setJobs)
      .catch((err) => setError(err instanceof ApiError ? err.message : "Impossible de charger l'historique"));
    api.training
      .registry()
      .then((r) => {
        setRegistry(r.entries);
        setRegistryError(null);
      })
      // AUDIT_ROADMAP.md, H4/D3 : un échec ici faisait disparaître le
      // panneau "Registre de modèles" sans aucun indice — indiscernable de
      // "rien n'a encore été promu".
      .catch((err) => setRegistryError(err instanceof ApiError ? err.message : "Registre indisponible"));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  function handleSelectionChange(keys: Set<string | number>) {
    setSelected(keys as Set<number>);
    setComparison(null);
  }

  async function handleCompare() {
    setComparing(true);
    setCompareError(null);
    try {
      setComparison(await api.training.compareJobs(Array.from(selected)));
    } catch (err) {
      setCompareError(err instanceof ApiError ? err.message : "Comparaison impossible");
    } finally {
      setComparing(false);
    }
  }

  async function handleBulkDelete() {
    if (selected.size === 0) return;
    setBulkDeleting(true);
    try {
      const { succeeded, failed } = await runBulkDelete(Array.from(selected), (id) => api.training.remove(id));
      if (failed === 0) {
        toast.push({ variant: "success", title: `${succeeded} entraînement${succeeded > 1 ? "s" : ""} supprimé${succeeded > 1 ? "s" : ""}` });
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
      setComparison(null);
      load();
    }
  }

  const metricKeys = useMemo(() => {
    if (!comparison) return [];
    const keys = new Set<string>();
    comparison.entries.forEach((e) =>
      Object.entries(e.metrics).forEach(([k, v]) => {
        if (typeof v === "number") keys.add(k);
      }),
    );
    // cv_score/headline en tête, puis le reste dans un ordre stable.
    return Array.from(keys).sort((a, b) => (a === "cv_score" ? -1 : b === "cv_score" ? 1 : a.localeCompare(b)));
  }, [comparison]);

  const filteredJobs = useMemo(() => {
    if (!jobs) return [];
    const term = search.trim().toLowerCase();
    return jobs.filter((job) => {
      if (statusFilter !== "all" && job.status !== statusFilter) return false;
      if (!term) return true;
      const haystack = `${job.dataset_name ?? ""} ${job.target_column} ${job.algorithm ?? ""}`.toLowerCase();
      return haystack.includes(term);
    });
  }, [jobs, search, statusFilter]);

  const columns = useMemo<TableColumn<TrainingJobSummary>[]>(
    () => [
      {
        key: "dataset",
        header: "Dataset → Cible",
        sortable: true,
        sortValue: (job) => job.dataset_name ?? "",
        sticky: true,
        render: (job) => (
          <div className="flex items-center gap-2.5 min-w-0">
            <ColorIconBadge icon={BrainCircuit} color={accentColorForId(job.id)} size="sm" />
            <div className="min-w-0">
              <p className="text-sm text-foreground truncate">
                {job.dataset_name ?? "Dataset"} <span className="text-muted-foreground">→</span> {job.target_column}
              </p>
            </div>
          </div>
        ),
      },
      {
        key: "algorithm",
        header: "Algorithme",
        sortable: true,
        sortValue: (job) => job.algorithm ?? "",
        render: (job) => job.algorithm ?? "—",
      },
      {
        key: "metric",
        header: "Métrique",
        align: "right",
        sortValue: (job) => job.headline_metric?.value ?? null,
        render: (job) =>
          job.headline_metric?.value !== null && job.headline_metric?.value !== undefined ? (
            <span className="text-xs text-muted-foreground tabular-nums">
              {job.headline_metric.name} = {job.headline_metric.value.toFixed(3)}
            </span>
          ) : (
            "—"
          ),
      },
      {
        key: "status",
        header: "Statut",
        sortable: true,
        sortValue: (job) => job.status,
        render: (job) => <JobStatusBadge status={job.status} />,
      },
      {
        key: "created_by",
        header: "Auteur",
        sortable: true,
        sortValue: (job) => job.created_by ?? "",
        render: (job) => job.created_by ?? "—",
      },
      {
        key: "created_at",
        header: "Date",
        align: "right",
        sortable: true,
        sortValue: (job) => job.created_at,
        render: (job) => formatDateTime(job.created_at),
      },
      {
        key: "actions",
        header: "",
        align: "right",
        render: (job) => (
          <Link to={`/training?job=${job.id}`} className="text-xs text-primary hover:text-primary/80 whitespace-nowrap">
            Voir →
          </Link>
        ),
      },
    ],
    [],
  );

  return (
    <AppShell pillarId="supervised">
      <PageHeader
        eyebrow="Historique"
        title="Entraînements"
        description="Sélectionnez au moins deux entraînements pour les comparer côte à côte."
        icon={History}
        color={pillarColor("supervised")}
        action={
          <Button onClick={handleCompare} disabled={selected.size < 2 || comparing}>
            <GitCompareArrows size={15} />
            {comparing ? "Comparaison…" : `Comparer (${selected.size})`}
          </Button>
        }
      />

      {error && <p className="text-sm text-destructive mb-4">{error}</p>}
      {registryError && <p className="text-sm text-destructive mb-4">{registryError}</p>}

      {registry && registry.length > 0 && (
        <Card className="p-5 mb-6">
          <SectionHeader
            icon={Award}
            color="violet"
            label="Registre de modèles"
            help="Les modèles explicitement promus (validation ou production) depuis la page Résultats d'un entraînement — pas un doublon de l'historique complet ci-dessous, seulement ce qui a été retenu."
          />
          <ul className="divide-y divide-border">
            {registry.map((entry) => (
              <li key={entry.model_id} className="py-2.5 flex items-center gap-3">
                <ColorIconBadge icon={Award} color={accentColorForId(entry.model_id)} size="sm" />
                <div className="min-w-0 flex-1">
                  <p className="text-sm text-foreground truncate">
                    {entry.dataset_name ?? "Dataset"} <span className="text-muted-foreground">→</span> {entry.target_column}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {entry.algorithm}
                    {entry.promoted_at ? ` · promu le ${formatDateTime(entry.promoted_at)}` : ""}
                  </p>
                </div>
                <div className="flex items-center gap-2 flex-shrink-0">
                  {entry.headline_metric?.value !== null && entry.headline_metric?.value !== undefined && (
                    <span className="text-xs text-muted-foreground tabular-nums">
                      {entry.headline_metric.name} = {entry.headline_metric.value.toFixed(3)}
                    </span>
                  )}
                  <Badge variant={entry.stage === "production" ? "success" : "warning"} dot>
                    {entry.stage === "production" ? "Production" : "Staging"}
                  </Badge>
                </div>
              </li>
            ))}
          </ul>
        </Card>
      )}

      <Card className="p-5 mb-6">
        <SectionHeader icon={History} color="blue" label="Tous les entraînements" />

        <div className="grid gap-3 sm:grid-cols-3 mb-4">
          <div className="relative sm:col-span-2">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground pointer-events-none" />
            <Input
              placeholder="Rechercher un dataset, une cible, un algorithme…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="pl-9"
            />
          </div>
          <Select aria-label="Filtrer par statut" value={statusFilter} onChange={(e) => setStatusFilter(e.target.value as JobStatus | "all")}>
            {STATUS_FILTER_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </Select>
        </div>

        <Table
          columns={columns}
          rows={filteredJobs}
          rowKey={(job) => job.id}
          caption="Historique de tous les entraînements supervisés"
          loading={jobs === null && !error}
          pageSize={15}
          selectable
          selectedKeys={selected}
          onSelectionChange={handleSelectionChange}
          emptyMessage={
            jobs && jobs.length > 0 ? "Aucun entraînement ne correspond à ces filtres." : "Aucun entraînement pour l'instant."
          }
        />
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

      {compareError && <p className="text-sm text-destructive mb-4">{compareError}</p>}

      {comparison && (
        <Card className="p-5 overflow-x-auto">
          <SectionHeader
            icon={GitCompareArrows}
            color="violet"
            label="Comparaison"
            help="Les lignes surlignées correspondent à des réglages qui diffèrent entre au moins deux des entraînements sélectionnés."
          />
          <table className="min-w-full text-sm border-separate border-spacing-0">
            <thead>
              <tr>
                <th className="text-left px-3 py-2 text-xs font-medium text-muted-foreground sticky left-0 bg-card">
                  &nbsp;
                </th>
                {comparison.entries.map((e) => (
                  <th key={e.job_id} className="text-left px-3 py-2 min-w-[160px]">
                    <p className="text-sm font-medium text-foreground truncate">{e.dataset_name ?? "Dataset"}</p>
                    <p className="text-xs text-muted-foreground truncate">→ {e.target_column}</p>
                    <div className="mt-1 flex items-center gap-1.5">
                      <JobStatusBadge status={e.status} />
                      {e.algorithm && <span className="text-xs text-muted-foreground">{e.algorithm}</span>}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              <tr className="bg-muted">
                <td className="px-3 py-2 text-xs font-semibold text-muted-foreground sticky left-0 bg-muted" colSpan={comparison.entries.length + 1}>
                  Métriques
                </td>
              </tr>
              {metricKeys.map((key) => (
                <tr key={key} className="border-t border-border/60">
                  <td className="px-3 py-2 text-xs text-muted-foreground sticky left-0 bg-card">{formatMetricLabel(key)}</td>
                  {comparison.entries.map((e) => {
                    const value = e.metrics[key];
                    return (
                      <td key={e.job_id} className="px-3 py-2 text-sm tabular-nums text-foreground">
                        {typeof value === "number" ? formatMetricValue(value) : "—"}
                      </td>
                    );
                  })}
                </tr>
              ))}

              <tr className="bg-muted">
                <td className="px-3 py-2 text-xs font-semibold text-muted-foreground sticky left-0 bg-muted" colSpan={comparison.entries.length + 1}>
                  Configuration
                </td>
              </tr>
              {CONFIG_ROWS.map((row) => {
                const differs = comparison.differing_config_fields.includes(row.key);
                return (
                  <tr key={row.key} className={`border-t border-border/60 ${differs ? "bg-warning/10" : ""}`}>
                    <td className={`px-3 py-2 text-xs text-muted-foreground sticky left-0 ${differs ? "bg-warning/15" : "bg-card"}`}>
                      {row.label}
                      {differs && <span className="ml-1 text-warning">●</span>}
                    </td>
                    {comparison.entries.map((e) => (
                      <td key={e.job_id} className="px-3 py-2 text-sm text-foreground">
                        {formatConfigValue(row.key, e.config[row.key])}
                      </td>
                    ))}
                  </tr>
                );
              })}
              <tr className="border-t border-border/60">
                <td className="px-3 py-2 text-xs text-muted-foreground sticky left-0 bg-card">Ingénierie de variables</td>
                {comparison.entries.map((e) => (
                  <td key={e.job_id} className="px-3 py-2 text-sm text-foreground">
                    {e.feature_engineering_active ? "Activée" : "Non appliquée"}
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </Card>
      )}
    </AppShell>
  );
}
