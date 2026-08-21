import { useCallback, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import {
  AlertTriangle,
  BrainCircuit,
  Boxes,
  History,
  ScatterChart,
  Search,
  Shapes,
  Sparkles,
  type LucideIcon,
} from "lucide-react";
import { api, type JobStatus } from "../api/client";
import AppShell from "../components/AppShell";
import { Card } from "../components/ui/Card";
import { ColorIconBadge, type AccentColor } from "../components/ui/ColorIconBadge";
import { ErrorNote } from "../components/ui/ErrorNote";
import { Input } from "../components/ui/Input";
import { PageHeader } from "../components/ui/PageHeader";
import { Select } from "../components/ui/Select";
import { JobStatusBadge } from "../components/ui/StatusBadge";
import { Table, type TableColumn } from "../components/ui/Table";
import { formatDateTime, formatPercent } from "../utils/format";

type JobKind = "supervised" | "clustering" | "dimensionality" | "anomalies" | "vision_classification" | "vision_anomalies";

interface HistoryEntry {
  key: string;
  kind: JobKind;
  id: number;
  datasetName: string;
  detailLabel: string;
  status: JobStatus;
  headline: string | null;
  createdBy: string | null;
  createdAt: string;
  href: string;
}

// Même palette icône/couleur/libellé que Dashboard.tsx (ACTIVITY_KIND_META) —
// une seule vérité pour "à quoi ressemble ce type d'analyse" dans toute l'app.
const KIND_META: Record<JobKind, { icon: LucideIcon; color: AccentColor; label: string }> = {
  supervised: { icon: BrainCircuit, color: "violet", label: "Entraînement" },
  clustering: { icon: Shapes, color: "rose", label: "Clustering" },
  dimensionality: { icon: ScatterChart, color: "blue", label: "Réduction de dimension" },
  anomalies: { icon: AlertTriangle, color: "amber", label: "Détection d'anomalies" },
  vision_classification: { icon: Boxes, color: "teal", label: "Classification d'images" },
  vision_anomalies: { icon: Sparkles, color: "violet", label: "Anomalies visuelles" },
};

const STATUS_OPTIONS: { value: JobStatus | "all"; label: string }[] = [
  { value: "all", label: "Tous les statuts" },
  { value: "queued", label: "En file" },
  { value: "running", label: "En cours" },
  { value: "completed", label: "Terminé" },
  { value: "failed", label: "Échec" },
  { value: "cancelled", label: "Annulé" },
];

const PERIOD_OPTIONS: { value: string; label: string; days: number | null }[] = [
  { value: "all", label: "Toute la période", days: null },
  { value: "7", label: "7 derniers jours", days: 7 },
  { value: "30", label: "30 derniers jours", days: 30 },
  { value: "90", label: "90 derniers jours", days: 90 },
];

/** Historique unifié, tous piliers confondus (Lot 7, §J.4) — nouvelle page
 * épinglée en plus des 3 historiques déjà existants (TrainingHistory,
 * UnsupervisedHistory, VisionHistory), qui restent inchangés et gardent
 * leurs fonctionnalités propres (comparaison inter-jobs, registre de
 * modèles, onglets par type). Celle-ci répond à un besoin différent : "tout
 * ce qui s'est passé, tous piliers confondus, avec des filtres" — choix
 * confirmé par l'utilisateur : "Nouvelle page globale en plus". Chaque
 * échec de chargement par type dégrade honnêtement (le reste s'affiche
 * quand même) plutôt que de faire échouer toute la page. */
export default function AllHistory() {
  const [rows, setRows] = useState<HistoryEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [kindFilter, setKindFilter] = useState<JobKind | "all">("all");
  const [statusFilter, setStatusFilter] = useState<JobStatus | "all">("all");
  const [authorFilter, setAuthorFilter] = useState("all");
  const [periodFilter, setPeriodFilter] = useState("all");

  const load = useCallback(async () => {
    const [supervised, clustering, dimensionality, anomalies, visionClassification, visionAnomalies] = await Promise.allSettled([
      api.training.listJobs(),
      api.clustering.listJobs(),
      api.dimensionality.listJobs(),
      api.anomalies.listJobs(),
      api.visionClassification.listJobs(),
      api.visionAnomalies.listJobs(),
    ]);

    const items: HistoryEntry[] = [];

    if (supervised.status === "fulfilled") {
      supervised.value.forEach((j) =>
        items.push({
          key: `supervised-${j.id}`,
          kind: "supervised",
          id: j.id,
          datasetName: j.dataset_name ?? "Dataset",
          detailLabel: `→ ${j.target_column}`,
          status: j.status,
          headline:
            j.headline_metric && j.headline_metric.value !== null && j.headline_metric.value !== undefined
              ? `${j.headline_metric.name} = ${j.headline_metric.value.toFixed(3)}`
              : null,
          createdBy: j.created_by,
          createdAt: j.created_at,
          href: `/training?job=${j.id}`,
        }),
      );
    }
    if (clustering.status === "fulfilled") {
      clustering.value.forEach((j) =>
        items.push({
          key: `clustering-${j.id}`,
          kind: "clustering",
          id: j.id,
          datasetName: j.dataset_name ?? "Dataset",
          detailLabel: j.n_clusters ? `${j.n_clusters} groupes` : (j.algorithm ?? "Clustering"),
          status: j.status,
          headline: j.silhouette !== null ? `silhouette = ${j.silhouette.toFixed(3)}` : null,
          createdBy: j.created_by,
          createdAt: j.created_at,
          href: `/clustering?job=${j.id}`,
        }),
      );
    }
    if (dimensionality.status === "fulfilled") {
      dimensionality.value.forEach((j) =>
        items.push({
          key: `dimensionality-${j.id}`,
          kind: "dimensionality",
          id: j.id,
          datasetName: j.dataset_name ?? "Dataset",
          detailLabel: j.algorithm ?? "Réduction de dimension",
          status: j.status,
          headline: j.total_variance_explained !== null ? `variance PCA = ${formatPercent(j.total_variance_explained)}` : null,
          createdBy: j.created_by,
          createdAt: j.created_at,
          href: `/reduction-dimension?job=${j.id}`,
        }),
      );
    }
    if (anomalies.status === "fulfilled") {
      anomalies.value.forEach((j) =>
        items.push({
          key: `anomalies-${j.id}`,
          kind: "anomalies",
          id: j.id,
          datasetName: j.dataset_name ?? "Dataset",
          detailLabel: j.n_anomalies_consensus !== null ? `${j.n_anomalies_consensus} atypiques` : "Détection d'anomalies",
          status: j.status,
          headline: j.anomaly_rate_consensus !== null ? formatPercent(j.anomaly_rate_consensus) : null,
          createdBy: j.created_by,
          createdAt: j.created_at,
          href: `/anomalies?job=${j.id}`,
        }),
      );
    }
    if (visionClassification.status === "fulfilled") {
      visionClassification.value.forEach((j) =>
        items.push({
          key: `vision_classification-${j.id}`,
          kind: "vision_classification",
          id: j.id,
          datasetName: j.vision_dataset_name ?? "Dataset",
          detailLabel: j.backbone_id,
          status: j.status,
          headline: j.test_accuracy !== null ? `exactitude = ${formatPercent(j.test_accuracy)}` : null,
          createdBy: j.created_by,
          createdAt: j.created_at,
          href: `/vision/classification?job=${j.id}`,
        }),
      );
    }
    if (visionAnomalies.status === "fulfilled") {
      visionAnomalies.value.forEach((j) =>
        items.push({
          key: `vision_anomalies-${j.id}`,
          kind: "vision_anomalies",
          id: j.id,
          datasetName: j.vision_dataset_name ?? "Dataset",
          detailLabel: j.model_id,
          status: j.status,
          headline: j.roc_auc !== null ? `AUC = ${j.roc_auc.toFixed(3)}` : null,
          createdBy: j.created_by,
          createdAt: j.created_at,
          href: `/vision/anomalies?job=${j.id}`,
        }),
      );
    }

    const failedCount = [supervised, clustering, dimensionality, anomalies, visionClassification, visionAnomalies].filter(
      (r) => r.status === "rejected",
    ).length;
    setError(
      failedCount > 0
        ? `${failedCount} type${failedCount > 1 ? "s" : ""} d'analyse n'ont pas pu être chargés — l'historique ci-dessous est peut-être incomplet.`
        : null,
    );

    items.sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());
    setRows(items);
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const authors = useMemo(() => {
    if (!rows) return [];
    const set = new Set(rows.map((r) => r.createdBy).filter((v): v is string => Boolean(v)));
    return Array.from(set).sort((a, b) => a.localeCompare(b));
  }, [rows]);

  const filteredRows = useMemo(() => {
    if (!rows) return [];
    const term = search.trim().toLowerCase();
    const period = PERIOD_OPTIONS.find((p) => p.value === periodFilter);
    const now = Date.now();
    return rows.filter((r) => {
      if (kindFilter !== "all" && r.kind !== kindFilter) return false;
      if (statusFilter !== "all" && r.status !== statusFilter) return false;
      if (authorFilter !== "all" && r.createdBy !== authorFilter) return false;
      if (period?.days && now - new Date(r.createdAt).getTime() > period.days * 86_400_000) return false;
      if (term) {
        const haystack = `${r.datasetName} ${r.detailLabel} ${KIND_META[r.kind].label}`.toLowerCase();
        if (!haystack.includes(term)) return false;
      }
      return true;
    });
  }, [rows, search, kindFilter, statusFilter, authorFilter, periodFilter]);

  const columns = useMemo<TableColumn<HistoryEntry>[]>(
    () => [
      {
        key: "kind",
        header: "Type",
        sortable: true,
        sticky: true,
        sortValue: (r) => KIND_META[r.kind].label,
        render: (r) => {
          const meta = KIND_META[r.kind];
          return (
            <div className="flex items-center gap-2.5">
              <ColorIconBadge icon={meta.icon} color={meta.color} size="sm" />
              <span className="text-sm text-foreground/90 whitespace-nowrap">{meta.label}</span>
            </div>
          );
        },
      },
      {
        key: "dataset",
        header: "Dataset",
        sortable: true,
        sortValue: (r) => r.datasetName,
        render: (r) => (
          <div className="min-w-0">
            <p className="text-sm text-foreground truncate">{r.datasetName}</p>
            <p className="text-xs text-muted-foreground truncate">{r.detailLabel}</p>
          </div>
        ),
      },
      {
        key: "headline",
        header: "Résultat",
        align: "right",
        render: (r) => (r.headline ? <span className="text-xs text-muted-foreground tabular-nums">{r.headline}</span> : "—"),
      },
      {
        key: "status",
        header: "Statut",
        sortable: true,
        sortValue: (r) => r.status,
        render: (r) => <JobStatusBadge status={r.status} />,
      },
      {
        key: "createdBy",
        header: "Auteur",
        sortable: true,
        sortValue: (r) => r.createdBy ?? "",
        render: (r) => r.createdBy ?? "—",
      },
      {
        key: "createdAt",
        header: "Date",
        align: "right",
        sortable: true,
        sortValue: (r) => r.createdAt,
        render: (r) => formatDateTime(r.createdAt),
      },
      {
        key: "actions",
        header: "",
        align: "right",
        render: (r) => (
          <Link to={r.href} className="text-xs text-primary hover:text-primary/80 whitespace-nowrap">
            Voir →
          </Link>
        ),
      },
    ],
    [],
  );

  return (
    <AppShell>
      <PageHeader
        eyebrow="Historique"
        title="Toutes les analyses"
        description="Tous vos entraînements et analyses ML, tous piliers confondus."
        icon={History}
        color="blue"
      />

      {error && <ErrorNote message={error} />}

      <Card className="p-5">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5 mb-5">
          <div className="relative lg:col-span-2">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground pointer-events-none" />
            <Input
              placeholder="Rechercher un dataset, une cible…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="pl-9"
            />
          </div>
          <Select value={kindFilter} onChange={(e) => setKindFilter(e.target.value as JobKind | "all")}>
            <option value="all">Tous les types</option>
            {(Object.keys(KIND_META) as JobKind[]).map((k) => (
              <option key={k} value={k}>
                {KIND_META[k].label}
              </option>
            ))}
          </Select>
          <Select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value as JobStatus | "all")}>
            {STATUS_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </Select>
          <Select value={periodFilter} onChange={(e) => setPeriodFilter(e.target.value)}>
            {PERIOD_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </Select>
          <Select value={authorFilter} onChange={(e) => setAuthorFilter(e.target.value)}>
            <option value="all">Tous les auteurs</option>
            {authors.map((a) => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
          </Select>
        </div>

        <Table
          columns={columns}
          rows={filteredRows}
          rowKey={(r) => r.key}
          caption="Historique de toutes les analyses ML, tous piliers confondus"
          loading={rows === null}
          pageSize={20}
          emptyMessage={rows && rows.length > 0 ? "Aucune analyse ne correspond à ces filtres." : "Aucune analyse pour l'instant."}
        />
      </Card>
    </AppShell>
  );
}
