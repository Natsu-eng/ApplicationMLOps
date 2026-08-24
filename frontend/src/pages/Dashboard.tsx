import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Link, useSearchParams } from "react-router-dom";
import {
  Activity,
  AlertTriangle,
  BrainCircuit,
  Boxes,
  ChevronDown,
  Database,
  FileSpreadsheet,
  LayoutDashboard,
  ScatterChart,
  Shapes,
  Sparkles,
  Trash2,
  Users,
  type LucideIcon,
} from "lucide-react";
import { ApiError, api, apiErrorReference, type DashboardSummary, type JobStatus, type TrainingJobSummary } from "../api/client";
import { useAuth } from "../contexts/AuthContext";
import AppShell from "../components/AppShell";
import { pillarColor } from "../config/pillars";
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

type ActivityKind = "supervised" | "clustering" | "dimensionality" | "anomalies" | "vision_classification" | "vision_anomalies";

interface ActivityItem {
  kind: ActivityKind;
  id: number;
  datasetName: string;
  detailLabel: string;
  createdAt: string;
  status: JobStatus;
  headline: string | null;
  href: string;
  /** Traçabilité (Lot 16B) — déjà porté par les 4 types de job résumé
   * (`created_by`), jamais affiché jusqu'ici sur le flux d'activité. */
  createdBy: string | null;
  /** Uniquement pour "supervised" — ouvre ModelResultModal en place plutôt
   * que de naviguer, comportement d'origine conservé tel quel. */
  raw?: TrainingJobSummary;
}

const ACTIVITY_KIND_META: Record<ActivityKind, { icon: LucideIcon; color: AccentColor; label: string }> = {
  supervised: { icon: BrainCircuit, color: "violet", label: "Entraînement" },
  clustering: { icon: Shapes, color: "rose", label: "Clustering" },
  dimensionality: { icon: ScatterChart, color: "blue", label: "Réduction de dimension" },
  anomalies: { icon: AlertTriangle, color: "amber", label: "Détection d'anomalies" },
  vision_classification: { icon: Boxes, color: "teal", label: "Classification d'images" },
  vision_anomalies: { icon: Sparkles, color: "violet", label: "Anomalies visuelles" },
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

  // Lot 4 (correctif I3, AUDIT_DATALAB_2026-08-16.md §C.2.4) — un seul
  // aller-retour (`GET /dashboard/summary`) remplace les 8 appels de liste
  // complets faits ici jusque-là (membres, datasets, et les 6 types de
  // job). Un échec devient forcément global (plus de dégradation fine par
  // pilier) : accepté — les 8 requêtes touchaient de toute façon la même
  // base de données, le risque de panne partielle était déjà faible, et le
  // gain de performance (1 requête réseau au lieu de 8, N+1 éliminé côté
  // serveur) l'emporte largement pour la page la plus visitée du produit.
  const [summary, setSummary] = useState<DashboardSummary | null>(null);
  const [summaryError, setSummaryError] = useState<string | null>(null);
  const [summaryErrorRef, setSummaryErrorRef] = useState<string | undefined>(undefined);
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

  const loadSummary = useCallback(async () => {
    try {
      setSummary(await api.dashboard.summary());
      setSummaryError(null);
      setSummaryErrorRef(undefined);
    } catch (err) {
      setSummaryError(err instanceof ApiError ? err.message : "Impossible de charger le tableau de bord");
      setSummaryErrorRef(apiErrorReference(err));
    }
  }, []);

  useEffect(() => {
    loadSummary();
  }, [loadSummary]);

  async function handleDeleteJob(id: number) {
    try {
      await api.training.remove(id);
      loadSummary();
    } catch {
      // best-effort — la liste se resynchronisera au prochain chargement
    }
  }

  const totalJobsCount = summary ? summary.supervised_count + summary.unsupervised_count + summary.vision_count : undefined;

  const activity = useMemo<ActivityItem[]>(() => {
    if (!summary) return [];
    const items: ActivityItem[] = [];
    summary.recent_supervised.forEach((j) =>
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
        createdBy: j.created_by,
        raw: j,
      }),
    );
    summary.recent_clustering.forEach((j) =>
      items.push({
        kind: "clustering",
        id: j.id,
        datasetName: j.dataset_name ?? "Dataset",
        detailLabel: j.n_clusters ? `${j.n_clusters} groupes` : (j.algorithm ?? "Clustering"),
        createdAt: j.created_at,
        status: j.status,
        headline: j.silhouette !== null ? `silhouette = ${j.silhouette.toFixed(3)}` : null,
        href: `/clustering?job=${j.id}`,
        createdBy: j.created_by,
      }),
    );
    summary.recent_dimensionality.forEach((j) =>
      items.push({
        kind: "dimensionality",
        id: j.id,
        datasetName: j.dataset_name ?? "Dataset",
        detailLabel: j.algorithm ?? "Réduction de dimension",
        createdAt: j.created_at,
        status: j.status,
        headline: j.total_variance_explained !== null ? `variance PCA = ${formatPercent(j.total_variance_explained)}` : null,
        href: `/reduction-dimension?job=${j.id}`,
        createdBy: j.created_by,
      }),
    );
    summary.recent_anomalies.forEach((j) =>
      items.push({
        kind: "anomalies",
        id: j.id,
        datasetName: j.dataset_name ?? "Dataset",
        detailLabel: j.n_anomalies_consensus !== null ? `${j.n_anomalies_consensus} atypiques` : "Détection d'anomalies",
        createdAt: j.created_at,
        status: j.status,
        headline: j.anomaly_rate_consensus !== null ? formatPercent(j.anomaly_rate_consensus) : null,
        href: `/anomalies?job=${j.id}`,
        createdBy: j.created_by,
      }),
    );
    summary.recent_vision_classification.forEach((j) =>
      items.push({
        kind: "vision_classification",
        id: j.id,
        datasetName: j.vision_dataset_name ?? "Dataset",
        detailLabel: j.backbone_id,
        createdAt: j.created_at,
        status: j.status,
        headline: j.test_accuracy !== null ? `exactitude = ${formatPercent(j.test_accuracy)}` : null,
        href: `/vision/classification?job=${j.id}`,
        createdBy: j.created_by,
      }),
    );
    summary.recent_vision_anomalies.forEach((j) =>
      items.push({
        kind: "vision_anomalies",
        id: j.id,
        datasetName: j.vision_dataset_name ?? "Dataset",
        detailLabel: j.model_id,
        createdAt: j.created_at,
        status: j.status,
        headline: j.roc_auc !== null ? `AUC = ${j.roc_auc.toFixed(3)}` : null,
        href: `/vision/anomalies?job=${j.id}`,
        createdBy: j.created_by,
      }),
    );
    items.sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());
    return items.slice(0, 6);
  }, [summary]);

  const recentDatasets = summary?.recent_datasets ?? [];

  if (!user) return null;

  return (
    <AppShell>
      <PageHeader
        eyebrow="Vue d'ensemble"
        title={`${greeting(new Date().getHours())}, ${user.nom.split(" ")[0]}`}
        description={`${user.organization_name} — l'activité récente de votre équipe, tous piliers confondus.`}
        icon={LayoutDashboard}
        color="blue"
        action={<NewAnalysisMenu />}
      />

      {/* Couleurs (Lot 2A correctif 3) : une tuile qui agrège plusieurs
          piliers (Datasets, Analyses ML dans son ensemble, En cours,
          Membres) reste neutre — aucune ne "appartient" à un seul pilier.
          Seules les 3 colonnes de la tuile "Analyses ML" portent chacune la
          couleur RÉELLE de leur pilier (`pillarColor`, source unique
          `config/pillars.ts`). */}
      <StatTileRow wide>
        <StatTile icon={Database} label="Datasets" value={summary?.datasets_count} color="neutral" delayMs={0} />
        <StatTile
          icon={Activity}
          label="Analyses ML"
          value={totalJobsCount}
          color="neutral"
          delayMs={60}
          split={[
            { label: "Supervisé", value: summary?.supervised_count, color: pillarColor("supervised") },
            { label: "Non supervisé", value: summary?.unsupervised_count, color: pillarColor("unsupervised") },
            { label: "Vision", value: summary?.vision_count, color: pillarColor("vision") },
          ]}
        />
        <StatTile icon={Activity} label="En cours" value={summary?.active_count} color="neutral" delayMs={120} />
        <StatTile icon={Users} label="Membres de l'équipe" value={summary?.members_count} color="neutral" delayMs={180} />
      </StatTileRow>

      <div className="grid gap-6 lg:grid-cols-2 mb-10">
        <Card className="p-5">
          <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
            <h2 className="text-h3 text-foreground">Dernière activité</h2>
            <div className="flex items-center gap-3 text-xs">
              <Link to="/training/history" className="text-primary hover:text-primary/80">
                Historique supervisé
              </Link>
              <Link to="/non-supervise/historique" className="text-primary hover:text-primary/80">
                Historique non supervisé
              </Link>
              <Link to="/vision/historique" className="text-primary hover:text-primary/80">
                Historique Vision
              </Link>
            </div>
          </div>

          {summaryError ? (
            <ErrorNote message={summaryError} reference={summaryErrorRef} />
          ) : !summary ? (
            <p className="text-sm text-muted-foreground">Chargement…</p>
          ) : activity.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              Aucune analyse pour l'instant — lancez-en une depuis{" "}
              <Link to="/training" className="text-primary underline underline-offset-2 hover:text-primary/80">
                Entraînement
              </Link>
              , un module de{" "}
              <Link to="/clustering" className="text-primary underline underline-offset-2 hover:text-primary/80">
                ML non supervisé
              </Link>
              , ou{" "}
              <Link to="/vision/classification" className="text-primary underline underline-offset-2 hover:text-primary/80">
                Vision
              </Link>
              .
            </p>
          ) : activity.length > 0 ? (
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
                      <div className="flex items-center gap-1 text-xs text-muted-foreground">
                        <span className="truncate">
                          {meta.label} · {formatDateTime(item.createdAt)}
                        </span>
                        {item.createdBy && (
                          <span className="flex-shrink-0 whitespace-nowrap text-foreground/70">
                            · {item.createdBy}
                          </span>
                        )}
                      </div>
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
          ) : null}
        </Card>

        <Card className="p-5">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-h3 text-foreground">Derniers datasets</h2>
            <Link to="/datasets" className="text-xs text-primary hover:text-primary/80">
              Voir tout
            </Link>
          </div>

          {summaryError ? (
            <ErrorNote message={summaryError} reference={summaryErrorRef} />
          ) : !summary ? (
            <p className="text-sm text-muted-foreground">Chargement…</p>
          ) : recentDatasets.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              Aucun dataset pour l'instant — importez-en un depuis{" "}
              <Link to="/datasets" className="text-primary underline underline-offset-2 hover:text-primary/80">
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

/** Point d'entrée "Nouvelle analyse" (retour utilisateur) — remplace un
 * bouton câblé en dur sur /training, qui n'atteignait que le pilier
 * supervisé et doublait des liens déjà présents plus bas dans la page.
 * Les 6 analyses réellement lançables (une par module actif, jamais "Mes
 * données"/"Historique" — ce menu lance une analyse, pas de la navigation
 * générale), groupées par pilier. Même coût en clics pour le supervisé
 * qu'avant (Nouvelle analyse → Entraînement) ; les 5 autres deviennent
 * enfin accessibles depuis le tableau de bord. */
const NEW_ANALYSIS_GROUPS: {
  pillar: string;
  color: AccentColor;
  items: { to: string; label: string; icon: LucideIcon }[];
}[] = [
  { pillar: "Supervisé", color: "violet", items: [{ to: "/training", label: "Entraînement", icon: BrainCircuit }] },
  {
    pillar: "Non supervisé",
    color: "rose",
    items: [
      { to: "/clustering", label: "Clustering", icon: Shapes },
      { to: "/reduction-dimension", label: "Réduction de dimension", icon: ScatterChart },
      { to: "/anomalies", label: "Détection d'anomalies", icon: AlertTriangle },
    ],
  },
  {
    pillar: "Vision",
    color: "teal",
    items: [
      { to: "/vision/classification", label: "Classification d'images", icon: Boxes },
      { to: "/vision/anomalies", label: "Anomalies visuelles", icon: Sparkles },
    ],
  },
];

function NewAnalysisMenu() {
  const [open, setOpen] = useState(false);
  // Position calculée à l'ouverture (Lot 10, retour utilisateur direct —
  // capture réelle) : ce menu vit dans le bandeau "Vue d'ensemble", une
  // `Card` (overflow-hidden inconditionnel, Card.tsx) — un menu simplement
  // `absolute` à l'intérieur y était rogné aux bords de la carte, presque
  // entièrement invisible. Porté vers `document.body` (même correctif que
  // Modal.tsx, Lot 5) avec une position `fixed` recalculée depuis le
  // bouton, pour échapper à CETTE carte ET à toute future ancêtre du même
  // genre — plus robuste qu'un simple changement d'`overflow` sur la carte.
  const [menuPos, setMenuPos] = useState<{ top: number; right: number } | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  const updatePosition = useCallback(() => {
    const rect = buttonRef.current?.getBoundingClientRect();
    if (!rect) return;
    setMenuPos({ top: rect.bottom + 8, right: window.innerWidth - rect.right });
  }, []);

  useEffect(() => {
    if (!open) return;
    updatePosition();
    function onPointerDown(e: MouseEvent) {
      const target = e.target as Node;
      if (containerRef.current?.contains(target)) return;
      if (menuRef.current?.contains(target)) return;
      setOpen(false);
    }
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    window.addEventListener("resize", updatePosition);
    window.addEventListener("scroll", updatePosition, true);
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("resize", updatePosition);
      window.removeEventListener("scroll", updatePosition, true);
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  return (
    <div className="relative" ref={containerRef}>
      <Button ref={buttonRef} onClick={() => setOpen((o) => !o)} aria-haspopup="menu" aria-expanded={open}>
        <BrainCircuit size={15} />
        Nouvelle analyse
        <ChevronDown size={14} className={`transition-transform ${open ? "rotate-180" : ""}`} />
      </Button>

      {open && menuPos && createPortal(
        <div
          ref={menuRef}
          role="menu"
          style={{ top: menuPos.top, right: menuPos.right }}
          className="fixed w-64 rounded-xl border border-border bg-card shadow-lg py-2 z-50"
        >
          {NEW_ANALYSIS_GROUPS.map((group, i) => (
            <div key={group.pillar} className={i > 0 ? "mt-1 pt-1 border-t border-border" : ""}>
              <p className="px-3 py-1 text-overline text-muted-foreground">{group.pillar}</p>
              {group.items.map((item) => {
                const Icon = item.icon;
                return (
                  <Link
                    key={item.to}
                    to={item.to}
                    role="menuitem"
                    onClick={() => setOpen(false)}
                    className="flex items-center gap-2.5 px-3 py-2 text-sm text-foreground hover:bg-muted transition-colors"
                  >
                    <ColorIconBadge icon={Icon} color={group.color} size="sm" />
                    {item.label}
                  </Link>
                );
              })}
            </div>
          ))}
        </div>,
        document.body,
      )}
    </div>
  );
}
