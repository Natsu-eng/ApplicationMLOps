import { useCallback, useEffect, useState, type FormEvent } from "react";
import { Link, useSearchParams } from "react-router-dom";
import {
  AlertCircle,
  AlertTriangle,
  ArrowDown,
  ArrowUp,
  Ban,
  Boxes,
  CircleDashed,
  Download,
  FileCode,
  FileJson,
  Gauge,
  ListChecks,
  Loader2,
  PlayCircle,
  RotateCcw,
  ScanSearch,
  Shapes,
  Sparkles,
  Target,
  Trash2,
  Trophy,
  Waves,
  Zap,
} from "lucide-react";
import {
  ApiError,
  api,
  type AlgorithmCatalogEntry,
  type ClusterAssignmentMethod,
  type ClusterCandidate,
  type ClusterPrediction,
  type ClusteringJobSummary,
  type ClusteringResult,
  type ColumnSchema,
  type DatasetSummary,
} from "../api/client";
import AppShell from "../components/AppShell";
import { pillarColor } from "../config/pillars";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { Input } from "../components/ui/Input";
import { Badge } from "../components/ui/Badge";
import { Switch } from "../components/ui/Switch";
import { accentSurfaceClass, accentValueTextClass, type AccentColor } from "../components/ui/ColorIconBadge";
import { PageHeader } from "../components/ui/PageHeader";
import { SectionHeader } from "../components/ui/SectionHeader";
import { Select } from "../components/ui/Select";
import { Table, type TableColumn } from "../components/ui/Table";
import { Tabs } from "../components/ui/Tabs";
import { ModelExportActions } from "../components/ui/ModelExportActions";
import { buildClusteringModelCard } from "../utils/clusteringModelCard";
import { DataQualityWarnings } from "../components/training/DataQualityWarnings";
import { ClusterProfileGrid } from "../components/clustering/ClusterProfileGrid";
import { DriftPanel } from "../components/shared/DriftPanel";
import { useJobEvents } from "../hooks/useJobEvents";
import { useConfirmAction } from "../hooks/useConfirmAction";
import { useIdempotencyKey } from "../hooks/useIdempotencyKey";
import { CHART_SERIES_COLORS } from "../theme/charts";
import {
  assessSilhouetteQuality,
  assessStabilityQuality,
  buildRecommendationExplanation,
  computeClusterDistribution,
} from "../utils/clusterQuality";
import { QUALITY_TONE_ACCENT } from "../utils/qualityAssessment";

const ACTIVE_STATUSES = new Set(["queued", "running"]);
const ACTIVE_JOB_STORAGE_KEY = "datalab_active_clustering_job_id";

type Phase = "configure" | "progress" | "results" | "failed" | "cancelled";

function phaseOf(job: ClusteringJobSummary | null): Phase {
  if (!job) return "configure";
  if (ACTIVE_STATUSES.has(job.status)) return "progress";
  if (job.status === "completed") return "results";
  return job.status === "cancelled" ? "cancelled" : "failed";
}

/** Pilier ML non supervisé — clustering (Lot 11+). Page unique volontairement
 * plus simple que le wizard supervisé (Training.tsx) : la configuration
 * d'un clustering est plus légère (pas de cible, pas de mode expert à ce
 * stade). Même state machine configure/progress/results/failed, même
 * persistance de session (sessionStorage) qu'un rafraîchissement en cours
 * de calcul ne fasse jamais perdre la progression. */
export default function Clustering() {
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [datasetsError, setDatasetsError] = useState<string | null>(null);
  const [activeJob, setActiveJob] = useState<ClusteringJobSummary | null>(null);
  const [restoringJob, setRestoringJob] = useState(true);
  const confirmDelete = useConfirmAction<true>();
  const [searchParams, setSearchParams] = useSearchParams();

  // Reprise d'un clustering — priorité au deep-link `?job=` (ex. depuis la
  // page Historique du pilier non supervisé), sinon la session en cours
  // (rafraîchissement pendant un calcul, comportement d'origine).
  useEffect(() => {
    const queryJobId = searchParams.get("job");
    const storedId = queryJobId ?? sessionStorage.getItem(ACTIVE_JOB_STORAGE_KEY);
    if (!storedId) {
      setRestoringJob(false);
      return;
    }
    api.clustering
      .getJob(Number(storedId))
      .then(setActiveJob)
      .catch(() => {
        sessionStorage.removeItem(ACTIVE_JOB_STORAGE_KEY);
        setSearchParams({}, { replace: true });
      })
      .finally(() => setRestoringJob(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (activeJob) sessionStorage.setItem(ACTIVE_JOB_STORAGE_KEY, String(activeJob.id));
    else sessionStorage.removeItem(ACTIVE_JOB_STORAGE_KEY);
  }, [activeJob]);

  function openJob(job: ClusteringJobSummary) {
    setActiveJob(job);
    setSearchParams({ job: String(job.id) }, { replace: false });
  }

  const loadDatasets = useCallback(async () => {
    try {
      const all = await api.datasets.list();
      setDatasets(all.filter((d) => d.status === "ready"));
      setDatasetsError(null);
    } catch (err) {
      setDatasetsError(err instanceof ApiError ? err.message : "Impossible de charger vos datasets");
    }
  }, []);

  useEffect(() => {
    loadDatasets();
  }, [loadDatasets]);

  const phase = phaseOf(activeJob);

  // Notifications SSE (Lot 7, §J.2) — remplace le polling setInterval.
  useJobEvents(
    phase === "progress" && activeJob ? `/clustering/jobs/${activeJob.id}/events` : null,
    (snapshot) => setActiveJob((prev) => (prev ? { ...prev, ...snapshot } : prev)),
  );

  function resetToConfigure() {
    setActiveJob(null);
    setSearchParams({}, { replace: false });
  }

  async function handleDeleteActiveJob() {
    if (!activeJob) return;
    try {
      await api.clustering.remove(activeJob.id);
    } catch {
      // best-effort — on repart en configuration dans tous les cas
    }
    resetToConfigure();
  }

  const [cancelling, setCancelling] = useState(false);
  async function handleCancelActiveJob() {
    if (!activeJob) return;
    setCancelling(true);
    try {
      setActiveJob(await api.clustering.cancel(activeJob.id));
    } catch {
      // best-effort — le prochain poll reflétera l'état réel
    } finally {
      setCancelling(false);
    }
  }

  const [rerunning, setRerunning] = useState(false);
  const [rerunError, setRerunError] = useState<string | null>(null);
  async function handleRerunActiveJob() {
    if (!activeJob) return;
    setRerunning(true);
    setRerunError(null);
    try {
      openJob(await api.clustering.rerun(activeJob.id));
    } catch (err) {
      setRerunError(err instanceof ApiError ? err.message : "Impossible de relancer ce clustering");
    } finally {
      setRerunning(false);
    }
  }

  const titles: Record<Phase, string> = {
    configure: "Découvrir des groupes",
    progress: "Recherche de groupes en cours",
    results: "Groupes découverts",
    failed: "Échec du clustering",
    cancelled: "Clustering annulé",
  };

  return (
    <AppShell pillarId="unsupervised">
      <PageHeader
        eyebrow="ML non supervisé"
        title={titles[phase]}
        description={
          phase === "configure"
            ? "Repérez des profils similaires dans vos données, sans savoir à l'avance ce que vous cherchez. Choisissez un dataset et les variables à analyser."
            : undefined
        }
        icon={Shapes}
        color={pillarColor("unsupervised")}
        action={
          phase !== "configure" ? (
            <div className="flex items-center gap-2">
              {(phase === "results" || phase === "failed" || phase === "cancelled") && (
                <>
                  <button
                    type="button"
                    onClick={() => confirmDelete.trigger(true, handleDeleteActiveJob)}
                    onMouseLeave={confirmDelete.reset}
                    aria-label={confirmDelete.isPending(true) ? "Confirmer la suppression" : "Supprimer ce clustering"}
                    title={confirmDelete.isPending(true) ? "Cliquer à nouveau pour confirmer" : "Supprimer ce clustering"}
                    className={`p-2 rounded-lg transition-colors ${
                      confirmDelete.isPending(true)
                        ? "text-destructive bg-destructive/15"
                        : "text-muted-foreground hover:text-destructive hover:bg-destructive/10"
                    }`}
                  >
                    <Trash2 size={16} />
                  </button>
                  <Button variant="secondary" size="sm" onClick={handleRerunActiveJob} disabled={rerunning}>
                    <RotateCcw size={14} />
                    {rerunning ? "Relance…" : "Relancer"}
                  </Button>
                </>
              )}
              <Button variant="secondary" size="sm" onClick={resetToConfigure}>
                <PlayCircle size={14} />
                Nouveau clustering
              </Button>
            </div>
          ) : undefined
        }
      />
      {rerunError && (
        <p className="text-sm text-destructive bg-destructive/10 border border-destructive/20 rounded-lg px-3 py-2 max-w-2xl mx-auto mb-4">
          {rerunError}
        </p>
      )}

      {restoringJob ? (
        <div className="flex items-center justify-center py-16 text-sm text-muted-foreground gap-2">
          <Loader2 size={16} className="animate-spin" />
          Reprise de votre session…
        </div>
      ) : phase === "configure" ? (
        <div className="max-w-2xl mx-auto">
          <ClusteringForm datasets={datasets} datasetsError={datasetsError} onJobCreated={openJob} />
        </div>
      ) : phase === "progress" && activeJob ? (
        <Card className="max-w-2xl mx-auto p-8 text-center">
          <Loader2 size={28} className="animate-spin mx-auto mb-4 text-primary" />
          <p className="text-sm text-foreground mb-1">{activeJob.progress_step ?? "Préparation…"}</p>
          <div className="w-full h-1.5 rounded-full bg-muted overflow-hidden mt-4 mb-4">
            <div
              className="h-full rounded-full bg-primary transition-all duration-500"
              style={{ width: `${activeJob.progress_percent}%` }}
            />
          </div>
          <Button variant="ghost" size="sm" onClick={handleCancelActiveJob} disabled={cancelling}>
            <Ban size={14} />
            {cancelling ? "Annulation…" : "Annuler ce clustering"}
          </Button>
        </Card>
      ) : (phase === "failed" || phase === "cancelled") && activeJob ? (
        <Card className="max-w-2xl mx-auto p-6">
          <div
            className={`flex items-start gap-3 rounded-lg p-4 border ${
              phase === "cancelled"
                ? "text-muted-foreground bg-muted border-border"
                : "text-destructive bg-destructive/10 border-destructive/20"
            }`}
          >
            {phase === "cancelled" ? (
              <Ban size={18} className="flex-shrink-0 mt-0.5" />
            ) : (
              <AlertCircle size={18} className="flex-shrink-0 mt-0.5" />
            )}
            <p className="text-sm">{activeJob.error_message ?? "Le clustering a échoué."}</p>
          </div>
        </Card>
      ) : phase === "results" && activeJob ? (
        <ClusteringResultView job={activeJob} />
      ) : null}

      {phase === "configure" && (
        <p className="text-xs text-muted-foreground text-center mt-6 max-w-2xl mx-auto">
          <Link to="/" className="text-primary hover:text-primary/80">
            Voir tous les objectifs
          </Link>
          .
        </p>
      )}
    </AppShell>
  );
}

function ClusteringForm({
  datasets,
  datasetsError,
  onJobCreated,
}: {
  datasets: DatasetSummary[];
  datasetsError: string | null;
  onJobCreated: (job: ClusteringJobSummary) => void;
}) {
  const [datasetId, setDatasetId] = useState<number | "">("");
  const [columns, setColumns] = useState<ColumnSchema[]>([]);
  const [selectedFeatures, setSelectedFeatures] = useState<Set<string>>(new Set());
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  // Idempotence (Phase 2, AUDIT_BACKEND_2026-08-23.md §F4).
  const idempotencyKey = useIdempotencyKey();

  // Mode expert — choix des algorithmes comparés (retour utilisateur direct :
  // "j'espère que côté interface l'utilisateur a le choix de choisir les
  // modèles" — jusqu'ici `algorithm_ids` existait déjà côté API mais
  // n'était exposé nulle part sur cette page). Décochée par défaut : le
  // comportement historique (sous-ensemble par défaut du registre) ne
  // change pas pour qui n'ouvre jamais ce panneau.
  const [expertMode, setExpertMode] = useState(false);
  const [algorithms, setAlgorithms] = useState<AlgorithmCatalogEntry[]>([]);
  const [selectedAlgorithmIds, setSelectedAlgorithmIds] = useState<Set<string>>(new Set());

  useEffect(() => {
    api.clustering
      .algorithmsCatalog()
      .then((res) => {
        setAlgorithms(res.algorithms);
        setSelectedAlgorithmIds(new Set(res.algorithms.filter((a) => a.is_default).map((a) => a.id)));
      })
      .catch(() => {
        // Catalogue indisponible : le mode expert reste utilisable en
        // dégradé (aucun algorithme à cocher) — jamais bloquant pour le
        // mode guidé, qui n'en dépend pas.
      });
  }, []);

  function toggleAlgorithm(id: string) {
    setSelectedAlgorithmIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  // Même seuil que `engine.py::_LINEAR_ONLY_ALGORITHM_IDS` — coût quasi
  // linéaire (KMeans/MiniBatchKMeans), sans le risque mémoire O(n²) du
  // hiérarchique/DBSCAN qui impose le plafond conservateur par défaut.
  // Affiché ici pour rendre le compromis visible AVANT le lancement, pas
  // seulement après coup dans le récapitulatif du résultat.
  const LINEAR_ONLY_ALGORITHM_IDS = new Set(["kmeans", "minibatch_kmeans"]);
  const unlocksHigherRowCap =
    expertMode && selectedAlgorithmIds.size > 0 && [...selectedAlgorithmIds].every((id) => LINEAR_ONLY_ALGORITHM_IDS.has(id));

  async function handleDatasetChange(id: string) {
    setError(null);
    if (!id) {
      setDatasetId("");
      setColumns([]);
      setSelectedFeatures(new Set());
      return;
    }
    const numericId = Number(id);
    setDatasetId(numericId);
    try {
      const detail = await api.datasets.get(numericId);
      setColumns(detail.columns);
      setSelectedFeatures(new Set(detail.columns.map((c) => c.name)));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Impossible de charger les colonnes");
    }
  }

  function toggleFeature(name: string) {
    setSelectedFeatures((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }

  // Toujours une exclusion, jamais un simple toggle (voir
  // DataQualityWarnings.tsx) — approuver une suggestion du contrôle qualité
  // doit exclure la colonne, jamais la réintégrer si déjà exclue.
  function excludeFeatures(names: string[]) {
    setSelectedFeatures((prev) => {
      const next = new Set(prev);
      names.forEach((n) => next.delete(n));
      return next;
    });
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (!datasetId || selectedFeatures.size === 0) return;
    setError(null);
    setIsSubmitting(true);
    try {
      const job = await api.clustering.createJob(
        {
          dataset_id: datasetId,
          feature_columns: Array.from(selectedFeatures),
          algorithm_ids: expertMode ? Array.from(selectedAlgorithmIds) : undefined,
        },
        idempotencyKey.current,
      );
      idempotencyKey.reset();
      onJobCreated(job);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Impossible de lancer le clustering");
    } finally {
      setIsSubmitting(false);
    }
  }

  if (datasetsError) {
    return (
      <Card className="p-5">
        <p className="text-sm text-destructive bg-destructive/10 border border-destructive/20 rounded-lg px-3 py-2">
          {datasetsError}
        </p>
      </Card>
    );
  }

  if (datasets.length === 0) {
    return (
      <Card className="p-5">
        <p className="text-sm text-muted-foreground">
          Aucun dataset prêt — importez-en un depuis{" "}
          <Link to="/datasets" className="text-primary hover:text-primary/80">
            Mes données
          </Link>
          .
        </p>
      </Card>
    );
  }

  return (
    <Card className="p-5">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="clustering-dataset" className="block text-sm text-muted-foreground mb-1">
            Jeu de données
          </label>
          <Select id="clustering-dataset" value={datasetId} onChange={(e) => handleDatasetChange(e.target.value)} required>
            <option value="">Choisir un dataset…</option>
            {datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name} ({d.row_count} lignes)
              </option>
            ))}
          </Select>
        </div>

        {columns.length > 0 && (
          <div>
            <label className="block text-sm text-muted-foreground mb-1.5">
              Variables à analyser ({selectedFeatures.size} sélectionnée{selectedFeatures.size > 1 ? "s" : ""})
            </label>
            <div className="max-h-56 overflow-y-auto rounded-lg border border-border divide-y divide-border/60">
              {columns.map((c) => (
                <label
                  key={c.name}
                  className="flex items-center gap-2.5 px-3 py-2 text-sm text-foreground/90 hover:bg-muted/50 cursor-pointer transition-colors"
                >
                  <input
                    type="checkbox"
                    checked={selectedFeatures.has(c.name)}
                    onChange={() => toggleFeature(c.name)}
                    className="accent-primary"
                  />
                  <span className="flex-1 truncate">{c.name}</span>
                  <span className="text-xs text-muted-foreground flex-shrink-0">{c.dtype}</span>
                </label>
              ))}
            </div>
            <p className="text-xs text-muted-foreground mt-1.5">
              Toutes les variables comptent à égalité une fois normalisées — décochez celles qui n'ont pas de sens
              pour comparer des observations entre elles (ex. un identifiant).
            </p>
          </div>
        )}

        {datasetId && (
          <div>
            <p className="block text-sm text-muted-foreground mb-1.5">Qualité des données</p>
            <DataQualityWarnings
              datasetId={datasetId}
              selectedFeatures={selectedFeatures}
              onExcludeColumns={excludeFeatures}
            />
          </div>
        )}

        <div className="flex items-center justify-between gap-4 pt-1 border-t border-border/60">
          <div>
            <p className="text-sm font-medium text-foreground">Mode expert</p>
            <p className="text-xs text-muted-foreground">
              Choisissez vous-même les algorithmes comparés, au lieu du sous-ensemble par défaut. Inutile pour un
              usage courant.
            </p>
          </div>
          <Switch checked={expertMode} onChange={setExpertMode} label="Activer le mode expert" />
        </div>

        {expertMode && (
          <div className="space-y-2.5 rounded-lg border border-primary/20 bg-primary/5 p-3.5">
            <p className="text-sm text-muted-foreground">
              Algorithmes comparés — {selectedAlgorithmIds.size} sélectionné{selectedAlgorithmIds.size > 1 ? "s" : ""}
            </p>
            {algorithms.length === 0 && <p className="text-xs text-muted-foreground">Chargement du catalogue…</p>}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5">
              {algorithms.map((algo) => {
                const checked = selectedAlgorithmIds.has(algo.id);
                const isLinear = LINEAR_ONLY_ALGORITHM_IDS.has(algo.id);
                return (
                  <label
                    key={algo.id}
                    className={`flex items-center gap-2 text-xs rounded-lg border px-2.5 py-2 cursor-pointer transition-colors ${
                      checked ? "border-primary/40 bg-primary/10 text-primary" : "border-border bg-card text-foreground/90 hover:border-input"
                    }`}
                  >
                    <input type="checkbox" className="accent-primary" checked={checked} onChange={() => toggleAlgorithm(algo.id)} />
                    <span className="flex-1 min-w-0">{algo.label}</span>
                    {isLinear ? (
                      <Badge variant="success">
                        <Zap size={10} className="mr-0.5" />
                        Rapide
                      </Badge>
                    ) : (
                      <Badge variant="neutral">
                        <AlertTriangle size={10} className="mr-0.5" />
                        Coûteux
                      </Badge>
                    )}
                  </label>
                );
              })}
            </div>
            <p className="text-xs text-muted-foreground flex items-start gap-1.5 pt-1">
              <Gauge size={13} className="flex-shrink-0 mt-0.5" />
              {unlocksHigherRowCap
                ? "Sélection limitée à des algorithmes rapides (coût quasi linéaire) : jusqu'à 20 000 lignes traitées sans échantillonnage, contre 5 000 dès qu'hiérarchique ou DBSCAN sont inclus (coût mémoire qui augmente avec le carré du nombre de lignes)."
                : "Hiérarchique et DBSCAN ont un coût mémoire qui augmente avec le carré du nombre de lignes : le plafond reste à 5 000 lignes traitées (échantillon représentatif, pas d'impact sur la fiabilité statistique). Décochez-les pour monter à 20 000."}
            </p>
          </div>
        )}

        {error && (
          <div className="flex items-center gap-2 text-sm text-destructive bg-destructive/10 border border-destructive/20 rounded-lg px-3 py-2">
            <AlertCircle size={15} className="flex-shrink-0" />
            {error}
          </div>
        )}

        <Button
          type="submit"
          disabled={!datasetId || selectedFeatures.size === 0 || (expertMode && selectedAlgorithmIds.size === 0) || isSubmitting}
          className="w-full"
        >
          {isSubmitting ? "Lancement…" : "Lancer le clustering"}
        </Button>
      </form>
    </Card>
  );
}

function formatCandidateParams(params: Record<string, unknown>): string {
  return Object.entries(params)
    .map(([key, value]) => `${key} = ${typeof value === "number" ? Number(value.toFixed(3)) : value}`)
    .join(", ");
}

/** "↑ meilleur"/"↓ meilleur" à côté du nom de la métrique — les 3 métriques
 * de qualité du clustering ne se lisent pas dans le même sens (silhouette et
 * Calinski-Harabasz : plus haut = mieux ; Davies-Bouldin : plus bas = mieux),
 * ambiguïté déjà signalée comme source de confusion. */
function MetricDirection({ better }: { better: "up" | "down" }) {
  const Icon = better === "up" ? ArrowUp : ArrowDown;
  return (
    <span
      className="inline-flex items-center text-caption text-muted-foreground/80"
      title={better === "up" ? "Plus haut = meilleur" : "Plus bas = meilleur"}
    >
      <Icon size={11} />
    </span>
  );
}

const CANDIDATE_COLUMNS: TableColumn<ClusterCandidate>[] = [
  {
    key: "rank",
    header: "#",
    render: (c) =>
      c.is_winner ? (
        <span className="inline-flex items-center gap-1 text-primary font-semibold">
          <Trophy size={13} /> {c.rank}
        </span>
      ) : (
        c.rank
      ),
  },
  { key: "algorithm", header: "Algorithme" },
  { key: "params", header: "Paramètres", render: (c) => formatCandidateParams(c.params), className: "text-muted-foreground" },
  { key: "n_clusters", header: "Groupes", align: "right" },
  {
    key: "silhouette",
    header: "Silhouette ↑",
    align: "right",
    render: (c) => (c.silhouette !== null ? c.silhouette.toFixed(3) : "—"),
  },
  {
    key: "davies_bouldin",
    header: "Davies-Bouldin ↓",
    align: "right",
    render: (c) => (c.davies_bouldin !== null ? c.davies_bouldin.toFixed(3) : "—"),
  },
  {
    key: "calinski_harabasz",
    header: "Calinski-Harabasz ↑",
    align: "right",
    render: (c) => (c.calinski_harabasz !== null ? c.calinski_harabasz.toFixed(0) : "—"),
  },
  {
    key: "noise_ratio",
    header: "Bruit",
    align: "right",
    render: (c) => (c.noise_ratio > 0 ? `${(c.noise_ratio * 100).toFixed(0)} %` : "—"),
  },
  {
    key: "composite_rank",
    header: "Rang composite ↓",
    align: "right",
    sortable: true,
    sortValue: (c) => c.composite_rank,
    render: (c) => (c.composite_rank !== null ? c.composite_rank.toFixed(2) : "—"),
    className: "font-medium",
  },
];

// Palette validée (theme/charts.ts) — jamais une teinte hex ad hoc pour une
// série de données, même dans une barre empilée custom hors Recharts.
const DISTRIBUTION_BAR_COLORS = CHART_SERIES_COLORS;
const DISTRIBUTION_NOISE_COLOR = "var(--text-muted)"; // neutre, jamais confondu avec un vrai segment

function ClusteringResultView({ job }: { job: ClusteringJobSummary }) {
  const jobId = job.id;
  const [result, setResult] = useState<ClusteringResult | null>(null);
  const [candidates, setCandidates] = useState<ClusterCandidate[]>([]);
  const [candidatesError, setCandidatesError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"comparaison" | "profils" | "assigner" | "derive">("profils");
  // Comparaison détaillée du top 3 (retour utilisateur direct : "propose
  // les 3 meilleurs modèles, résultats propres pour chaque, laisse le
  // choix à l'utilisateur") — clé = rang (string, contrainte de `Tabs<T
  // extends string>`), jamais l'algorithme seul (deux candidats du top 3
  // peuvent partager le même algorithme, ex. K-Means k=2 et k=3).
  const [selectedTopRank, setSelectedTopRank] = useState<string>("1");
  // Export en lot (retour utilisateur direct : "l'entreprise a 50 000
  // lignes, la plateforme n'en clusterise que 5 000, comment couvrir le
  // reste ?") — applique le modèle déjà entraîné à la TOTALITÉ du dataset
  // d'origine, pas seulement l'échantillon (voir
  // services/clustering_inference.py::assign_clusters_batch).
  const [exportingAssignments, setExportingAssignments] = useState(false);
  const [exportAssignmentsError, setExportAssignmentsError] = useState<string | null>(null);

  async function handleExportAssignments() {
    setExportingAssignments(true);
    setExportAssignmentsError(null);
    try {
      await api.clustering.exportAssignments(jobId);
    } catch (err) {
      setExportAssignmentsError(err instanceof ApiError ? err.message : "Impossible d'exporter les assignations");
    } finally {
      setExportingAssignments(false);
    }
  }

  // Fiche modèle (retour utilisateur direct : "on peut télécharger le
  // modèle mais pas un json... qui suit le modèle") — construite
  // ENTIÈREMENT à partir de `result`, déjà en mémoire, jamais un second
  // appel réseau. Voir `utils/clusteringModelCard.ts`.
  function handleExportModelCard() {
    if (!result) return;
    const card = buildClusteringModelCard(job, result);
    const blob = new Blob([JSON.stringify(card, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `clustering_fiche_modele_job${jobId}.json`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  useEffect(() => {
    api.clustering
      .getResult(jobId)
      .then(setResult)
      .catch((err) => setError(err instanceof ApiError ? err.message : "Résultat indisponible"));
    api.clustering
      .getCandidates(jobId)
      .then(setCandidates)
      .catch((err) => setCandidatesError(err instanceof ApiError ? err.message : "Impossible de charger le classement des configurations"));
  }, [jobId]);

  if (error) return <p className="text-sm text-destructive text-center">{error}</p>;
  if (!result) return <p className="text-sm text-muted-foreground text-center">Chargement…</p>;

  const winnerCandidate = candidates.find((c) => c.is_winner);
  // Top 3 avec résultats complets — le backend n'en calcule que 3 (voir
  // engine.py::TOP_N_WITH_FULL_RESULTS), `cluster_profiles !== null` est
  // donc un filtre suffisant, jamais un simple `.slice(0, 3)` qui
  // supposerait un ordre déjà garanti par le backend sans le vérifier ici.
  const topCandidatesWithProfiles = candidates.filter((c) => c.cluster_profiles !== null);
  const selectedTopCandidate = topCandidatesWithProfiles.find((c) => String(c.rank) === selectedTopRank);
  const quality = assessSilhouetteQuality(result.metrics.silhouette);
  const stabilityAri = result.model_card.stability_ari;
  const stability = assessStabilityQuality(typeof stabilityAri === "number" ? stabilityAri : null);
  const recommendation = winnerCandidate ? buildRecommendationExplanation(winnerCandidate, candidates) : null;
  const totalSamples = Number(result.model_card.n_samples) || result.profiles.reduce((sum, p) => sum + p.size, 0);
  const distribution = computeClusterDistribution(result.profiles, totalSamples);
  const noiseEntry = distribution.find((d) => d.isNoise);
  const isDensityAlgorithm = result.model_card.family === "densite";

  const noiseBudgetExceededForAll = Boolean(result.model_card.noise_budget_exceeded_for_all);

  return (
    <div className="max-w-4xl mx-auto space-y-5">
      {noiseBudgetExceededForAll && (
        <div className="flex items-start gap-3 text-warning bg-warning/10 border border-warning/20 rounded-lg p-4">
          <AlertCircle size={18} className="flex-shrink-0 mt-0.5" />
          <p className="text-sm">
            Aucune configuration testée ne structure une part suffisante de vos données (plus de la moitié des
            observations classées atypiques dans chaque cas) — le résultat ci-dessous reste le moins mauvais essai,
            à interpréter avec prudence. Essayez avec d'autres variables, ou vérifiez qu'une vraie structure de
            groupes existe dans vos données.
          </p>
        </div>
      )}

      <Card className={`p-5 ${accentSurfaceClass("rose")}`}>
        <SectionHeader
          icon={Boxes}
          color="rose"
          label={`${result.algorithm} — ${result.n_clusters} groupes découverts`}
          help="Silhouette : de -1 à 1, proche de 1 = groupes bien séparés et cohérents. Davies-Bouldin : plus bas = mieux. Calinski-Harabasz : plus haut = mieux — les trois se recoupent, ne se remplacent pas."
        />
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          {result.metrics.silhouette !== null && (
            <MetricTile label="Silhouette" direction="up" value={result.metrics.silhouette.toFixed(3)} color="rose" />
          )}
          {result.metrics.davies_bouldin !== null && (
            <MetricTile label="Davies-Bouldin" direction="down" value={result.metrics.davies_bouldin.toFixed(3)} color="blue" />
          )}
          {result.metrics.calinski_harabasz !== null && (
            <MetricTile label="Calinski-Harabasz" direction="up" value={result.metrics.calinski_harabasz.toFixed(0)} color="teal" />
          )}
        </div>

        <div className={`rounded-xl border p-3 mt-4 ${accentSurfaceClass(QUALITY_TONE_ACCENT[quality.tone])}`}>
          <span className="inline-flex items-center text-overline px-2 py-0.5 rounded-full bg-card/80 text-foreground">
            {quality.label}
          </span>
          <p className="text-xs text-foreground/70 mt-1.5">{quality.caveat}</p>
          {recommendation && (
            <p className="text-xs text-foreground/80 mt-2 pt-2 border-t border-border/40">{recommendation}</p>
          )}
        </div>
        {typeof stabilityAri === "number" && (
          <div className={`rounded-xl border p-3 mt-3 ${accentSurfaceClass(QUALITY_TONE_ACCENT[stability.tone])}`}>
            <span className="inline-flex items-center text-overline px-2 py-0.5 rounded-full bg-card/80 text-foreground">
              {stability.label} ({stabilityAri.toFixed(2)})
            </span>
            <p className="text-xs text-foreground/70 mt-1.5">{stability.caveat}</p>
          </div>
        )}
        {Boolean(result.model_card.sampled) && (
          <div className="mt-3 pt-3 border-t border-border/60">
            <p className="text-xs text-muted-foreground">
              Calculé sur un échantillon de {String(result.model_card.n_samples_used)} observations sur{" "}
              {String(result.model_card.n_samples_total)} au total — tirage aléatoire déterministe et reproductible.
              Statistiquement, la précision dépend de la taille de l'échantillon, pas du pourcentage couvert : ce
              chiffre n'est pas moins fiable qu'un échantillon plus large.{" "}
              {result.model_card.row_cap_linear_only
                ? "Plafond relevé à 20 000 lignes ici (algorithmes rapides uniquement, mode expert)."
                : "Le plafond reste à 5 000 lignes tant qu'un algorithme hiérarchique ou DBSCAN est comparé (coût mémoire qui augmente avec le carré du nombre de lignes) — le mode expert permet de le relever à 20 000 en ne comparant que des algorithmes rapides."}
            </p>
            <p className="text-xs text-muted-foreground mt-1.5">
              Pour un cluster sur chacune de vos {String(result.model_card.n_samples_total)} lignes réelles (pas
              seulement l'échantillon), exportez les assignations ci-dessous.
            </p>
          </div>
        )}
      </Card>

      <div className="flex items-center gap-2 flex-wrap">
        <ModelExportActions
          onExportArtifact={() => api.clustering.exportModel(jobId)}
          exportConfig={{
            algorithm: result.algorithm,
            n_clusters: result.n_clusters,
            feature_columns: job.feature_columns,
            metrics: result.metrics,
            model_card: result.model_card,
          }}
          configFilename={`clustering_config_job${jobId}.json`}
        />
        <Button variant="secondary" size="sm" onClick={handleExportAssignments} loading={exportingAssignments}>
          <Download size={14} />
          Exporter les assignations (CSV, toutes les lignes)
        </Button>
        <Button variant="secondary" size="sm" onClick={handleExportModelCard}>
          <FileJson size={14} />
          Fiche modèle (JSON)
        </Button>
        <Button variant="secondary" size="sm" onClick={() => api.clustering.exportDeploymentScript(jobId)}>
          <FileCode size={14} />
          Script de déploiement (.py)
        </Button>
      </div>
      {exportAssignmentsError && <p className="text-xs text-destructive">{exportAssignmentsError}</p>}
      <p className="text-xs text-muted-foreground -mt-2">
        Pour déployer ce modèle en dehors de DataLab Pro : téléchargez l'artefact ET le script de déploiement,
        placez-les dans le même dossier — le script recharge l'artefact et assigne un cluster, sans dépendre de
        cette plateforme (voir l'en-tête du script pour l'installation des bibliothèques nécessaires).
      </p>

      <Tabs
        items={[
          { id: "profils" as const, label: "Profils de segments", icon: Sparkles },
          { id: "comparaison" as const, label: "Comparaison", icon: ListChecks },
          { id: "assigner" as const, label: "Assigner", icon: Target },
          { id: "derive" as const, label: "Dérive", icon: Waves },
        ]}
        active={activeTab}
        onChange={setActiveTab}
        urlParam="onglet"
      />

      {activeTab === "comparaison" && (
        candidatesError ? (
          <p className="text-sm text-destructive text-center">{candidatesError}</p>
        ) : (
          candidates.length > 0 && (
            <div className="space-y-6">
              <div>
                <SectionHeader
                  icon={ListChecks}
                  color="blue"
                  label={`Configurations comparées (${candidates.length})`}
                  help="Plusieurs algorithmes et nombres de groupes sont testés à chaque lancement, classés sur un rang composite qui combine les 3 métriques de qualité (silhouette, Davies-Bouldin, Calinski-Harabasz) — jamais un seul critère isolé, qui peut sacrifier la compacité des groupes pour un gain marginal ailleurs. ↑ = plus haut est meilleur, ↓ = plus bas est meilleur."
                />
                <Table columns={CANDIDATE_COLUMNS} rows={candidates} rowKey={(c) => `${c.algorithm}-${c.rank}`} highlightRow={(c) => c.is_winner} />
              </div>

              {topCandidatesWithProfiles.length > 1 && (
                <div>
                  <SectionHeader
                    icon={Trophy}
                    color="violet"
                    label="Top 3 en détail — comparez avant de choisir"
                    help="Résultats complets (profils de segments) pour les 3 meilleures configurations. Le classement global retient la 1ʳᵉ, mais rien n'oblige à s'y limiter — un nombre de groupes légèrement différent peut être plus pertinent pour votre usage métier ; comparez, puis relancez cette configuration précise si vous préférez une autre entrée du tableau ci-dessus."
                  />
                  <Tabs
                    items={topCandidatesWithProfiles.map((c) => ({
                      id: String(c.rank),
                      label: `#${c.rank} — ${c.algorithm}`,
                      icon: c.is_winner ? Trophy : Boxes,
                    }))}
                    active={selectedTopRank}
                    onChange={setSelectedTopRank}
                  />
                  {selectedTopCandidate?.cluster_profiles && (
                    <div className="mt-4 space-y-3">
                      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground bg-muted/50 rounded-lg px-3 py-2">
                        <span>
                          Silhouette <strong className="text-foreground tabular-nums">{selectedTopCandidate.silhouette?.toFixed(3) ?? "—"}</strong>
                          {selectedTopCandidate.rank_silhouette !== null && ` (rang ${selectedTopCandidate.rank_silhouette})`}
                        </span>
                        <span>
                          Davies-Bouldin <strong className="text-foreground tabular-nums">{selectedTopCandidate.davies_bouldin?.toFixed(3) ?? "—"}</strong>
                          {selectedTopCandidate.rank_davies_bouldin !== null && ` (rang ${selectedTopCandidate.rank_davies_bouldin})`}
                        </span>
                        <span>
                          Calinski-Harabasz <strong className="text-foreground tabular-nums">{selectedTopCandidate.calinski_harabasz?.toFixed(0) ?? "—"}</strong>
                          {selectedTopCandidate.rank_calinski_harabasz !== null && ` (rang ${selectedTopCandidate.rank_calinski_harabasz})`}
                        </span>
                        {selectedTopCandidate.noise_count !== null && selectedTopCandidate.noise_count > 0 && (
                          <span>{selectedTopCandidate.noise_count} observation{selectedTopCandidate.noise_count > 1 ? "s" : ""} atypique{selectedTopCandidate.noise_count > 1 ? "s" : ""}</span>
                        )}
                      </div>
                      <ClusterProfileGrid profiles={selectedTopCandidate.cluster_profiles} />
                    </div>
                  )}
                </div>
              )}
            </div>
          )
        )
      )}

      {activeTab === "profils" && distribution.length > 0 && (
        <Card className="p-5">
          <SectionHeader
            icon={CircleDashed}
            color="amber"
            label="Répartition"
            help="Taille de chaque segment retenu, plus les observations non rattachées à un groupe (le cas échéant) — les pourcentages totalisent toujours 100 %."
          />
          <div className="flex h-3 rounded-full overflow-hidden mb-4 bg-muted">
            {distribution.map((d, i) => (
              <div
                key={d.id}
                style={{
                  width: `${d.pct}%`,
                  backgroundColor: d.isNoise ? DISTRIBUTION_NOISE_COLOR : DISTRIBUTION_BAR_COLORS[i % DISTRIBUTION_BAR_COLORS.length],
                }}
                title={`${d.label} — ${d.count} (${d.pct.toFixed(1)} %)`}
              />
            ))}
          </div>
          <div className="space-y-2">
            {distribution.map((d, i) => (
              <div key={d.id} className="flex items-center gap-2.5 text-sm">
                <span
                  className="size-2.5 rounded-full flex-shrink-0"
                  style={{
                    backgroundColor: d.isNoise
                      ? DISTRIBUTION_NOISE_COLOR
                      : DISTRIBUTION_BAR_COLORS[i % DISTRIBUTION_BAR_COLORS.length],
                  }}
                />
                <span className={`flex-1 ${d.isNoise ? "text-muted-foreground italic" : "text-foreground/90"}`}>{d.label}</span>
                <span className="text-muted-foreground tabular-nums">{d.count}</span>
                <span className="font-medium tabular-nums w-14 text-right">{d.pct.toFixed(1)} %</span>
              </div>
            ))}
          </div>
          {noiseEntry && (
            <p className="text-xs text-muted-foreground mt-4 pt-3 border-t border-border/60">
              <strong className="text-foreground/80">Observations atypiques / non rattachées</strong> — {noiseEntry.count}{" "}
              observation{noiseEntry.count > 1 ? "s" : ""} ({noiseEntry.pct.toFixed(1)} %){" "}
              {isDensityAlgorithm
                ? "que l'algorithme par densité retenu (proche des voisinages, type DBSCAN) considère comme du bruit — trop isolées pour appartenir à un groupe, pas nécessairement une anomalie métier."
                : "non rattachées à un groupe par l'algorithme retenu — à ne pas confondre automatiquement avec une anomalie métier."}
            </p>
          )}
        </Card>
      )}

      {activeTab === "profils" && (
      <div>
        <SectionHeader icon={Sparkles} color="violet" label="Profils de segments" help="Chaque groupe décrit par sa taille et ce qui le distingue le plus du reste de la population — jamais une simple étiquette numérique." />
        <ClusterProfileGrid profiles={result.profiles} />
      </div>
      )}

      {activeTab === "assigner" && <ClusterAssignmentForm jobId={jobId} featureColumns={job.feature_columns} />}

      {activeTab === "derive" && <DriftPanel pillar="clustering" jobId={jobId} />}

      <Card className="p-4 flex items-center justify-between gap-3 flex-wrap">
        <div className="flex items-start gap-3">
          <ScanSearch size={16} className="text-muted-foreground flex-shrink-0 mt-0.5" />
          <p className="text-xs text-muted-foreground">
            Prolongez l'analyse sur ces mêmes variables : visualisez-les en 2 dimensions, ou repérez les observations
            les plus atypiques.
          </p>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <Link
            to={`/reduction-dimension?dataset_id=${job.dataset_id}&features=${encodeURIComponent(job.feature_columns.join(","))}`}
          >
            <Button variant="secondary" size="sm">
              <ScanSearch size={14} />
              Visualiser en 2D
            </Button>
          </Link>
          <Link to={`/anomalies?dataset_id=${job.dataset_id}&features=${encodeURIComponent(job.feature_columns.join(","))}`}>
            <Button variant="secondary" size="sm">
              <AlertCircle size={14} />
              Détecter les anomalies
            </Button>
          </Link>
        </div>
      </Card>
    </div>
  );
}

// Explication de la méthode d'assignation — voir
// services/clustering_inference.py pour le raisonnement complet derrière
// chaque cas (aucun algorithme du registre n'a de méthode d'assignation
// "gratuite" hors K-Means/K-Means rapide, voir la docstring du module).
const ASSIGNMENT_METHOD_LABELS: Record<ClusterAssignmentMethod, string> = {
  exact: "Assignation exacte — même critère qu'à l'entraînement (distance au centroïde le plus proche).",
  approximate_centroid:
    "Assignation approchée par centroïde le plus proche — l'algorithme retenu (hiérarchique) n'a pas de méthode d'assignation native pour de nouvelles observations.",
  approximate_nearest_core:
    "Assignation approchée par voisinage le plus proche — l'algorithme retenu (densité) n'a pas de méthode d'assignation native pour de nouvelles observations.",
  unsupported: "Assignation indisponible pour ce clustering — relancez un clustering pour en bénéficier.",
};

function ClusterAssignmentForm({ jobId, featureColumns }: { jobId: number; featureColumns: string[] }) {
  const [values, setValues] = useState<Record<string, string>>({});
  const [prediction, setPrediction] = useState<ClusterPrediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setPrediction(null);
    setIsSubmitting(true);
    try {
      const result = await api.clustering.predict(jobId, values);
      setPrediction(result);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Impossible d'assigner cette observation");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <Card className="p-5">
      <SectionHeader
        icon={Target}
        color="violet"
        label="Assigner une nouvelle observation"
        help="Indiquez les valeurs d'une nouvelle observation pour voir à quel groupe déjà découvert elle se rattache le mieux, sans relancer un clustering complet."
      />
      <form onSubmit={handleSubmit} className="space-y-3">
        <div className="grid sm:grid-cols-2 gap-3">
          {featureColumns.map((col) => (
            <div key={col}>
              <label htmlFor={`assign-${col}`} className="block text-xs text-muted-foreground mb-1">
                {col}
              </label>
              <Input
                id={`assign-${col}`}
                value={values[col] ?? ""}
                onChange={(e) => setValues((prev) => ({ ...prev, [col]: e.target.value }))}
                required
              />
            </div>
          ))}
        </div>
        {error && (
          <div className="flex items-center gap-2 text-sm text-destructive bg-destructive/10 border border-destructive/20 rounded-lg px-3 py-2">
            <AlertCircle size={15} className="flex-shrink-0" />
            {error}
          </div>
        )}
        <Button type="submit" variant="secondary" size="sm" disabled={isSubmitting}>
          {isSubmitting ? "Assignation…" : "Assigner"}
        </Button>
      </form>
      {prediction && (
        <div
          className={`rounded-lg border p-3 mt-3 ${accentSurfaceClass(prediction.cluster_id !== null ? "teal" : "amber")}`}
        >
          <p className="text-sm text-foreground">
            {prediction.cluster_id !== null
              ? `Rattachée au segment ${prediction.cluster_id + 1}.`
              : prediction.assignment_method === "unsupported"
                ? "Assignation indisponible pour ce clustering."
                : "Observation atypique — non rattachée à un groupe."}
          </p>
          <p className="text-xs text-muted-foreground mt-1">{ASSIGNMENT_METHOD_LABELS[prediction.assignment_method]}</p>
        </div>
      )}
    </Card>
  );
}

function MetricTile({
  label,
  value,
  color,
  direction,
}: {
  label: string;
  value: string;
  color: AccentColor;
  /** Sens d'amélioration de la métrique — les 3 métriques de clustering ne
   * se lisent pas toutes dans le même sens, source de confusion signalée. */
  direction?: "up" | "down";
}) {
  return (
    <div className={`rounded-xl border px-4 py-3 ${accentSurfaceClass(color)}`}>
      <p className="text-xs text-muted-foreground mb-1 inline-flex items-center gap-1">
        {label}
        {direction && <MetricDirection better={direction} />}
      </p>
      <p className={`text-xl font-bold tabular-nums ${accentValueTextClass(color)}`}>{value}</p>
    </div>
  );
}
