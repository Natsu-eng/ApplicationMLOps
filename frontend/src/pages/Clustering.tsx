import { useCallback, useEffect, useState, type FormEvent } from "react";
import { Link, useSearchParams } from "react-router-dom";
import {
  AlertCircle,
  ArrowDown,
  ArrowUp,
  Ban,
  Boxes,
  CircleDashed,
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
} from "lucide-react";
import {
  ApiError,
  api,
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
import { ColorIconBadge, accentSurfaceClass, accentValueTextClass, type AccentColor } from "../components/ui/ColorIconBadge";
import { PageHeader } from "../components/ui/PageHeader";
import { SectionHeader } from "../components/ui/SectionHeader";
import { Select } from "../components/ui/Select";
import { Table, type TableColumn } from "../components/ui/Table";
import { DataQualityWarnings } from "../components/training/DataQualityWarnings";
import { useConfirmAction } from "../hooks/useConfirmAction";
import { CHART_SERIES_COLORS } from "../theme/charts";
import {
  assessSilhouetteQuality,
  assessStabilityQuality,
  buildRecommendationExplanation,
  computeClusterDistribution,
} from "../utils/clusterQuality";
import { QUALITY_TONE_ACCENT } from "../utils/qualityAssessment";

const ACTIVE_STATUSES = new Set(["queued", "running"]);
const POLL_INTERVAL_MS = 3000;
const ACTIVE_JOB_STORAGE_KEY = "datalab_active_clustering_job_id";

type Phase = "configure" | "progress" | "results" | "failed" | "cancelled";

function phaseOf(job: ClusteringJobSummary | null): Phase {
  if (!job) return "configure";
  if (ACTIVE_STATUSES.has(job.status)) return "progress";
  if (job.status === "completed") return "results";
  return job.status === "cancelled" ? "cancelled" : "failed";
}

const CANDIDATE_COLORS: AccentColor[] = ["rose", "blue", "teal", "amber", "violet"];

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

  useEffect(() => {
    if (phase !== "progress" || !activeJob) return;
    const interval = setInterval(async () => {
      try {
        setActiveJob(await api.clustering.getJob(activeJob.id));
      } catch {
        // silencieux — nouvelle tentative au prochain tick
      }
    }, POLL_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [phase, activeJob]);

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
      const job = await api.clustering.createJob({
        dataset_id: datasetId,
        feature_columns: Array.from(selectedFeatures),
      });
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

        {error && (
          <div className="flex items-center gap-2 text-sm text-destructive bg-destructive/10 border border-destructive/20 rounded-lg px-3 py-2">
            <AlertCircle size={15} className="flex-shrink-0" />
            {error}
          </div>
        )}

        <Button type="submit" disabled={!datasetId || selectedFeatures.size === 0 || isSubmitting} className="w-full">
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
];

// Palette validée (theme/charts.ts) — jamais une teinte hex ad hoc pour une
// série de données, même dans une barre empilée custom hors Recharts.
const DISTRIBUTION_BAR_COLORS = CHART_SERIES_COLORS;
const DISTRIBUTION_NOISE_COLOR = "#94a3b8"; // slate-400 — neutre, jamais confondu avec un vrai segment

function ClusteringResultView({ job }: { job: ClusteringJobSummary }) {
  const jobId = job.id;
  const [result, setResult] = useState<ClusteringResult | null>(null);
  const [candidates, setCandidates] = useState<ClusterCandidate[]>([]);
  const [candidatesError, setCandidatesError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

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
          <span
            className={`inline-flex items-center text-overline px-2 py-0.5 rounded-full bg-card/80 ${accentValueTextClass(QUALITY_TONE_ACCENT[quality.tone])}`}
          >
            {quality.label}
          </span>
          <p className="text-xs text-foreground/70 mt-1.5">{quality.caveat}</p>
          {recommendation && (
            <p className="text-xs text-foreground/80 mt-2 pt-2 border-t border-border/40">{recommendation}</p>
          )}
        </div>
        {typeof stabilityAri === "number" && (
          <div className={`rounded-xl border p-3 mt-3 ${accentSurfaceClass(QUALITY_TONE_ACCENT[stability.tone])}`}>
            <span
              className={`inline-flex items-center text-overline px-2 py-0.5 rounded-full bg-card/80 ${accentValueTextClass(QUALITY_TONE_ACCENT[stability.tone])}`}
            >
              {stability.label} ({stabilityAri.toFixed(2)})
            </span>
            <p className="text-xs text-foreground/70 mt-1.5">{stability.caveat}</p>
          </div>
        )}
        {Boolean(result.model_card.sampled) && (
          <p className="text-xs text-muted-foreground mt-3">
            Calculé sur un échantillon de {String(result.model_card.n_samples_used)} observations sur{" "}
            {String(result.model_card.n_samples_total)} au total (dataset volumineux — échantillonnage déterministe,
            reproductible).
          </p>
        )}
      </Card>

      {candidatesError ? (
        <p className="text-sm text-destructive text-center">{candidatesError}</p>
      ) : (
        candidates.length > 0 && (
          <div>
            <SectionHeader
              icon={ListChecks}
              color="blue"
              label={`Configurations comparées (${candidates.length})`}
              help="Plusieurs algorithmes et nombres de groupes sont testés à chaque lancement, classés sur le score de silhouette — jamais un seul essai lancé à l'aveugle. ↑ = plus haut est meilleur, ↓ = plus bas est meilleur."
            />
            <Table columns={CANDIDATE_COLUMNS} rows={candidates} rowKey={(c) => `${c.algorithm}-${c.rank}`} highlightRow={(c) => c.is_winner} />
          </div>
        )
      )}

      {distribution.length > 0 && (
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

      <div>
        <SectionHeader icon={Sparkles} color="violet" label="Profils de segments" help="Chaque groupe décrit par sa taille et ce qui le distingue le plus du reste de la population — jamais une simple étiquette numérique." />
        <div className="grid sm:grid-cols-2 gap-4">
          {result.profiles.map((profile, i) => {
            const color = CANDIDATE_COLORS[i % CANDIDATE_COLORS.length];
            return (
              <Card key={profile.cluster_id} className={`p-4 ${accentSurfaceClass(color)}`}>
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <ColorIconBadge icon={Boxes} color={color} size="sm" />
                    <span className="text-sm font-medium text-foreground">Segment {profile.cluster_id + 1}</span>
                  </div>
                  <span className={`text-sm font-semibold tabular-nums ${accentValueTextClass(color)}`}>
                    {profile.size_pct.toFixed(1)} %
                  </span>
                </div>
                <p className="text-xs text-muted-foreground mb-2">
                  {profile.size} observation{profile.size > 1 ? "s" : ""}
                </p>
                {profile.differentiating_variables.length > 0 && (
                  <div className="space-y-1 mb-2">
                    <p className="text-overline uppercase text-muted-foreground">
                      Variables différenciantes
                    </p>
                    {profile.differentiating_variables.slice(0, 3).map((varName) => {
                      const stat = profile.numeric_summary[varName];
                      if (!stat) return null;
                      return (
                        <p key={varName} className="text-xs text-foreground/90">
                          <span className="font-medium">{varName}</span> : moyenne {stat.mean.toFixed(2)}{" "}
                          <span className="text-muted-foreground">
                            ({stat.z_score > 0 ? "+" : ""}
                            {stat.z_score.toFixed(1)}σ vs population)
                          </span>
                        </p>
                      );
                    })}
                  </div>
                )}
                {Object.entries(profile.categorical_summary).length > 0 && (
                  <div className="space-y-1">
                    {Object.entries(profile.categorical_summary)
                      .slice(0, 2)
                      .map(([col, cat]) => (
                        <p key={col} className="text-xs text-foreground/90">
                          <span className="font-medium">{col}</span> dominant : {cat.top_category} (
                          {cat.top_pct.toFixed(0)} %
                          {cat.lift !== null && (cat.lift >= 1.5 || cat.lift <= 0.67) && (
                            <span className="text-muted-foreground">
                              {" "}
                              — vs {cat.population_pct.toFixed(0)} % sur l'ensemble, ×{cat.lift.toFixed(1)}
                            </span>
                          )}
                          )
                        </p>
                      ))}
                  </div>
                )}
              </Card>
            );
          })}
        </div>
      </div>

      <ClusterAssignmentForm jobId={jobId} featureColumns={job.feature_columns} />

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
      <p className={`text-xl font-semibold tabular-nums ${accentValueTextClass(color)}`}>{value}</p>
    </div>
  );
}
