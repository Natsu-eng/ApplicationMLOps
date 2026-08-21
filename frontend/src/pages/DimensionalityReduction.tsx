import { useCallback, useEffect, useState, type FormEvent } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { AlertCircle, Ban, Info, Loader2, PlayCircle, RotateCcw, ScatterChart as ScatterChartIcon, Sparkles, Trash2 } from "lucide-react";
import {
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  ApiError,
  api,
  type AlgorithmCatalogEntry,
  type ColumnSchema,
  type DatasetSummary,
  type DimensionalityColorByResponse,
  type DimensionalityJobSummary,
  type DimensionalityPoint,
  type DimensionalityResult,
} from "../api/client";
import AppShell from "../components/AppShell";
import { pillarColor } from "../config/pillars";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { accentSurfaceClass, accentValueTextClass, type AccentColor } from "../components/ui/ColorIconBadge";
import { PageHeader } from "../components/ui/PageHeader";
import { SectionHeader } from "../components/ui/SectionHeader";
import { Select } from "../components/ui/Select";
import { Table, type TableColumn } from "../components/ui/Table";
import { useConfirmAction } from "../hooks/useConfirmAction";
import { CHART_GRID_STROKE, CHART_SERIES_COLORS, CHART_TICK_STYLE, CHART_TOOLTIP_STYLE } from "../theme/charts";
import { binIndexForValue, computeQuantileEdges, formatBinLabel } from "../utils/quantileBins";
import { DataQualityWarnings } from "../components/training/DataQualityWarnings";
import { assessTrustworthinessQuality } from "../utils/dimensionalityQuality";
import { QUALITY_TONE_ACCENT } from "../utils/qualityAssessment";

const ACTIVE_STATUSES = new Set(["queued", "running"]);
const POLL_INTERVAL_MS = 3000;
const ACTIVE_JOB_STORAGE_KEY = "datalab_active_dimensionality_job_id";

type Phase = "configure" | "progress" | "results" | "failed" | "cancelled";

function phaseOf(job: DimensionalityJobSummary | null): Phase {
  if (!job) return "configure";
  if (ACTIVE_STATUSES.has(job.status)) return "progress";
  if (job.status === "completed") return "results";
  return job.status === "cancelled" ? "cancelled" : "failed";
}

/** Pilier ML non supervisé — réduction de dimension (Lot 13). Même state
 * machine et même persistance de session que Clustering.tsx. Accepte un
 * pré-remplissage dataset + colonnes via query params (`?dataset_id=&
 * features=`, séparateur virgule) — lien croisé depuis un résultat de
 * clustering, volontairement sans transmission de label de cluster (couplage
 * nul entre les deux moteurs backend). */
export default function DimensionalityReduction() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [datasetsError, setDatasetsError] = useState<string | null>(null);
  const [activeJob, setActiveJob] = useState<DimensionalityJobSummary | null>(null);
  const [restoringJob, setRestoringJob] = useState(true);
  const confirmDelete = useConfirmAction<true>();

  // Reprise d'un calcul — priorité au deep-link `?job=` (ex. depuis la page
  // Historique du pilier non supervisé), sinon la session en cours.
  useEffect(() => {
    const queryJobId = searchParams.get("job");
    const storedId = queryJobId ?? sessionStorage.getItem(ACTIVE_JOB_STORAGE_KEY);
    if (!storedId) {
      setRestoringJob(false);
      return;
    }
    api.dimensionality
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

  function openJob(job: DimensionalityJobSummary) {
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
        setActiveJob(await api.dimensionality.getJob(activeJob.id));
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
      await api.dimensionality.remove(activeJob.id);
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
      setActiveJob(await api.dimensionality.cancel(activeJob.id));
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
      openJob(await api.dimensionality.rerun(activeJob.id));
    } catch (err) {
      setRerunError(err instanceof ApiError ? err.message : "Impossible de relancer ce calcul");
    } finally {
      setRerunning(false);
    }
  }

  const titles: Record<Phase, string> = {
    configure: "Visualiser vos données en 2D",
    progress: "Calcul de la projection en cours",
    results: "Projection calculée",
    failed: "Échec du calcul",
    cancelled: "Calcul annulé",
  };

  return (
    <AppShell pillarId="unsupervised">
      <PageHeader
        eyebrow="ML non supervisé"
        title={titles[phase]}
        description={
          phase === "configure"
            ? "Ramenez vos données à 2 dimensions pour les visualiser en un coup d'œil — utile pour repérer visuellement des groupes ou des points isolés."
            : undefined
        }
        icon={ScatterChartIcon}
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
                    aria-label={confirmDelete.isPending(true) ? "Confirmer la suppression" : "Supprimer ce calcul"}
                    title={confirmDelete.isPending(true) ? "Cliquer à nouveau pour confirmer" : "Supprimer ce calcul"}
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
                Nouveau calcul
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
          <DimensionalityForm
            datasets={datasets}
            datasetsError={datasetsError}
            onJobCreated={openJob}
            initialDatasetId={searchParams.get("dataset_id")}
            initialFeatures={searchParams.get("features")}
          />
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
            {cancelling ? "Annulation…" : "Annuler ce calcul"}
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
            <p className="text-sm">{activeJob.error_message ?? "Le calcul a échoué."}</p>
          </div>
        </Card>
      ) : phase === "results" && activeJob ? (
        <DimensionalityResultView jobId={activeJob.id} />
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

function DimensionalityForm({
  datasets,
  datasetsError,
  onJobCreated,
  initialDatasetId,
  initialFeatures,
}: {
  datasets: DatasetSummary[];
  datasetsError: string | null;
  onJobCreated: (job: DimensionalityJobSummary) => void;
  initialDatasetId: string | null;
  initialFeatures: string | null;
}) {
  const [datasetId, setDatasetId] = useState<number | "">("");
  const [columns, setColumns] = useState<ColumnSchema[]>([]);
  const [selectedFeatures, setSelectedFeatures] = useState<Set<string>>(new Set());
  const [algorithms, setAlgorithms] = useState<AlgorithmCatalogEntry[]>([]);
  const [algorithmId, setAlgorithmId] = useState("pca");
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [prefillApplied, setPrefillApplied] = useState(false);

  useEffect(() => {
    api.dimensionality
      .algorithmsCatalog()
      .then((res) => {
        setAlgorithms(res.algorithms);
        const preferred = res.algorithms.find((a) => a.is_default);
        if (preferred) setAlgorithmId(preferred.id);
      })
      .catch(() => setError("Impossible de charger le catalogue de méthodes — réessayez dans un instant."));
  }, []);

  const handleDatasetChange = useCallback(async (id: string, preselect?: Set<string>) => {
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
      setSelectedFeatures(preselect && preselect.size > 0 ? preselect : new Set(detail.columns.map((c) => c.name)));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Impossible de charger les colonnes");
    }
  }, []);

  // Pré-remplissage depuis le lien croisé "Visualiser en 2D" (résultat de
  // clustering) — une seule fois, dès que la liste de datasets est chargée.
  useEffect(() => {
    if (prefillApplied || datasets.length === 0 || !initialDatasetId) return;
    setPrefillApplied(true);
    const exists = datasets.some((d) => String(d.id) === initialDatasetId);
    if (!exists) return;
    const preselect = new Set((initialFeatures ?? "").split(",").filter(Boolean));
    handleDatasetChange(initialDatasetId, preselect);
  }, [datasets, initialDatasetId, initialFeatures, prefillApplied, handleDatasetChange]);

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
      const job = await api.dimensionality.createJob({
        dataset_id: datasetId,
        feature_columns: Array.from(selectedFeatures),
        algorithm_id: algorithmId,
      });
      onJobCreated(job);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Impossible de lancer le calcul");
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
          <label htmlFor="dimred-dataset" className="block text-sm text-muted-foreground mb-1">
            Jeu de données
          </label>
          <Select id="dimred-dataset" value={datasetId} onChange={(e) => handleDatasetChange(e.target.value)} required>
            <option value="">Choisir un dataset…</option>
            {datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name} ({d.row_count} lignes)
              </option>
            ))}
          </Select>
        </div>

        {algorithms.length > 0 && (
          <div>
            <label htmlFor="dimred-algorithm" className="block text-sm text-muted-foreground mb-1">
              Méthode
            </label>
            <Select id="dimred-algorithm" value={algorithmId} onChange={(e) => setAlgorithmId(e.target.value)}>
              {algorithms.map((a) => (
                <option key={a.id} value={a.id}>
                  {a.label}
                </option>
              ))}
            </Select>
          </div>
        )}

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
          {isSubmitting ? "Lancement…" : "Calculer la projection"}
        </Button>
      </form>
    </Card>
  );
}

const COLOR_NONE = "__none__";

function buildSeries(
  points: DimensionalityPoint[],
  colorBy: DimensionalityColorByResponse | null
): { label: string; color: string; points: DimensionalityPoint[] }[] {
  if (!colorBy) {
    return [{ label: "Observations", color: CHART_SERIES_COLORS[0], points }];
  }
  if (colorBy.kind === "categorical") {
    const categories = Array.from(
      new Set(points.map((p) => colorBy.values[String(p.row_index)])).values()
    ).map((v) => (v === null || v === undefined ? "—" : String(v)));
    const uniqueCategories = Array.from(new Set(categories)).sort();
    return uniqueCategories.map((cat, i) => ({
      label: cat,
      color: CHART_SERIES_COLORS[i % CHART_SERIES_COLORS.length],
      points: points.filter((p) => {
        const raw = colorBy.values[String(p.row_index)];
        const label = raw === null || raw === undefined ? "—" : String(raw);
        return label === cat;
      }),
    }));
  }
  const numericValues = points
    .map((p) => colorBy.values[String(p.row_index)])
    .filter((v): v is number => typeof v === "number");
  const edges = computeQuantileEdges(numericValues, 5);
  const binCount = Math.max(1, edges.length - 1);
  const groups: DimensionalityPoint[][] = Array.from({ length: binCount }, () => []);
  for (const p of points) {
    const v = colorBy.values[String(p.row_index)];
    if (typeof v !== "number") continue;
    const idx = Math.min(binIndexForValue(v, edges), binCount - 1);
    groups[idx].push(p);
  }
  return groups
    .map((pts, i) => ({ label: formatBinLabel(edges, i), color: CHART_SERIES_COLORS[i % CHART_SERIES_COLORS.length], points: pts }))
    .filter((series) => series.points.length > 0);
}

const LOADING_COLUMNS: TableColumn<{ feature: string; pc1: number; pc2: number }>[] = [
  { key: "feature", header: "Variable" },
  { key: "pc1", header: "PC1", align: "right", render: (r) => r.pc1.toFixed(3) },
  { key: "pc2", header: "PC2", align: "right", render: (r) => r.pc2.toFixed(3) },
];

function DimensionalityResultView({ jobId }: { jobId: number }) {
  const [result, setResult] = useState<DimensionalityResult | null>(null);
  const [points, setPoints] = useState<DimensionalityPoint[]>([]);
  const [pointsError, setPointsError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [colorByColumn, setColorByColumn] = useState<string>(COLOR_NONE);
  const [colorByData, setColorByData] = useState<DimensionalityColorByResponse | null>(null);

  useEffect(() => {
    api.dimensionality
      .getResult(jobId)
      .then(setResult)
      .catch((err) => setError(err instanceof ApiError ? err.message : "Résultat indisponible"));
    api.dimensionality
      .getPoints(jobId)
      .then(setPoints)
      .catch((err) => setPointsError(err instanceof ApiError ? err.message : "Impossible de charger les points de la projection"));
  }, [jobId]);

  useEffect(() => {
    if (colorByColumn === COLOR_NONE) {
      setColorByData(null);
      return;
    }
    api.dimensionality
      .getColorBy(jobId, colorByColumn)
      .then(setColorByData)
      .catch(() => setColorByData(null));
  }, [jobId, colorByColumn]);

  if (error) return <p className="text-sm text-destructive text-center">{error}</p>;
  if (!result) return <p className="text-sm text-muted-foreground text-center">Chargement…</p>;

  const series = buildSeries(points, colorByData);
  const quality = assessTrustworthinessQuality(result.trustworthiness_primary);

  return (
    <div className="max-w-4xl mx-auto space-y-5">
      <Card className={`p-5 ${accentSurfaceClass("blue")}`}>
        <SectionHeader
          icon={ScatterChartIcon}
          color="blue"
          label={result.algorithm}
          help="Conservation des voisinages : proche de 1, les points proches sur cette projection étaient déjà proches dans les données d'origine (mesure réelle, sklearn.manifold.trustworthiness) — ne dit rien des distances entre groupes éloignés. Variance expliquée : propre à la PCA, part de l'information des données conservée sur ces 2 axes."
        />
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          {result.algorithm_id === "pca" && (
            <MetricTile label="Variance expliquée" value={`${(result.total_variance_explained * 100).toFixed(1)} %`} color="blue" />
          )}
          <MetricTile label="Conservation des voisinages" value={result.trustworthiness_primary.toFixed(3)} color="teal" />
        </div>
        <div className={`rounded-xl border p-3 mt-4 ${accentSurfaceClass(QUALITY_TONE_ACCENT[quality.tone])}`}>
          <span
            className={`inline-flex items-center text-overline px-2 py-0.5 rounded-full bg-card/80 ${accentValueTextClass(QUALITY_TONE_ACCENT[quality.tone])}`}
          >
            {quality.label}
          </span>
          <p className="text-xs text-foreground/70 mt-1.5">{quality.caveat}</p>
        </div>
        {result.algorithm_id !== "pca" && (
          <p className={`text-xs mt-3 rounded-lg border px-3 py-2 ${accentSurfaceClass("violet")}`}>
            <span className={`font-medium ${accentValueTextClass("violet")}`}>Référence PCA</span>
            <span className="text-foreground/70">
              {" "}
              (calculée en plus, à titre de repère — n'est pas la méthode utilisée pour cette projection) — variance
              expliquée : {(result.total_variance_explained * 100).toFixed(1)} % · conservation des voisinages PCA :{" "}
              {result.trustworthiness_pca.toFixed(3)}.
            </span>
          </p>
        )}
        {result.sampled && (
          <p className="text-xs text-muted-foreground mt-3">
            Calculé sur un échantillon de {result.n_samples_used} observations sur {result.n_samples_total} au total
            (dataset volumineux — échantillonnage déterministe, reproductible).
          </p>
        )}
        {Array.isArray(result.model_card.categorical_columns) && result.model_card.categorical_columns.length > 0 && (
          <p className="text-xs text-muted-foreground mt-3">
            {result.model_card.categorical_columns.length} variable
            {result.model_card.categorical_columns.length > 1 ? "s" : ""} catégorielle
            {result.model_card.categorical_columns.length > 1 ? "s" : ""} encodée
            {result.model_card.categorical_columns.length > 1 ? "s" : ""} en {String(result.model_card.n_categorical_dimensions)}{" "}
            dimension{Number(result.model_card.n_categorical_dimensions) > 1 ? "s" : ""} (sur{" "}
            {String(result.model_card.n_dimensions_after_encoding)} au total) avant projection — une variable à
            beaucoup de valeurs distinctes peut peser plus lourd dans les distances calculées.
          </p>
        )}
      </Card>

      <Card className={`p-4 flex items-start gap-3 ${accentSurfaceClass("amber")}`}>
        <Info size={16} className={`flex-shrink-0 mt-0.5 ${accentValueTextClass("amber")}`} />
        <p className="text-xs text-foreground/70">{result.distance_fidelity_note}</p>
      </Card>

      <Card className="p-5">
        <div className="flex items-center justify-between flex-wrap gap-3 mb-3">
          <SectionHeader icon={Sparkles} color="violet" label="Projection en 2 dimensions" />
          <div className="flex items-center gap-2">
            <label htmlFor="dimred-color-by" className="text-xs text-muted-foreground flex-shrink-0">
              Colorer par
            </label>
            <div className="w-44">
              <Select id="dimred-color-by" value={colorByColumn} onChange={(e) => setColorByColumn(e.target.value)}>
                <option value={COLOR_NONE}>Aucune</option>
                {result.feature_columns.map((col) => (
                  <option key={col} value={col}>
                    {col}
                  </option>
                ))}
              </Select>
            </div>
          </div>
        </div>
        {pointsError ? (
          <p className="text-sm text-destructive text-center py-8">{pointsError}</p>
        ) : (
          <ResponsiveContainer width="100%" height={360}>
            <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID_STROKE} />
              <XAxis type="number" dataKey="x" tick={CHART_TICK_STYLE} name="Axe 1" />
              <YAxis type="number" dataKey="y" tick={CHART_TICK_STYLE} name="Axe 2" />
              <RechartsTooltip {...CHART_TOOLTIP_STYLE} cursor={{ strokeDasharray: "3 3" }} />
              {series.length > 1 && <Legend wrapperStyle={{ fontSize: 11 }} />}
              {series.map((s) => (
                <Scatter key={s.label} name={s.label} data={s.points} fill={s.color} fillOpacity={0.75} />
              ))}
            </ScatterChart>
          </ResponsiveContainer>
        )}
      </Card>

      {result.algorithm_id === "pca" && result.loadings.length > 0 && (
        <div>
          <SectionHeader
            icon={Sparkles}
            color="amber"
            label="Variables les plus contributives (PCA)"
            help="Poids de chaque variable sur les 2 axes de la PCA — n'a de sens que pour une projection PCA (PC1/PC2 sont des combinaisons linéaires des variables). t-SNE/UMAP n'ont pas d'équivalent : leurs axes ne s'interprètent pas de cette façon."
          />
          <Table
            columns={LOADING_COLUMNS}
            rows={result.loadings.map((l) => ({ feature: l.feature, pc1: l.pc1, pc2: l.pc2 }))}
            rowKey={(r) => r.feature}
          />
        </div>
      )}
    </div>
  );
}

function MetricTile({ label, value, color }: { label: string; value: string; color: AccentColor }) {
  return (
    <div className={`rounded-xl border px-4 py-3 ${accentSurfaceClass(color)}`}>
      <p className="text-xs text-muted-foreground mb-1">{label}</p>
      <p className={`text-xl font-semibold tabular-nums ${accentValueTextClass(color)}`}>{value}</p>
    </div>
  );
}
