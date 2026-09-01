import { useCallback, useEffect, useMemo, useState, type FormEvent } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { AlertCircle, AlertTriangle, Ban, BarChart3, CheckCircle2, Download, FileCode, FileJson, Loader2, PlayCircle, RotateCcw, Search, SlidersHorizontal, Target, Trash2 } from "lucide-react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip as RechartsTooltip, XAxis, YAxis } from "recharts";
import {
  ApiError,
  api,
  type AnomalyAgreement,
  type AnomalyJobSummary,
  type AnomalyObservation,
  type AnomalyResult,
  type AnomalyScore,
  type ColumnSchema,
  type DatasetSummary,
} from "../api/client";
import AppShell from "../components/AppShell";
import { pillarColor } from "../config/pillars";
import { Badge } from "../components/ui/Badge";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { accentSurfaceClass, accentValueTextClass, type AccentColor } from "../components/ui/ColorIconBadge";
import { Input } from "../components/ui/Input";
import { Modal } from "../components/ui/Modal";
import { PageHeader } from "../components/ui/PageHeader";
import { SectionHeader } from "../components/ui/SectionHeader";
import { Select } from "../components/ui/Select";
import { Table, type TableColumn } from "../components/ui/Table";
import { Tabs } from "../components/ui/Tabs";
import { ModelExportActions } from "../components/ui/ModelExportActions";
import { useConfirmAction } from "../hooks/useConfirmAction";
import { useIdempotencyKey } from "../hooks/useIdempotencyKey";
import { CHART_COLOR_PRIMARY, CHART_GRID_STROKE, CHART_TICK_STYLE_SM, CHART_TOOLTIP_STYLE } from "../theme/charts";
import { DataQualityWarnings } from "../components/training/DataQualityWarnings";
import { useJobEvents } from "../hooks/useJobEvents";
import { assessConsensusQuality } from "../utils/anomalyQuality";
import { buildAnomalyModelCard } from "../utils/anomalyModelCard";
import { QUALITY_TONE_ACCENT } from "../utils/qualityAssessment";

const ACTIVE_STATUSES = new Set(["queued", "running"]);
const ACTIVE_JOB_STORAGE_KEY = "datalab_active_anomaly_job_id";
const DEFAULT_TOP_N = 50;
const MAX_TOP_N = 200;
const DEFAULT_CONTAMINATION_PCT = 5;

type Phase = "configure" | "progress" | "results" | "failed" | "cancelled";

function phaseOf(job: AnomalyJobSummary | null): Phase {
  if (!job) return "configure";
  if (ACTIVE_STATUSES.has(job.status)) return "progress";
  if (job.status === "completed") return "results";
  return job.status === "cancelled" ? "cancelled" : "failed";
}

const AGREEMENT_LABELS: Record<AnomalyAgreement, string> = {
  both: "Confirmée par les 2 méthodes",
  isolation_forest_only: "Isolation Forest seul",
  lof_only: "LOF seul",
  none: "Non classée comme anomalie",
};

// Sous ce seuil, un écart-type n'est pas une "contribution" lisible — le
// backend renvoie toujours les 5 variables au z-score le plus élevé en
// valeur absolue (services/anomaly_training.py::_build_numeric_deviations),
// même quand aucune n'est réellement notable (ex. un rang bas, une
// observation par ailleurs banale). Affichage uniquement — ne change rien
// au calcul ni au classement.
const NOT_SIGNIFICANT_Z_THRESHOLD = 0.5;

const AGREEMENT_VARIANTS: Record<AnomalyAgreement, "danger" | "warning" | "neutral"> = {
  both: "danger",
  isolation_forest_only: "warning",
  lof_only: "warning",
  none: "neutral",
};

/** Pilier ML non supervisé — détection d'anomalies (Lot 14). Même
 * architecture que Clustering.tsx/DimensionalityReduction.tsx. Isolation
 * Forest et LOF tournent toujours ensemble (jamais de choix d'algorithme,
 * voir services/anomaly_registry.py côté backend) — le seul réglage exposé
 * est le nombre d'observations à classer. */
export default function AnomalyDetection() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [datasetsError, setDatasetsError] = useState<string | null>(null);
  const [activeJob, setActiveJob] = useState<AnomalyJobSummary | null>(null);
  const [restoringJob, setRestoringJob] = useState(true);
  const confirmDelete = useConfirmAction<true>();

  // Reprise d'une détection — priorité au deep-link `?job=` (ex. depuis la
  // page Historique du pilier non supervisé), sinon la session en cours.
  useEffect(() => {
    const queryJobId = searchParams.get("job");
    const storedId = queryJobId ?? sessionStorage.getItem(ACTIVE_JOB_STORAGE_KEY);
    if (!storedId) {
      setRestoringJob(false);
      return;
    }
    api.anomalies
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

  function openJob(job: AnomalyJobSummary) {
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
    phase === "progress" && activeJob ? `/anomalies/jobs/${activeJob.id}/events` : null,
    (snapshot) => setActiveJob((prev) => (prev ? { ...prev, ...snapshot } : prev)),
  );

  function resetToConfigure() {
    setActiveJob(null);
    setSearchParams({}, { replace: false });
  }

  async function handleDeleteActiveJob() {
    if (!activeJob) return;
    try {
      await api.anomalies.remove(activeJob.id);
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
      setActiveJob(await api.anomalies.cancel(activeJob.id));
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
      openJob(await api.anomalies.rerun(activeJob.id));
    } catch (err) {
      setRerunError(err instanceof ApiError ? err.message : "Impossible de relancer cette détection");
    } finally {
      setRerunning(false);
    }
  }

  const titles: Record<Phase, string> = {
    configure: "Repérer les observations atypiques",
    progress: "Détection en cours",
    results: "Observations atypiques détectées",
    failed: "Échec de la détection",
    cancelled: "Détection annulée",
  };

  return (
    <AppShell pillarId="unsupervised">
      <PageHeader
        eyebrow="ML non supervisé"
        title={titles[phase]}
        description={
          phase === "configure"
            ? "Repérez les observations qui s'écartent le plus du reste de vos données — deux méthodes complémentaires (Isolation Forest, LOF) sont toujours comparées ensemble."
            : undefined
        }
        icon={AlertTriangle}
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
                    aria-label={confirmDelete.isPending(true) ? "Confirmer la suppression" : "Supprimer cette détection"}
                    title={confirmDelete.isPending(true) ? "Cliquer à nouveau pour confirmer" : "Supprimer cette détection"}
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
                Nouvelle détection
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
          <AnomalyForm
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
            {cancelling ? "Annulation…" : "Annuler cette détection"}
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
            <p className="text-sm">{activeJob.error_message ?? "La détection a échoué."}</p>
          </div>
        </Card>
      ) : phase === "results" && activeJob ? (
        <AnomalyResultView jobId={activeJob.id} featureColumns={activeJob.feature_columns} datasetName={activeJob.dataset_name} />
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

function AnomalyForm({
  datasets,
  datasetsError,
  onJobCreated,
  initialDatasetId,
  initialFeatures,
}: {
  datasets: DatasetSummary[];
  datasetsError: string | null;
  onJobCreated: (job: AnomalyJobSummary) => void;
  initialDatasetId: string | null;
  initialFeatures: string | null;
}) {
  const [datasetId, setDatasetId] = useState<number | "">("");
  const [columns, setColumns] = useState<ColumnSchema[]>([]);
  const [selectedFeatures, setSelectedFeatures] = useState<Set<string>>(new Set());
  const [topN, setTopN] = useState(DEFAULT_TOP_N);
  // `null` = réglage automatique (comportement historique, formule du papier
  // original de chaque algorithme) — même principe que le mode guidé du
  // reste du produit : pas de paramètre à régler par défaut, réglable
  // explicitement seulement si l'utilisateur active le mode manuel.
  const [contaminationPct, setContaminationPct] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [prefillApplied, setPrefillApplied] = useState(false);
  // Idempotence (Phase 2, AUDIT_BACKEND_2026-08-23.md §F4).
  const idempotencyKey = useIdempotencyKey();

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

  // Pré-remplissage depuis un lien croisé (résultat de clustering) — une
  // seule fois, dès que la liste de datasets est chargée.
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
      const job = await api.anomalies.createJob(
        {
          dataset_id: datasetId,
          feature_columns: Array.from(selectedFeatures),
          top_n: topN,
          contamination: contaminationPct !== null ? contaminationPct / 100 : undefined,
        },
        idempotencyKey.current,
      );
      idempotencyKey.reset();
      onJobCreated(job);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Impossible de lancer la détection");
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
          <label htmlFor="anomaly-dataset" className="block text-sm text-muted-foreground mb-1">
            Jeu de données
          </label>
          <Select id="anomaly-dataset" value={datasetId} onChange={(e) => handleDatasetChange(e.target.value)} required>
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

        <div>
          <label htmlFor="anomaly-top-n" className="block text-sm text-muted-foreground mb-1">
            Nombre d'observations à classer
          </label>
          <Input
            id="anomaly-top-n"
            type="number"
            min={1}
            max={MAX_TOP_N}
            value={topN}
            onChange={(e) => setTopN(Math.min(MAX_TOP_N, Math.max(1, Number(e.target.value) || 1)))}
          />
        </div>

        <div>
          <label className="flex items-center gap-2 text-sm text-muted-foreground mb-1.5">
            <input
              type="checkbox"
              checked={contaminationPct !== null}
              onChange={(e) => setContaminationPct(e.target.checked ? DEFAULT_CONTAMINATION_PCT : null)}
              className="accent-primary"
            />
            Régler moi-même la proportion attendue d'anomalies
          </label>
          {contaminationPct !== null && (
            <>
              <Input
                id="anomaly-contamination"
                type="number"
                min={1}
                max={50}
                step={1}
                value={contaminationPct}
                onChange={(e) => setContaminationPct(Math.min(50, Math.max(1, Number(e.target.value) || 1)))}
              />
              <p className="text-xs text-muted-foreground mt-1">
                % de vos données attendu comme atypique — par défaut, ce seuil est déduit automatiquement de vos
                données (recommandé si vous ne savez pas à l'avance quelle proportion est concernée).
              </p>
            </>
          )}
        </div>

        {error && (
          <div className="flex items-center gap-2 text-sm text-destructive bg-destructive/10 border border-destructive/20 rounded-lg px-3 py-2">
            <AlertCircle size={15} className="flex-shrink-0" />
            {error}
          </div>
        )}

        <Button type="submit" disabled={!datasetId || selectedFeatures.size === 0 || isSubmitting} className="w-full">
          {isSubmitting ? "Lancement…" : "Lancer la détection"}
        </Button>
      </form>
    </Card>
  );
}

/** Compte, à partir de l'histogramme RÉEL déjà chargé (`score_histogram`),
 * combien d'observations ont un score de consensus au moins égal à
 * `threshold` — approximation par bin entier (un bin n'est compté que si sa
 * borne basse atteint le seuil), jamais une fausse précision décimale que
 * les données binned ne permettent pas. Purement exploratoire : le vrai
 * seuil de décision (`agreement`/`is_anomaly_*`) résulte d'un ET entre deux
 * modèles distincts (Isolation Forest, LOF), pas d'une simple coupure sur ce
 * score continu — ce curseur aide à explorer LA DISTRIBUTION, il ne
 * recalcule jamais la décision réelle des deux modèles. */
function countAtOrAboveThreshold(histogram: AnomalyResult["score_histogram"], threshold: number): number {
  let count = 0;
  for (let i = 0; i < histogram.counts.length; i++) {
    if (histogram.bin_edges[i] >= threshold) count += histogram.counts[i];
  }
  return count;
}

/** "Où placer le curseur" (Lot 7, Anomalies.html) — construit uniquement à
 * partir de `score_histogram`, déjà chargé pour le graphe ci-dessus mais
 * jusqu'ici jamais exploité au-delà d'un histogramme statique. Aucun appel
 * réseau supplémentaire : tout se recalcule en direct côté client au
 * déplacement du curseur. */
function ThresholdExplorer({ result }: { result: AnomalyResult }) {
  const { bin_edges } = result.score_histogram;
  const min = bin_edges[0] ?? 0;
  const max = bin_edges[bin_edges.length - 1] ?? 1;
  const [threshold, setThreshold] = useState(min);

  const count = useMemo(
    () => countAtOrAboveThreshold(result.score_histogram, threshold),
    [result.score_histogram, threshold],
  );
  const pct = result.n_samples_used > 0 ? (count / result.n_samples_used) * 100 : 0;

  return (
    <Card className="p-5">
      <SectionHeader
        icon={SlidersHorizontal}
        color="blue"
        label="Où placer le curseur"
        help="Déplace un seuil exploratoire sur la distribution réelle des scores de consensus déjà calculée — pas le seuil de décision effectif du modèle (qui résulte d'un accord entre Isolation Forest ET LOF, pas d'une simple coupure sur ce score), un outil pour comprendre la distribution avant de régler la proportion attendue d'anomalies à l'étape suivante."
      />
      <input
        type="range"
        min={min}
        max={max}
        step={(max - min) / 100 || 0.01}
        value={threshold}
        onChange={(e) => setThreshold(Number(e.target.value))}
        className="w-full accent-primary"
        aria-label="Seuil exploratoire de score de consensus"
      />
      <div className="flex items-center justify-between mt-2">
        <span className="text-xs text-muted-foreground">
          Seuil : <span className="tabular-nums text-foreground">{threshold.toFixed(2)}</span>
        </span>
        <span className="text-xs text-muted-foreground">
          <span className="tabular-nums font-medium text-foreground">{count}</span> ligne{count > 1 ? "s" : ""} au-dessus
          {" · "}
          <span className="tabular-nums">{pct.toFixed(1)} %</span>
        </span>
      </div>
      <p className="text-xs text-muted-foreground mt-3 pt-3 border-t border-border/60">
        À titre de repère : la décision retenue (accord Isolation Forest + LOF) signale actuellement{" "}
        <span className="tabular-nums text-foreground">{result.n_anomalies_consensus}</span> ligne
        {result.n_anomalies_consensus > 1 ? "s" : ""} (
        <span className="tabular-nums">{(result.anomaly_rate_consensus * 100).toFixed(1)} %</span>).
      </p>
    </Card>
  );
}

function topDeviation(obs: AnomalyObservation): string | null {
  const entries = Object.entries(obs.numeric_deviations);
  if (entries.length === 0) return null;
  const [name, stat] = entries[0];
  if (Math.abs(stat.z_score) < NOT_SIGNIFICANT_Z_THRESHOLD) return null;
  return `${name} (${stat.z_score > 0 ? "+" : ""}${stat.z_score.toFixed(1)}σ)`;
}

function observationColumns(onOpenDetail: (obs: AnomalyObservation) => void): TableColumn<AnomalyObservation>[] {
  return [
    { key: "rank", header: "#", align: "right" },
    { key: "row_index", header: "Ligne", align: "right", render: (o) => o.row_index + 1 },
    {
      key: "consensus_score",
      header: "Score d'anomalie",
      align: "right",
      render: (o) => o.consensus_score.toFixed(3),
    },
    {
      key: "agreement",
      header: "Décision",
      render: (o) => <Badge variant={AGREEMENT_VARIANTS[o.agreement]}>{AGREEMENT_LABELS[o.agreement]}</Badge>,
    },
    {
      key: "deviation",
      header: "Variable la plus explicative",
      render: (o) => topDeviation(o) ?? "Aucune contribution significative",
      className: "text-muted-foreground",
    },
    {
      key: "detail",
      header: "",
      align: "right",
      render: (o) => (
        <button type="button" onClick={() => onOpenDetail(o)} className="text-primary hover:text-primary/80 text-xs font-medium">
          Détails
        </button>
      ),
    },
  ];
}

function AnomalyResultView({
  jobId,
  featureColumns,
  datasetName,
}: {
  jobId: number;
  featureColumns: string[];
  datasetName: string | null;
}) {
  const [result, setResult] = useState<AnomalyResult | null>(null);
  const [observations, setObservations] = useState<AnomalyObservation[]>([]);
  const [observationsError, setObservationsError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [detailObservation, setDetailObservation] = useState<AnomalyObservation | null>(null);
  const [activeTab, setActiveTab] = useState<"distribution" | "observations" | "noter">("observations");

  useEffect(() => {
    api.anomalies
      .getResult(jobId)
      .then(setResult)
      .catch((err) => setError(err instanceof ApiError ? err.message : "Résultat indisponible"));
    api.anomalies
      .getObservations(jobId)
      .then(setObservations)
      .catch((err) => setObservationsError(err instanceof ApiError ? err.message : "Impossible de charger le classement des observations"));
  }, [jobId]);

  if (error) return <p className="text-sm text-destructive text-center">{error}</p>;
  if (!result) return <p className="text-sm text-muted-foreground text-center">Chargement…</p>;

  // Comptes entiers, pas les taux (évite toute ambiguïté d'arrondi flottant
  // sur un "0 %" qui ne serait pas vraiment zéro) — vrai seulement quand
  // aucune des deux méthodes n'a rien flaggé du tout.
  const noAnomaliesDetected = result.n_anomalies_isolation_forest === 0 && result.n_anomalies_lof === 0;

  const histogramData = result.score_histogram.counts.map((count, i) => ({
    range: `${result.score_histogram.bin_edges[i].toFixed(2)}–${result.score_histogram.bin_edges[i + 1].toFixed(2)}`,
    count,
  }));

  const quality = assessConsensusQuality(result.anomaly_rate_consensus);

  // Fiche modèle (retour utilisateur direct : "on peut télécharger le
  // modèle mais pas un json... qui suit le modèle") — construite
  // ENTIÈREMENT à partir de `result`, déjà en mémoire, jamais un second
  // appel réseau. Voir `utils/anomalyModelCard.ts`.
  function handleExportModelCard() {
    // Garde défensive pure — ce bouton n'est rendu que dans la branche où
    // `result` est déjà non nul (voir plus bas), mais `result` reste
    // `AnomalyResult | null` dans la fermeture de cette fonction (déclarée
    // avant le rétrécissement de type par le rendu conditionnel).
    if (!result) return;
    const card = buildAnomalyModelCard(featureColumns, datasetName, result);
    const blob = new Blob([JSON.stringify(card, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `anomalies_fiche_modele_job${jobId}.json`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="max-w-4xl mx-auto space-y-5">
      <Card className={`p-5 ${accentSurfaceClass("amber")}`}>
        <SectionHeader
          icon={AlertTriangle}
          color="amber"
          label="Taux d'observations atypiques"
          help="Isolation Forest et LOF sont deux méthodes complémentaires évaluées systématiquement ensemble — le consensus (les deux d'accord) est le signal le plus fiable, jamais un seul essai lancé à l'aveugle."
        />
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          <MetricTile label="Isolation Forest" value={`${(result.anomaly_rate_isolation_forest * 100).toFixed(1)} %`} color="blue" />
          <MetricTile label="LOF" value={`${(result.anomaly_rate_lof * 100).toFixed(1)} %`} color="teal" />
          <MetricTile label="Consensus (les 2 méthodes)" value={`${(result.anomaly_rate_consensus * 100).toFixed(1)} %`} color="amber" />
        </div>
        <div className={`rounded-xl border p-3 mt-4 ${accentSurfaceClass(QUALITY_TONE_ACCENT[quality.tone])}`}>
          <span className="inline-flex items-center text-overline px-2 py-0.5 rounded-full bg-card/80 text-foreground">
            {quality.label}
          </span>
          <p className="text-xs text-foreground/70 mt-1.5">{quality.caveat}</p>
        </div>
        {result.sampled && (
          <p className="text-xs text-muted-foreground mt-3">
            Calculé sur un échantillon de {result.n_samples_used} observations sur {result.n_samples_total} au total
            (dataset volumineux — échantillonnage déterministe, reproductible).
          </p>
        )}
        <p className="text-xs text-muted-foreground mt-3">
          {result.model_card.contamination === "auto"
            ? "Proportion attendue d'anomalies déduite automatiquement de vos données."
            : `Proportion attendue d'anomalies réglée manuellement à ${(Number(result.model_card.contamination) * 100).toFixed(0)} %.`}
        </p>
      </Card>

      <div className="flex items-center gap-2 flex-wrap">
        <ModelExportActions
          onExportArtifact={() => api.anomalies.exportModel(jobId)}
          exportConfig={{
            feature_columns: featureColumns,
            n_anomalies_consensus: result.n_anomalies_consensus,
            anomaly_rate_consensus: result.anomaly_rate_consensus,
            model_card: result.model_card,
          }}
          configFilename={`anomalies_config_job${jobId}.json`}
        />
        <Button variant="secondary" size="sm" onClick={() => api.anomalies.exportScores(jobId)}>
          <Download size={14} />
          Exporter les scores (CSV, toutes les lignes)
        </Button>
        <Button variant="secondary" size="sm" onClick={handleExportModelCard}>
          <FileJson size={14} />
          Fiche modèle (JSON)
        </Button>
        <Button variant="secondary" size="sm" onClick={() => api.anomalies.exportDeploymentScript(jobId)}>
          <FileCode size={14} />
          Script de déploiement (.py)
        </Button>
      </div>
      <p className="text-xs text-muted-foreground -mt-2">
        Pour déployer ce modèle en dehors de DataLab Pro : téléchargez l'artefact ET le script de déploiement,
        placez-les dans le même dossier — le script recharge l'artefact et note de nouvelles observations, sans
        dépendre de cette plateforme (voir l'en-tête du script pour l'installation des bibliothèques nécessaires).
      </p>

      <Tabs
        items={[
          { id: "observations" as const, label: "Observations", icon: Search },
          { id: "distribution" as const, label: "Distribution des scores", icon: BarChart3 },
          { id: "noter" as const, label: "Noter une observation", icon: Target },
        ]}
        active={activeTab}
        onChange={setActiveTab}
        urlParam="onglet"
      />

      {activeTab === "noter" && <AnomalyScoreForm jobId={jobId} featureColumns={featureColumns} />}

      {activeTab === "distribution" && (
        <>
          <Card className="p-5">
            <SectionHeader
              icon={BarChart3}
              color="blue"
              label="Distribution des scores de consensus"
              help="Score continu de 0 à 1 (moyenne des rangs Isolation Forest/LOF) — plus un score est élevé, plus l'observation est atypique par rapport au reste du jeu de données."
            />
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={histogramData} margin={{ top: 8, right: 8, bottom: 40, left: 0 }}>
                <CartesianGrid stroke={CHART_GRID_STROKE} vertical={false} />
                <XAxis
                  dataKey="range"
                  tick={CHART_TICK_STYLE_SM}
                  interval={3}
                  angle={-40}
                  textAnchor="end"
                  height={55}
                />
                <YAxis tick={CHART_TICK_STYLE_SM} allowDecimals={false} />
                <RechartsTooltip {...CHART_TOOLTIP_STYLE} />
                <Bar dataKey="count" fill={CHART_COLOR_PRIMARY} radius={[3, 3, 0, 0]} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <ThresholdExplorer result={result} />
        </>
      )}

      {activeTab === "observations" && (
      <div>
        <SectionHeader
          icon={Search}
          color="rose"
          label={
            noAnomaliesDetected
              ? `Observations aux scores d'anomalie les plus élevés (${observations.length})`
              : `Observations les plus atypiques (${observations.length})`
          }
          help="Classées par score de consensus décroissant — 'Confirmée par les 2 méthodes' est le signal le plus fiable."
        />
        {noAnomaliesDetected && (
          <div className="flex items-start gap-3 text-success bg-success/10 border border-success/20 rounded-lg p-4 mb-4">
            <CheckCircle2 size={18} className="flex-shrink-0 mt-0.5" />
            <p className="text-sm">
              Aucune anomalie détectée selon les seuils actuels. Le classement ci-dessous reste affiché à titre
              indicatif — ce sont les observations les moins "typiques" du lot, pas des anomalies confirmées par
              Isolation Forest ou LOF.
            </p>
          </div>
        )}
        {observationsError ? (
          <p className="text-sm text-destructive text-center">{observationsError}</p>
        ) : (
          <Table
            columns={observationColumns(setDetailObservation)}
            rows={observations}
            rowKey={(o) => o.row_index}
            highlightRow={(o) => o.agreement === "both"}
            pageSize={20}
          />
        )}
      </div>
      )}

      {detailObservation && (
        <Modal title={`Ligne ${detailObservation.row_index + 1} — détail`} onClose={() => setDetailObservation(null)}>
          <div className="space-y-4">
            <div>
              <p className="text-xs uppercase tracking-wide text-muted-foreground mb-2">Variables numériques les plus déviantes</p>
              {Object.entries(detailObservation.numeric_deviations).length === 0 ? (
                <p className="text-sm text-muted-foreground">Aucune variable numérique.</p>
              ) : (
                <div className="space-y-1.5">
                  {Object.entries(detailObservation.numeric_deviations).map(([col, stat]) => (
                    <div key={col} className="flex items-center justify-between text-sm">
                      <span className="text-foreground/90">{col}</span>
                      <span className="text-muted-foreground tabular-nums">
                        {stat.value.toFixed(2)} ({stat.z_score > 0 ? "+" : ""}
                        {stat.z_score.toFixed(1)}σ)
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
            {Object.entries(detailObservation.categorical_flags).length > 0 && (
              <div>
                <p className="text-xs uppercase tracking-wide text-muted-foreground mb-2">Valeurs catégorielles rares</p>
                <div className="space-y-1.5">
                  {Object.entries(detailObservation.categorical_flags).map(([col, flag]) => (
                    <div key={col} className="flex items-center justify-between text-sm">
                      <span className="text-foreground/90">{col}</span>
                      <span className="text-muted-foreground">
                        {flag.value} ({flag.population_pct.toFixed(1)} % de la population)
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </Modal>
      )}
    </div>
  );
}

/** Note une NOUVELLE observation (Lot 6B, §F.2 — jusqu'ici, comme le
 * clustering avant lui, une détection entraînée ne pouvait jamais être
 * réutilisée sur une nouvelle observation) — même pattern que
 * `Clustering.tsx::ClusterAssignmentForm`. */
function AnomalyScoreForm({ jobId, featureColumns }: { jobId: number; featureColumns: string[] }) {
  const [values, setValues] = useState<Record<string, string>>({});
  const [score, setScore] = useState<AnomalyScore | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setScore(null);
    setIsSubmitting(true);
    try {
      const result = await api.anomalies.predict(jobId, values);
      setScore(result);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Impossible de noter cette observation");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <Card className="p-5">
      <SectionHeader
        icon={Target}
        color="amber"
        label="Noter une nouvelle observation"
        help="Indiquez les valeurs d'une nouvelle observation pour voir si elle ressort comme atypique par rapport aux données déjà analysées, sans relancer une détection complète."
      />
      <form onSubmit={handleSubmit} className="space-y-3">
        <div className="grid sm:grid-cols-2 gap-3">
          {featureColumns.map((col) => (
            <div key={col}>
              <label htmlFor={`score-${col}`} className="block text-xs text-muted-foreground mb-1">
                {col}
              </label>
              <Input
                id={`score-${col}`}
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
          {isSubmitting ? "Notation…" : "Noter"}
        </Button>
      </form>
      {score && (
        <div className={`rounded-lg border p-3 mt-3 ${accentSurfaceClass(score.is_anomaly_consensus ? "amber" : "teal")}`}>
          <p className="text-sm text-foreground">
            {score.is_anomaly_consensus
              ? "Observation atypique (consensus des deux méthodes)."
              : "Observation dans la norme."}
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            Score de consensus : {score.consensus_score.toFixed(2)} — {AGREEMENT_LABELS[score.agreement]}
          </p>
        </div>
      )}
    </Card>
  );
}

function MetricTile({ label, value, color }: { label: string; value: string; color: AccentColor }) {
  return (
    <div className={`rounded-xl border px-4 py-3 ${accentSurfaceClass(color)}`}>
      <p className="text-xs text-muted-foreground mb-1">{label}</p>
      <p className={`text-xl font-bold tabular-nums ${accentValueTextClass(color)}`}>{value}</p>
    </div>
  );
}
