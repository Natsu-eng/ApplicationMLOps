import { useEffect, useState, type FormEvent } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { AlertCircle, AlertTriangle, Loader2, PlayCircle, Sparkles, Target, Trash2 } from "lucide-react";
import { Line, LineChart, CartesianGrid, ResponsiveContainer, Tooltip as RechartsTooltip, XAxis, YAxis } from "recharts";
import {
  ApiError,
  api,
  type VisionAnomalyExample,
  type VisionAnomalyJobSummary,
  type VisionAnomalyModelOption,
  type VisionAnomalyResult,
} from "../api/client";
import AppShell from "../components/AppShell";
import { pillarColor } from "../config/pillars";
import { Badge } from "../components/ui/Badge";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { accentSurfaceClass, accentValueTextClass, type AccentColor } from "../components/ui/ColorIconBadge";
import { Heatmap } from "../components/ui/Heatmap";
import { Input } from "../components/ui/Input";
import { PageHeader } from "../components/ui/PageHeader";
import { SectionHeader } from "../components/ui/SectionHeader";
import { Select } from "../components/ui/Select";
import { VisionDatasetPicker } from "../components/vision/VisionDatasetPicker";
import { VisionImage } from "../components/vision/VisionImage";
import { useConfirmAction } from "../hooks/useConfirmAction";
import { CHART_GRID_STROKE, CHART_SERIES_COLORS, CHART_TICK_STYLE_SM, CHART_TOOLTIP_STYLE } from "../theme/charts";

const ACTIVE_STATUSES = new Set(["queued", "running"]);
const POLL_INTERVAL_MS = 3000;
const ACTIVE_JOB_STORAGE_KEY = "datalab_active_vision_anomaly_job_id";

type Phase = "configure" | "progress" | "results" | "failed";

function phaseOf(job: VisionAnomalyJobSummary | null): Phase {
  if (!job) return "configure";
  if (ACTIVE_STATUSES.has(job.status)) return "progress";
  return job.status === "completed" ? "results" : "failed";
}

/** Pilier Vision — détection d'anomalies visuelles (structure normal/défaut,
 * sous-lot C). Même architecture que VisionClassification.tsx/AnomalyDetection.tsx. */
export default function VisionAnomalies() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [activeJob, setActiveJob] = useState<VisionAnomalyJobSummary | null>(null);
  const [activeDatasetId, setActiveDatasetId] = useState<number | null>(null);
  const [restoringJob, setRestoringJob] = useState(true);
  const confirmDelete = useConfirmAction<true>();

  useEffect(() => {
    const queryJobId = searchParams.get("job");
    const storedId = queryJobId ?? sessionStorage.getItem(ACTIVE_JOB_STORAGE_KEY);
    if (!storedId) {
      setRestoringJob(false);
      return;
    }
    api.visionAnomalies
      .getJob(Number(storedId))
      .then((job) => {
        setActiveJob(job);
        setActiveDatasetId(job.vision_dataset_id);
      })
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

  function openJob(job: VisionAnomalyJobSummary) {
    setActiveJob(job);
    setActiveDatasetId(job.vision_dataset_id);
    setSearchParams({ job: String(job.id) }, { replace: false });
  }

  const phase = phaseOf(activeJob);

  useEffect(() => {
    if (phase !== "progress" || !activeJob) return;
    const interval = setInterval(async () => {
      try {
        setActiveJob(await api.visionAnomalies.getJob(activeJob.id));
      } catch {
        // silencieux — nouvelle tentative au prochain tick
      }
    }, POLL_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [phase, activeJob]);

  function resetToConfigure() {
    setActiveJob(null);
    setActiveDatasetId(null);
    setSearchParams({}, { replace: false });
  }

  async function handleDeleteActiveJob() {
    if (!activeJob) return;
    try {
      await api.visionAnomalies.remove(activeJob.id);
    } catch {
      // best-effort — on repart en configuration dans tous les cas
    }
    resetToConfigure();
  }

  const titles: Record<Phase, string> = {
    configure: "Détecter des défauts visuels",
    progress: "Entraînement en cours",
    results: "Défauts détectés",
    failed: "Échec de l'entraînement",
  };

  return (
    <AppShell pillarId="vision">
      <PageHeader
        eyebrow="Vision"
        title={titles[phase]}
        description={
          phase === "configure"
            ? "Entraînez un modèle à reconnaître des pièces normales à partir de photos sans défaut, pour repérer automatiquement celles qui s'en écartent (structure normal/défaut)."
            : undefined
        }
        icon={AlertTriangle}
        color={pillarColor("vision")}
        action={
          phase !== "configure" ? (
            <div className="flex items-center gap-2">
              {(phase === "results" || phase === "failed") && (
                <button
                  type="button"
                  onClick={() => confirmDelete.trigger(true, handleDeleteActiveJob)}
                  onMouseLeave={confirmDelete.reset}
                  aria-label={confirmDelete.isPending(true) ? "Confirmer la suppression" : "Supprimer cet entraînement"}
                  title={confirmDelete.isPending(true) ? "Cliquer à nouveau pour confirmer" : "Supprimer cet entraînement"}
                  className={`p-2 rounded-lg transition-colors ${
                    confirmDelete.isPending(true)
                      ? "text-destructive bg-destructive/15"
                      : "text-muted-foreground hover:text-destructive hover:bg-destructive/10"
                  }`}
                >
                  <Trash2 size={16} />
                </button>
              )}
              <Button variant="secondary" size="sm" onClick={resetToConfigure}>
                <PlayCircle size={14} />
                Nouvel entraînement
              </Button>
            </div>
          ) : undefined
        }
      />

      {restoringJob ? (
        <div className="flex items-center justify-center py-16 text-sm text-muted-foreground gap-2">
          <Loader2 size={16} className="animate-spin" />
          Reprise de votre session…
        </div>
      ) : phase === "configure" ? (
        <div className="max-w-2xl mx-auto">
          <AnomalyVisionForm onJobCreated={openJob} />
        </div>
      ) : phase === "progress" && activeJob ? (
        <Card className="max-w-2xl mx-auto p-8 text-center">
          <Loader2 size={28} className="animate-spin mx-auto mb-4 text-primary" />
          <p className="text-sm text-foreground mb-1">{activeJob.progress_step ?? "Préparation…"}</p>
          <div className="w-full h-1.5 rounded-full bg-muted overflow-hidden mt-4">
            <div
              className="h-full rounded-full bg-primary transition-all duration-500"
              style={{ width: `${activeJob.progress_percent}%` }}
            />
          </div>
        </Card>
      ) : phase === "failed" && activeJob ? (
        <Card className="max-w-2xl mx-auto p-6">
          <div className="flex items-start gap-3 text-destructive bg-destructive/10 border border-destructive/20 rounded-lg p-4">
            <AlertCircle size={18} className="flex-shrink-0 mt-0.5" />
            <p className="text-sm">{activeJob.error_message ?? "L'entraînement a échoué."}</p>
          </div>
        </Card>
      ) : phase === "results" && activeJob && activeDatasetId ? (
        <AnomalyVisionResultView jobId={activeJob.id} datasetId={activeDatasetId} />
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

function AnomalyVisionForm({ onJobCreated }: { onJobCreated: (job: VisionAnomalyJobSummary) => void }) {
  const [datasetId, setDatasetId] = useState<number | "">("");
  const [models, setModels] = useState<VisionAnomalyModelOption[]>([]);
  const [modelId, setModelId] = useState("");
  const [numEpochs, setNumEpochs] = useState(15);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    api.visionAnomalies.models().then((list) => {
      setModels(list);
      if (list.length > 0) setModelId(list[0].id);
    });
  }, []);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (!datasetId) return;
    setError(null);
    setIsSubmitting(true);
    try {
      const job = await api.visionAnomalies.createJob({
        vision_dataset_id: datasetId,
        model_id: modelId,
        num_epochs: numEpochs,
      });
      onJobCreated(job);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Impossible de lancer l'entraînement");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <Card className="p-5">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm text-muted-foreground mb-1.5">Dataset d'images (structure normal/défaut)</label>
          <VisionDatasetPicker structureType="mvtec_ad" value={datasetId} onChange={(id) => setDatasetId(id)} />
        </div>

        {models.length > 0 && (
          <div>
            <label htmlFor="va-model" className="block text-sm text-muted-foreground mb-1">
              Modèle
            </label>
            <Select id="va-model" value={modelId} onChange={(e) => setModelId(e.target.value)}>
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </Select>
          </div>
        )}

        <div>
          <label htmlFor="va-epochs" className="block text-sm text-muted-foreground mb-1">
            Nombre d'époques
          </label>
          <Input
            id="va-epochs"
            type="number"
            min={1}
            max={50}
            value={numEpochs}
            onChange={(e) => setNumEpochs(Math.min(50, Math.max(1, Number(e.target.value) || 1)))}
          />
        </div>

        {error && (
          <div className="flex items-center gap-2 text-sm text-destructive bg-destructive/10 border border-destructive/20 rounded-lg px-3 py-2">
            <AlertCircle size={15} className="flex-shrink-0" />
            {error}
          </div>
        )}

        <Button type="submit" disabled={!datasetId || isSubmitting} className="w-full">
          {isSubmitting ? "Lancement…" : "Lancer l'entraînement"}
        </Button>
      </form>
    </Card>
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

function AnomalyVisionResultView({ jobId, datasetId }: { jobId: number; datasetId: number }) {
  const [result, setResult] = useState<VisionAnomalyResult | null>(null);
  const [examples, setExamples] = useState<VisionAnomalyExample[]>([]);
  const [examplesError, setExamplesError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.visionAnomalies
      .getResult(jobId)
      .then(setResult)
      .catch((err) => setError(err instanceof ApiError ? err.message : "Résultat indisponible"));
    api.visionAnomalies
      .getExamples(jobId)
      .then(setExamples)
      .catch((err) => setExamplesError(err instanceof ApiError ? err.message : "Exemples indisponibles"));
  }, [jobId]);

  if (error) return <p className="text-sm text-destructive text-center">{error}</p>;
  if (!result) return <p className="text-sm text-muted-foreground text-center">Chargement…</p>;

  const historyData = result.history.map((h) => ({
    epoch: h.epoch + 1,
    "Perte (train)": h.train_loss,
    "Perte (validation)": h.val_loss,
  }));

  // Lot 0.2 (correctif C2, AUDIT_DATALAB_2026-08-16.md) — absent (undefined)
  // sur les modèles entraînés avant ce correctif : pas de bannière dans ce
  // cas, rétrocompatibilité par absence plutôt qu'une alerte inventée.
  const calibrationStatus = result.model_card.threshold_calibration_status;
  const calibrationMessage = result.model_card.threshold_calibration_message;
  const calibrationBiased = calibrationStatus === "degraded" && typeof calibrationMessage === "string";

  return (
    <div className="max-w-4xl mx-auto space-y-5">
      {calibrationBiased && (
        <div className="rounded-lg border border-warning/20 bg-warning/10 p-3 flex items-start gap-2">
          <AlertTriangle size={16} className="flex-shrink-0 mt-0.5 text-warning" />
          <p className="text-sm text-warning">{calibrationMessage}</p>
        </div>
      )}

      <Card className={`p-5 ${accentSurfaceClass("amber")}`}>
        <SectionHeader
          icon={Target}
          color="amber"
          label="Performance de détection"
          help="Le seuil de détection est calibré automatiquement (J de Youden) sur une partie des images de test (calibration), puis les métriques ci-dessous sont calculées sur l'autre partie (évaluation) — jamais sur les mêmes images que celles ayant servi à choisir le seuil."
        />
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <MetricTile label="ROC-AUC" value={result.roc_auc.toFixed(3)} color="amber" />
          <MetricTile label="Exactitude" value={`${(result.test_accuracy * 100).toFixed(1)} %`} color="violet" />
          <MetricTile label="Précision" value={`${(result.test_precision * 100).toFixed(1)} %`} color="blue" />
          <MetricTile label="Rappel" value={`${(result.test_recall * 100).toFixed(1)} %`} color="teal" />
        </div>
        <p className="text-xs text-muted-foreground mt-3">
          {result.n_train} images normales d'entraînement · {result.n_val} de validation · {result.n_test} de test
          {result.n_calibration != null &&
            result.n_evaluation != null &&
            ` (${result.n_calibration} pour la calibration du seuil, ${result.n_evaluation} pour l'évaluation)`}
          {Array.isArray(result.model_card.defect_categories) &&
            ` — catégories de défaut : ${(result.model_card.defect_categories as string[]).join(", ")}`}
          .{Boolean(result.model_card.time_capped) && " Entraînement arrêté par le garde-fou de temps CPU."}
        </p>
      </Card>

      <Card className="p-5">
        <SectionHeader icon={Sparkles} color="blue" label="Courbe d'apprentissage (reconstruction)" />
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={historyData} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID_STROKE} vertical={false} />
            <XAxis dataKey="epoch" tick={CHART_TICK_STYLE_SM} label={{ value: "Époque", position: "insideBottom", offset: -2 }} />
            <YAxis tick={CHART_TICK_STYLE_SM} />
            <RechartsTooltip {...CHART_TOOLTIP_STYLE} />
            <Line type="monotone" dataKey="Perte (train)" stroke={CHART_SERIES_COLORS[0]} dot={false} isAnimationActive={false} />
            <Line type="monotone" dataKey="Perte (validation)" stroke={CHART_SERIES_COLORS[1]} dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      <Card className="p-5">
        <SectionHeader icon={AlertTriangle} color="rose" label="Matrice de confusion" help="0 = normal, 1 = défaut." />
        <Heatmap xLabels={["Normal", "Défaut"]} yLabels={["Normal", "Défaut"]} matrix={result.confusion_matrix} variant="sequential" />
      </Card>

      <div>
        <SectionHeader
          icon={AlertTriangle}
          color="amber"
          label={`Exemples les plus atypiques (${examples.length})`}
          help="Triés par score d'anomalie décroissant — la carte de chaleur et le masque indiquent où se situe l'écart par rapport aux pièces normales."
        />
        {examplesError ? (
          <p className="text-sm text-destructive text-center">{examplesError}</p>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2">
            {examples.map((example) => (
              <AnomalyExampleCard key={example.relative_path} example={example} datasetId={datasetId} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function AnomalyExampleCard({ example, datasetId }: { example: VisionAnomalyExample; datasetId: number }) {
  const isDefect = example.true_label === 1;
  const correctlyDetected = example.true_label === example.predicted_label;
  return (
    <Card className="p-3">
      <div className="flex items-center justify-between mb-2">
        <Badge variant={isDefect ? "warning" : "neutral"}>{isDefect ? example.defect_category : "Normal"}</Badge>
        <Badge variant={correctlyDetected ? "success" : "danger"}>
          {correctlyDetected ? "Détection correcte" : "Détection erronée"}
        </Badge>
      </div>
      <div className="grid grid-cols-3 gap-2">
        <div>
          <p className="text-caption text-muted-foreground mb-1">Image</p>
          <VisionImage
            datasetId={datasetId}
            path={example.relative_path}
            alt="image source"
            className="w-full aspect-square object-cover rounded-lg border border-border"
          />
        </div>
        <div>
          <p className="text-caption text-muted-foreground mb-1">Zones erronées (superposées)</p>
          <img
            src={example.heatmap_png}
            alt="Image avec carte d'erreur superposée"
            className="w-full aspect-square object-cover rounded-lg border border-border"
          />
        </div>
        <div>
          <p className="text-caption text-muted-foreground mb-1">Masque du défaut</p>
          <img src={example.mask_png} alt="masque binaire" className="w-full aspect-square object-cover rounded-lg border border-border" />
        </div>
      </div>
      <div className="flex items-center gap-2 mt-2">
        <span
          className="inline-block h-1.5 w-10 rounded-full flex-shrink-0"
          style={{ background: "linear-gradient(to right, #0000cc, #00cc66, #cc0000)" }}
          aria-hidden="true"
        />
        <p className="text-caption text-muted-foreground">Bleu = normal · Rouge = zone la plus atypique</p>
      </div>
      <p className="text-xs text-muted-foreground mt-2 tabular-nums">Score d'anomalie : {example.anomaly_score.toFixed(4)}</p>
    </Card>
  );
}
