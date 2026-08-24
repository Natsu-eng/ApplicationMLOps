import { useEffect, useState, type FormEvent } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { AlertCircle, AlertTriangle, Ban, ChevronLeft, ChevronRight, Loader2, PlayCircle, RotateCcw, Sparkles, Target, Trash2 } from "lucide-react";
import { Line, LineChart, CartesianGrid, ResponsiveContainer, Tooltip as RechartsTooltip, XAxis, YAxis } from "recharts";
import {
  ApiError,
  api,
  type AugmentationPreset,
  type VisionAnomalyExample,
  type VisionAnomalyJobSummary,
  type VisionAnomalyModelOption,
  type VisionAnomalyResult,
  type VisionDatasetDetail,
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
import { Tabs } from "../components/ui/Tabs";
import { ModelExportActions } from "../components/ui/ModelExportActions";
import { VisionDatasetPicker } from "../components/vision/VisionDatasetPicker";
import { useJobEvents } from "../hooks/useJobEvents";
import { VisionImage } from "../components/vision/VisionImage";
import {
  AUGMENTATION_PRESET_INFO,
  AugmentationPresetPicker,
  AugmentationPreviewGallery,
  Fact,
  SplitRatioControl,
  StepContent,
} from "../components/vision/VisionWizard";
import { WizardStepper } from "../components/ui/WizardStepper";
import { useConfirmAction } from "../hooks/useConfirmAction";
import { useIdempotencyKey } from "../hooks/useIdempotencyKey";
import { CHART_GRID_STROKE, CHART_SERIES_COLORS, CHART_TICK_STYLE_SM, CHART_TOOLTIP_STYLE } from "../theme/charts";

/** Étapes du wizard (Lot 6A) — parité avec VisionClassification.tsx : même
 * structure à 4 étapes, mêmes composants partagés (VisionWizard.tsx).
 * "Mode expert" ici n'a pas de case à cocher pour l'activer/désactiver
 * (contrairement à la classification) — les réglages d'anomalies sont déjà
 * peu nombreux, les regrouper sous une étape dédiée suffit à la clarté
 * sans ajouter un interrupteur qui masquerait/démasquerait 4 champs. */
const STEP_LABELS = [
  { number: 1, label: "Données & modèle" },
  { number: 2, label: "Augmentation" },
  { number: 3, label: "Mode expert" },
  { number: 4, label: "Lancement" },
];

const MODEL_HINTS: Record<string, string> = {
  conv_autoencoder: "Le plus rapide — bon point de départ pour la plupart des datasets.",
  denoising_autoencoder: "Ajoute du bruit pendant l'entraînement pour apprendre des traits plus robustes — utile si vos photos varient en netteté/éclairage.",
  conv_vae: "Régularise l'espace latent (variationnel) — peut mieux généraliser sur un petit jeu de données \"normales\", au prix d'un entraînement légèrement plus lent.",
};

const ACTIVE_STATUSES = new Set(["queued", "running"]);
const ACTIVE_JOB_STORAGE_KEY = "datalab_active_vision_anomaly_job_id";

type Phase = "configure" | "progress" | "results" | "failed" | "cancelled";

function phaseOf(job: VisionAnomalyJobSummary | null): Phase {
  if (!job) return "configure";
  if (ACTIVE_STATUSES.has(job.status)) return "progress";
  if (job.status === "completed") return "results";
  return job.status === "cancelled" ? "cancelled" : "failed";
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

  // Notifications SSE (Lot 7, §J.2) — remplace le polling setInterval.
  useJobEvents(
    phase === "progress" && activeJob ? `/vision/anomalies/jobs/${activeJob.id}/events` : null,
    (snapshot) => setActiveJob((prev) => (prev ? { ...prev, ...snapshot } : prev)),
  );

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

  const [cancelling, setCancelling] = useState(false);
  async function handleCancelActiveJob() {
    if (!activeJob) return;
    setCancelling(true);
    try {
      setActiveJob(await api.visionAnomalies.cancel(activeJob.id));
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
      openJob(await api.visionAnomalies.rerun(activeJob.id));
    } catch (err) {
      setRerunError(err instanceof ApiError ? err.message : "Impossible de relancer cet entraînement");
    } finally {
      setRerunning(false);
    }
  }

  const titles: Record<Phase, string> = {
    configure: "Détecter des défauts visuels",
    progress: "Entraînement en cours",
    results: "Défauts détectés",
    failed: "Échec de l'entraînement",
    cancelled: "Entraînement annulé",
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
              {(phase === "results" || phase === "failed" || phase === "cancelled") && (
                <>
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
                <Button variant="secondary" size="sm" onClick={handleRerunActiveJob} disabled={rerunning}>
                  <RotateCcw size={14} />
                  {rerunning ? "Relance…" : "Relancer"}
                </Button>
                </>
              )}
              <Button variant="secondary" size="sm" onClick={resetToConfigure}>
                <PlayCircle size={14} />
                Nouvel entraînement
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
        <div className="max-w-3xl mx-auto">
          <AnomalyVisionForm onJobCreated={openJob} />
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
            {cancelling ? "Annulation…" : "Annuler cet entraînement"}
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
  const [datasetDetail, setDatasetDetail] = useState<VisionDatasetDetail | null>(null);
  const [models, setModels] = useState<VisionAnomalyModelOption[]>([]);
  const [modelId, setModelId] = useState("");
  const [numEpochs, setNumEpochs] = useState(15);
  const [batchSize, setBatchSize] = useState(16);
  const [learningRate, setLearningRate] = useState(1e-3);
  const [maskPercentile, setMaskPercentile] = useState(0.97);
  const [augmentationPreset, setAugmentationPreset] = useState<AugmentationPreset>("aucune");
  // Part de train/good/ réservée à la validation (répartition, Lot 6A) —
  // pas de "test" : contrairement à la classification, test/ est un
  // dossier séparé du dataset (structure normal/défaut), jamais un split.
  const [valRatio, setValRatio] = useState(0.15);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  // Idempotence (Phase 2, AUDIT_BACKEND_2026-08-23.md §F4).
  const idempotencyKey = useIdempotencyKey();

  const [activeStep, setActiveStep] = useState(1);
  const [maxReachedStep, setMaxReachedStep] = useState(1);

  function goToStep(step: number) {
    if (step <= maxReachedStep) setActiveStep(step);
  }
  function goNext() {
    setActiveStep((s) => {
      const next = Math.min(s + 1, STEP_LABELS.length);
      setMaxReachedStep((m) => Math.max(m, next));
      return next;
    });
  }
  function goPrev() {
    setActiveStep((s) => Math.max(s - 1, 1));
  }

  useEffect(() => {
    api.visionAnomalies.models().then((list) => {
      setModels(list);
      if (list.length > 0) setModelId(list[0].id);
    });
  }, []);

  function handleDatasetChange(id: number | "", detail: VisionDatasetDetail | null) {
    setDatasetId(id);
    setDatasetDetail(detail);
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (!datasetId) return;
    setError(null);
    setIsSubmitting(true);
    try {
      const job = await api.visionAnomalies.createJob(
        {
          vision_dataset_id: datasetId,
          model_id: modelId,
          num_epochs: numEpochs,
          batch_size: batchSize,
          learning_rate: learningRate,
          mask_percentile: maskPercentile,
          augmentation_preset: augmentationPreset,
          val_ratio: valRatio,
        },
        idempotencyKey.current,
      );
      idempotencyKey.reset();
      onJobCreated(job);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Impossible de lancer l'entraînement");
    } finally {
      setIsSubmitting(false);
    }
  }

  const selectedModelLabel = models.find((m) => m.id === modelId)?.label ?? "—";
  const step1Valid = Boolean(datasetId && modelId);

  return (
    <form onSubmit={handleSubmit}>
      <WizardStepper steps={STEP_LABELS} activeStep={activeStep} maxReachedStep={maxReachedStep} onSelect={goToStep} />

      <Card className="p-5 mt-4">
        {activeStep === 1 && (
          <StepContent
            title="Choisissez vos données et le modèle"
            description="Sélectionnez un dataset structuré normal/défaut (train/good + test/good + test/<défaut>), puis le modèle de reconstruction à entraîner."
          >
            <div>
              <label className="block text-sm text-muted-foreground mb-1.5">Dataset d'images (structure normal/défaut)</label>
              <VisionDatasetPicker structureType="mvtec_ad" value={datasetId} onChange={handleDatasetChange} />
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
                {MODEL_HINTS[modelId] && <p className="text-xs text-muted-foreground mt-1">{MODEL_HINTS[modelId]}</p>}
              </div>
            )}

            {datasetDetail && (
              <div>
                <label className="block text-sm text-muted-foreground mb-1.5">Répartition de train/good/</label>
                <SplitRatioControl
                  totalImages={datasetDetail.n_images}
                  splits={[{ key: "val", label: "Validation", ratio: valRatio, onChange: setValRatio, min: 0.05, max: 0.4 }]}
                />
              </div>
            )}
          </StepContent>
        )}

        {activeStep === 2 && (
          <StepContent
            title="Augmentation des données"
            description="Diversifie artificiellement vos images normales d'entraînement (retournements, rotations légères...) — appliquée uniquement à train/good/, jamais à test/."
          >
            <AugmentationPresetPicker value={augmentationPreset} onChange={setAugmentationPreset} />
            <AugmentationPreviewGallery datasetId={datasetId} preset={augmentationPreset} />
          </StepContent>
        )}

        {activeStep === 3 && (
          <StepContent
            title="Mode expert"
            description="Réglages fins de l'entraînement — les valeurs par défaut conviennent à la plupart des datasets."
          >
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

            <div>
              <label htmlFor="va-batch-size" className="block text-sm text-muted-foreground mb-1">
                Taille de lot (batch size)
              </label>
              <Input
                id="va-batch-size"
                type="number"
                min={1}
                max={128}
                value={batchSize}
                onChange={(e) => setBatchSize(Math.min(128, Math.max(1, Number(e.target.value) || 1)))}
              />
            </div>

            <div>
              <label htmlFor="va-learning-rate" className="block text-sm text-muted-foreground mb-1">
                Taux d'apprentissage — {learningRate}
              </label>
              <input
                id="va-learning-rate"
                type="range"
                min={0.0001}
                max={0.01}
                step={0.0001}
                value={learningRate}
                onChange={(e) => setLearningRate(Number(e.target.value))}
                className="w-full accent-primary"
              />
            </div>

            <div>
              <label htmlFor="va-mask-percentile" className="block text-sm text-muted-foreground mb-1">
                Sensibilité de la carte de défaut — percentile {Math.round(maskPercentile * 100)}
              </label>
              <input
                id="va-mask-percentile"
                type="range"
                min={0.9}
                max={0.995}
                step={0.005}
                value={maskPercentile}
                onChange={(e) => setMaskPercentile(Number(e.target.value))}
                className="w-full accent-primary"
              />
              <p className="text-xs text-muted-foreground mt-1">
                Plus haut = seules les zones les plus atypiques sont marquées comme défaut sur la carte de
                chaleur (moins de zones surlignées, plus précises).
              </p>
            </div>
          </StepContent>
        )}

        {activeStep === 4 && (
          <StepContent title="Prêt à lancer" description="Vérifiez le récapitulatif, puis lancez l'entraînement.">
            <div className="rounded-xl border border-border bg-muted p-4">
              <p className="text-xs uppercase tracking-wide text-muted-foreground mb-3">Récapitulatif</p>
              <dl className="grid grid-cols-2 gap-y-2.5 text-sm">
                <Fact label="Dataset" value={datasetDetail ? `${datasetDetail.n_images} images` : "—"} />
                <Fact label="Modèle" value={selectedModelLabel} />
                <Fact label="Époques" value={String(numEpochs)} />
                <Fact label="Augmentation" value={AUGMENTATION_PRESET_INFO[augmentationPreset].label} />
                <Fact label="Validation" value={`${Math.round(valRatio * 100)} % de train/good/`} />
              </dl>
            </div>

            {error && (
              <div className="flex items-center gap-2 text-sm text-destructive bg-destructive/10 border border-destructive/20 rounded-lg px-3 py-2">
                <AlertCircle size={15} className="flex-shrink-0" />
                {error}
              </div>
            )}

            <Button type="submit" disabled={!step1Valid || isSubmitting} className="w-full" size="md">
              <PlayCircle size={16} />
              {isSubmitting ? "Lancement…" : "Lancer l'entraînement"}
            </Button>
          </StepContent>
        )}

        <div className="flex items-center justify-between pt-5 mt-5 border-t border-border">
          {activeStep > 1 ? (
            <Button type="button" variant="secondary" size="sm" onClick={goPrev}>
              <ChevronLeft size={14} />
              Précédent
            </Button>
          ) : (
            <span />
          )}
          {activeStep < STEP_LABELS.length && (
            <Button type="button" size="sm" onClick={goNext} disabled={activeStep === 1 && !step1Valid}>
              Continuer
              <ChevronRight size={14} />
            </Button>
          )}
        </div>
      </Card>
    </form>
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

function AnomalyVisionResultView({ jobId, datasetId }: { jobId: number; datasetId: number }) {
  const [result, setResult] = useState<VisionAnomalyResult | null>(null);
  const [examples, setExamples] = useState<VisionAnomalyExample[]>([]);
  const [examplesError, setExamplesError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"performance" | "exemples">("performance");

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

      <ModelExportActions
        onExportArtifact={() => api.visionAnomalies.exportModel(jobId)}
        exportConfig={{
          threshold: result.threshold,
          roc_auc: result.roc_auc,
          test_accuracy: result.test_accuracy,
          test_precision: result.test_precision,
          test_recall: result.test_recall,
          test_f1: result.test_f1,
          model_card: result.model_card,
        }}
        configFilename={`vision_anomalies_config_job${jobId}.json`}
      />

      <Tabs
        items={[
          { id: "performance" as const, label: "Performance", icon: Sparkles },
          { id: "exemples" as const, label: "Exemples", icon: AlertTriangle },
        ]}
        active={activeTab}
        onChange={setActiveTab}
        urlParam="onglet"
      />

      {activeTab === "performance" && (
        <>
          <Card className="p-5">
            <SectionHeader icon={Sparkles} color="blue" label="Courbe d'apprentissage (reconstruction)" />
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={historyData} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
                <CartesianGrid stroke={CHART_GRID_STROKE} vertical={false} />
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
        </>
      )}

      {activeTab === "exemples" && (
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
      )}
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
        {/* Couleurs fixes volontaires (Lot 8, revue du grep de couleurs en
            dur porté depuis le Lot 1) : reproduit exactement la palette
            "jet" appliquée côté serveur au PNG ci-dessus
            (backend/domains/vision/localization.py::_apply_colormap,
            bleu=faible/rouge=fort) — jamais les jetons de thème, qui
            rendraient cette légende fausse par rapport à l'image réellement
            affichée au-dessus. */}
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
