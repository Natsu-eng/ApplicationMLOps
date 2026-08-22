import { useEffect, useRef, useState, type FormEvent } from "react";
import { Link, useSearchParams } from "react-router-dom";
import {
  AlertCircle,
  Ban,
  Boxes,
  ChevronLeft,
  ChevronRight,
  Loader2,
  PlayCircle,
  RotateCcw,
  Sparkles,
  Target,
  Trash2,
} from "lucide-react";
import { Line, LineChart, CartesianGrid, ResponsiveContainer, Tooltip as RechartsTooltip, XAxis, YAxis } from "recharts";
import {
  ApiError,
  api,
  type AugmentationPreset,
  type GradCamExplanation,
  type VisionBackbone,
  type VisionClassificationJobSummary,
  type VisionClassificationResult,
  type VisionDatasetDetail,
  type VisionPredictionExample,
} from "../api/client";
import AppShell from "../components/AppShell";
import { pillarColor } from "../config/pillars";
import { Badge } from "../components/ui/Badge";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { accentSurfaceClass, accentValueTextClass, type AccentColor } from "../components/ui/ColorIconBadge";
import EvaluationCharts from "../components/training/EvaluationCharts";
import { Input } from "../components/ui/Input";
import { PageHeader } from "../components/ui/PageHeader";
import { SectionHeader } from "../components/ui/SectionHeader";
import { Select } from "../components/ui/Select";
import { Switch } from "../components/ui/Switch";
import { VisionDatasetPicker } from "../components/vision/VisionDatasetPicker";
import { useJobEvents } from "../hooks/useJobEvents";
import { VisionImage } from "../components/vision/VisionImage";
import {
  AUGMENTATION_PRESET_INFO,
  AugmentationPresetPicker,
  AugmentationPreviewGallery,
  ClassImbalanceBanner,
  Fact,
  SplitRatioControl,
  StepContent,
  StepperNav,
} from "../components/vision/VisionWizard";
import { useConfirmAction } from "../hooks/useConfirmAction";
import { CHART_GRID_STROKE, CHART_SERIES_COLORS, CHART_TICK_STYLE_SM, CHART_TOOLTIP_STYLE } from "../theme/charts";

/** Étapes du wizard horizontal (Lot 6A, correctif I10) — porte le pattern
 * de `Training.tsx` (pastilles numérotées, navigation par étapes, mode
 * expert replié par défaut) au pilier Vision, jusqu'ici asymétrique (un
 * simple formulaire à plat). Pas de "Qualité des données" (pas d'EDA pour
 * des images) — 4 étapes plutôt que 5, même esprit. */
const STEP_LABELS = [
  { number: 1, label: "Données & modèle" },
  { number: 2, label: "Augmentation" },
  { number: 3, label: "Mode expert" },
  { number: 4, label: "Lancement" },
];

const ACTIVE_STATUSES = new Set(["queued", "running"]);
const ACTIVE_JOB_STORAGE_KEY = "datalab_active_vision_classification_job_id";

type Phase = "configure" | "progress" | "results" | "failed" | "cancelled";

function phaseOf(job: VisionClassificationJobSummary | null): Phase {
  if (!job) return "configure";
  if (ACTIVE_STATUSES.has(job.status)) return "progress";
  if (job.status === "completed") return "results";
  return job.status === "cancelled" ? "cancelled" : "failed";
}

/** Pilier Vision — classification d'images par transfer learning (sous-lot
 * B) + Grad-CAM (sous-lot D). Même architecture que AnomalyDetection.tsx. */
export default function VisionClassification() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [activeJob, setActiveJob] = useState<VisionClassificationJobSummary | null>(null);
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
    api.visionClassification
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

  function openJob(job: VisionClassificationJobSummary) {
    setActiveJob(job);
    setActiveDatasetId(job.vision_dataset_id);
    setSearchParams({ job: String(job.id) }, { replace: false });
  }

  const phase = phaseOf(activeJob);

  // Notifications SSE (Lot 7, §J.2) — remplace le polling setInterval.
  useJobEvents(
    phase === "progress" && activeJob ? `/vision/classification/jobs/${activeJob.id}/events` : null,
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
      await api.visionClassification.remove(activeJob.id);
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
      setActiveJob(await api.visionClassification.cancel(activeJob.id));
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
      openJob(await api.visionClassification.rerun(activeJob.id));
    } catch (err) {
      setRerunError(err instanceof ApiError ? err.message : "Impossible de relancer cet entraînement");
    } finally {
      setRerunning(false);
    }
  }

  const titles: Record<Phase, string> = {
    configure: "Classifier des images",
    progress: "Entraînement en cours",
    results: "Résultats de la classification",
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
            ? "Entraînez un modèle à reconnaître des catégories d'images à partir d'exemples déjà classés — par transfer learning, à partir d'un modèle pré-entraîné."
            : undefined
        }
        icon={Boxes}
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
          <ClassificationForm onJobCreated={openJob} />
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
        <ClassificationResultView jobId={activeJob.id} datasetId={activeDatasetId} />
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

function ClassificationForm({ onJobCreated }: { onJobCreated: (job: VisionClassificationJobSummary) => void }) {
  const [datasetId, setDatasetId] = useState<number | "">("");
  const [datasetDetail, setDatasetDetail] = useState<VisionDatasetDetail | null>(null);
  const [backbones, setBackbones] = useState<VisionBackbone[]>([]);
  const [backboneId, setBackboneId] = useState("");
  const [numEpochs, setNumEpochs] = useState(8);
  const [batchSize, setBatchSize] = useState(16);
  const [learningRate, setLearningRate] = useState(1e-3);
  const [dropoutRate, setDropoutRate] = useState(0.3);
  const [freezeBackbone, setFreezeBackbone] = useState(true);
  const [unfreezeAfterEpoch, setUnfreezeAfterEpoch] = useState<number | "">("");
  const [classWeighting, setClassWeighting] = useState(true);
  const [earlyStoppingPatience, setEarlyStoppingPatience] = useState<number | "">(3);
  const [useLrScheduler, setUseLrScheduler] = useState(true);
  const [augmentationPreset, setAugmentationPreset] = useState<AugmentationPreset>("standard");
  // Vrai tant que l'utilisateur n'a pas choisi lui-même un preset — permet
  // à la recommandation (I9) de s'appliquer automatiquement au choix du
  // dataset SANS écraser un choix déjà fait explicitement.
  const [augmentationTouched, setAugmentationTouched] = useState(false);
  // Répartition personnalisée (Lot 6A) — 15/15 reproduit le 70/15/15
  // historique (défaut inchangé pour qui ne touche jamais ces curseurs).
  const [valRatio, setValRatio] = useState(0.15);
  const [testRatio, setTestRatio] = useState(0.15);

  // Mode expert (même esprit que Training.tsx/ExpertModePanel) — replié
  // par défaut, chaque manette démarre à la même valeur que le mode
  // guidé : l'activer sans rien changer ne modifie aucun résultat.
  const [expertMode, setExpertMode] = useState(false);

  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

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
    api.visionClassification.backbones().then((list) => {
      setBackbones(list);
      if (list.length > 0) setBackboneId(list[0].id);
    });
  }, []);

  function handleDatasetChange(id: number | "", detail: VisionDatasetDetail | null) {
    setDatasetId(id);
    setDatasetDetail(detail);
    if (!augmentationTouched && detail?.recommended_augmentation_preset) {
      setAugmentationPreset(detail.recommended_augmentation_preset);
    }
  }

  function handleAugmentationChange(preset: AugmentationPreset) {
    setAugmentationTouched(true);
    setAugmentationPreset(preset);
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (!datasetId) return;
    setError(null);
    setIsSubmitting(true);
    try {
      const job = await api.visionClassification.createJob({
        vision_dataset_id: datasetId,
        backbone_id: backboneId,
        num_epochs: numEpochs,
        batch_size: batchSize,
        learning_rate: learningRate,
        dropout_rate: dropoutRate,
        freeze_backbone: freezeBackbone,
        unfreeze_after_epoch: unfreezeAfterEpoch === "" ? null : unfreezeAfterEpoch,
        class_weighting: classWeighting,
        early_stopping_patience: earlyStoppingPatience === "" ? null : earlyStoppingPatience,
        use_lr_scheduler: useLrScheduler,
        augmentation_preset: augmentationPreset,
        val_ratio: valRatio,
        test_ratio: testRatio,
      });
      onJobCreated(job);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Impossible de lancer l'entraînement");
    } finally {
      setIsSubmitting(false);
    }
  }

  const selectedBackboneLabel = backbones.find((b) => b.id === backboneId)?.label ?? "—";
  const step1Valid = Boolean(datasetId && backboneId);

  return (
    <form onSubmit={handleSubmit}>
      <StepperNav steps={STEP_LABELS} activeStep={activeStep} maxReachedStep={maxReachedStep} onSelect={goToStep} />

      <Card className="p-5 mt-4">
        {activeStep === 1 && (
          <StepContent
            title="Choisissez vos données et le modèle"
            description="Sélectionnez un dataset d'images déjà classées par dossier, puis le modèle pré-entraîné à affiner."
          >
            <div>
              <label className="block text-sm text-muted-foreground mb-1.5">Dataset d'images</label>
              <VisionDatasetPicker structureType="classification" value={datasetId} onChange={handleDatasetChange} />
            </div>

            <ClassImbalanceBanner classDistribution={datasetDetail?.class_distribution} />

            {backbones.length > 0 && (
              <div>
                <label htmlFor="vc-backbone" className="block text-sm text-muted-foreground mb-1">
                  Modèle pré-entraîné (transfer learning)
                </label>
                <Select id="vc-backbone" value={backboneId} onChange={(e) => setBackboneId(e.target.value)}>
                  {backbones.map((b) => (
                    <option key={b.id} value={b.id}>
                      {b.label}
                    </option>
                  ))}
                </Select>
                <p className="text-xs text-muted-foreground mt-1">
                  Un modèle plus léger (MobileNet) entraîne plus vite ; un modèle plus profond (ResNet,
                  EfficientNet, DenseNet) peut être plus précis sur un dataset plus riche, au prix d'un
                  entraînement plus long.
                </p>
              </div>
            )}

            {datasetDetail && (
              <div>
                <label className="block text-sm text-muted-foreground mb-1.5">Répartition des données</label>
                <SplitRatioControl
                  totalImages={datasetDetail.n_images}
                  splits={[
                    { key: "val", label: "Validation", ratio: valRatio, onChange: setValRatio, min: 0.05, max: 0.4 },
                    { key: "test", label: "Test", ratio: testRatio, onChange: setTestRatio, min: 0.05, max: 0.4 },
                  ]}
                />
              </div>
            )}
          </StepContent>
        )}

        {activeStep === 2 && (
          <StepContent
            title="Augmentation des données"
            description="Diversifie artificiellement vos images d'entraînement (retournements, rotations légères...) pour réduire le sur-apprentissage. La recommandation est déjà sélectionnée — changez-la si besoin."
          >
            <AugmentationPresetPicker
              value={augmentationPreset}
              onChange={handleAugmentationChange}
              recommendedPreset={datasetDetail?.recommended_augmentation_preset}
            />

            <AugmentationPreviewGallery datasetId={datasetId} preset={augmentationPreset} />
          </StepContent>
        )}

        {activeStep === 3 && (
          <StepContent
            title="Mode expert"
            description="Par défaut, des réglages standards s'appliquent (pondération de classes, arrêt anticipé et taux d'apprentissage adaptatif déjà activés). Activez ce mode pour tout contrôler."
          >
            <div>
              <label htmlFor="vc-epochs" className="block text-sm text-muted-foreground mb-1">
                Nombre d'époques
              </label>
              <Input
                id="vc-epochs"
                type="number"
                min={1}
                max={30}
                value={numEpochs}
                onChange={(e) => setNumEpochs(Math.min(30, Math.max(1, Number(e.target.value) || 1)))}
              />
            </div>

            <div className="flex items-center justify-between rounded-lg border border-border px-3 py-2.5">
              <div>
                <p className="text-sm text-foreground">Mode expert</p>
                <p className="text-xs text-muted-foreground">
                  Contrôle direct du taux d'apprentissage, de la taille de lot, du dégel progressif du tronc, de
                  la pondération de classes, de l'arrêt anticipé et du scheduler.
                </p>
              </div>
              <Switch checked={expertMode} onChange={setExpertMode} label="Mode expert" />
            </div>

            {expertMode && (
              <>
                <div className="flex items-center justify-between rounded-lg border border-border px-3 py-2.5">
                  <div>
                    <p className="text-sm text-foreground">Geler le tronc pré-entraîné</p>
                    <p className="text-xs text-muted-foreground">
                      Recommandé — seule la tête de classification est entraînée, plus rapide et plus fiable
                      avec peu d'images.
                    </p>
                  </div>
                  <Switch checked={freezeBackbone} onChange={setFreezeBackbone} label="Geler le tronc pré-entraîné" />
                </div>

                {freezeBackbone && (
                  <div>
                    <label htmlFor="vc-unfreeze" className="block text-sm text-muted-foreground mb-1">
                      Dégeler à partir de l'époque <span className="text-muted-foreground">(optionnel)</span>
                    </label>
                    <Input
                      id="vc-unfreeze"
                      type="number"
                      min={0}
                      max={numEpochs - 1}
                      placeholder="Jamais"
                      value={unfreezeAfterEpoch}
                      onChange={(e) => setUnfreezeAfterEpoch(e.target.value === "" ? "" : Number(e.target.value))}
                    />
                    <p className="text-xs text-muted-foreground mt-1">
                      Fine-tuning complet du tronc à partir de cette époque — affine le modèle pré-entraîné sur
                      vos images, au prix d'un entraînement plus lent.
                    </p>
                  </div>
                )}

                <div>
                  <label htmlFor="vc-batch-size" className="block text-sm text-muted-foreground mb-1">
                    Taille de lot (batch size)
                  </label>
                  <Input
                    id="vc-batch-size"
                    type="number"
                    min={1}
                    max={128}
                    value={batchSize}
                    onChange={(e) => setBatchSize(Math.min(128, Math.max(1, Number(e.target.value) || 1)))}
                  />
                </div>

                <div>
                  <label htmlFor="vc-learning-rate" className="block text-sm text-muted-foreground mb-1">
                    Taux d'apprentissage — {learningRate}
                  </label>
                  <input
                    id="vc-learning-rate"
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
                  <label htmlFor="vc-dropout" className="block text-sm text-muted-foreground mb-1">
                    Dropout — {Math.round(dropoutRate * 100)} %
                  </label>
                  <input
                    id="vc-dropout"
                    type="range"
                    min={0}
                    max={0.9}
                    step={0.05}
                    value={dropoutRate}
                    onChange={(e) => setDropoutRate(Number(e.target.value))}
                    className="w-full accent-primary"
                  />
                </div>

                <div className="flex items-center justify-between rounded-lg border border-border px-3 py-2.5">
                  <div>
                    <p className="text-sm text-foreground">Pondération de classes</p>
                    <p className="text-xs text-muted-foreground">
                      Corrige un dataset déséquilibré — sans elle, le modèle peut apprendre à toujours prédire
                      la classe majoritaire.
                    </p>
                  </div>
                  <Switch checked={classWeighting} onChange={setClassWeighting} label="Pondération de classes" />
                </div>

                <div className="flex items-center justify-between rounded-lg border border-border px-3 py-2.5">
                  <div>
                    <p className="text-sm text-foreground">Taux d'apprentissage adaptatif</p>
                    <p className="text-xs text-muted-foreground">Réduit le taux d'apprentissage quand la progression stagne.</p>
                  </div>
                  <Switch checked={useLrScheduler} onChange={setUseLrScheduler} label="Taux d'apprentissage adaptatif" />
                </div>

                <div>
                  <label htmlFor="vc-patience" className="block text-sm text-muted-foreground mb-1">
                    Arrêt anticipé — patience (époques) <span className="text-muted-foreground">(optionnel)</span>
                  </label>
                  <Input
                    id="vc-patience"
                    type="number"
                    min={1}
                    max={10}
                    placeholder="Désactivé"
                    value={earlyStoppingPatience}
                    onChange={(e) => setEarlyStoppingPatience(e.target.value === "" ? "" : Number(e.target.value))}
                  />
                  <p className="text-xs text-muted-foreground mt-1">
                    Arrête l'entraînement si aucune amélioration depuis ce nombre d'époques — économise du temps
                    sans changer quel modèle est retenu (les meilleurs poids sont toujours conservés).
                  </p>
                </div>
              </>
            )}
          </StepContent>
        )}

        {activeStep === 4 && (
          <StepContent title="Prêt à lancer" description="Vérifiez le récapitulatif, puis lancez l'entraînement.">
            <div className="rounded-xl border border-border bg-muted p-4">
              <p className="text-xs uppercase tracking-wide text-muted-foreground mb-3">Récapitulatif</p>
              <dl className="grid grid-cols-2 gap-y-2.5 text-sm">
                <Fact label="Dataset" value={datasetDetail ? `${datasetDetail.n_images} images` : "—"} />
                <Fact label="Modèle" value={selectedBackboneLabel} />
                <Fact label="Époques" value={String(numEpochs)} />
                <Fact label="Augmentation" value={AUGMENTATION_PRESET_INFO[augmentationPreset].label} />
                <Fact
                  label="Répartition"
                  value={`${Math.round((1 - valRatio - testRatio) * 100)} / ${Math.round(valRatio * 100)} / ${Math.round(testRatio * 100)} %`}
                />
                <Fact label="Pondération de classes" value={classWeighting ? "Activée" : "Désactivée"} />
                <Fact label="Arrêt anticipé" value={earlyStoppingPatience === "" ? "Désactivé" : `${earlyStoppingPatience} époques`} />
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
      <p className={`text-xl font-semibold tabular-nums ${accentValueTextClass(color)}`}>{value}</p>
    </div>
  );
}

function ClassificationResultView({ jobId, datasetId }: { jobId: number; datasetId: number }) {
  const [result, setResult] = useState<VisionClassificationResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.visionClassification
      .getResult(jobId)
      .then(setResult)
      .catch((err) => setError(err instanceof ApiError ? err.message : "Résultat indisponible"));
  }, [jobId]);

  if (error) return <p className="text-sm text-destructive text-center">{error}</p>;
  if (!result) return <p className="text-sm text-muted-foreground text-center">Chargement…</p>;

  const historyData = result.history.map((h) => ({
    epoch: h.epoch + 1,
    "Perte (train)": h.train_loss,
    "Perte (validation)": h.val_loss,
    "Exactitude (train)": h.train_accuracy,
    "Exactitude (validation)": h.val_accuracy,
  }));

  const incorrectExamples = result.examples.filter((e) => !e.correct);
  const correctExamples = result.examples.filter((e) => e.correct);

  return (
    <div className="max-w-4xl mx-auto space-y-5">
      <Card className={`p-5 ${accentSurfaceClass("violet")}`}>
        <SectionHeader
          icon={Target}
          color="violet"
          label="Performance sur le jeu de test"
          help="Calculée sur les images de test, jamais vues pendant l'entraînement — moyenne macro entre les classes (précision/rappel/F1)."
        />
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <MetricTile label="Exactitude" value={`${(result.test_accuracy * 100).toFixed(1)} %`} color="violet" />
          <MetricTile label="Précision" value={`${(result.test_precision_macro * 100).toFixed(1)} %`} color="blue" />
          <MetricTile label="Rappel" value={`${(result.test_recall_macro * 100).toFixed(1)} %`} color="teal" />
          <MetricTile label="F1" value={`${(result.test_f1_macro * 100).toFixed(1)} %`} color="amber" />
          {result.test_roc_auc != null && (
            <MetricTile label="ROC-AUC" value={result.test_roc_auc.toFixed(3)} color="rose" />
          )}
        </div>
        <p className="text-xs text-muted-foreground mt-3">
          {result.n_train} images d'entraînement · {result.n_val} de validation · {result.n_test} de test —{" "}
          {result.class_names.length} classes ({result.class_names.join(", ")}).
          {Boolean(result.model_card.time_capped) && " Entraînement arrêté par le garde-fou de temps CPU."}
        </p>
      </Card>

      <Card className="p-5">
        <SectionHeader icon={Sparkles} color="blue" label="Courbes d'apprentissage" />
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={historyData} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID_STROKE} vertical={false} />
            <XAxis dataKey="epoch" tick={CHART_TICK_STYLE_SM} label={{ value: "Époque", position: "insideBottom", offset: -2 }} />
            <YAxis tick={CHART_TICK_STYLE_SM} />
            <RechartsTooltip {...CHART_TOOLTIP_STYLE} />
            <Line type="monotone" dataKey="Exactitude (train)" stroke={CHART_SERIES_COLORS[0]} dot={false} isAnimationActive={false} />
            <Line type="monotone" dataKey="Exactitude (validation)" stroke={CHART_SERIES_COLORS[1]} dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      {/* Réutilise EvaluationCharts.tsx tel quel (matrice de confusion +
          ROC + PR) — même composant que le tabulaire, jamais un second
          composant de graphique à maintenir en parallèle (Lot 6A,
          correctif 16G). roc_curves/pr_curves absents (undefined) sur les
          modèles entraînés avant ce correctif : les deux Card ROC/PR ne
          s'affichent simplement pas (EvaluationCharts gère déjà ce cas
          pour le tabulaire). */}
      <EvaluationCharts
        taskType="classification"
        evaluation={{
          confusion_matrix: result.confusion_matrix,
          class_names: result.class_names,
          roc_curves: result.roc_curves,
          pr_curves: result.pr_curves,
        }}
      />

      {incorrectExamples.length > 0 && (
        <div>
          <SectionHeader icon={AlertCircle} color="rose" label={`Erreurs de classification (${incorrectExamples.length})`} />
          <ExampleGrid examples={incorrectExamples} datasetId={datasetId} />
        </div>
      )}

      {correctExamples.length > 0 && (
        <div>
          <SectionHeader icon={Target} color="teal" label={`Exemples corrects (${correctExamples.length})`} />
          <ExampleGrid examples={correctExamples} datasetId={datasetId} />
        </div>
      )}

      <GradCamPanel jobId={jobId} />
    </div>
  );
}

function ExampleGrid({ examples, datasetId }: { examples: VisionPredictionExample[]; datasetId: number }) {
  return (
    <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 gap-3">
      {examples.map((example) => (
        <div key={example.relative_path} className="space-y-1">
          <VisionImage
            datasetId={datasetId}
            path={example.relative_path}
            alt={example.relative_path}
            className={`w-full aspect-square object-cover rounded-lg border-2 ${example.correct ? "border-success/40" : "border-destructive/40"}`}
          />
          <p className="text-caption text-muted-foreground truncate" title={example.relative_path}>
            {example.correct ? example.predicted_label : `${example.true_label} → ${example.predicted_label}`}
          </p>
        </div>
      ))}
    </div>
  );
}

function GradCamPanel({ jobId }: { jobId: number }) {
  const [explanation, setExplanation] = useState<GradCamExplanation | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  async function handleFile(files: FileList | null) {
    const file = files?.[0];
    if (!file) return;
    setError(null);
    setExplanation(null);
    setPreviewUrl(URL.createObjectURL(file));
    setIsSubmitting(true);
    try {
      setExplanation(await api.visionClassification.explain(jobId, file));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Impossible de générer l'explication");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <Card className="p-5">
      <SectionHeader
        icon={Sparkles}
        color="amber"
        label="Pourquoi cette prédiction ? (Grad-CAM)"
        help="Superpose une carte de chaleur sur l'image : les zones les plus chaudes sont celles qui ont le plus influencé la classe prédite par le modèle."
      />
      <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={(e) => handleFile(e.target.files)} />
      <Button variant="secondary" size="sm" type="button" onClick={() => fileInputRef.current?.click()}>
        Choisir une image à expliquer
      </Button>

      {isSubmitting && <p className="text-sm text-muted-foreground mt-3">Calcul en cours…</p>}
      {error && (
        <div className="flex items-center gap-2 text-sm text-destructive bg-destructive/10 border border-destructive/20 rounded-lg px-3 py-2 mt-3">
          <AlertCircle size={15} className="flex-shrink-0" />
          {error}
        </div>
      )}

      {explanation && previewUrl && (
        <div className="mt-4 space-y-3">
          <div className="flex items-center gap-2 flex-wrap">
            <Badge variant="primary">Prédiction : {explanation.predicted_label}</Badge>
            {Object.entries(explanation.probabilities).map(([label, proba]) => (
              <Badge key={label} variant="neutral">
                {label} : {(proba * 100).toFixed(0)} %
              </Badge>
            ))}
          </div>
          <div className="max-w-sm">
            <img
              src={explanation.heatmap_png}
              alt="Image avec carte de chaleur Grad-CAM superposée"
              className="w-full aspect-square object-cover rounded-lg border border-border"
            />
            <div className="flex items-center gap-2 mt-2">
              {/* Couleurs fixes volontaires (Lot 8, revue du grep de couleurs
                  en dur porté depuis le Lot 1) : reproduit exactement la
                  palette "jet" appliquée côté serveur au PNG ci-dessus
                  (backend/domains/vision/localization.py::_apply_colormap,
                  bleu=faible/rouge=fort) — jamais les jetons de thème, qui
                  rendraient cette légende fausse par rapport à l'image
                  réellement affichée au-dessus. */}
              <span
                className="inline-block h-2 w-16 rounded-full flex-shrink-0"
                style={{ background: "linear-gradient(to right, #0000cc, #00cc66, #cc0000)" }}
                aria-hidden="true"
              />
              <p className="text-xs text-muted-foreground">
                Bleu = faible influence · Rouge = zones qui ont le plus influencé la classe "
                {explanation.target_label}"
              </p>
            </div>
          </div>
        </div>
      )}
    </Card>
  );
}
