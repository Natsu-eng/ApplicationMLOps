import { useEffect, useRef, useState, type FormEvent } from "react";
import { Link, useSearchParams } from "react-router-dom";
import {
  Activity,
  AlertCircle,
  Ban,
  Boxes,
  ChevronLeft,
  ChevronRight,
  FileCode,
  FileJson,
  Loader2,
  PlayCircle,
  RotateCcw,
  Sparkles,
  Target,
  Trash2,
  Trophy,
  Wand2,
} from "lucide-react";
import { Line, LineChart, CartesianGrid, ResponsiveContainer, Tooltip as RechartsTooltip, XAxis, YAxis } from "recharts";
import {
  ApiError,
  api,
  type AugmentationPreset,
  type BackboneComparisonCandidate,
  type GradCamBatchItem,
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
import { BulkActionBar } from "../components/ui/BulkActionBar";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { accentSurfaceClass, accentValueTextClass, type AccentColor } from "../components/ui/ColorIconBadge";
import EvaluationCharts from "../components/training/EvaluationCharts";
import { CalibrationChart } from "../components/training/ReliabilityDiagnostics";
import { Input } from "../components/ui/Input";
import { PageHeader } from "../components/ui/PageHeader";
import { SectionHeader } from "../components/ui/SectionHeader";
import { Select } from "../components/ui/Select";
import { Switch } from "../components/ui/Switch";
import { Tabs } from "../components/ui/Tabs";
import { Table, type TableColumn } from "../components/ui/Table";
import { LabelWithHelp } from "../components/ui/Tooltip";
import { ModelExportActions } from "../components/ui/ModelExportActions";
import { VisionDatasetPicker } from "../components/vision/VisionDatasetPicker";
import { useJobEvents } from "../hooks/useJobEvents";
import { VisionImage } from "../components/vision/VisionImage";
import { buildVisionClassificationModelCard } from "../utils/visionClassificationModelCard";
import {
  AUGMENTATION_PRESET_INFO,
  AugmentationPresetPicker,
  AugmentationPreviewGallery,
  ClassImbalanceBanner,
  Fact,
  ImageSizePicker,
  SplitRatioControl,
  StepContent,
} from "../components/vision/VisionWizard";
import { WizardStepper } from "../components/ui/WizardStepper";
import { useConfirmAction } from "../hooks/useConfirmAction";
import { useIdempotencyKey } from "../hooks/useIdempotencyKey";
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
// Même plafond que `router.py::MAX_EXPLAIN_BATCH_SIZE` — affiché ici pour
// que l'utilisateur voie la limite AVANT de se heurter à un rejet serveur.
const MAX_EXPLAIN_BATCH_SIZE = 12;
// Même plafond que `services/engine.py::MAX_BACKBONES_PER_COMPARISON` —
// comparatif de backbones (mode expert), affiché ici pour la même raison.
const MAX_BACKBONES_PER_COMPARISON = 4;

// Lot 16F — palier de vitesse déjà calculé côté serveur (voir
// services/registry.py::speed_tier), jamais recalculé ici : seul le libellé
// et la couleur d'affichage sont une décision d'UI.
const SPEED_TIER_LABELS: Record<VisionBackbone["speed_tier"], string> = {
  rapide: "Rapide",
  modere: "Modéré",
  lent: "Lent",
};
const SPEED_TIER_BADGE_VARIANT: Record<VisionBackbone["speed_tier"], "success" | "neutral" | "warning"> = {
  rapide: "success",
  modere: "neutral",
  lent: "warning",
};

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
        <ClassificationResultView jobId={activeJob.id} datasetId={activeDatasetId} datasetName={activeJob.vision_dataset_name} />
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
  // Mode expert : comparatif de backbones (retour utilisateur direct — parité
  // avec la comparaison multi-modèles du ML tabulaire) — replié par défaut,
  // un seul backbone (comportement historique) tant que non activé.
  const [comparisonMode, setComparisonMode] = useState(false);
  const [comparisonBackboneIds, setComparisonBackboneIds] = useState<Set<string>>(new Set());
  // Mode expert : résolution d'entrée (retour utilisateur direct — "vision
  // n'offre pas de réduire/augmenter la taille des images") — 224 =
  // comportement historique inchangé.
  const [imageSize, setImageSize] = useState(224);
  const [numEpochs, setNumEpochs] = useState(8);
  const [batchSize, setBatchSize] = useState(16);
  const [learningRate, setLearningRate] = useState(1e-3);
  const [weightDecay, setWeightDecay] = useState(0);
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
      const job = await api.visionClassification.createJob(
        {
          vision_dataset_id: datasetId,
          backbone_id: backboneId,
          // Mode expert : comparatif (retour utilisateur direct) — n'envoyé
          // que si réellement activé ET au moins 2 backbones cochés, jamais
          // un tableau à 1 élément (rejeté côté serveur de toute façon, mais
          // autant ne pas l'envoyer — même garde que le mode expert du ML
          // tabulaire, voir trainingPayload.ts).
          backbone_ids:
            comparisonMode && comparisonBackboneIds.size >= 2 ? Array.from(comparisonBackboneIds) : undefined,
          image_size: imageSize,
          num_epochs: numEpochs,
          batch_size: batchSize,
          learning_rate: learningRate,
          weight_decay: weightDecay,
          dropout_rate: dropoutRate,
          freeze_backbone: freezeBackbone,
          unfreeze_after_epoch: unfreezeAfterEpoch === "" ? null : unfreezeAfterEpoch,
          class_weighting: classWeighting,
          early_stopping_patience: earlyStoppingPatience === "" ? null : earlyStoppingPatience,
          use_lr_scheduler: useLrScheduler,
          augmentation_preset: augmentationPreset,
          val_ratio: valRatio,
          test_ratio: testRatio,
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

  const selectedBackbone = backbones.find((b) => b.id === backboneId);
  const selectedBackboneLabel = selectedBackbone?.label ?? "—";
  const step1Valid = Boolean(
    datasetId && (comparisonMode ? comparisonBackboneIds.size >= 2 : backboneId),
  );

  function toggleComparisonBackbone(id: string) {
    setComparisonBackboneIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else if (next.size < MAX_BACKBONES_PER_COMPARISON) {
        next.add(id);
      }
      return next;
    });
  }

  return (
    <form onSubmit={handleSubmit}>
      <WizardStepper steps={STEP_LABELS} activeStep={activeStep} maxReachedStep={maxReachedStep} onSelect={goToStep} />

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
                <div className="flex items-center justify-between gap-3 mb-1">
                  <label htmlFor="vc-backbone" className="block text-sm text-muted-foreground">
                    Modèle pré-entraîné (transfer learning)
                  </label>
                  <Switch
                    checked={comparisonMode}
                    onChange={(v) => {
                      setComparisonMode(v);
                      if (v) setComparisonBackboneIds(new Set(backboneId ? [backboneId] : []));
                    }}
                    label="Comparer plusieurs modèles"
                  />
                </div>

                {!comparisonMode ? (
                  <>
                    <Select id="vc-backbone" value={backboneId} onChange={(e) => setBackboneId(e.target.value)}>
                      {backbones.map((b) => (
                        <option key={b.id} value={b.id}>
                          {b.label} — {SPEED_TIER_LABELS[b.speed_tier]} ({b.params_millions.toFixed(1)} M paramètres)
                        </option>
                      ))}
                    </Select>
                    {selectedBackbone && (
                      <div className="flex items-center gap-2 mt-1.5">
                        <Badge variant={SPEED_TIER_BADGE_VARIANT[selectedBackbone.speed_tier]}>
                          {SPEED_TIER_LABELS[selectedBackbone.speed_tier]}
                        </Badge>
                        <span className="text-xs text-muted-foreground">
                          {selectedBackbone.params_millions.toFixed(1)} M paramètres
                        </span>
                      </div>
                    )}
                    <p className="text-xs text-muted-foreground mt-1">
                      Un modèle plus léger (MobileNet) entraîne plus vite ; un modèle plus profond (ResNet,
                      EfficientNet, DenseNet) peut être plus précis sur un dataset plus riche, au prix d'un
                      entraînement plus long. Aucun GPU n'est disponible en production aujourd'hui — tout
                      entraînement tourne sur CPU ; un modèle "Lent" en profitera le plus le jour où un GPU
                      dédié sera mis en place.
                    </p>
                  </>
                ) : (
                  <div>
                    <p className="text-xs text-muted-foreground mb-2">
                      Chaque modèle coché sera entraîné avec les mêmes réglages, puis le meilleur sur la
                      validation sera automatiquement retenu — jusqu'à {MAX_BACKBONES_PER_COMPARISON} modèles
                      ({comparisonBackboneIds.size}/{MAX_BACKBONES_PER_COMPARISON} sélectionnés). L'entraînement
                      prendra environ {comparisonBackboneIds.size || 1}× plus longtemps qu'un seul modèle.
                    </p>
                    <div className="space-y-1.5">
                      {backbones.map((b) => {
                        const checked = comparisonBackboneIds.has(b.id);
                        const disabled = !checked && comparisonBackboneIds.size >= MAX_BACKBONES_PER_COMPARISON;
                        return (
                          <label
                            key={b.id}
                            className={`flex items-center gap-2 rounded-lg border px-2.5 py-2 text-sm cursor-pointer transition-colors ${
                              checked ? "border-primary/40 bg-primary/10 text-primary" : "border-border text-foreground/90"
                            } ${disabled ? "opacity-50 cursor-not-allowed" : ""}`}
                          >
                            <input
                              type="checkbox"
                              className="accent-primary"
                              checked={checked}
                              disabled={disabled}
                              onChange={() => toggleComparisonBackbone(b.id)}
                            />
                            <span className="flex-1">{b.label}</span>
                            <Badge variant={SPEED_TIER_BADGE_VARIANT[b.speed_tier]}>
                              {SPEED_TIER_LABELS[b.speed_tier]}
                            </Badge>
                          </label>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            )}

            <ImageSizePicker value={imageSize} onChange={setImageSize} defaultValue={224} />

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

            <AugmentationPreviewGallery datasetId={datasetId} preset={augmentationPreset} imageSize={imageSize} />
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

                <div>
                  <label htmlFor="vc-weight-decay" className="block text-sm text-muted-foreground mb-1">
                    <LabelWithHelp
                      label={`Régularisation L2 (weight decay) — ${weightDecay}`}
                      help="Pénalise les poids trop grands pendant l'entraînement, en plus du dropout — un second levier contre le sur-apprentissage. 0 = désactivée (comportement historique)."
                    />
                  </label>
                  <input
                    id="vc-weight-decay"
                    type="range"
                    min={0}
                    max={0.01}
                    step={0.0005}
                    value={weightDecay}
                    onChange={(e) => setWeightDecay(Number(e.target.value))}
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
      <p className={`text-xl font-bold tabular-nums ${accentValueTextClass(color)}`}>{value}</p>
    </div>
  );
}

// Mode expert : classement des backbones comparés (retour utilisateur direct
// — parité avec le classement multi-modèles du ML tabulaire) — n'apparaît
// que sur un job lancé avec `backbone_ids` (≥ 2 entrées).
const BACKBONE_CANDIDATE_COLUMNS: TableColumn<BackboneComparisonCandidate>[] = [
  { key: "backbone_label", header: "Backbone", render: (c) => c.backbone_label },
  {
    key: "best_val_accuracy",
    header: "Exactitude (validation)",
    render: (c) => `${(c.best_val_accuracy * 100).toFixed(1)} %`,
    sortValue: (c) => c.best_val_accuracy,
  },
  {
    key: "best_val_loss",
    header: "Perte (validation)",
    render: (c) => c.best_val_loss.toFixed(4),
    sortValue: (c) => c.best_val_loss,
  },
  {
    key: "test_accuracy",
    header: "Exactitude (test)",
    render: (c) => `${(c.test_accuracy * 100).toFixed(1)} %`,
    sortValue: (c) => c.test_accuracy,
  },
  { key: "num_epochs_run", header: "Époques", render: (c) => String(c.num_epochs_run) },
  {
    key: "training_seconds",
    header: "Durée",
    render: (c) => `${Math.round(c.training_seconds)} s${c.time_capped ? " (plafonnée)" : ""}`,
  },
];

function BackboneComparisonCard({ candidates }: { candidates: BackboneComparisonCandidate[] }) {
  return (
    <Card className={`p-5 ${accentSurfaceClass("amber")}`}>
      <SectionHeader
        icon={Trophy}
        color="amber"
        label={`Comparatif de backbones (${candidates.length})`}
        help="Chaque backbone a été entraîné avec les mêmes réglages. Le retenu (surligné) est celui dont la meilleure époque généralise le mieux — perte de validation minimale, jamais le score de test (réservé à l'évaluation finale, pas au choix entre candidats)."
      />
      <Table
        columns={BACKBONE_CANDIDATE_COLUMNS}
        rows={candidates}
        rowKey={(c) => c.backbone_id}
        highlightRow={(c) => c.selected}
      />
    </Card>
  );
}

function ClassificationResultView({
  jobId,
  datasetId,
  datasetName,
}: {
  jobId: number;
  datasetId: number;
  datasetName: string | null;
}) {
  const [result, setResult] = useState<VisionClassificationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"performance" | "exemples" | "fiabilite" | "gradcam">("performance");

  // Sélection multiple pour Grad-CAM en lot (retour utilisateur direct :
  // "Grad-CAM devrait supporter le batch, pas une image à la fois") — vit
  // ici (au-dessus de l'onglet "Exemples") plutôt que dans ExampleGrid, la
  // sélection doit survivre en passant de "Exemples" à "Grad-CAM".
  const [selectedForExplain, setSelectedForExplain] = useState<Set<string>>(new Set());
  const [batchExplainResults, setBatchExplainResults] = useState<GradCamBatchItem[] | null>(null);
  const [batchExplainLoading, setBatchExplainLoading] = useState(false);
  const [batchExplainError, setBatchExplainError] = useState<string | null>(null);

  useEffect(() => {
    api.visionClassification
      .getResult(jobId)
      .then(setResult)
      .catch((err) => setError(err instanceof ApiError ? err.message : "Résultat indisponible"));
  }, [jobId]);

  function toggleExplainSelection(path: string) {
    setSelectedForExplain((prev) => {
      const next = new Set(prev);
      if (next.has(path)) {
        next.delete(path);
      } else if (next.size < MAX_EXPLAIN_BATCH_SIZE) {
        next.add(path);
      }
      return next;
    });
  }

  async function handleExplainSelected() {
    setBatchExplainLoading(true);
    setBatchExplainError(null);
    try {
      const res = await api.visionClassification.explainDatasetExamples(jobId, Array.from(selectedForExplain));
      setBatchExplainResults(res.results);
      setSelectedForExplain(new Set());
      setActiveTab("gradcam");
    } catch (err) {
      setBatchExplainError(err instanceof ApiError ? err.message : "Impossible de générer les explications");
    } finally {
      setBatchExplainLoading(false);
    }
  }

  if (error) return <p className="text-sm text-destructive text-center">{error}</p>;
  if (!result) return <p className="text-sm text-muted-foreground text-center">Chargement…</p>;

  const historyData = result.history.map((h) => ({
    epoch: h.epoch + 1,
    "Perte (train)": h.train_loss,
    "Perte (validation)": h.val_loss,
    "Exactitude (train)": h.train_accuracy,
    "Exactitude (validation)": h.val_accuracy,
  }));

  // Fiche modèle (retour utilisateur direct : "on peut télécharger le
  // modèle mais pas un json... qui suit le modèle") — construite
  // ENTIÈREMENT à partir de `result`, déjà en mémoire, jamais un second
  // appel réseau. Voir `utils/visionClassificationModelCard.ts`.
  function handleExportModelCard() {
    // Garde défensive pure — le rétrécissement de type effectué plus haut
    // (`if (!result) return`) ne traverse pas la fermeture de cette
    // fonction imbriquée ; `result` y reste `VisionClassificationResult |
    // null` du point de vue de TypeScript.
    if (!result) return;
    const card = buildVisionClassificationModelCard(datasetName, result);
    const blob = new Blob([JSON.stringify(card, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `vision_classification_fiche_modele_job${jobId}.json`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

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

      {Array.isArray(result.model_card.candidates) && result.model_card.candidates.length > 1 && (
        <BackboneComparisonCard candidates={result.model_card.candidates as BackboneComparisonCandidate[]} />
      )}

      <div className="flex items-center gap-2 flex-wrap">
        <ModelExportActions
          onExportArtifact={() => api.visionClassification.exportModel(jobId)}
          exportConfig={{
            class_names: result.class_names,
            test_accuracy: result.test_accuracy,
            test_precision_macro: result.test_precision_macro,
            test_recall_macro: result.test_recall_macro,
            test_f1_macro: result.test_f1_macro,
            model_card: result.model_card,
          }}
          configFilename={`vision_classification_config_job${jobId}.json`}
        />
        <Button variant="secondary" size="sm" onClick={handleExportModelCard}>
          <FileJson size={14} />
          Fiche modèle (JSON)
        </Button>
        <Button variant="secondary" size="sm" onClick={() => api.visionClassification.exportDeploymentScript(jobId)}>
          <FileCode size={14} />
          Script de déploiement (.py)
        </Button>
      </div>
      <p className="text-xs text-muted-foreground -mt-2">
        Pour déployer ce modèle en dehors de DataLab Pro : téléchargez l'artefact ET le script de déploiement,
        placez-les dans le même dossier — le script reconstruit l'architecture du réseau et prédit sur de nouvelles
        images, sans dépendre de cette plateforme (voir l'en-tête du script pour l'installation des bibliothèques
        nécessaires).
      </p>

      <Tabs
        items={[
          { id: "performance" as const, label: "Performance", icon: Sparkles },
          { id: "exemples" as const, label: "Exemples", icon: Target },
          { id: "fiabilite" as const, label: "Fiabilité", icon: Activity },
          { id: "gradcam" as const, label: "Grad-CAM", icon: AlertCircle },
        ]}
        active={activeTab}
        onChange={setActiveTab}
        urlParam="onglet"
      />

      {activeTab === "performance" && (
        <>
          <Card className="p-5">
            <SectionHeader icon={Sparkles} color="blue" label="Courbes d'apprentissage" />
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={historyData} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
                <CartesianGrid stroke={CHART_GRID_STROKE} vertical={false} />
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
        </>
      )}

      {activeTab === "exemples" && (
        <>
          <p className="text-xs text-muted-foreground">
            Cochez jusqu'à {MAX_EXPLAIN_BATCH_SIZE} images pour les expliquer d'un coup (Grad-CAM) — utile pour
            comparer plusieurs erreurs sans les ré-uploader une par une.
          </p>
          {incorrectExamples.length > 0 && (
            <div>
              <SectionHeader icon={AlertCircle} color="rose" label={`Erreurs de classification (${incorrectExamples.length})`} />
              <ExampleGrid
                examples={incorrectExamples}
                datasetId={datasetId}
                selectedForExplain={selectedForExplain}
                onToggleExplainSelection={toggleExplainSelection}
              />
            </div>
          )}

          {correctExamples.length > 0 && (
            <div>
              <SectionHeader icon={Target} color="teal" label={`Exemples corrects (${correctExamples.length})`} />
              <ExampleGrid
                examples={correctExamples}
                datasetId={datasetId}
                selectedForExplain={selectedForExplain}
                onToggleExplainSelection={toggleExplainSelection}
              />
            </div>
          )}

          <BulkActionBar count={selectedForExplain.size} onClear={() => setSelectedForExplain(new Set())}>
            <Button size="sm" onClick={handleExplainSelected} loading={batchExplainLoading}>
              <Wand2 size={14} />
              Expliquer (Grad-CAM)
            </Button>
          </BulkActionBar>
          {batchExplainError && <p className="text-xs text-destructive">{batchExplainError}</p>}
        </>
      )}

      {activeTab === "fiabilite" && <ReliabilityTab result={result} />}

      {activeTab === "gradcam" && (
        <GradCamPanel jobId={jobId} batchResults={batchExplainResults} onClearBatch={() => setBatchExplainResults(null)} />
      )}
    </div>
  );
}

/** Onglet "Fiabilité" (retour utilisateur : "d'autres fonctionnalités
 * modernes que les autres plateformes n'offrent pas") — le modèle est-il
 * "sûr à raison" ? Une confiance de 90 % qui n'est vraie que 60 % du temps
 * est un vrai risque en production, invisible dans la seule exactitude déjà
 * affichée dans l'onglet Performance. Réutilise CalibrationChart.tsx tel
 * quel (déjà validé côté tabulaire, `ModelResultModal.tsx`) — jamais un
 * second composant de graphique de calibration à maintenir en parallèle. */
function ReliabilityTab({ result }: { result: VisionClassificationResult }) {
  const status = result.model_card.calibration_status;
  const isEmpty = !result.calibration || Object.keys(result.calibration).length === 0;
  const isDegraded = typeof status === "object" && status !== null && "status" in status && status.status === "degraded";
  const degradedMessage =
    isDegraded && "message" in status && typeof status.message === "string" ? status.message : null;

  return (
    <Card className="p-5">
      <SectionHeader
        icon={Activity}
        color="teal"
        label="Courbe de calibration"
        help="Compare la probabilité annoncée par le modèle à la fréquence réellement observée sur le jeu de test — le modèle est-il « sûr à raison » ? Un point sur la diagonale signifie une confiance fiable."
      />
      {status === undefined || (!isDegraded && isEmpty) ? (
        <p className="text-xs text-muted-foreground italic">Non disponible pour ce modèle — réentraînez-le pour l'obtenir.</p>
      ) : isDegraded ? (
        <p className="text-xs text-muted-foreground italic">{degradedMessage ?? "Calcul indisponible pour ce modèle."}</p>
      ) : (
        <CalibrationChart calibration={result.calibration ?? {}} />
      )}
    </Card>
  );
}

function ExampleGrid({
  examples,
  datasetId,
  selectedForExplain,
  onToggleExplainSelection,
}: {
  examples: VisionPredictionExample[];
  datasetId: number;
  selectedForExplain: Set<string>;
  onToggleExplainSelection: (path: string) => void;
}) {
  return (
    <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 gap-3">
      {examples.map((example) => {
        const checked = selectedForExplain.has(example.relative_path);
        return (
          <div key={example.relative_path} className="space-y-1">
            <label className="relative block cursor-pointer">
              <VisionImage
                datasetId={datasetId}
                path={example.relative_path}
                alt={example.relative_path}
                className={`w-full aspect-square object-cover rounded-lg border-2 transition-colors ${
                  checked ? "border-primary" : example.correct ? "border-success/40" : "border-destructive/40"
                }`}
              />
              <input
                type="checkbox"
                checked={checked}
                onChange={() => onToggleExplainSelection(example.relative_path)}
                aria-label={`Sélectionner ${example.relative_path} pour Grad-CAM`}
                className="absolute top-1.5 left-1.5 accent-primary h-4 w-4 rounded shadow"
              />
            </label>
            <p className="text-caption text-muted-foreground truncate" title={example.relative_path}>
              {example.correct ? example.predicted_label : `${example.true_label} → ${example.predicted_label}`}
            </p>
          </div>
        );
      })}
    </div>
  );
}

/** Légende de la palette "jet" (bleu=faible/rouge=fort — couleurs fixes
 * volontaires, Lot 8 : reproduit exactement la palette appliquée côté
 * serveur, `backend/domains/vision/localization.py::_apply_colormap`,
 * jamais les jetons de thème, qui rendraient cette légende fausse) —
 * extraite pour être partagée par l'explication unique (upload) ET la
 * grille du batch, jamais dupliquée. */
function GradCamColorLegend({ targetLabel }: { targetLabel: string }) {
  return (
    <div className="flex items-center gap-2 mt-2">
      <span
        className="inline-block h-2 w-16 rounded-full flex-shrink-0"
        style={{ background: "linear-gradient(to right, #0000cc, #00cc66, #cc0000)" }}
        aria-hidden="true"
      />
      <p className="text-xs text-muted-foreground">
        Bleu = faible influence · Rouge = zones qui ont le plus influencé la classe "{targetLabel}"
      </p>
    </div>
  );
}

function GradCamPanel({
  jobId,
  batchResults,
  onClearBatch,
}: {
  jobId: number;
  /** Résultats du lot lancé depuis l'onglet "Exemples" (retour utilisateur
   * direct : "Grad-CAM devrait supporter le batch") — `null` tant qu'aucun
   * lot n'a été demandé, indépendant de l'explication par upload ci-dessous
   * (les deux peuvent coexister). */
  batchResults: GradCamBatchItem[] | null;
  onClearBatch: () => void;
}) {
  const [explanation, setExplanation] = useState<GradCamExplanation | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  // Résultats d'un upload de PLUSIEURS images externes à la fois (retour
  // utilisateur direct : "possible d'expliquer plusieurs images en même
  // temps ?") — distinct de `batchResults` (prop, images déjà dans le
  // dataset) même si l'affichage réutilise le même motif de grille.
  const [uploadedBatchResults, setUploadedBatchResults] = useState<GradCamBatchItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  async function handleFile(files: FileList | null) {
    const selected = Array.from(files ?? []);
    if (selected.length === 0) return;
    if (selected.length > MAX_EXPLAIN_BATCH_SIZE) {
      setError(`${MAX_EXPLAIN_BATCH_SIZE} images maximum par lot — sélectionnez-en moins.`);
      return;
    }
    setError(null);
    setExplanation(null);
    setUploadedBatchResults(null);
    setIsSubmitting(true);
    try {
      if (selected.length === 1) {
        setPreviewUrl(URL.createObjectURL(selected[0]));
        setExplanation(await api.visionClassification.explain(jobId, selected[0]));
      } else {
        setPreviewUrl(null);
        const res = await api.visionClassification.explainBatch(jobId, selected);
        setUploadedBatchResults(res.results);
      }
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Impossible de générer l'explication");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className="space-y-5">
      {batchResults && (
        <Card className="p-5">
          <div className="flex items-start justify-between gap-3">
            <SectionHeader
              icon={Wand2}
              color="violet"
              label={`Explications en lot (${batchResults.length})`}
              help="Images sélectionnées depuis l'onglet « Exemples » — même modèle chargé une seule fois pour tout le lot."
            />
            <Button variant="secondary" size="sm" onClick={onClearBatch}>
              Effacer
            </Button>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
            {batchResults.map((item) => (
              <div key={item.relative_path} className="space-y-1.5">
                {item.error ? (
                  <div className="aspect-square rounded-lg border border-destructive/30 bg-destructive/5 flex items-center justify-center p-2">
                    <p className="text-xs text-destructive text-center">{item.error}</p>
                  </div>
                ) : (
                  <img
                    src={item.heatmap_png ?? undefined}
                    alt={`Grad-CAM pour ${item.relative_path}`}
                    className="w-full aspect-square object-cover rounded-lg border border-border"
                  />
                )}
                <p className="text-caption text-muted-foreground truncate" title={item.relative_path}>
                  {item.relative_path}
                </p>
                {item.predicted_label && <Badge variant="primary">{item.predicted_label}</Badge>}
              </div>
            ))}
          </div>
          {batchResults.some((item) => !item.error) && <GradCamColorLegend targetLabel="classe prédite (par image)" />}
        </Card>
      )}

      <Card className="p-5">
        <SectionHeader
          icon={Sparkles}
          color="amber"
          label="Expliquer une ou plusieurs images externes"
          help={`Superpose une carte de chaleur sur l'image : les zones les plus chaudes sont celles qui ont le plus influencé la classe prédite par le modèle. Sélectionnez plusieurs fichiers à la fois pour les expliquer d'un coup (jusqu'à ${MAX_EXPLAIN_BATCH_SIZE}). Pour expliquer des images déjà dans le dataset, sélectionnez-les depuis l'onglet « Exemples ».`}
        />
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          multiple
          className="hidden"
          onChange={(e) => handleFile(e.target.files)}
        />
        <Button variant="secondary" size="sm" type="button" onClick={() => fileInputRef.current?.click()}>
          Choisir une ou plusieurs images à expliquer
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
              <GradCamColorLegend targetLabel={explanation.target_label} />
            </div>
          </div>
        )}

        {uploadedBatchResults && (
          <div className="mt-4 space-y-3">
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
              {uploadedBatchResults.map((item, i) => (
                <div key={`${item.relative_path}-${i}`} className="space-y-1.5">
                  {item.error ? (
                    <div className="aspect-square rounded-lg border border-destructive/30 bg-destructive/5 flex items-center justify-center p-2">
                      <p className="text-xs text-destructive text-center">{item.error}</p>
                    </div>
                  ) : (
                    <img
                      src={item.heatmap_png ?? undefined}
                      alt={`Grad-CAM pour ${item.relative_path}`}
                      className="w-full aspect-square object-cover rounded-lg border border-border"
                    />
                  )}
                  <p className="text-caption text-muted-foreground truncate" title={item.relative_path}>
                    {item.relative_path}
                  </p>
                  {item.predicted_label && <Badge variant="primary">{item.predicted_label}</Badge>}
                </div>
              ))}
            </div>
            {uploadedBatchResults.some((item) => !item.error) && (
              <GradCamColorLegend targetLabel="classe prédite (par image)" />
            )}
          </div>
        )}
      </Card>
    </div>
  );
}
