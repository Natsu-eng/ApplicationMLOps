import { useEffect, useRef, useState, type FormEvent } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { AlertCircle, Boxes, Loader2, PlayCircle, Sparkles, Target, Trash2 } from "lucide-react";
import { Line, LineChart, CartesianGrid, ResponsiveContainer, Tooltip as RechartsTooltip, XAxis, YAxis } from "recharts";
import {
  ApiError,
  api,
  type GradCamExplanation,
  type VisionBackbone,
  type VisionClassificationJobSummary,
  type VisionClassificationResult,
  type VisionPredictionExample,
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
import { Switch } from "../components/ui/Switch";
import { VisionDatasetPicker } from "../components/vision/VisionDatasetPicker";
import { VisionImage } from "../components/vision/VisionImage";
import { useConfirmAction } from "../hooks/useConfirmAction";
import { CHART_GRID_STROKE, CHART_SERIES_COLORS, CHART_TICK_STYLE_SM, CHART_TOOLTIP_STYLE } from "../theme/charts";

const ACTIVE_STATUSES = new Set(["queued", "running"]);
const POLL_INTERVAL_MS = 3000;
const ACTIVE_JOB_STORAGE_KEY = "datalab_active_vision_classification_job_id";

type Phase = "configure" | "progress" | "results" | "failed";

function phaseOf(job: VisionClassificationJobSummary | null): Phase {
  if (!job) return "configure";
  if (ACTIVE_STATUSES.has(job.status)) return "progress";
  return job.status === "completed" ? "results" : "failed";
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

  useEffect(() => {
    if (phase !== "progress" || !activeJob) return;
    const interval = setInterval(async () => {
      try {
        setActiveJob(await api.visionClassification.getJob(activeJob.id));
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
      await api.visionClassification.remove(activeJob.id);
    } catch {
      // best-effort — on repart en configuration dans tous les cas
    }
    resetToConfigure();
  }

  const titles: Record<Phase, string> = {
    configure: "Classifier des images",
    progress: "Entraînement en cours",
    results: "Résultats de la classification",
    failed: "Échec de l'entraînement",
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
          <ClassificationForm onJobCreated={openJob} />
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
  const [backbones, setBackbones] = useState<VisionBackbone[]>([]);
  const [backboneId, setBackboneId] = useState("");
  const [numEpochs, setNumEpochs] = useState(8);
  const [freezeBackbone, setFreezeBackbone] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    api.visionClassification.backbones().then((list) => {
      setBackbones(list);
      if (list.length > 0) setBackboneId(list[0].id);
    });
  }, []);

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
        freeze_backbone: freezeBackbone,
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
          <label className="block text-sm text-muted-foreground mb-1.5">Dataset d'images</label>
          <VisionDatasetPicker structureType="classification" value={datasetId} onChange={(id) => setDatasetId(id)} />
        </div>

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
          </div>
        )}

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
            <p className="text-sm text-foreground">Geler le tronc pré-entraîné</p>
            <p className="text-xs text-muted-foreground">
              Recommandé — seule la tête de classification est entraînée, plus rapide et plus fiable avec peu
              d'images.
            </p>
          </div>
          <Switch checked={freezeBackbone} onChange={setFreezeBackbone} label="Geler le tronc pré-entraîné" />
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

      <Card className="p-5">
        <SectionHeader icon={Boxes} color="teal" label="Matrice de confusion" help="Lignes = classe réelle, colonnes = classe prédite." />
        <Heatmap xLabels={result.class_names} yLabels={result.class_names} matrix={result.confusion_matrix} variant="sequential" />
      </Card>

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
