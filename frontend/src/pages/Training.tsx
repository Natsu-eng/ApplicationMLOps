import { useCallback, useEffect, useRef, useState, type FormEvent, type ReactNode } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { AlertCircle, Ban, BrainCircuit, ChevronLeft, ChevronRight, Loader2, PlayCircle, RotateCcw, Target, Trash2 } from "lucide-react";
import {
  ApiError,
  api,
  type ColumnSchema,
  type DatasetSummary,
  type DurationEstimate,
  type FeatureEngineeringSpec,
  type HyperparameterOverrides,
  type TargetSuggestion,
  type TrainingJobSummary,
} from "../api/client";
import AppShell from "../components/AppShell";
import { pillarColor } from "../config/pillars";
import { ClassRebalancingSuggestion } from "../components/training/ClassRebalancingSuggestion";
import { DataQualityWarnings, EXCLUDABLE_CODES, excludableColumnsOf } from "../components/training/DataQualityWarnings";
import {
  DEFAULT_CQR_ALPHA,
  DEFAULT_CV_FOLDS,
  DEFAULT_SEED,
  ExpertModePanel,
} from "../components/training/ExpertModePanel";
import { FeatureEngineeringSuggestions } from "../components/training/FeatureEngineeringSuggestions";
import { ModelResultView } from "../components/training/ModelResultModal";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { PageHeader } from "../components/ui/PageHeader";
import { Select } from "../components/ui/Select";
import { WizardStepper } from "../components/ui/WizardStepper";
import { useConfirmAction } from "../hooks/useConfirmAction";
import { useIdempotencyKey } from "../hooks/useIdempotencyKey";
import { formatDateTime, formatDuration } from "../utils/format";
import { useJobEvents } from "../hooks/useJobEvents";
import { buildTrainingJobPayload } from "../utils/trainingPayload";

const DEFAULT_OPTUNA_TRIALS = 20; // `api.core.config.Settings.optuna_trials_default`

const ACTIVE_STATUSES = new Set(["queued", "running"]);
const ACTIVE_JOB_STORAGE_KEY = "datalab_active_training_job_id";
// Estimation de durée calculée AVANT le lancement (étape 5 du wizard,
// `api.training.estimateDuration`) — persistée à travers le job pour rester
// affichable pendant la progression (Lot 6, Progression.html : "Fin
// estimée"). Jamais recalculée pendant l'exécution : une seule estimation,
// dérivée de l'historique réel de l'organisation, pas une fausse mise à
// jour en direct qu'on ne sait pas produire.
const ACTIVE_JOB_ETA_STORAGE_KEY = "datalab_active_training_job_eta_seconds";

/** Étapes du wizard horizontal (refonte UI) — même contenu/ordre que le
 * pipeline guidé existant (Lot E1-ter), rebaptisées pour tenir dans une
 * pastille compacte. */
const STEP_LABELS = [
  { number: 1, label: "Données & cible" },
  { number: 2, label: "Qualité des données" },
  { number: 3, label: "Améliorations" },
  { number: 4, label: "Mode expert" },
  { number: 5, label: "Lancement" },
];

type Phase = "configure" | "progress" | "results" | "failed" | "cancelled";

function phaseOf(job: TrainingJobSummary | null): Phase {
  if (!job) return "configure";
  if (ACTIVE_STATUSES.has(job.status)) return "progress";
  if (job.status === "completed") return "results";
  return job.status === "cancelled" ? "cancelled" : "failed";
}

/** Page dédiée à l'entraînement (Lot E1-ter, sur demande explicite) : pas
 * d'historique à côté — configurer, lancer, puis voir le résultat EN PLACE
 * sur cette même page. L'historique complet des entraînements passés vit
 * sur le tableau de bord ("Derniers entraînements"), pas ici. */
export default function Training() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [datasetsError, setDatasetsError] = useState<string | null>(null);
  const [activeJob, setActiveJob] = useState<TrainingJobSummary | null>(null);
  const [etaSeconds, setEtaSeconds] = useState<number | null>(null);
  const [restoringJob, setRestoringJob] = useState(true);
  const confirmDelete = useConfirmAction<true>();

  // Persistance de l'entraînement actif à travers un rafraîchissement de
  // page (sessionStorage) — signalé comme "comportement ambigu à clarifier
  // avant d'y toucher" dans backend/workflow.md, jamais traité depuis.
  // Avant ce correctif : rafraîchir pendant qu'un job tournait réellement
  // côté serveur (RQ/worker) faisait perdre tout l'état React, renvoyait
  // silencieusement au formulaire de configuration comme si de rien
  // n'était, alors que l'entraînement continuait en tâche de fond —
  // aucun moyen de retrouver sa progression sans passer par le tableau de
  // bord. sessionStorage plutôt que localStorage : le job actif ne doit
  // resurgir que dans CETTE session de navigation (cet onglet, jusqu'à sa
  // fermeture), jamais des jours plus tard dans un nouvel onglet.
  //
  // Deep-link `?job=` (Lot 7, §J.2) — priorité sur sessionStorage, même
  // ordre que les 5 autres pages de job : permet à l'historique unifié de
  // rouvrir un entraînement précis, y compris depuis un lien partagé.
  useEffect(() => {
    const queryJobId = searchParams.get("job");
    const storedId = queryJobId ?? sessionStorage.getItem(ACTIVE_JOB_STORAGE_KEY);
    if (!storedId) {
      setRestoringJob(false);
      return;
    }
    api.training
      .getJob(Number(storedId))
      .then((job) => {
        setActiveJob(job);
        const storedEta = sessionStorage.getItem(ACTIVE_JOB_ETA_STORAGE_KEY);
        setEtaSeconds(storedEta ? Number(storedEta) : null);
      })
      .catch(() => {
        sessionStorage.removeItem(ACTIVE_JOB_STORAGE_KEY);
        sessionStorage.removeItem(ACTIVE_JOB_ETA_STORAGE_KEY);
        setSearchParams({}, { replace: true });
      })
      .finally(() => setRestoringJob(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (activeJob) {
      sessionStorage.setItem(ACTIVE_JOB_STORAGE_KEY, String(activeJob.id));
    } else {
      sessionStorage.removeItem(ACTIVE_JOB_STORAGE_KEY);
    }
  }, [activeJob]);

  function openJob(job: TrainingJobSummary, estimatedSeconds: number | null = null) {
    setActiveJob(job);
    setEtaSeconds(estimatedSeconds);
    if (estimatedSeconds !== null) {
      sessionStorage.setItem(ACTIVE_JOB_ETA_STORAGE_KEY, String(estimatedSeconds));
    } else {
      sessionStorage.removeItem(ACTIVE_JOB_ETA_STORAGE_KEY);
    }
    setSearchParams({ job: String(job.id) }, { replace: false });
  }

  const loadDatasets = useCallback(async () => {
    try {
      const all = await api.datasets.list();
      setDatasets(all.filter((d) => d.status === "ready"));
      setDatasetsError(null);
    } catch (err) {
      // AUDIT_ROADMAP.md, H4/D3 : avant ce correctif, un échec réseau ici
      // était indiscernable de "vous n'avez encore aucun dataset" — les deux
      // affichaient le même formulaire vide, sans indice pour l'utilisateur.
      setDatasetsError(err instanceof ApiError ? err.message : "Impossible de charger vos datasets");
    }
  }, []);

  useEffect(() => {
    loadDatasets();
  }, [loadDatasets]);

  const phase = phaseOf(activeJob);

  // Notifications SSE (Lot 7, §J.2) — remplace le polling setInterval : la
  // page bascule d'elle-même vers la vue résultat (ou échec) dès que le
  // flux serveur pousse un changement de statut.
  useJobEvents(
    phase === "progress" && activeJob ? `/training/jobs/${activeJob.id}/events` : null,
    (snapshot) => setActiveJob((prev) => (prev ? { ...prev, ...snapshot } : prev)),
  );

  function resetToConfigure() {
    setActiveJob(null);
    setEtaSeconds(null);
    sessionStorage.removeItem(ACTIVE_JOB_ETA_STORAGE_KEY);
    setSearchParams({}, { replace: false });
  }

  async function handleDeleteActiveJob() {
    if (!activeJob) return;
    try {
      await api.training.remove(activeJob.id);
    } catch {
      // best-effort — on repart en configuration dans tous les cas
    }
    resetToConfigure();
  }

  async function handleCancelActiveJob() {
    if (!activeJob) return;
    try {
      setActiveJob(await api.training.cancel(activeJob.id));
    } catch {
      // le prochain poll (ou un rafraîchissement) reflétera l'état réel
    }
  }

  const [rerunning, setRerunning] = useState(false);
  const [rerunError, setRerunError] = useState<string | null>(null);
  async function handleRerunActiveJob() {
    if (!activeJob) return;
    setRerunning(true);
    setRerunError(null);
    try {
      openJob(await api.training.rerun(activeJob.id));
    } catch (err) {
      setRerunError(err instanceof ApiError ? err.message : "Impossible de relancer cet entraînement");
    } finally {
      setRerunning(false);
    }
  }

  const titles: Record<Phase, string> = {
    configure: "Entraîner un modèle",
    progress: "Entraînement en cours",
    results: "Résultat de l'entraînement",
    failed: "Échec de l'entraînement",
    cancelled: "Entraînement annulé",
  };

  return (
    <AppShell pillarId="supervised">
      <PageHeader
        eyebrow="Entraînement"
        title={titles[phase]}
        description={
          phase === "configure" ? "Objectif : prédire une valeur ou une catégorie. Nous vous guidons pas à pas." : undefined
        }
        icon={BrainCircuit}
        color={pillarColor("supervised")}
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
        <p className="text-sm text-destructive bg-destructive/10 border border-destructive/20 rounded-lg px-3 py-2 max-w-xl mx-auto mb-4">
          {rerunError}
        </p>
      )}

      {restoringJob ? (
        <div className="flex items-center justify-center py-16 text-sm text-muted-foreground gap-2">
          <Loader2 size={16} className="animate-spin" />
          Reprise de votre session…
        </div>
      ) : phase === "configure" && (
        <div className="max-w-3xl mx-auto">
          <TrainingForm datasets={datasets} datasetsError={datasetsError} onJobCreated={openJob} />
          <p className="text-xs text-muted-foreground text-center mt-4">
            Vos entraînements précédents restent consultables depuis le{" "}
            <Link to="/dashboard" className="text-primary underline underline-offset-2 hover:text-primary/80">
              tableau de bord
            </Link>
            .
          </p>
        </div>
      )}

      {phase === "progress" && activeJob && (
        <TrainingProgress job={activeJob} etaSeconds={etaSeconds} onCancel={handleCancelActiveJob} />
      )}

      {(phase === "failed" || phase === "cancelled") && activeJob && <TrainingFailed job={activeJob} />}

      {phase === "results" && activeJob && (
        <div className="max-w-4xl mx-auto">
          <ModelResultView job={activeJob} />
        </div>
      )}
    </AppShell>
  );
}

interface ProgressJournalEntry {
  time: string;
  step: string;
  percent: number;
}

/** Journal de progression (Lot 6, Progression.html) — reconstruit CÔTÉ
 * CLIENT à partir des vrais évènements SSE déjà reçus (`job.progress_step`/
 * `progress_percent`, voir `useJobEvents` et
 * `backend/domains/training/worker.py::_make_progress_callback`) : chaque
 * transition distincte de `progress_step` devient une ligne horodatée. Pas
 * de nouvel endpoint ni de stockage serveur — uniquement l'historique des
 * évènements que cette page a déjà reçus pendant qu'elle était ouverte
 * (rouvrir la page après un rafraîchissement repart avec un journal vide,
 * seul l'état COURANT du job est repersisté). Le tableau détaillé par
 * modèle, le graphe de convergence Optuna et les jauges CPU/mémoire de la
 * maquette n'existent PAS ici : le backend ne renvoie qu'une seule chaîne de
 * progression globale (pas de score par modèle avant la fin du job, pas de
 * télémétrie ressources) — les fabriquer aurait affiché des nombres inventés,
 * contraire au principe déjà établi de ce code (jamais une UI qui a l'air
 * fonctionnelle sans l'être).
 */
function TrainingProgress({
  job,
  etaSeconds,
  onCancel,
}: {
  job: TrainingJobSummary;
  etaSeconds: number | null;
  onCancel: () => void;
}) {
  const [cancelling, setCancelling] = useState(false);
  const [now, setNow] = useState(() => Date.now());
  const [journal, setJournal] = useState<ProgressJournalEntry[]>([]);
  const lastStepRef = useRef<string | null>(null);

  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    if (!job.progress_step || job.progress_step === lastStepRef.current) return;
    lastStepRef.current = job.progress_step;
    setJournal((prev) => [
      ...prev,
      { time: new Date().toLocaleTimeString("fr-FR"), step: job.progress_step as string, percent: job.progress_percent },
    ]);
  }, [job.progress_step, job.progress_percent]);

  const startedAt = job.started_at ?? job.created_at;
  const remainingSeconds =
    etaSeconds !== null
      ? Math.max(0, etaSeconds - Math.floor((now - new Date(startedAt).getTime()) / 1000))
      : null;

  return (
    <Card className="max-w-2xl mx-auto p-8 text-center">
      <div className="mx-auto mb-4 h-14 w-14 rounded-2xl bg-primary/10 border border-primary/20 flex items-center justify-center">
        <Loader2 className="text-primary animate-spin" size={26} />
      </div>
      <h2 className="text-base font-medium text-foreground">
        {job.dataset_name ?? "Dataset"} <span className="text-muted-foreground">→</span> {job.target_column}
      </h2>
      <p className="text-xs text-muted-foreground mb-6">
        {job.task_type === "regression" ? "Régression" : "Classification"} · lancé {formatDateTime(job.created_at)}
      </p>
      <div className="h-2 rounded-full bg-muted overflow-hidden mb-2">
        <div
          className="h-full rounded-full bg-primary transition-all duration-500"
          style={{ width: `${Math.max(job.progress_percent, 4)}%` }}
        />
      </div>
      <div className="flex items-center justify-center gap-3 mb-3">
        <p className="num text-xs text-muted-foreground">{job.progress_percent}%</p>
        {remainingSeconds !== null && (
          <>
            <span className="text-muted-foreground/40">·</span>
            <p className="num text-xs text-muted-foreground">
              reste ≈ {formatDuration(remainingSeconds)}
            </p>
          </>
        )}
      </div>
      <p className="text-sm text-muted-foreground">
        {job.progress_step ?? "En attente d'un worker disponible…"}
      </p>
      <p className="text-xs text-muted-foreground mt-4 mb-4">
        La durée dépend de la taille du dataset et du nombre d'essais — cette page se met à jour
        automatiquement, vous pouvez aussi la quitter et revenir plus tard.
      </p>

      {journal.length > 0 && (
        <div className="text-left rounded-lg border border-border bg-muted p-3 max-h-40 overflow-y-auto mb-4">
          <p className="text-overline text-muted-foreground mb-2">Journal de cette session</p>
          <ul className="space-y-1">
            {journal.map((entry, i) => (
              <li key={i} className="num text-caption text-muted-foreground">
                <span className="text-muted-foreground/70">{entry.time}</span> — {entry.step}
              </li>
            ))}
          </ul>
        </div>
      )}

      <Button
        variant="ghost"
        size="sm"
        onClick={() => {
          setCancelling(true);
          onCancel();
        }}
        disabled={cancelling}
      >
        <Ban size={14} />
        {cancelling ? "Annulation…" : "Annuler cet entraînement"}
      </Button>
    </Card>
  );
}

function TrainingFailed({ job }: { job: TrainingJobSummary }) {
  const cancelled = job.status === "cancelled";
  return (
    <Card className="max-w-xl mx-auto p-8 text-center">
      <div
        className={`mx-auto mb-4 h-14 w-14 rounded-2xl border flex items-center justify-center ${
          cancelled ? "bg-muted border-border" : "bg-destructive/10 border-destructive/20"
        }`}
      >
        {cancelled ? (
          <Ban className="text-muted-foreground" size={26} />
        ) : (
          <AlertCircle className="text-destructive" size={26} />
        )}
      </div>
      <h2 className="text-base font-medium text-foreground">
        {job.dataset_name ?? "Dataset"} <span className="text-muted-foreground">→</span> {job.target_column}
      </h2>
      <p className="text-xs text-muted-foreground mb-4">
        {job.task_type === "regression" ? "Régression" : "Classification"} · lancé {formatDateTime(job.created_at)}
      </p>
      {job.error_message && (
        <p
          className={`text-sm rounded-lg px-3 py-2 text-left border ${
            cancelled
              ? "text-muted-foreground bg-muted border-border"
              : "text-destructive bg-destructive/10 border-destructive/20"
          }`}
        >
          {job.error_message}
        </p>
      )}
    </Card>
  );
}

function TrainingForm({
  datasets,
  datasetsError,
  onJobCreated,
}: {
  datasets: DatasetSummary[];
  datasetsError: string | null;
  onJobCreated: (job: TrainingJobSummary, estimatedSeconds?: number | null) => void;
}) {
  const [datasetId, setDatasetId] = useState<number | "">("");
  const [columns, setColumns] = useState<ColumnSchema[]>([]);
  const [targetColumn, setTargetColumn] = useState("");
  const [targetSuggestions, setTargetSuggestions] = useState<TargetSuggestion[]>([]);
  const [groupColumn, setGroupColumn] = useState("");
  const [selectedFeatures, setSelectedFeatures] = useState<Set<string>>(new Set());
  const [showFeaturePicker, setShowFeaturePicker] = useState(false);
  const [optunaTrials, setOptunaTrials] = useState(DEFAULT_OPTUNA_TRIALS);
  const [testSize, setTestSize] = useState(0.2);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Idempotence (Phase 2, AUDIT_BACKEND_2026-08-23.md §F4) — voir
  // hooks/useIdempotencyKey.ts.
  const idempotencyKey = useIdempotencyKey();
  const [featureEngineering, setFeatureEngineering] = useState<Pick<
    FeatureEngineeringSpec,
    "upstream" | "pipeline"
  > | null>(null);
  const [classRebalancing, setClassRebalancing] = useState(false);

  // Mode expert (Lot E2) — replié par défaut ; chaque manette démarre à la
  // même valeur que le mode guidé (voir ExpertModePanel), donc l'activer
  // sans rien changer ne modifie aucun résultat.
  const [expertMode, setExpertMode] = useState(false);
  const [cvFolds, setCvFolds] = useState(DEFAULT_CV_FOLDS);
  const [seed, setSeed] = useState(DEFAULT_SEED);
  const [cqrAlpha, setCqrAlpha] = useState(DEFAULT_CQR_ALPHA);
  const [selectedModelIds, setSelectedModelIds] = useState<Set<string>>(new Set());
  // Mode expert hyperparamètres (retour utilisateur direct : "laisser le
  // choix sur les hyperparamètres, profondeur des arbres etc.") — clé = id
  // du modèle, valeur = {nom_hyperparamètre: valeur fixée}. Vide par
  // défaut : comportement strictement inchangé (recherche automatique).
  const [hyperparameterOverrides, setHyperparameterOverrides] = useState<HyperparameterOverrides>({});
  const [durationEstimate, setDurationEstimate] = useState<DurationEstimate | null>(null);

  // Estimation de durée avant lancement (Lot 7, §J.1) — dérivée de
  // l'historique réel de l'organisation (services/duration_estimate.py),
  // jamais une constante inventée. Recalculée à chaque changement pertinent,
  // affichée seulement à l'étape récapitulative (voir plus bas).
  useEffect(() => {
    if (!datasetId) {
      setDurationEstimate(null);
      return;
    }
    const nModels = expertMode ? Math.max(1, selectedModelIds.size) : 4;
    api.training
      .estimateDuration(datasetId, nModels, optunaTrials, cvFolds)
      .then(setDurationEstimate)
      .catch(() => setDurationEstimate(null));
  }, [datasetId, expertMode, selectedModelIds.size, optunaTrials, cvFolds]);

  // Wizard horizontal (refonte UI) — une étape visible à la fois, navigable
  // par les pastilles ou Précédent/Continuer. `maxReachedStep` autorise à
  // revenir sur une étape déjà vue sans perdre sa progression, mais empêche
  // de sauter en avant sur une étape jamais atteinte.
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

  async function handleDatasetChange(id: string) {
    setError(null);
    setTargetColumn("");
    setGroupColumn("");
    setTargetSuggestions([]);
    if (!id) {
      setDatasetId("");
      setColumns([]);
      return;
    }
    const numericId = Number(id);
    setDatasetId(numericId);
    try {
      const detail = await api.datasets.get(numericId);
      setColumns(detail.columns);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Impossible de charger les colonnes");
    }
    // Suggestion de cible (Lot 7, §J.1) — best-effort, jamais bloquant :
    // une erreur ici ne doit jamais empêcher de choisir une cible à la main.
    api.datasets
      .targetSuggestions(numericId)
      .then((res) => setTargetSuggestions(res.suggestions))
      .catch(() => setTargetSuggestions([]));
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (!datasetId || !targetColumn) return;
    setError(null);
    setIsSubmitting(true);
    try {
      const job = await api.training.createJob(
        buildTrainingJobPayload({
          datasetId,
          targetColumn,
          featureColumns: Array.from(selectedFeatures),
          groupColumn,
          optunaTrials,
          cvFolds,
          testSize,
          seed,
          cqrAlpha,
          featureEngineering,
          classRebalancing,
          expertMode,
          selectedModelIds,
          hyperparameterOverrides,
        }),
        idempotencyKey.current,
      );
      idempotencyKey.reset(); // succès — la PROCHAINE soumission est une nouvelle tentative distincte
      onJobCreated(
        job,
        durationEstimate?.status === "estimated" ? durationEstimate.estimated_seconds : null,
      );
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Impossible de lancer l'entraînement");
    } finally {
      setIsSubmitting(false);
    }
  }

  const otherColumns = columns.filter((c) => c.name !== targetColumn);

  // Par défaut, toutes les variables sauf la cible et la colonne de groupe
  // (une colonne de groupe sert à identifier des échantillons répétés, pas
  // à prédire — l'inclure comme feature fuiterait l'identité du groupe).
  //
  // Retour utilisateur direct — diagnostic de cohérence du wizard,
  // objectif 1/4 : "l'étape 1 doit détecter PAR DÉFAUT les colonnes à
  // exclure (identifiants, quasi-constantes, doublons)", pas seulement
  // laisser l'utilisateur les découvrir à l'étape 2 puis revenir en
  // arrière. Dès que dataset+cible sont choisis, le même contrôle qualité
  // qu'à l'étape 2 tourne une fois en arrière-plan et pré-exclut les
  // colonnes SANS AUCUNE ambiguïté (`EXCLUDABLE_CODES` — colonne
  // constante/quasi-constante, cardinalité excessive proche d'un
  // identifiant, doublon exact) : la MÊME règle que le bouton "Exclure"
  // manuel de l'étape 2, jamais une nouvelle heuristique inventée ici.
  // Les vraies alertes qui exigent un jugement humain (fuite de cible,
  // déséquilibre des classes...) ne sont JAMAIS auto-exclues — seulement
  // signalées, l'utilisateur décide. Rien n'est verrouillé : la case reste
  // cochable à tout moment à l'étape 1 pour réintégrer une colonne.
  // Best-effort, jamais bloquant : une erreur réseau laisse simplement la
  // sélection par défaut "tout inclus", le contrôle qualité de l'étape 2
  // reste disponible pour exclure manuellement.
  useEffect(() => {
    setSelectedFeatures(new Set(otherColumns.filter((c) => c.name !== groupColumn).map((c) => c.name)));

    if (!datasetId || !targetColumn) return;
    let cancelled = false;
    api.datasets
      .qualityCheck(datasetId, targetColumn, groupColumn || undefined)
      .then((res) => {
        if (cancelled) return;
        const toExclude = new Set(res.warnings.filter((w) => EXCLUDABLE_CODES.has(w.code)).flatMap(excludableColumnsOf));
        if (toExclude.size === 0) return;
        setSelectedFeatures((prev) => {
          const next = new Set(prev);
          toExclude.forEach((c) => next.delete(c));
          return next;
        });
      })
      .catch(() => {
        // Dégradation silencieuse assumée — voir commentaire ci-dessus.
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [targetColumn, groupColumn, columns, datasetId]);

  function toggleFeature(name: string) {
    setSelectedFeatures((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }

  // Approuver une suggestion d'exclusion (contrôle qualité, étape 2) doit
  // toujours RETIRER la colonne — jamais un toggle : ré-approuver la même
  // suggestion (ou "Tout exclure" appelé deux fois) reste sans effet, jamais
  // une réintégration accidentelle.
  function excludeFeatures(names: string[]) {
    setSelectedFeatures((prev) => {
      const next = new Set(prev);
      names.forEach((name) => next.delete(name));
      return next;
    });
  }

  const selectedDataset = datasets.find((d) => d.id === datasetId);
  const step1Valid = Boolean(datasetId && targetColumn && selectedFeatures.size > 0);

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
    <form onSubmit={handleSubmit}>
      <WizardStepper steps={STEP_LABELS} activeStep={activeStep} maxReachedStep={maxReachedStep} onSelect={goToStep} />

      <Card className="p-5 mt-4">
        {activeStep === 1 && (
          <StepContent title="Choisissez vos données" description="Sélectionnez le jeu de données, la colonne à prédire et, si besoin, une colonne de regroupement.">
            <div>
              <label htmlFor="training-dataset" className="block text-sm text-muted-foreground mb-1">
                Jeu de données
              </label>
              <Select
                id="training-dataset"
                value={datasetId}
                onChange={(e) => handleDatasetChange(e.target.value)}
                required
              >
                <option value="">Choisir un dataset…</option>
                {datasets.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name} ({d.row_count} lignes)
                  </option>
                ))}
              </Select>
            </div>

            {columns.length > 0 && (
              <>
                <div>
                  <label htmlFor="training-target" className="block text-sm text-muted-foreground mb-1">
                    Colonne à prédire (cible)
                  </label>
                  <Select id="training-target" value={targetColumn} onChange={(e) => setTargetColumn(e.target.value)} required>
                    <option value="">Choisir une colonne…</option>
                    {columns.map((c) => (
                      <option key={c.name} value={c.name}>
                        {c.name} ({c.dtype})
                      </option>
                    ))}
                  </Select>
                  <p className="text-xs text-muted-foreground mt-1">
                    C'est la valeur que le modèle apprendra à prédire. Classification ou régression détectée
                    automatiquement selon cette colonne.
                  </p>
                  {targetSuggestions.length > 0 && !targetColumn && (
                    <div className="mt-2">
                      <p className="text-xs text-muted-foreground mb-1.5">
                        Colonnes les plus plausibles pour ce dataset :
                      </p>
                      <div className="flex flex-wrap gap-1.5">
                        {targetSuggestions.map((s) => (
                          <button
                            key={s.column}
                            type="button"
                            onClick={() => setTargetColumn(s.column)}
                            title={s.reasons.join(" · ")}
                            className="inline-flex items-center gap-1 rounded-full border border-primary/20 bg-primary/5 px-2.5 py-1 text-xs font-medium text-primary hover:bg-primary/10 transition-colors"
                          >
                            <Target size={11} />
                            {s.column}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {targetColumn && (
                  <div>
                    <label htmlFor="training-group-column" className="block text-sm text-muted-foreground mb-1">
                      Colonne de regroupement <span className="text-muted-foreground">(optionnel)</span>
                    </label>
                    <Select id="training-group-column" value={groupColumn} onChange={(e) => setGroupColumn(e.target.value)}>
                      <option value="">Aucune — split classique</option>
                      {otherColumns.map((c) => (
                        <option key={c.name} value={c.name}>
                          {c.name}
                        </option>
                      ))}
                    </Select>
                    <p className="text-xs text-muted-foreground mt-1">
                      Empêche qu'un même groupe (ex. un client) apparaisse à la fois en entraînement et en
                      test — un garde-fou anti-fuite. Si plusieurs lignes partagent un même échantillon
                      (mesures répétées), indiquez la colonne qui les identifie.
                    </p>
                  </div>
                )}

                {targetColumn && (
                  <div>
                    <button
                      type="button"
                      onClick={() => setShowFeaturePicker((v) => !v)}
                      className="text-sm text-primary hover:text-primary/80"
                    >
                      Choisir les variables utilisées
                      <span className="text-muted-foreground">
                        {" "}
                        ({selectedFeatures.size} sélectionnée{selectedFeatures.size > 1 ? "s" : ""})
                      </span>
                    </button>
                    {showFeaturePicker && (
                      <div className="mt-2 max-h-40 overflow-y-auto rounded-lg border border-border bg-muted p-2 space-y-1">
                        {otherColumns
                          .filter((c) => c.name !== groupColumn)
                          .map((c) => (
                            <label
                              key={c.name}
                              className="flex items-center gap-2 text-xs text-muted-foreground px-1 py-0.5"
                            >
                              <input
                                type="checkbox"
                                checked={selectedFeatures.has(c.name)}
                                onChange={() => toggleFeature(c.name)}
                                className="accent-primary"
                              />
                              {c.name} <span className="text-muted-foreground">({c.dtype})</span>
                            </label>
                          ))}
                      </div>
                    )}
                    <p className="text-xs text-muted-foreground mt-1">
                      Par défaut, toutes les variables sauf la cible sont utilisées — décochez celles à
                      exclure (ex. un identifiant sans valeur prédictive, ou une colonne signalée par le
                      contrôle qualité à l'étape suivante).
                    </p>
                  </div>
                )}
              </>
            )}
          </StepContent>
        )}

        {activeStep === 2 && targetColumn && datasetId && (
          <StepContent
            title="Qualité des données"
            description="Nous avons analysé vos colonnes. Voici ce qu'il faut savoir avant d'entraîner — expliqué simplement."
          >
            <DataQualityWarnings
              datasetId={datasetId}
              targetColumn={targetColumn}
              groupColumn={groupColumn || undefined}
              selectedFeatures={selectedFeatures}
              onExcludeColumns={excludeFeatures}
            />
          </StepContent>
        )}

        {activeStep === 3 && targetColumn && datasetId && (
          <StepContent title="Améliorations automatiques" description="Cochez les transformations à appliquer. Les recommandées sont déjà activées.">
            <ClassRebalancingSuggestion
              datasetId={datasetId}
              targetColumn={targetColumn}
              groupColumn={groupColumn || undefined}
              onChange={setClassRebalancing}
            />
            <FeatureEngineeringSuggestions
              datasetId={datasetId}
              targetColumn={targetColumn}
              groupColumn={groupColumn || undefined}
              selectedFeatures={selectedFeatures}
              onChange={setFeatureEngineering}
            />
          </StepContent>
        )}

        {activeStep === 4 && (
          <StepContent title="Mode expert" description="Par défaut, nous choisissons et réglons les modèles pour vous. Activez ce mode pour tout contrôler.">
            <div>
              <label htmlFor="training-test-size" className="block text-sm text-muted-foreground mb-1">
                Part du jeu de test — {Math.round(testSize * 100)} %
              </label>
              <input
                id="training-test-size"
                type="range"
                min={0.1}
                max={0.4}
                step={0.05}
                value={testSize}
                onChange={(e) => setTestSize(Number(e.target.value))}
                className="w-full accent-primary"
              />
            </div>

            <ExpertModePanel
              expertMode={expertMode}
              onExpertModeChange={setExpertMode}
              optunaTrials={optunaTrials}
              onOptunaTrialsChange={setOptunaTrials}
              cvFolds={cvFolds}
              onCvFoldsChange={setCvFolds}
              seed={seed}
              onSeedChange={setSeed}
              cqrAlpha={cqrAlpha}
              onCqrAlphaChange={setCqrAlpha}
              selectedModelIds={selectedModelIds}
              onSelectedModelIdsChange={setSelectedModelIds}
              classRebalancing={classRebalancing}
              onClassRebalancingChange={setClassRebalancing}
              hyperparameterOverrides={hyperparameterOverrides}
              onHyperparameterOverridesChange={setHyperparameterOverrides}
            />
          </StepContent>
        )}

        {activeStep === 5 && (
          <StepContent title="Prêt à lancer" description="Vérifiez le récapitulatif, puis lancez l'entraînement.">
            <div className="rounded-xl border border-border bg-muted p-4">
              <p className="text-xs uppercase tracking-wide text-muted-foreground mb-3">Récapitulatif</p>
              <dl className="grid grid-cols-2 gap-y-2.5 text-sm">
                <Fact label="Données" value={selectedDataset?.name ?? "—"} />
                <Fact label="Cible" value={targetColumn} mono />
                <Fact
                  label="Modèles comparés"
                  value={expertMode ? `${selectedModelIds.size} sélectionné${selectedModelIds.size > 1 ? "s" : ""}` : "Sélection automatique"}
                />
                <Fact
                  label="Hyperparamètres fixés"
                  value={
                    expertMode && Object.keys(hyperparameterOverrides).length > 0
                      ? `${Object.values(hyperparameterOverrides).reduce((n, params) => n + Object.keys(params).length, 0)}`
                      : "Aucun (recherche automatique)"
                  }
                />
                <Fact label="Rééquilibrage des classes" value={classRebalancing ? "Activé" : "Désactivé"} />
                <Fact label="Ingénierie de variables" value={featureEngineering ? "Activée" : "Non appliquée"} />
                <Fact label="Variables utilisées" value={String(selectedFeatures.size)} />
                <Fact
                  label="Durée estimée"
                  value={
                    durationEstimate?.status === "estimated" && durationEstimate.estimated_seconds !== null
                      ? formatDuration(durationEstimate.estimated_seconds)
                      : "Indisponible"
                  }
                />
              </dl>
              {durationEstimate?.status === "degraded" && (
                <p className="text-xs text-muted-foreground mt-3 pt-3 border-t border-border/60">
                  {durationEstimate.message ?? "Estimation indisponible."} L'estimation se construit à partir de vos
                  entraînements terminés.
                </p>
              )}
              {durationEstimate?.status === "estimated" && (
                <p className="text-xs text-muted-foreground mt-3 pt-3 border-t border-border/60">
                  Estimation fondée sur {durationEstimate.based_on_n_jobs} entraînement
                  {durationEstimate.based_on_n_jobs > 1 ? "s" : ""} précédent
                  {durationEstimate.based_on_n_jobs > 1 ? "s" : ""} de votre organisation — repère indicatif, pas une
                  garantie.
                </p>
              )}
            </div>

            {error && (
              <p className="text-sm text-destructive bg-destructive/10 border border-destructive/20 rounded-lg px-3 py-2">
                {error}
              </p>
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
            <Button
              type="button"
              size="sm"
              onClick={goNext}
              disabled={activeStep === 1 && !step1Valid}
            >
              Continuer
              <ChevronRight size={14} />
            </Button>
          )}
        </div>
      </Card>
    </form>
  );
}

// Barre d'étapes : voir components/ui/WizardStepper.tsx (partagée avec les
// wizards Vision — avant cette refonte, dupliquée ici à l'identique malgré
// une note dans VisionWizard.tsx affirmant l'avoir déjà extraite).

/** Contenu d'une étape du wizard — titre + description en langage clair,
 * puis les champs propres à l'étape. */
function StepContent({
  title,
  description,
  children,
}: {
  title: string;
  description?: string;
  children: ReactNode;
}) {
  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-h3 text-foreground">{title}</h3>
        {description && <p className="text-xs text-muted-foreground mt-0.5">{description}</p>}
      </div>
      {children}
    </div>
  );
}

function Fact({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div>
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd className={`text-foreground ${mono ? "font-mono text-xs" : "text-sm"}`}>{value}</dd>
    </div>
  );
}
