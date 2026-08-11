import { useCallback, useEffect, useRef, useState, type FormEvent, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { AlertCircle, Check, ChevronLeft, ChevronRight, Loader2, PlayCircle, Trash2 } from "lucide-react";
import {
  ApiError,
  api,
  type ColumnSchema,
  type DatasetSummary,
  type FeatureEngineeringSpec,
  type TrainingJobSummary,
} from "../api/client";
import AppShell from "../components/AppShell";
import { ClassRebalancingSuggestion } from "../components/training/ClassRebalancingSuggestion";
import { DataQualityWarnings } from "../components/training/DataQualityWarnings";
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
import { formatDateTime } from "../utils/format";
import { buildTrainingJobPayload } from "../utils/trainingPayload";

const DEFAULT_OPTUNA_TRIALS = 20; // `api.core.config.Settings.optuna_trials_default`

const ACTIVE_STATUSES = new Set(["queued", "running"]);
const POLL_INTERVAL_MS = 3000;

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

type Phase = "configure" | "progress" | "results" | "failed";

function phaseOf(job: TrainingJobSummary | null): Phase {
  if (!job) return "configure";
  if (ACTIVE_STATUSES.has(job.status)) return "progress";
  return job.status === "completed" ? "results" : "failed";
}

/** Page dédiée à l'entraînement (Lot E1-ter, sur demande explicite) : pas
 * d'historique à côté — configurer, lancer, puis voir le résultat EN PLACE
 * sur cette même page. L'historique complet des entraînements passés vit
 * sur le tableau de bord ("Derniers entraînements"), pas ici. */
export default function Training() {
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [activeJob, setActiveJob] = useState<TrainingJobSummary | null>(null);
  const [confirmingDelete, setConfirmingDelete] = useState(false);

  const loadDatasets = useCallback(async () => {
    try {
      const all = await api.datasets.list();
      setDatasets(all.filter((d) => d.status === "ready"));
    } catch {
      // silencieux — le formulaire affichera simplement "aucun dataset"
    }
  }, []);

  useEffect(() => {
    loadDatasets();
  }, [loadDatasets]);

  const phase = phaseOf(activeJob);

  // Poll le job actif tant qu'il est en file/en cours — la page bascule
  // d'elle-même vers la vue résultat (ou échec) dès que le statut change.
  useEffect(() => {
    if (phase !== "progress" || !activeJob) return;
    const interval = setInterval(async () => {
      try {
        setActiveJob(await api.training.getJob(activeJob.id));
      } catch {
        // silencieux — nouvelle tentative au prochain tick
      }
    }, POLL_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [phase, activeJob]);

  function resetToConfigure() {
    setActiveJob(null);
    setConfirmingDelete(false);
  }

  async function handleDeleteActiveJob() {
    if (!activeJob) return;
    if (!confirmingDelete) {
      setConfirmingDelete(true);
      return;
    }
    try {
      await api.training.remove(activeJob.id);
    } catch {
      // best-effort — on repart en configuration dans tous les cas
    }
    resetToConfigure();
  }

  const titles: Record<Phase, string> = {
    configure: "Entraîner un modèle",
    progress: "Entraînement en cours",
    results: "Résultat de l'entraînement",
    failed: "Échec de l'entraînement",
  };

  return (
    <AppShell pillarId="supervised">
      <div className="mb-8 flex items-start justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-widest text-primary font-semibold mb-1">
            Entraînement
          </p>
          <h1 className="text-2xl font-serif text-slate-900">{titles[phase]}</h1>
          {phase === "configure" && (
            <p className="text-sm text-slate-500 mt-1">
              Objectif : prédire une valeur ou une catégorie. Nous vous guidons pas à pas.
            </p>
          )}
        </div>

        {phase !== "configure" && (
          <div className="flex items-center gap-2 flex-shrink-0">
            {(phase === "results" || phase === "failed") && (
              <button
                type="button"
                onClick={handleDeleteActiveJob}
                onMouseLeave={() => setConfirmingDelete(false)}
                aria-label={confirmingDelete ? "Confirmer la suppression" : "Supprimer cet entraînement"}
                title={confirmingDelete ? "Cliquer à nouveau pour confirmer" : "Supprimer cet entraînement"}
                className={`p-2 rounded-lg transition-colors ${
                  confirmingDelete
                    ? "text-rose-700 bg-rose-100"
                    : "text-slate-400 hover:text-rose-600 hover:bg-rose-50"
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
        )}
      </div>

      {phase === "configure" && (
        <div className="max-w-3xl mx-auto">
          <TrainingForm datasets={datasets} onJobCreated={setActiveJob} />
          <p className="text-xs text-slate-400 text-center mt-4">
            Vos entraînements précédents restent consultables depuis le{" "}
            <Link to="/dashboard" className="text-primary hover:text-primary/80">
              tableau de bord
            </Link>
            .
          </p>
        </div>
      )}

      {phase === "progress" && activeJob && <TrainingProgress job={activeJob} />}

      {phase === "failed" && activeJob && <TrainingFailed job={activeJob} />}

      {phase === "results" && activeJob && (
        <div className="max-w-4xl mx-auto">
          <ModelResultView job={activeJob} />
        </div>
      )}
    </AppShell>
  );
}

function TrainingProgress({ job }: { job: TrainingJobSummary }) {
  return (
    <Card className="max-w-xl mx-auto p-8 text-center">
      <div className="mx-auto mb-4 h-14 w-14 rounded-2xl bg-primary/10 border border-primary/20 flex items-center justify-center">
        <Loader2 className="text-primary animate-spin" size={26} />
      </div>
      <h2 className="text-base font-medium text-slate-900">
        {job.dataset_name ?? "Dataset"} <span className="text-slate-400">→</span> {job.target_column}
      </h2>
      <p className="text-xs text-slate-500 mb-6">
        {job.task_type === "regression" ? "Régression" : "Classification"} · lancé {formatDateTime(job.created_at)}
      </p>
      <div className="h-2 rounded-full bg-slate-100 overflow-hidden mb-2">
        <div
          className="h-full rounded-full bg-primary transition-all duration-500"
          style={{ width: `${Math.max(job.progress_percent, 4)}%` }}
        />
      </div>
      <p className="text-xs text-slate-400 tabular-nums mb-3">{job.progress_percent}%</p>
      <p className="text-sm text-slate-600">
        {job.progress_step ?? "En attente d'un worker disponible…"}
      </p>
      <p className="text-xs text-slate-400 mt-4">
        La durée dépend de la taille du dataset et du nombre d'essais — cette page se met à jour
        automatiquement, vous pouvez aussi la quitter et revenir plus tard.
      </p>
    </Card>
  );
}

function TrainingFailed({ job }: { job: TrainingJobSummary }) {
  return (
    <Card className="max-w-xl mx-auto p-8 text-center">
      <div className="mx-auto mb-4 h-14 w-14 rounded-2xl bg-rose-50 border border-rose-200 flex items-center justify-center">
        <AlertCircle className="text-rose-600" size={26} />
      </div>
      <h2 className="text-base font-medium text-slate-900">
        {job.dataset_name ?? "Dataset"} <span className="text-slate-400">→</span> {job.target_column}
      </h2>
      <p className="text-xs text-slate-500 mb-4">
        {job.task_type === "regression" ? "Régression" : "Classification"} · lancé {formatDateTime(job.created_at)}
      </p>
      {job.error_message && (
        <p className="text-sm text-rose-700 bg-rose-50 border border-rose-200 rounded-lg px-3 py-2 text-left">
          {job.error_message}
        </p>
      )}
    </Card>
  );
}

function TrainingForm({
  datasets,
  onJobCreated,
}: {
  datasets: DatasetSummary[];
  onJobCreated: (job: TrainingJobSummary) => void;
}) {
  const [datasetId, setDatasetId] = useState<number | "">("");
  const [columns, setColumns] = useState<ColumnSchema[]>([]);
  const [targetColumn, setTargetColumn] = useState("");
  const [groupColumn, setGroupColumn] = useState("");
  const [selectedFeatures, setSelectedFeatures] = useState<Set<string>>(new Set());
  const [showFeaturePicker, setShowFeaturePicker] = useState(false);
  const [optunaTrials, setOptunaTrials] = useState(DEFAULT_OPTUNA_TRIALS);
  const [testSize, setTestSize] = useState(0.2);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
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
        }),
      );
      onJobCreated(job);
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
  useEffect(() => {
    setSelectedFeatures(new Set(otherColumns.filter((c) => c.name !== groupColumn).map((c) => c.name)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [targetColumn, groupColumn, columns]);

  function toggleFeature(name: string) {
    setSelectedFeatures((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }

  const selectedDataset = datasets.find((d) => d.id === datasetId);
  const step1Valid = Boolean(datasetId && targetColumn && selectedFeatures.size > 0);

  if (datasets.length === 0) {
    return (
      <Card className="p-5">
        <p className="text-sm text-slate-500">
          Aucun dataset prêt — importez-en un depuis <span className="text-primary">Mes données</span>.
        </p>
      </Card>
    );
  }

  return (
    <form onSubmit={handleSubmit}>
      <StepperNav steps={STEP_LABELS} activeStep={activeStep} maxReachedStep={maxReachedStep} onSelect={goToStep} />

      <Card className="p-5 mt-4">
        {activeStep === 1 && (
          <StepContent title="Choisissez vos données" description="Sélectionnez le jeu de données, la colonne à prédire et, si besoin, une colonne de regroupement.">
            <div>
              <label className="block text-sm text-slate-600 mb-1">Jeu de données</label>
              <select
                value={datasetId}
                onChange={(e) => handleDatasetChange(e.target.value)}
                required
                className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-primary/40"
              >
                <option value="">Choisir un dataset…</option>
                {datasets.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name} ({d.row_count} lignes)
                  </option>
                ))}
              </select>
            </div>

            {columns.length > 0 && (
              <>
                <div>
                  <label className="block text-sm text-slate-600 mb-1">Colonne à prédire (cible)</label>
                  <select
                    value={targetColumn}
                    onChange={(e) => setTargetColumn(e.target.value)}
                    required
                    className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-primary/40"
                  >
                    <option value="">Choisir une colonne…</option>
                    {columns.map((c) => (
                      <option key={c.name} value={c.name}>
                        {c.name} ({c.dtype})
                      </option>
                    ))}
                  </select>
                  <p className="text-xs text-slate-400 mt-1">
                    C'est la valeur que le modèle apprendra à prédire. Classification ou régression détectée
                    automatiquement selon cette colonne.
                  </p>
                </div>

                {targetColumn && (
                  <div>
                    <label className="block text-sm text-slate-600 mb-1">
                      Colonne de regroupement <span className="text-slate-400">(optionnel)</span>
                    </label>
                    <select
                      value={groupColumn}
                      onChange={(e) => setGroupColumn(e.target.value)}
                      className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-primary/40"
                    >
                      <option value="">Aucune — split classique</option>
                      {otherColumns.map((c) => (
                        <option key={c.name} value={c.name}>
                          {c.name}
                        </option>
                      ))}
                    </select>
                    <p className="text-xs text-slate-400 mt-1">
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
                      <span className="text-slate-400">
                        {" "}
                        ({selectedFeatures.size} sélectionnée{selectedFeatures.size > 1 ? "s" : ""})
                      </span>
                    </button>
                    {showFeaturePicker && (
                      <div className="mt-2 max-h-40 overflow-y-auto rounded-lg border border-slate-200 bg-slate-50 p-2 space-y-1">
                        {otherColumns
                          .filter((c) => c.name !== groupColumn)
                          .map((c) => (
                            <label
                              key={c.name}
                              className="flex items-center gap-2 text-xs text-slate-600 px-1 py-0.5"
                            >
                              <input
                                type="checkbox"
                                checked={selectedFeatures.has(c.name)}
                                onChange={() => toggleFeature(c.name)}
                                className="accent-primary"
                              />
                              {c.name} <span className="text-slate-400">({c.dtype})</span>
                            </label>
                          ))}
                      </div>
                    )}
                    <p className="text-xs text-slate-400 mt-1">
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
            <DataQualityWarnings datasetId={datasetId} targetColumn={targetColumn} groupColumn={groupColumn || undefined} />
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
              onChange={setFeatureEngineering}
            />
          </StepContent>
        )}

        {activeStep === 4 && (
          <StepContent title="Mode expert" description="Par défaut, nous choisissons et réglons les modèles pour vous. Activez ce mode pour tout contrôler.">
            <div>
              <label className="block text-sm text-slate-600 mb-1">
                Part du jeu de test — {Math.round(testSize * 100)} %
              </label>
              <input
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
            />
          </StepContent>
        )}

        {activeStep === 5 && (
          <StepContent title="Prêt à lancer" description="Vérifiez le récapitulatif, puis lancez l'entraînement.">
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
              <p className="text-xs uppercase tracking-wide text-slate-500 mb-3">Récapitulatif</p>
              <dl className="grid grid-cols-2 gap-y-2.5 text-sm">
                <Fact label="Données" value={selectedDataset?.name ?? "—"} />
                <Fact label="Cible" value={targetColumn} mono />
                <Fact
                  label="Modèles comparés"
                  value={expertMode ? `${selectedModelIds.size} sélectionné${selectedModelIds.size > 1 ? "s" : ""}` : "Sélection automatique"}
                />
                <Fact label="Rééquilibrage des classes" value={classRebalancing ? "Activé" : "Désactivé"} />
                <Fact label="Ingénierie de variables" value={featureEngineering ? "Activée" : "Non appliquée"} />
                <Fact label="Variables utilisées" value={String(selectedFeatures.size)} />
              </dl>
            </div>

            {error && (
              <p className="text-sm text-rose-700 bg-rose-50 border border-rose-200 rounded-lg px-3 py-2">
                {error}
              </p>
            )}

            <Button type="submit" disabled={!step1Valid || isSubmitting} className="w-full" size="md">
              <PlayCircle size={16} />
              {isSubmitting ? "Lancement…" : "Lancer l'entraînement"}
            </Button>
          </StepContent>
        )}

        <div className="flex items-center justify-between pt-5 mt-5 border-t border-slate-200">
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

/** Wizard horizontal (refonte UI, calqué sur la maquette de référence) —
 * pastilles numérotées reliées par des chevrons, navigables (une étape déjà
 * atteinte reste cliquable, jamais celles pas encore vues). Défile
 * horizontalement au besoin (5 pastilles ne tiennent pas toujours sur un
 * petit écran) — flèches gauche/droite en plus du scroll tactile/molette. */
function StepperNav({
  steps,
  activeStep,
  maxReachedStep,
  onSelect,
}: {
  steps: { number: number; label: string }[];
  activeStep: number;
  maxReachedStep: number;
  onSelect: (step: number) => void;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  function scrollBy(delta: number) {
    scrollRef.current?.scrollBy({ left: delta, behavior: "smooth" });
  }

  return (
    <div className="flex items-center gap-1">
      <button
        type="button"
        onClick={() => scrollBy(-180)}
        aria-label="Défiler vers la gauche"
        className="flex-shrink-0 h-7 w-7 flex items-center justify-center rounded-full text-slate-400 hover:bg-slate-100 transition-colors"
      >
        <ChevronLeft size={16} />
      </button>
      <div ref={scrollRef} className="flex items-center gap-2 overflow-x-auto py-1 scroll-smooth">
        {steps.map((step, i) => (
          <div key={step.number} className="flex items-center gap-2 flex-shrink-0">
            <StepPill
              number={step.number}
              label={step.label}
              state={step.number < activeStep ? "done" : step.number === activeStep ? "current" : "pending"}
              disabled={step.number > maxReachedStep}
              onClick={() => onSelect(step.number)}
            />
            {i < steps.length - 1 && <ChevronRight size={14} className="text-slate-300 flex-shrink-0" />}
          </div>
        ))}
      </div>
      <button
        type="button"
        onClick={() => scrollBy(180)}
        aria-label="Défiler vers la droite"
        className="flex-shrink-0 h-7 w-7 flex items-center justify-center rounded-full text-slate-400 hover:bg-slate-100 transition-colors"
      >
        <ChevronRight size={16} />
      </button>
    </div>
  );
}

function StepPill({
  number,
  label,
  state,
  disabled,
  onClick,
}: {
  number: number;
  label: string;
  state: "done" | "current" | "pending";
  disabled: boolean;
  onClick: () => void;
}) {
  const pillStyle = {
    done: "border-success/30 bg-success/10 text-success",
    current: "border-primary/30 bg-primary/10 text-primary",
    pending: "border-slate-200 text-slate-400",
  }[state];
  const circleStyle = {
    done: "bg-success text-white",
    current: "bg-primary text-white",
    pending: "bg-white border border-slate-300 text-slate-400",
  }[state];

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`flex items-center gap-2 rounded-full border pl-1.5 pr-3 py-1.5 text-sm font-medium whitespace-nowrap transition-colors disabled:cursor-not-allowed ${pillStyle}`}
    >
      <span className={`h-5 w-5 rounded-full flex items-center justify-center text-[11px] font-semibold flex-shrink-0 ${circleStyle}`}>
        {state === "done" ? <Check size={12} strokeWidth={3} /> : number}
      </span>
      {label}
    </button>
  );
}

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
        <h3 className="text-sm font-medium text-slate-800">{title}</h3>
        {description && <p className="text-xs text-slate-500 mt-0.5">{description}</p>}
      </div>
      {children}
    </div>
  );
}

function Fact({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div>
      <dt className="text-xs text-slate-500">{label}</dt>
      <dd className={`text-slate-800 ${mono ? "font-mono text-xs" : "text-sm"}`}>{value}</dd>
    </div>
  );
}
