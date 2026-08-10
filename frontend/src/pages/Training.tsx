import { useCallback, useEffect, useState, type FormEvent, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { AlertCircle, ChevronDown, Loader2, PlayCircle, Trash2 } from "lucide-react";
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
import { FeatureEngineeringSuggestions } from "../components/training/FeatureEngineeringSuggestions";
import { ModelResultView } from "../components/training/ModelResultModal";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { formatDateTime } from "../utils/format";

const ACTIVE_STATUSES = new Set(["queued", "running"]);
const POLL_INTERVAL_MS = 3000;

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
          <p className="text-xs uppercase tracking-widest text-teal-600 font-semibold mb-1">
            Entraînement
          </p>
          <h1 className="text-2xl font-serif text-slate-900">{titles[phase]}</h1>
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
        <div className="max-w-2xl mx-auto">
          <TrainingForm datasets={datasets} onJobCreated={setActiveJob} />
          <p className="text-xs text-slate-400 text-center mt-4">
            Vos entraînements précédents restent consultables depuis le{" "}
            <Link to="/dashboard" className="text-teal-600 hover:text-teal-700">
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
      <div className="mx-auto mb-4 h-14 w-14 rounded-2xl bg-teal-50 border border-teal-200 flex items-center justify-center">
        <Loader2 className="text-teal-600 animate-spin" size={26} />
      </div>
      <h2 className="text-base font-medium text-slate-900">
        {job.dataset_name ?? "Dataset"} <span className="text-slate-400">→</span> {job.target_column}
      </h2>
      <p className="text-xs text-slate-500 mb-6">
        {job.task_type === "regression" ? "Régression" : "Classification"} · lancé {formatDateTime(job.created_at)}
      </p>
      <div className="h-2 rounded-full bg-slate-100 overflow-hidden mb-2">
        <div
          className="h-full rounded-full bg-teal-600 transition-all duration-500"
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
  const [optunaTrials, setOptunaTrials] = useState(20);
  const [testSize, setTestSize] = useState(0.2);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [featureEngineering, setFeatureEngineering] = useState<Pick<
    FeatureEngineeringSpec,
    "upstream" | "pipeline"
  > | null>(null);
  const [classRebalancing, setClassRebalancing] = useState(false);

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
      const job = await api.training.createJob({
        dataset_id: datasetId,
        target_column: targetColumn,
        feature_columns: Array.from(selectedFeatures),
        group_column: groupColumn || undefined,
        optuna_trials: optunaTrials,
        test_size: testSize,
        feature_engineering: featureEngineering ?? undefined,
        class_rebalancing: classRebalancing,
      });
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

  return (
    <Card className="p-5">
      <h2 className="text-sm font-medium text-slate-800 mb-5">Nouvel entraînement</h2>

      {datasets.length === 0 ? (
        <p className="text-sm text-slate-500">
          Aucun dataset prêt — importez-en un depuis <span className="text-teal-600">Mes données</span>.
        </p>
      ) : (
        <form onSubmit={handleSubmit} className="relative">
          {/* Ligne de progression du pipeline guidé — pleine hauteur en fond,
              chaque pastille de `Step` (opaque) la recouvre localement, ce
              qui donne l'effet d'un fil reliant les étapes sans avoir à
              savoir laquelle est la dernière (contenu conditionnel). */}
          <div className="absolute left-[15px] top-3 bottom-3 w-px bg-slate-200" aria-hidden="true" />

          <div className="space-y-6 relative">
            <Step number={1} title="Sélection des données">
              <div>
                <label className="block text-sm text-slate-600 mb-1">Dataset</label>
                <select
                  value={datasetId}
                  onChange={(e) => handleDatasetChange(e.target.value)}
                  required
                  className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-teal-500/40"
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
                    <label className="block text-sm text-slate-600 mb-1">Colonne cible à prédire</label>
                    <select
                      value={targetColumn}
                      onChange={(e) => setTargetColumn(e.target.value)}
                      required
                      className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-teal-500/40"
                    >
                      <option value="">Choisir une colonne…</option>
                      {columns.map((c) => (
                        <option key={c.name} value={c.name}>
                          {c.name} ({c.dtype})
                        </option>
                      ))}
                    </select>
                    <p className="text-xs text-slate-400 mt-1">
                      Classification ou régression détectée automatiquement selon cette colonne.
                    </p>
                  </div>

                  {targetColumn && (
                    <div>
                      <label className="block text-sm text-slate-600 mb-1">
                        Colonne de groupe <span className="text-slate-400">(optionnel — anti-fuite)</span>
                      </label>
                      <select
                        value={groupColumn}
                        onChange={(e) => setGroupColumn(e.target.value)}
                        className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-teal-500/40"
                      >
                        <option value="">Aucune — split classique</option>
                        {otherColumns.map((c) => (
                          <option key={c.name} value={c.name}>
                            {c.name}
                          </option>
                        ))}
                      </select>
                      <p className="text-xs text-slate-400 mt-1">
                        Si plusieurs lignes partagent un même échantillon (mesures répétées), indiquez la
                        colonne qui les identifie : train et test ne partageront jamais le même groupe.
                      </p>
                    </div>
                  )}

                  {targetColumn && (
                    <div>
                      <button
                        type="button"
                        onClick={() => setShowFeaturePicker((v) => !v)}
                        className="text-sm text-teal-600 hover:text-teal-700"
                      >
                        {showFeaturePicker ? "Masquer" : "Choisir"} les variables utilisées
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
                                  className="accent-teal-500"
                                />
                                {c.name} <span className="text-slate-400">({c.dtype})</span>
                              </label>
                            ))}
                        </div>
                      )}
                      <p className="text-xs text-slate-400 mt-1">
                        Par défaut, toutes les variables sauf la cible sont utilisées — décochez celles à
                        exclure (ex. un identifiant sans valeur prédictive, ou une colonne signalée par le
                        contrôle qualité ci-dessous).
                      </p>
                    </div>
                  )}
                </>
              )}
            </Step>

            {targetColumn && datasetId && (
              <Step
                number={2}
                title="Contrôle qualité"
                description="Avertissements détectés sur ce dataset pour cette cible — jamais bloquants, toujours à lire avant de lancer."
              >
                <div className="rounded-xl border border-slate-200 bg-slate-50/60 p-3">
                  <DataQualityWarnings
                    datasetId={datasetId}
                    targetColumn={targetColumn}
                    groupColumn={groupColumn || undefined}
                  />
                </div>
              </Step>
            )}

            {targetColumn && datasetId && (
              <Step number={3} title="Améliorations automatiques suggérées">
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
              </Step>
            )}

            {targetColumn && (
              <Step number={4} title="Réglages avancés" optional>
                {/* Emplacement réservé au futur Mode Expert (Lot E2) : ce
                    <details> accueillera le choix guidé/expert par pilier —
                    non implémenté ici, uniquement l'emplacement. */}
                <details className="group rounded-xl border border-slate-200">
                  <summary className="flex items-center justify-between cursor-pointer list-none px-3 py-2.5 text-sm text-slate-600">
                    Essais d'optimisation, jeu de test
                    <ChevronDown size={14} className="text-slate-400 transition-transform group-open:rotate-180" />
                  </summary>
                  <div className="px-3 pb-3 pt-3 space-y-4 border-t border-slate-200">
                    <div>
                      <label className="block text-sm text-slate-600 mb-1">
                        Recherche d'hyperparamètres — {optunaTrials} essais
                      </label>
                      <input
                        type="range"
                        min={5}
                        max={60}
                        step={5}
                        value={optunaTrials}
                        onChange={(e) => setOptunaTrials(Number(e.target.value))}
                        className="w-full accent-teal-500"
                      />
                      <p className="text-xs text-slate-400 mt-1">
                        Plus élevé = recherche plus fine, mais entraînement plus long.
                      </p>
                    </div>

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
                        className="w-full accent-teal-500"
                      />
                    </div>
                  </div>
                </details>
              </Step>
            )}

            {targetColumn && (
              <Step number={5} title="Lancer l'entraînement">
                <p className="text-xs text-slate-500">
                  La durée dépend de la taille du dataset et du nombre d'essais — cette page affichera la
                  progression puis le résultat dès le lancement.
                </p>

                {error && (
                  <p className="text-sm text-rose-700 bg-rose-50 border border-rose-200 rounded-lg px-3 py-2">
                    {error}
                  </p>
                )}

                <Button
                  type="submit"
                  disabled={!targetColumn || selectedFeatures.size === 0 || isSubmitting}
                  className="w-full"
                  size="md"
                >
                  <PlayCircle size={16} />
                  {isSubmitting ? "Lancement…" : "Lancer l'entraînement"}
                </Button>
              </Step>
            )}
          </div>
        </form>
      )}
    </Card>
  );
}

/** Étape numérotée du pipeline guidé (Lot E1-ter) — pastille pleine
 * (recouvre la ligne de connexion en fond, voir `TrainingForm`), titre et
 * contenu libre. `optional` grise légèrement le libellé pour signaler une
 * étape qu'on peut ignorer sans conséquence (réglages avancés). */
function Step({
  number,
  title,
  description,
  optional = false,
  children,
}: {
  number: number;
  title: string;
  description?: string;
  optional?: boolean;
  children: ReactNode;
}) {
  return (
    <div className="relative pl-9">
      <div
        className={`absolute left-0 top-0 h-[30px] w-[30px] rounded-full flex items-center justify-center text-xs font-semibold ${
          optional ? "bg-slate-100 text-slate-500 border border-slate-300" : "bg-teal-600 text-white"
        }`}
      >
        {number}
      </div>
      <div className="pt-1 space-y-3">
        <div>
          <h3 className="text-sm font-medium text-slate-800">
            {title}
            {optional && <span className="text-slate-400 font-normal"> (optionnel)</span>}
          </h3>
          {description && <p className="text-xs text-slate-500 mt-0.5">{description}</p>}
        </div>
        {children}
      </div>
    </div>
  );
}
