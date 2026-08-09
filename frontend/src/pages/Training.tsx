import { useCallback, useEffect, useRef, useState, type FormEvent, type MouseEvent } from "react";
import { AlertCircle, PlayCircle, Sparkles, Trash2 } from "lucide-react";
import {
  ApiError,
  api,
  type ColumnSchema,
  type DatasetSummary,
  type TrainingJobSummary,
} from "../api/client";
import AppShell from "../components/AppShell";
import { DataQualityWarnings } from "../components/training/DataQualityWarnings";
import ModelResultModal from "../components/training/ModelResultModal";
import { Badge } from "../components/ui/Badge";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { formatDateTime } from "../utils/format";

const ACTIVE_STATUSES = new Set(["queued", "running"]);
const POLL_INTERVAL_MS = 3000;

export default function Training() {
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [jobs, setJobs] = useState<TrainingJobSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [viewingJob, setViewingJob] = useState<TrainingJobSummary | null>(null);

  const loadDatasets = useCallback(async () => {
    try {
      const all = await api.datasets.list();
      setDatasets(all.filter((d) => d.status === "ready"));
    } catch {
      // silencieux — le formulaire affichera simplement "aucun dataset"
    }
  }, []);

  const loadJobs = useCallback(async () => {
    try {
      setJobs(await api.training.listJobs());
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Impossible de charger les entraînements");
    }
  }, []);

  useEffect(() => {
    loadDatasets();
    loadJobs();
  }, [loadDatasets, loadJobs]);

  // Poll uniquement tant qu'un job est en file ou en cours — évite de solliciter
  // l'API pour rien une fois que tout est terminé.
  const hasActiveJob = jobs.some((j) => ACTIVE_STATUSES.has(j.status));
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  useEffect(() => {
    if (hasActiveJob && !pollRef.current) {
      pollRef.current = setInterval(loadJobs, POLL_INTERVAL_MS);
    }
    if (!hasActiveJob && pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
      pollRef.current = null;
    };
  }, [hasActiveJob, loadJobs]);

  return (
    <AppShell>
      <div className="mb-8">
        <p className="text-xs uppercase tracking-widest text-teal-400/90 font-semibold mb-1">
          Entraînement
        </p>
        <h1 className="text-2xl font-serif text-slate-100">Entraîner un modèle</h1>
      </div>

      <div className="grid gap-6 lg:grid-cols-5">
        <div className="lg:col-span-2">
          <TrainingForm
            datasets={datasets}
            onJobCreated={() => {
              loadJobs();
            }}
          />
        </div>

        <div className="lg:col-span-3">
          <p className="text-xs uppercase tracking-wide text-slate-500 mb-3">
            Historique — {jobs.length} entraînement{jobs.length > 1 ? "s" : ""}
          </p>

          {error && (
            <div className="flex items-center gap-2 text-sm text-rose-300 bg-rose-500/10 border border-rose-500/20 rounded-lg px-3 py-2 mb-4">
              <AlertCircle size={15} className="flex-shrink-0" />
              {error}
            </div>
          )}

          {jobs.length === 0 ? (
            <Card className="p-10 text-center">
              <Sparkles className="mx-auto mb-3 text-slate-700" size={28} />
              <p className="text-sm text-slate-500">
                Aucun entraînement pour l'instant — configurez-en un à gauche.
              </p>
            </Card>
          ) : (
            <div className="space-y-3">
              {jobs.map((job) => (
                <JobCard
                  key={job.id}
                  job={job}
                  onView={() => setViewingJob(job)}
                  onDelete={async () => {
                    try {
                      await api.training.remove(job.id);
                    } catch (err) {
                      // Toujours rafraîchir même en cas d'échec : si la suppression
                      // a déjà réussi ailleurs (404), la carte doit disparaître.
                      setError(err instanceof ApiError ? err.message : "Suppression impossible");
                    }
                    loadJobs();
                  }}
                />
              ))}
            </div>
          )}
        </div>
      </div>

      {viewingJob && <ModelResultModal job={viewingJob} onClose={() => setViewingJob(null)} />}
    </AppShell>
  );
}

function statusBadge(job: TrainingJobSummary) {
  switch (job.status) {
    case "completed":
      return <Badge variant="success">Terminé</Badge>;
    case "failed":
      return <Badge variant="danger">Échec</Badge>;
    case "running":
      return <Badge variant="warning">En cours</Badge>;
    default:
      return <Badge variant="neutral">En file</Badge>;
  }
}

function JobCard({
  job,
  onView,
  onDelete,
}: {
  job: TrainingJobSummary;
  onView: () => void;
  onDelete: () => void;
}) {
  const isActive = ACTIVE_STATUSES.has(job.status);
  const isCompleted = job.status === "completed";
  const [isDeleting, setIsDeleting] = useState(false);
  const [confirming, setConfirming] = useState(false);

  async function handleDelete(event: MouseEvent) {
    event.stopPropagation();
    if (!confirming) {
      setConfirming(true);
      return;
    }
    setIsDeleting(true);
    try {
      await onDelete();
    } finally {
      setIsDeleting(false);
    }
  }

  return (
    <Card
      interactive={isCompleted}
      onClick={isCompleted ? onView : undefined}
      className={`p-4 ${isCompleted ? "cursor-pointer" : ""}`}
    >
      <div className="flex items-start justify-between gap-3 mb-2">
        <div className="min-w-0">
          <p className="text-sm font-medium text-slate-100 truncate">
            {job.dataset_name ?? "Dataset"} <span className="text-slate-600">→</span> {job.target_column}
          </p>
          <p className="text-xs text-slate-500 mt-0.5">
            {job.task_type === "regression" ? "Régression" : "Classification"} · {formatDateTime(job.created_at)}
            {job.created_by ? ` · ${job.created_by}` : ""}
          </p>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          {statusBadge(job)}
          <button
            type="button"
            onClick={handleDelete}
            onMouseLeave={() => setConfirming(false)}
            disabled={isDeleting}
            aria-label={confirming ? "Confirmer la suppression" : "Supprimer"}
            title={confirming ? "Cliquer à nouveau pour confirmer" : "Supprimer cet entraînement"}
            className={`p-1 rounded-md transition-colors ${
              confirming
                ? "text-rose-300 bg-rose-500/15"
                : "text-slate-600 hover:text-rose-300 hover:bg-rose-500/10"
            }`}
          >
            <Trash2 size={13} />
          </button>
        </div>
      </div>

      {isActive && (
        <div className="mt-3">
          <div className="h-1.5 rounded-full bg-slate-800 overflow-hidden">
            <div
              className="h-full rounded-full bg-teal-500 transition-all duration-500"
              style={{ width: `${Math.max(job.progress_percent, 4)}%` }}
            />
          </div>
          <p className="text-xs text-slate-500 mt-1.5">
            {job.progress_step ?? "En attente d'un worker disponible…"}
          </p>
        </div>
      )}

      {job.status === "failed" && job.error_message && (
        <p className="text-xs text-rose-400 mt-2">{job.error_message}</p>
      )}

      {isCompleted && job.headline_metric && (
        <p className="text-xs text-slate-400 mt-2">
          <Badge variant="accent">{job.algorithm}</Badge>{" "}
          {job.headline_metric.name} ={" "}
          <span className="tabular-nums text-slate-200">
            {job.headline_metric.value?.toFixed(3) ?? "—"}
          </span>
          <span className="text-slate-600"> · cliquer pour le détail</span>
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
  onJobCreated: () => void;
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
      await api.training.createJob({
        dataset_id: datasetId,
        target_column: targetColumn,
        feature_columns: Array.from(selectedFeatures),
        group_column: groupColumn || undefined,
        optuna_trials: optunaTrials,
        test_size: testSize,
      });
      onJobCreated();
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
      <h2 className="text-sm font-medium text-slate-200 mb-4">Nouvel entraînement</h2>

      {datasets.length === 0 ? (
        <p className="text-sm text-slate-500">
          Aucun dataset prêt — importez-en un depuis <span className="text-teal-400">Mes données</span>.
        </p>
      ) : (
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm text-slate-400 mb-1">Dataset</label>
            <select
              value={datasetId}
              onChange={(e) => handleDatasetChange(e.target.value)}
              required
              className="w-full rounded-lg border border-slate-700 bg-slate-950/60 px-3 py-2 text-sm text-slate-100 focus:outline-none focus:ring-2 focus:ring-teal-500/50"
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
                <label className="block text-sm text-slate-400 mb-1">Colonne cible à prédire</label>
                <select
                  value={targetColumn}
                  onChange={(e) => setTargetColumn(e.target.value)}
                  required
                  className="w-full rounded-lg border border-slate-700 bg-slate-950/60 px-3 py-2 text-sm text-slate-100 focus:outline-none focus:ring-2 focus:ring-teal-500/50"
                >
                  <option value="">Choisir une colonne…</option>
                  {columns.map((c) => (
                    <option key={c.name} value={c.name}>
                      {c.name} ({c.dtype})
                    </option>
                  ))}
                </select>
                <p className="text-xs text-slate-600 mt-1">
                  Classification ou régression détectée automatiquement selon cette colonne.
                </p>
              </div>

              {targetColumn && (
                <div>
                  <label className="block text-sm text-slate-400 mb-1">
                    Colonne de groupe <span className="text-slate-600">(optionnel — anti-fuite)</span>
                  </label>
                  <select
                    value={groupColumn}
                    onChange={(e) => setGroupColumn(e.target.value)}
                    className="w-full rounded-lg border border-slate-700 bg-slate-950/60 px-3 py-2 text-sm text-slate-100 focus:outline-none focus:ring-2 focus:ring-teal-500/50"
                  >
                    <option value="">Aucune — split classique</option>
                    {otherColumns.map((c) => (
                      <option key={c.name} value={c.name}>
                        {c.name}
                      </option>
                    ))}
                  </select>
                  <p className="text-xs text-slate-600 mt-1">
                    Si plusieurs lignes partagent un même échantillon (mesures répétées), indiquez la
                    colonne qui les identifie : train et test ne partageront jamais le même groupe.
                  </p>
                </div>
              )}

              {targetColumn && datasetId && (
                <DataQualityWarnings
                  datasetId={datasetId}
                  targetColumn={targetColumn}
                  groupColumn={groupColumn || undefined}
                />
              )}

              {targetColumn && (
                <div>
                  <button
                    type="button"
                    onClick={() => setShowFeaturePicker((v) => !v)}
                    className="text-sm text-teal-400 hover:text-teal-300"
                  >
                    {showFeaturePicker ? "Masquer" : "Choisir"} les variables utilisées
                    <span className="text-slate-600"> ({selectedFeatures.size} sélectionnée{selectedFeatures.size > 1 ? "s" : ""})</span>
                  </button>
                  {showFeaturePicker && (
                    <div className="mt-2 max-h-40 overflow-y-auto rounded-lg border border-slate-800 bg-slate-950/40 p-2 space-y-1">
                      {otherColumns
                        .filter((c) => c.name !== groupColumn)
                        .map((c) => (
                          <label key={c.name} className="flex items-center gap-2 text-xs text-slate-300 px-1 py-0.5">
                            <input
                              type="checkbox"
                              checked={selectedFeatures.has(c.name)}
                              onChange={() => toggleFeature(c.name)}
                              className="accent-teal-500"
                            />
                            {c.name} <span className="text-slate-600">({c.dtype})</span>
                          </label>
                        ))}
                    </div>
                  )}
                  <p className="text-xs text-slate-600 mt-1">
                    Par défaut, toutes les variables sauf la cible sont utilisées — décochez celles à
                    exclure (ex. un identifiant sans valeur prédictive).
                  </p>
                </div>
              )}

              <div>
                <label className="block text-sm text-slate-400 mb-1">
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
                <p className="text-xs text-slate-600 mt-1">
                  Plus élevé = recherche plus fine, mais entraînement plus long.
                </p>
              </div>

              <div>
                <label className="block text-sm text-slate-400 mb-1">
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
            </>
          )}

          {error && (
            <p className="text-sm text-rose-300 bg-rose-500/10 border border-rose-500/20 rounded-lg px-3 py-2">
              {error}
            </p>
          )}

          <Button
            type="submit"
            disabled={!targetColumn || selectedFeatures.size === 0 || isSubmitting}
            className="w-full"
          >
            <PlayCircle size={16} />
            {isSubmitting ? "Lancement…" : "Lancer l'entraînement"}
          </Button>
        </form>
      )}
    </Card>
  );
}
