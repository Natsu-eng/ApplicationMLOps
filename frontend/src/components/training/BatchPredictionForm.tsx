import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";
import { AlertCircle, Ban, Download, History, Loader2, Trash2, UploadCloud } from "lucide-react";
import { ApiError, api, type BatchPredictionJobSummary } from "../../api/client";
import { Button } from "../ui/Button";
import { Card } from "../ui/Card";
import { SectionHeader } from "../ui/SectionHeader";
import { Table, type TableColumn } from "../ui/Table";
import { JobStatusBadge } from "../ui/StatusBadge";
import { useJobEvents } from "../../hooks/useJobEvents";
import { useConfirmAction } from "../../hooks/useConfirmAction";
import { formatDateTime } from "../../utils/format";

const ACTIVE_STATUSES = new Set(["queued", "running"]);

/** Prédiction en lot (retour utilisateur direct : "batch prediction —
 * upload d'un fichier, prédictions pour toutes les lignes") — même esprit
 * que `PredictionForm.tsx` (une observation à la fois) mais pour un fichier
 * entier, traité en tâche de fond (voir `POST /training/jobs/{id}/predict-batch`).
 * Historique scopé à CE job (filtré côté client — pas de query serveur
 * dédiée, l'historique par organisation reste de taille raisonnable). */
export default function BatchPredictionForm({ jobId }: { jobId: number }) {
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [activeBatch, setActiveBatch] = useState<BatchPredictionJobSummary | null>(null);
  const [history, setHistory] = useState<BatchPredictionJobSummary[] | null>(null);
  const [showHistory, setShowHistory] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const confirmDelete = useConfirmAction<number>();

  const loadHistory = useCallback(async () => {
    try {
      const all = await api.training.listBatchPredictions();
      setHistory(all.filter((b) => b.training_job_id === jobId));
    } catch {
      setHistory([]);
    }
  }, [jobId]);

  useEffect(() => {
    if (showHistory && history === null) loadHistory();
  }, [showHistory, history, loadHistory]);

  // Notifications de progression (Lot 7, §J.2, même mécanisme que les
  // autres jobs) — désactivées dès que le lot actif n'est plus queued/running.
  const activeBatchId = activeBatch && ACTIVE_STATUSES.has(activeBatch.status) ? activeBatch.id : null;
  useJobEvents(
    activeBatchId !== null ? `/training/batch-predictions/${activeBatchId}/events` : null,
    (snapshot) => {
      setActiveBatch((prev) => (prev ? { ...prev, ...snapshot } : prev));
      if (!ACTIVE_STATUSES.has(snapshot.status) && activeBatchId !== null) {
        // Le lot vient de se terminer (succès, échec ou annulation) —
        // rafraîchit la liste si elle est déjà affichée, pour que la ligne
        // corresponde à l'état final sans attendre un nouveau montage.
        setHistory((prev) =>
          prev ? prev.map((b) => (b.id === activeBatchId ? { ...b, ...snapshot } : b)) : prev,
        );
      }
    },
  );

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (!file) return;
    setError(null);
    setIsSubmitting(true);
    try {
      const batch = await api.training.createBatchPrediction(jobId, file);
      setActiveBatch(batch);
      setHistory((prev) => (prev ? [batch, ...prev] : prev));
      setFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Impossible de lancer la prédiction en lot");
    } finally {
      setIsSubmitting(false);
    }
  }

  async function handleCancel() {
    if (!activeBatch) return;
    const updated = await api.training.cancelBatchPrediction(activeBatch.id);
    setActiveBatch(updated);
  }

  async function handleDelete(batchId: number) {
    await api.training.removeBatchPrediction(batchId);
    setHistory((prev) => prev?.filter((b) => b.id !== batchId) ?? null);
    if (activeBatch?.id === batchId) setActiveBatch(null);
  }

  const columns: TableColumn<BatchPredictionJobSummary>[] = [
    { key: "input_filename", header: "Fichier", render: (b) => b.input_filename },
    { key: "status", header: "Statut", render: (b) => <JobStatusBadge status={b.status} /> },
    { key: "n_rows", header: "Lignes", align: "right", render: (b) => (b.n_rows != null ? String(b.n_rows) : "—") },
    { key: "created_at", header: "Lancé", render: (b) => formatDateTime(b.created_at) },
    {
      key: "actions",
      header: "",
      align: "right",
      render: (b) => (
        <div className="flex items-center justify-end gap-2">
          {b.status === "completed" && (
            <button
              type="button"
              onClick={() => api.training.downloadBatchPredictionResult(b.id, `predictions_${b.input_filename}`)}
              className="text-primary hover:text-primary/80"
              aria-label={`Télécharger le résultat de ${b.input_filename}`}
            >
              <Download size={14} />
            </button>
          )}
          <button
            type="button"
            onClick={() => confirmDelete.trigger(b.id, () => handleDelete(b.id))}
            onMouseLeave={confirmDelete.reset}
            className={confirmDelete.isPending(b.id) ? "text-destructive" : "text-muted-foreground hover:text-destructive"}
            aria-label={confirmDelete.isPending(b.id) ? `Confirmer la suppression de ${b.input_filename}` : `Supprimer ${b.input_filename}`}
            title={confirmDelete.isPending(b.id) ? "Cliquez à nouveau pour confirmer" : "Supprimer"}
          >
            <Trash2 size={14} />
          </button>
        </div>
      ),
    },
  ];

  return (
    <div className="space-y-5">
      <Card className="p-5">
        <SectionHeader
          icon={UploadCloud}
          color="violet"
          label="Prédiction en lot"
          help="Uploadez un fichier (csv/xlsx/xls/parquet/json) contenant les mêmes colonnes que celles utilisées à l'entraînement — une prédiction est calculée pour CHAQUE ligne, résultat téléchargeable une fois terminé."
        />

        {!activeBatch || !ACTIVE_STATUSES.has(activeBatch.status) ? (
          <form onSubmit={handleSubmit} className="space-y-3">
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv,.xlsx,.xls,.parquet,.json"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              className="block w-full text-sm text-muted-foreground file:mr-3 file:rounded-lg file:border-0 file:bg-primary/10 file:px-3 file:py-1.5 file:text-primary file:text-sm hover:file:bg-primary/20"
            />
            {error && (
              <div className="flex items-center gap-2 text-sm text-destructive bg-destructive/10 border border-destructive/20 rounded-lg px-3 py-2">
                <AlertCircle size={15} className="flex-shrink-0" />
                {error}
              </div>
            )}
            <Button type="submit" disabled={!file || isSubmitting} size="sm">
              {isSubmitting ? "Envoi…" : "Lancer la prédiction en lot"}
            </Button>
          </form>
        ) : (
          <div className="rounded-lg border border-border bg-muted p-4">
            <div className="flex items-center justify-between mb-2">
              <p className="text-sm text-foreground flex items-center gap-2">
                <Loader2 size={14} className="animate-spin text-primary" />
                {activeBatch.input_filename}
              </p>
              <Button type="button" variant="ghost" size="sm" onClick={handleCancel}>
                <Ban size={13} />
                Annuler
              </Button>
            </div>
            <div className="h-1.5 rounded-full bg-border overflow-hidden mb-1.5">
              <div
                className="h-full rounded-full bg-primary transition-all duration-500"
                style={{ width: `${Math.max(activeBatch.progress_percent, 4)}%` }}
              />
            </div>
            <p className="text-xs text-muted-foreground">
              {activeBatch.progress_step ?? "En file d'attente"} — {activeBatch.progress_percent}%
            </p>
          </div>
        )}

        {activeBatch?.status === "failed" && (
          <div className="mt-3 flex items-center gap-2 text-sm text-destructive bg-destructive/10 border border-destructive/20 rounded-lg px-3 py-2">
            <AlertCircle size={15} className="flex-shrink-0" />
            {activeBatch.error_message}
          </div>
        )}
        {activeBatch?.status === "completed" && (
          <div className="mt-3 flex items-center justify-between gap-2 text-sm bg-success/10 border border-success/20 rounded-lg px-3 py-2">
            <span className="text-foreground">
              Terminé — {activeBatch.n_rows} ligne{(activeBatch.n_rows ?? 0) > 1 ? "s" : ""} prédite
              {(activeBatch.n_rows ?? 0) > 1 ? "s" : ""}.
            </span>
            <Button
              type="button"
              size="sm"
              onClick={() =>
                api.training.downloadBatchPredictionResult(activeBatch.id, `predictions_${activeBatch.input_filename}`)
              }
            >
              <Download size={13} />
              Télécharger
            </Button>
          </div>
        )}
      </Card>

      <div>
        <button
          type="button"
          onClick={() => setShowHistory((v) => !v)}
          className="flex items-center gap-1.5 text-sm text-primary hover:underline underline-offset-2"
        >
          <History size={14} />
          {showHistory ? "Masquer l'historique" : "Voir l'historique des prédictions en lot"}
        </button>
        {showHistory && (
          <div className="mt-3">
            {history === null ? (
              <p className="text-sm text-muted-foreground">Chargement…</p>
            ) : history.length === 0 ? (
              <p className="text-sm text-muted-foreground">Aucune prédiction en lot pour ce modèle pour l'instant.</p>
            ) : (
              <Table columns={columns} rows={history} rowKey={(b) => b.id} pageSize={10} />
            )}
          </div>
        )}
      </div>
    </div>
  );
}
