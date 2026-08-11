import { useCallback, useEffect, useRef, useState, type DragEvent } from "react";
import { AlertCircle, ChartColumn, Eye, FileSpreadsheet, Trash2, UploadCloud } from "lucide-react";
import { ApiError, api, type DatasetSummary, type PreviewResponse } from "../api/client";
import AppShell from "../components/AppShell";
import EdaModal from "../components/datasets/EdaModal";
import { Badge } from "../components/ui/Badge";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { ColorIconBadge, accentBarClass, accentColorForId } from "../components/ui/ColorIconBadge";
import { Modal } from "../components/ui/Modal";
import { formatDate, formatFileSize } from "../utils/format";

const ACCEPTED_EXTENSIONS = ".csv,.parquet,.xlsx,.xls,.json";

export default function Datasets() {
  const [datasets, setDatasets] = useState<DatasetSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [previewing, setPreviewing] = useState<DatasetSummary | null>(null);
  const [exploring, setExploring] = useState<DatasetSummary | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const load = useCallback(async () => {
    try {
      setDatasets(await api.datasets.list());
      setError(null);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Impossible de charger les datasets");
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  async function handleFiles(files: FileList | null) {
    const file = files?.[0];
    if (!file) return;
    setIsUploading(true);
    setError(null);
    try {
      await api.datasets.upload(file);
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Échec de l'upload");
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  async function handleDelete(id: number) {
    try {
      await api.datasets.remove(id);
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Suppression impossible");
    }
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDragging(false);
    handleFiles(event.dataTransfer.files);
  }

  return (
    <AppShell pillarId="supervised">
      <div className="mb-6 flex items-end justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-widest text-primary font-semibold mb-1">
            Données
          </p>
          <h1 className="text-2xl font-serif text-slate-900">Mes données</h1>
        </div>
        {datasets && datasets.length > 0 && (
          <Badge variant="neutral">
            {datasets.length} dataset{datasets.length > 1 ? "s" : ""}
          </Badge>
        )}
      </div>

      {/* Bande d'upload compacte — l'information consultée en permanence,
          c'est la grille des datasets ci-dessous, pas cette zone (E1-ter :
          l'upload prenait auparavant la moitié de l'écran et repoussait la
          grille hors-vue). Le dépôt par glisser-déposer reste actif sur
          toute la bande. */}
      <Card
        className={`p-4 mb-6 border-dashed flex items-center gap-4 transition-colors ${
          isDragging ? "border-primary/60 bg-primary/5" : ""
        }`}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={onDrop}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept={ACCEPTED_EXTENSIONS}
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
        <UploadCloud className="text-slate-400 flex-shrink-0" size={22} />
        <div className="min-w-0 flex-1">
          <p className="text-sm text-slate-600">Glissez un fichier ici, ou parcourez</p>
          <p className="text-xs text-slate-400">CSV, Parquet, Excel, JSON — 200 Mo max</p>
        </div>
        <Button
          variant="secondary"
          size="sm"
          onClick={() => fileInputRef.current?.click()}
          disabled={isUploading}
          className="flex-shrink-0"
        >
          {isUploading ? "Envoi…" : "Parcourir"}
        </Button>
      </Card>

      {error && (
        <div className="flex items-center gap-2 text-sm text-rose-700 bg-rose-50 border border-rose-200 rounded-lg px-3 py-2 mb-6">
          <AlertCircle size={15} className="flex-shrink-0" />
          {error}
        </div>
      )}

      {datasets === null ? (
        <p className="text-sm text-slate-500">Chargement…</p>
      ) : datasets.length === 0 ? (
        <Card className="p-10 text-center">
          <FileSpreadsheet className="mx-auto mb-3 text-slate-300" size={32} />
          <p className="text-sm text-slate-500">
            Aucun dataset pour l'instant — importez votre premier fichier ci-dessus.
          </p>
        </Card>
      ) : (
        // Plafonné à 3 colonnes (pas 4) — au-delà, la carte devient trop
        // étroite pour les 3 boutons d'action (Aperçu/Explorer/Supprimer),
        // qui se tronquent ou débordent (bug réel signalé en usage).
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {datasets.map((dataset) => (
            <DatasetCard
              key={dataset.id}
              dataset={dataset}
              onPreview={() => setPreviewing(dataset)}
              onExplore={() => setExploring(dataset)}
              onDelete={() => handleDelete(dataset.id)}
            />
          ))}
        </div>
      )}

      {previewing && <PreviewModal dataset={previewing} onClose={() => setPreviewing(null)} />}
      {exploring && <EdaModal dataset={exploring} onClose={() => setExploring(null)} />}
    </AppShell>
  );
}

function StatusBadge({ status }: { status: string }) {
  if (status === "ready") return <Badge variant="success" dot>Prêt</Badge>;
  if (status === "error") return <Badge variant="danger" dot>Erreur</Badge>;
  return <Badge variant="primary" dot pulse>Analyse…</Badge>;
}

function DatasetCard({
  dataset,
  onPreview,
  onExplore,
  onDelete,
}: {
  dataset: DatasetSummary;
  onPreview: () => void;
  onExplore: () => void;
  onDelete: () => void;
}) {
  const color = accentColorForId(dataset.id);
  return (
    <Card interactive className="group overflow-hidden flex flex-col">
      <div className={`h-1.5 ${accentBarClass(color)}`} aria-hidden="true" />
      <div className="p-5 flex flex-col flex-1">
        <div className="flex items-start justify-between gap-2 mb-3">
          <div className="flex items-start gap-3 min-w-0">
            <ColorIconBadge icon={FileSpreadsheet} color={color} size="sm" />
            <div className="min-w-0">
              <p className="text-sm font-medium text-slate-900 truncate" title={dataset.name}>
                {dataset.name}
              </p>
              <p className="text-xs text-slate-500 mt-0.5">
                {formatDate(dataset.created_at)}
                {dataset.uploaded_by ? ` · ${dataset.uploaded_by}` : ""}
              </p>
            </div>
          </div>
          <StatusBadge status={dataset.status} />
        </div>

        {dataset.status === "error" ? (
          <p className="text-xs text-rose-600 mb-3 line-clamp-2">{dataset.error_message}</p>
        ) : (
          <div className="flex gap-4 text-xs text-slate-500 mb-3 tabular-nums">
            <span>{dataset.row_count ?? "—"} lignes</span>
            <span>{dataset.column_count ?? "—"} colonnes</span>
            <span>{formatFileSize(dataset.file_size_bytes)}</span>
          </div>
        )}

        {/* flex-1 + min-w-0 sur Aperçu/Explorer : ces deux boutons se
            partagent l'espace et rétrécissent (texte tronqué) plutôt que de
            pousser le bouton Supprimer hors de la carte — bug réel constaté
            sur une grille à 4 colonnes (carte étroite), où le bouton
            Supprimer se retrouvait coupé par overflow-hidden. */}
        <div className="mt-auto flex items-center gap-1.5 pt-3 border-t border-slate-200">
          <Button
            variant="ghost"
            size="sm"
            onClick={onPreview}
            disabled={dataset.status !== "ready"}
            className="flex-1 min-w-0"
          >
            <Eye size={14} className="flex-shrink-0" /> <span className="truncate">Aperçu</span>
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={onExplore}
            disabled={dataset.status !== "ready"}
            className="flex-1 min-w-0"
          >
            <ChartColumn size={14} className="flex-shrink-0" /> <span className="truncate">Explorer</span>
          </Button>
          <Button
            variant="danger"
            size="sm"
            onClick={onDelete}
            aria-label="Supprimer"
            className="flex-shrink-0"
          >
            <Trash2 size={14} />
          </Button>
        </div>
      </div>
    </Card>
  );
}

function PreviewModal({ dataset, onClose }: { dataset: DatasetSummary; onClose: () => void }) {
  const [data, setData] = useState<PreviewResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.datasets
      .preview(dataset.id)
      .then(setData)
      .catch((err) => setError(err instanceof ApiError ? err.message : "Aperçu indisponible"));
  }, [dataset.id]);

  return (
    <Modal title={dataset.name} onClose={onClose}>
      {error && <p className="text-sm text-rose-600">{error}</p>}
      {!data && !error && <p className="text-sm text-slate-500">Chargement…</p>}
      {data && (
        <Card className="p-4 overflow-x-auto">
          <table className="min-w-full text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-200">
                {data.columns.map((col) => (
                  <th
                    key={col}
                    className="text-left font-medium text-slate-500 px-3 py-2 whitespace-nowrap"
                  >
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.rows.map((row, i) => (
                <tr key={i} className="border-b border-slate-100">
                  {data.columns.map((col) => (
                    <td key={col} className="px-3 py-1.5 text-slate-600 whitespace-nowrap">
                      {row[col] === null || row[col] === undefined ? "—" : String(row[col])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <p className="text-xs text-slate-400 mt-3">
            {data.sample_size} lignes affichées sur {data.row_count ?? "?"}
          </p>
        </Card>
      )}
    </Modal>
  );
}
