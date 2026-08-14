import { useCallback, useEffect, useRef, useState, type DragEvent } from "react";
import { useSearchParams } from "react-router-dom";
import { AlertCircle, ChartColumn, Database, Eye, FileSpreadsheet, Trash2, UploadCloud } from "lucide-react";
import { ApiError, api, type DatasetSummary, type PreviewResponse } from "../api/client";
import AppShell from "../components/AppShell";
import EdaModal from "../components/datasets/EdaModal";
import { Badge } from "../components/ui/Badge";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { ColorIconBadge, accentBarClass, accentColorForId } from "../components/ui/ColorIconBadge";
import { Modal } from "../components/ui/Modal";
import { PageHeader } from "../components/ui/PageHeader";
import { DatasetStatusBadge } from "../components/ui/StatusBadge";
import { useConfirmAction } from "../hooks/useConfirmAction";
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

  // Deep-linking (AUDIT_ROADMAP.md, H20/D12) — `?preview=<id>` /
  // `?explore=<id>` synchronisent l'URL avec la modale ouverte, dans les
  // deux sens : un rafraîchissement ou un lien partagé rouvre la même vue.
  const [searchParams, setSearchParams] = useSearchParams();

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

  useEffect(() => {
    const previewId = searchParams.get("preview");
    const exploreId = searchParams.get("explore");
    const id = previewId ?? exploreId;
    if (!id) return;
    api.datasets
      .get(Number(id))
      .then((dataset) => (previewId ? setPreviewing(dataset) : setExploring(dataset)))
      .catch(() => setSearchParams({}, { replace: true }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function openPreview(dataset: DatasetSummary) {
    setPreviewing(dataset);
    setSearchParams({ preview: String(dataset.id) }, { replace: false });
  }

  function closePreview() {
    setPreviewing(null);
    setSearchParams({}, { replace: false });
  }

  function openExplore(dataset: DatasetSummary) {
    setExploring(dataset);
    setSearchParams({ explore: String(dataset.id) }, { replace: false });
  }

  function closeExplore() {
    setExploring(null);
    setSearchParams({}, { replace: false });
  }

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
    // Suppression destructive : confirmation à deux clics obligatoire
    // (AUDIT_ROADMAP.md, H4/D2 — avant ce correctif, un seul clic
    // supprimait le dataset immédiatement, seule action destructrice de
    // l'app sans garde-fou, alors que le motif existe déjà ailleurs).
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
      <PageHeader
        eyebrow="Données"
        title="Mes données"
        description="Importez, explorez et préparez vos jeux de données avant de les utiliser pour un entraînement."
        icon={Database}
        color="teal"
        action={
          datasets && datasets.length > 0 ? (
            <Badge variant="neutral">
              {datasets.length} dataset{datasets.length > 1 ? "s" : ""}
            </Badge>
          ) : undefined
        }
      />

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
        <UploadCloud className="text-muted-foreground flex-shrink-0" size={22} />
        <div className="min-w-0 flex-1">
          <p className="text-sm text-foreground">Glissez un fichier ici, ou parcourez</p>
          <p className="text-xs text-muted-foreground">CSV, Parquet, Excel, JSON — 200 Mo max</p>
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
        <div className="flex items-center gap-2 text-sm text-destructive bg-destructive/10 border border-destructive/20 rounded-lg px-3 py-2 mb-6">
          <AlertCircle size={15} className="flex-shrink-0" />
          {error}
        </div>
      )}

      {datasets === null ? (
        error ? null : <p className="text-sm text-muted-foreground">Chargement…</p>
      ) : datasets.length === 0 ? (
        <Card className="p-10 text-center">
          <FileSpreadsheet className="mx-auto mb-3 text-muted-foreground/50" size={32} />
          <p className="text-sm text-muted-foreground">
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
              onPreview={() => openPreview(dataset)}
              onExplore={() => openExplore(dataset)}
              onDelete={() => handleDelete(dataset.id)}
            />
          ))}
        </div>
      )}

      {previewing && <PreviewModal dataset={previewing} onClose={closePreview} />}
      {exploring && <EdaModal dataset={exploring} onClose={closeExplore} />}
    </AppShell>
  );
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
  // Couleur décorative par IDENTITÉ (pas par statut) — revenu en arrière sur
  // retour utilisateur direct : la quasi-totalité des datasets sont "Prêt"
  // en usage réel, un accent piloté par le statut rendait la grille
  // monochrome (toutes les cartes teal). Le statut réel reste lisible via
  // StatusBadge (texte + puce), qui n'a jamais changé — variété visuelle et
  // information de statut sur deux canaux séparés plutôt que confondus.
  const color = accentColorForId(dataset.id);
  const confirmDelete = useConfirmAction<number>();
  const pending = confirmDelete.isPending(dataset.id);
  return (
    <Card interactive className="group overflow-hidden flex flex-col">
      <div className={`h-1.5 ${accentBarClass(color)}`} aria-hidden="true" />
      <div className="p-5 flex flex-col flex-1">
        <div className="flex items-start justify-between gap-2 mb-3">
          <div className="flex items-start gap-3 min-w-0">
            <ColorIconBadge icon={FileSpreadsheet} color={color} size="sm" />
            <div className="min-w-0">
              <p className="text-sm font-medium text-foreground truncate" title={dataset.name}>
                {dataset.name}
              </p>
              <p className="text-xs text-muted-foreground mt-0.5">
                {formatDate(dataset.created_at)}
                {dataset.uploaded_by ? ` · ${dataset.uploaded_by}` : ""}
              </p>
            </div>
          </div>
          <DatasetStatusBadge status={dataset.status} />
        </div>

        {dataset.status === "error" ? (
          <p className="text-xs text-destructive mb-3 line-clamp-2">{dataset.error_message}</p>
        ) : (
          <div className="flex gap-4 text-xs text-muted-foreground mb-3 tabular-nums">
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
        <div className="mt-auto flex items-center gap-1.5 pt-3 border-t border-border">
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
            onClick={() => confirmDelete.trigger(dataset.id, onDelete)}
            onMouseLeave={confirmDelete.reset}
            aria-label={pending ? "Confirmer la suppression" : "Supprimer"}
            title={pending ? "Cliquer à nouveau pour confirmer" : "Supprimer"}
            className={`flex-shrink-0 transition-shadow ${pending ? "ring-2 ring-destructive/60" : ""}`}
          >
            <Trash2 size={14} />
            {pending && <span className="text-xs">Confirmer ?</span>}
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
      {error && <p className="text-sm text-destructive">{error}</p>}
      {!data && !error && <p className="text-sm text-muted-foreground">Chargement…</p>}
      {data && (
        <Card className="p-4 overflow-x-auto">
          <table className="min-w-full text-xs border-collapse">
            <thead>
              <tr className="border-b border-border">
                {data.columns.map((col) => (
                  <th
                    key={col}
                    className="text-left font-medium text-muted-foreground px-3 py-2 whitespace-nowrap"
                  >
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.rows.map((row, i) => (
                <tr key={i} className="border-b border-border/60 hover:bg-muted/40 transition-colors">
                  {data.columns.map((col) => (
                    <td key={col} className="px-3 py-1.5 text-foreground/90 whitespace-nowrap">
                      {row[col] === null || row[col] === undefined ? "—" : String(row[col])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <p className="text-xs text-muted-foreground mt-3">
            {data.sample_size} lignes affichées sur {data.row_count ?? "?"}
          </p>
        </Card>
      )}
    </Modal>
  );
}
