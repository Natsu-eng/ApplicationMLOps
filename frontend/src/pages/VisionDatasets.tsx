import { useCallback, useEffect, useRef, useState, type DragEvent } from "react";
import { AlertCircle, Database, Images, Trash2, UploadCloud } from "lucide-react";
import { ApiError, api, type VisionDatasetDetail, type VisionDatasetSummary } from "../api/client";
import AppShell from "../components/AppShell";
import { pillarColor } from "../config/pillars";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { PageHeader } from "../components/ui/PageHeader";
import { DatasetStatusBadge } from "../components/ui/StatusBadge";
import { Table, type TableColumn } from "../components/ui/Table";
import { useConfirmAction } from "../hooks/useConfirmAction";
import { VisionDatasetExplorer } from "../components/vision/VisionDatasetExplorer";
import { formatDateTime } from "../utils/format";

// "mvtec_ad" reste la valeur technique stockée en base (structure_type,
// jamais renommée — voir DECISIONS.md D0.3/D6A.x) : seul le LIBELLÉ
// affiché change, la structure détectée (train/good + test/good +
// test/<défaut>) est générique, pas exclusive au jeu de données industriel
// MVTec AD au sens strict.
const STRUCTURE_LABELS: Record<string, string> = {
  classification: "Classification",
  mvtec_ad: "Normal / défaut",
};

/** Page "Mes données Vision" (Lot 16E) — jusqu'ici absente : la gestion des
 * datasets image n'existait qu'intégrée au wizard de chaque module
 * (`VisionDatasetPicker.tsx`, décision volontaire du Lot 15 sous-lot A),
 * sans aucun moyen de lister/supprimer un dataset en dehors d'un
 * entraînement en cours de configuration — rupture de parité avec "Mes
 * données" (pilier supervisé) et "Explorer mes données" (pilier non
 * supervisé), qui ont chacun leur page dédiée. Réutilise
 * `VisionDatasetExplorer` (Lot 16C) pour l'exploration détaillée. */
export default function VisionDatasets() {
  const [datasets, setDatasets] = useState<VisionDatasetSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [exploring, setExploring] = useState<VisionDatasetDetail | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);
  const confirmDelete = useConfirmAction<number>();

  const load = useCallback(() => {
    api.visionDatasets
      .list()
      .then(setDatasets)
      .catch((err) => setError(err instanceof ApiError ? err.message : "Impossible de charger vos datasets"));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    // `webkitdirectory`/`directory` : attributs non standard, absents du
    // typage JSX de React — posés impérativement plutôt qu'un cast de
    // props (même résultat, jamais besoin de `as any` sur le JSX).
    folderInputRef.current?.setAttribute("webkitdirectory", "");
    folderInputRef.current?.setAttribute("directory", "");
  }, []);

  async function handleArchiveFiles(files: FileList | null) {
    const file = files?.[0];
    if (!file) return;
    setIsUploading(true);
    setUploadError(null);
    try {
      await api.visionDatasets.upload(file);
      load();
    } catch (err) {
      setUploadError(err instanceof ApiError ? err.message : "Échec de l'upload");
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  // Lot 6A (correctif I9/import général) — sélection d'un DOSSIER entier
  // (input `webkitdirectory`, pas de glisser-déposer : la traversée d'une
  // arborescence de dossiers déposée exigerait l'API DataTransferItem
  // (`webkitGetAsEntry`), un chantier séparé — le bouton "Importer un
  // dossier" reste le chemin principal, robuste sur tous les navigateurs
  // qui supportent `webkitdirectory`).
  async function handleFolderFiles(files: FileList | null) {
    if (!files || files.length === 0) return;
    setIsUploading(true);
    setUploadError(null);
    try {
      await api.visionDatasets.uploadFolder(Array.from(files));
      load();
    } catch (err) {
      setUploadError(err instanceof ApiError ? err.message : "Échec de l'upload");
    } finally {
      setIsUploading(false);
      if (folderInputRef.current) folderInputRef.current.value = "";
    }
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDragging(false);
    handleArchiveFiles(event.dataTransfer.files);
  }

  async function handleDelete(id: number) {
    try {
      await api.visionDatasets.remove(id);
    } catch {
      // best-effort — la liste est de toute façon rechargée
    }
    load();
  }

  async function handleExplore(id: number) {
    try {
      setExploring(await api.visionDatasets.get(id));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Impossible de charger ce dataset");
    }
  }

  const columns: TableColumn<VisionDatasetSummary>[] = [
    { key: "name", header: "Nom", render: (d) => <span className="font-medium text-foreground">{d.name}</span> },
    {
      key: "structure_type",
      header: "Structure",
      render: (d) => STRUCTURE_LABELS[d.structure_type] ?? d.structure_type ?? "—",
    },
    { key: "n_images", header: "Images", align: "right" },
    { key: "n_classes", header: "Classes", align: "right", render: (d) => d.n_classes ?? "—" },
    { key: "status", header: "Statut", render: (d) => <DatasetStatusBadge status={d.status} /> },
    { key: "uploaded_by", header: "Importé par", render: (d) => d.uploaded_by ?? "—", className: "text-muted-foreground" },
    { key: "created_at", header: "Date", render: (d) => formatDateTime(d.created_at), className: "text-muted-foreground" },
    {
      key: "actions",
      header: "",
      align: "right",
      render: (d) => (
        <div className="flex items-center justify-end gap-2">
          {d.status === "ready" && (
            <button
              type="button"
              onClick={() => handleExplore(d.id)}
              className="inline-flex items-center gap-1 text-primary hover:text-primary/80 text-xs font-medium"
            >
              <Images size={12} />
              Explorer
            </button>
          )}
          <button
            type="button"
            onClick={() => confirmDelete.trigger(d.id, () => handleDelete(d.id))}
            onMouseLeave={confirmDelete.reset}
            aria-label={confirmDelete.isPending(d.id) ? "Confirmer la suppression" : "Supprimer ce dataset"}
            title={confirmDelete.isPending(d.id) ? "Cliquer à nouveau pour confirmer" : "Supprimer ce dataset"}
            className={`p-1.5 rounded-md transition-colors ${
              confirmDelete.isPending(d.id)
                ? "text-destructive bg-destructive/15"
                : "text-muted-foreground/60 hover:text-destructive hover:bg-destructive/10"
            }`}
          >
            <Trash2 size={13} />
          </button>
        </div>
      ),
    },
  ];

  return (
    <AppShell pillarId="vision">
      <PageHeader
        eyebrow="Vision"
        title="Mes données"
        description="Importez, explorez et gérez vos datasets d'images — dossiers de classes pour la classification, ou structure normal/défaut (train/good + test/good + test/<défaut>) pour la détection d'anomalies visuelles."
        icon={Database}
        color={pillarColor("vision")}
      />

      <Card
        className={`p-4 mb-5 border-dashed flex items-center gap-4 transition-colors ${isDragging ? "border-primary/60 bg-primary/5" : ""}`}
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
          accept=".zip,.tar,.tar.gz,.tgz"
          className="hidden"
          onChange={(e) => handleArchiveFiles(e.target.files)}
        />
        <input ref={folderInputRef} type="file" multiple className="hidden" onChange={(e) => handleFolderFiles(e.target.files)} />
        <UploadCloud className="text-muted-foreground flex-shrink-0" size={20} />
        <div className="min-w-0 flex-1">
          <p className="text-sm text-foreground">Glissez une archive (.zip, .tar, .tar.gz) ici, ou parcourez</p>
          <p className="text-xs text-muted-foreground">
            Structure détectée automatiquement — un dossier par classe, ou train/good + test/good + test/&lt;défaut&gt;
          </p>
        </div>
        <Button variant="secondary" size="sm" type="button" onClick={() => fileInputRef.current?.click()} disabled={isUploading} className="flex-shrink-0">
          {isUploading ? "Envoi…" : "Parcourir"}
        </Button>
        <Button variant="secondary" size="sm" type="button" onClick={() => folderInputRef.current?.click()} disabled={isUploading} className="flex-shrink-0">
          Importer un dossier
        </Button>
      </Card>

      {uploadError && (
        <div className="flex items-center gap-2 text-sm text-destructive bg-destructive/10 border border-destructive/20 rounded-lg px-3 py-2 mb-5">
          <AlertCircle size={15} className="flex-shrink-0" />
          {uploadError}
        </div>
      )}

      {error ? (
        <p className="text-sm text-destructive">{error}</p>
      ) : datasets === null ? (
        <p className="text-sm text-muted-foreground">Chargement…</p>
      ) : datasets.length === 0 ? (
        <Card className="p-5">
          <p className="text-sm text-muted-foreground">Aucun dataset d'images importé pour l'instant.</p>
        </Card>
      ) : (
        <Table columns={columns} rows={datasets} rowKey={(d) => d.id} />
      )}

      {exploring && <VisionDatasetExplorer dataset={exploring} onClose={() => setExploring(null)} />}
    </AppShell>
  );
}
