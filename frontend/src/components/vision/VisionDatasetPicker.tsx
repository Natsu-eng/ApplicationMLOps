import { useCallback, useEffect, useRef, useState, type DragEvent } from "react";
import { AlertCircle, Images, UploadCloud } from "lucide-react";
import {
  ApiError,
  api,
  type VisionDatasetDetail,
  type VisionDatasetStructureType,
  type VisionDatasetSummary,
} from "../../api/client";
import { Button } from "../ui/Button";
import { Select } from "../ui/Select";
import { VisionDatasetExplorer } from "./VisionDatasetExplorer";

// "mvtec_ad" reste la valeur technique stockée en base — voir
// DECISIONS.md D0.3/D6A.x : seul le libellé change, structure générique,
// pas exclusive au jeu de données industriel MVTec AD au sens strict.
const STRUCTURE_LABELS: Record<VisionDatasetStructureType, string> = {
  classification: "un dossier par classe",
  mvtec_ad: "Normal / défaut (train/good + test/good + test/<défaut>)",
};

/** Sélecteur de dataset image partagé — upload (archive .zip/.tar/.tar.gz
 * ou dossier, Lot 6A) OU choix d'un dataset déjà prêt, filtré par
 * structure attendue (classification vs normal/défaut). Pas de page
 * dédiée séparée (décision actée au sous-lot A) : ce composant est
 * intégré directement dans le wizard de chaque module vision. */
export function VisionDatasetPicker({
  structureType,
  value,
  onChange,
}: {
  structureType: VisionDatasetStructureType;
  value: number | "";
  onChange: (id: number | "", detail: VisionDatasetDetail | null) => void;
}) {
  const [datasets, setDatasets] = useState<VisionDatasetSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [selectedDetail, setSelectedDetail] = useState<VisionDatasetDetail | null>(null);
  const [exploring, setExploring] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

  const load = useCallback(async () => {
    try {
      const all = await api.visionDatasets.list();
      setDatasets(all.filter((d) => d.status === "ready" && d.structure_type === structureType));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Impossible de charger vos datasets");
    }
  }, [structureType]);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    // Attributs non standard, absents du typage JSX de React (voir
    // VisionDatasets.tsx, même motif) — posés impérativement.
    folderInputRef.current?.setAttribute("webkitdirectory", "");
    folderInputRef.current?.setAttribute("directory", "");
  }, []);

  function applyUploadedDataset(detail: VisionDatasetDetail) {
    if (detail.status === "error") {
      setError(detail.error_message ?? "Structure de l'archive non reconnue");
    } else if (detail.structure_type !== structureType) {
      setError(
        `Ce dataset a été détecté comme "${STRUCTURE_LABELS[detail.structure_type]}" — attendu ici : "${STRUCTURE_LABELS[structureType]}".`,
      );
    } else {
      onChange(detail.id, detail);
      setSelectedDetail(detail);
    }
  }

  async function handleArchiveFiles(files: FileList | null) {
    const file = files?.[0];
    if (!file) return;
    setIsUploading(true);
    setError(null);
    try {
      applyUploadedDataset(await api.visionDatasets.upload(file));
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Échec de l'upload");
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  async function handleFolderFiles(files: FileList | null) {
    if (!files || files.length === 0) return;
    setIsUploading(true);
    setError(null);
    try {
      applyUploadedDataset(await api.visionDatasets.uploadFolder(Array.from(files)));
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Échec de l'upload");
    } finally {
      setIsUploading(false);
      if (folderInputRef.current) folderInputRef.current.value = "";
    }
  }

  async function handleSelect(id: string) {
    if (!id) {
      onChange("", null);
      setSelectedDetail(null);
      return;
    }
    const numericId = Number(id);
    onChange(numericId, null);
    try {
      setSelectedDetail(await api.visionDatasets.get(numericId));
    } catch {
      // silencieux — le job sera de toute façon revalidé côté serveur à la création
    }
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDragging(false);
    handleArchiveFiles(event.dataTransfer.files);
  }

  return (
    <div className="space-y-3">
      {/* Filet (pas une Card) — ce sélecteur est toujours intégré DANS la Card
          du formulaire appelant (Lot 2A correctif 3) : une Card ici créerait
          une carte dans la carte. Une bordure pointillée à faible opacité
          suffit à signaler la zone de dépôt sans dupliquer le chrome. */}
      <div
        className={`rounded-card border border-dashed border-border/70 p-4 flex items-center gap-4 transition-colors ${isDragging ? "border-primary/60 bg-primary/5" : ""}`}
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
        <input
          ref={folderInputRef}
          type="file"
          multiple
          className="hidden"
          onChange={(e) => handleFolderFiles(e.target.files)}
        />
        <UploadCloud className="text-muted-foreground flex-shrink-0" size={20} />
        <div className="min-w-0 flex-1">
          <p className="text-sm text-foreground">Glissez une archive (.zip, .tar, .tar.gz) ici, ou parcourez</p>
          <p className="text-xs text-muted-foreground">Structure attendue : {STRUCTURE_LABELS[structureType]}</p>
        </div>
        <Button
          variant="secondary"
          size="sm"
          type="button"
          onClick={() => fileInputRef.current?.click()}
          disabled={isUploading}
          className="flex-shrink-0"
        >
          {isUploading ? "Envoi…" : "Parcourir"}
        </Button>
        <Button
          variant="secondary"
          size="sm"
          type="button"
          onClick={() => folderInputRef.current?.click()}
          disabled={isUploading}
          className="flex-shrink-0"
        >
          Dossier
        </Button>
      </div>

      {datasets.length > 0 && (
        <div>
          <label htmlFor="vision-dataset-select" className="block text-sm text-muted-foreground mb-1">
            Ou choisir un dataset déjà importé
          </label>
          <Select id="vision-dataset-select" value={value} onChange={(e) => handleSelect(e.target.value)}>
            <option value="">Choisir…</option>
            {datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name} ({d.n_images} images{d.n_classes ? `, ${d.n_classes} classes` : ""})
              </option>
            ))}
          </Select>
        </div>
      )}

      {error && (
        <div className="flex items-center gap-2 text-sm text-destructive bg-destructive/10 border border-destructive/20 rounded-lg px-3 py-2">
          <AlertCircle size={15} className="flex-shrink-0" />
          {error}
        </div>
      )}

      {selectedDetail && (
        <div className="rounded-lg border border-border bg-muted/40 px-3 py-2 text-xs text-muted-foreground">
          <div className="flex items-start justify-between gap-3">
            <p>
              {selectedDetail.n_images} images
              {selectedDetail.n_classes ? `, ${selectedDetail.n_classes} classes` : ""} —{" "}
              {Object.entries(selectedDetail.class_distribution)
                .map(([name, count]) => `${name} (${count})`)
                .join(" · ")}
            </p>
            <button
              type="button"
              onClick={() => setExploring(true)}
              className="flex-shrink-0 inline-flex items-center gap-1 text-primary hover:text-primary/80 font-medium"
            >
              <Images size={12} />
              Explorer
            </button>
          </div>
          {selectedDetail.validation_report.warnings.length > 0 && (
            <ul className="mt-1.5 space-y-0.5">
              {selectedDetail.validation_report.warnings.map((w) => (
                <li key={w} className="text-warning">
                  ⚠ {w}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {exploring && selectedDetail && (
        <VisionDatasetExplorer dataset={selectedDetail} onClose={() => setExploring(false)} />
      )}
    </div>
  );
}
