import { useCallback, useEffect, useRef, useState, type DragEvent } from "react";
import { AlertCircle, UploadCloud } from "lucide-react";
import {
  ApiError,
  api,
  type VisionDatasetDetail,
  type VisionDatasetStructureType,
  type VisionDatasetSummary,
} from "../../api/client";
import { Card } from "../ui/Card";
import { Button } from "../ui/Button";
import { Select } from "../ui/Select";

const STRUCTURE_LABELS: Record<VisionDatasetStructureType, string> = {
  classification: "un dossier par classe",
  mvtec_ad: "MVTec AD (train/good + test/good + test/<défaut>)",
};

/** Sélecteur de dataset image partagé — upload (ZIP) OU choix d'un dataset
 * déjà prêt, filtré par structure attendue (classification vs MVTec AD).
 * Pas de page dédiée séparée (décision actée au sous-lot A) : ce composant
 * est intégré directement dans le wizard de chaque module vision. */
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
  const fileInputRef = useRef<HTMLInputElement>(null);

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

  async function handleFiles(files: FileList | null) {
    const file = files?.[0];
    if (!file) return;
    setIsUploading(true);
    setError(null);
    try {
      const detail = await api.visionDatasets.upload(file);
      await load();
      if (detail.status === "error") {
        setError(detail.error_message ?? "Structure du ZIP non reconnue");
      } else if (detail.structure_type !== structureType) {
        setError(
          `Ce ZIP a été détecté comme "${STRUCTURE_LABELS[detail.structure_type]}" — attendu ici : "${STRUCTURE_LABELS[structureType]}".`,
        );
      } else {
        onChange(detail.id, detail);
        setSelectedDetail(detail);
      }
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Échec de l'upload");
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
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
    handleFiles(event.dataTransfer.files);
  }

  return (
    <div className="space-y-3">
      <Card
        className={`p-4 border-dashed flex items-center gap-4 transition-colors ${isDragging ? "border-primary/60 bg-primary/5" : ""}`}
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
          accept=".zip"
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
        <UploadCloud className="text-muted-foreground flex-shrink-0" size={20} />
        <div className="min-w-0 flex-1">
          <p className="text-sm text-foreground">Glissez une archive .zip ici, ou parcourez</p>
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
      </Card>

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
          {selectedDetail.n_images} images
          {selectedDetail.n_classes ? `, ${selectedDetail.n_classes} classes` : ""} —{" "}
          {Object.entries(selectedDetail.class_distribution)
            .map(([name, count]) => `${name} (${count})`)
            .join(" · ")}
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
    </div>
  );
}
