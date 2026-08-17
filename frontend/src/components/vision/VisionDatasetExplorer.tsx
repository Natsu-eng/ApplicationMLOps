import { useEffect, useState } from "react";
import { AlertTriangle, Images } from "lucide-react";
import { api, type VisionDatasetDetail, type VisionDatasetImageList } from "../../api/client";
import { Modal } from "../ui/Modal";
import { Tabs, type TabItem } from "../ui/Tabs";
import { VisionImage } from "./VisionImage";

type TabId = "apercu" | "qualite";

const TABS: TabItem<TabId>[] = [
  { id: "apercu", label: "Aperçu", icon: Images },
  { id: "qualite", label: "Qualité", icon: AlertTriangle },
];

/** Exploration complète d'un dataset Vision (Lot 16C) — galerie de
 * miniatures par classe + rapport de qualité détaillé (corrompues,
 * doublons, sous-dimensionnées, déséquilibre). Le rapport est déjà calculé
 * intégralement côté backend depuis le Lot 15 sous-lot A
 * (`VisionDatasetDetail.validation_report`) — ce composant ne fait
 * qu'enfin l'afficher en détail, jamais un nouveau calcul. */
export function VisionDatasetExplorer({ dataset, onClose }: { dataset: VisionDatasetDetail; onClose: () => void }) {
  const [activeTab, setActiveTab] = useState<TabId>("apercu");
  return (
    <Modal title={`Explorer — ${dataset.name}`} onClose={onClose} size="xl">
      <Tabs items={TABS} active={activeTab} onChange={setActiveTab} />
      <div className="mt-4">
        {activeTab === "apercu" ? <DatasetGallery dataset={dataset} /> : <DatasetQualityReport dataset={dataset} />}
      </div>
    </Modal>
  );
}

function DatasetGallery({ dataset }: { dataset: VisionDatasetDetail }) {
  const classNames = Object.keys(dataset.class_distribution);
  if (classNames.length === 0) {
    return <p className="text-sm text-muted-foreground">Aucune classe à afficher.</p>;
  }
  return (
    <div className="space-y-6 max-h-[65vh] overflow-y-auto pr-1">
      {classNames.map((className) => (
        <ClassGallerySection
          key={className}
          datasetId={dataset.id}
          className_={className}
          count={dataset.class_distribution[className]}
        />
      ))}
    </div>
  );
}

function ClassGallerySection({
  datasetId,
  className_,
  count,
}: {
  datasetId: number;
  className_: string;
  count: number;
}) {
  const [images, setImages] = useState<VisionDatasetImageList | null>(null);

  useEffect(() => {
    setImages(null);
    api.visionDatasets.listImages(datasetId, className_).then(setImages).catch(() => setImages({ class_name: className_, total: 0, paths: [] }));
  }, [datasetId, className_]);

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <p className="text-sm font-medium text-foreground">{className_}</p>
        <p className="text-xs text-muted-foreground tabular-nums">
          {count} image{count > 1 ? "s" : ""}
        </p>
      </div>
      {!images ? (
        <div className="grid grid-cols-6 sm:grid-cols-10 gap-2">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="aspect-square rounded-md bg-muted animate-pulse" />
          ))}
        </div>
      ) : (
        <>
          <div className="grid grid-cols-6 sm:grid-cols-10 gap-2">
            {images.paths.map((path) => (
              <VisionImage
                key={path}
                datasetId={datasetId}
                path={path}
                alt={path}
                className="w-full aspect-square object-cover rounded-md border border-border"
              />
            ))}
          </div>
          {images.total > images.paths.length && (
            <p className="text-caption text-muted-foreground mt-1.5">
              {images.paths.length} sur {images.total} images affichées.
            </p>
          )}
        </>
      )}
    </div>
  );
}

function DatasetQualityReport({ dataset }: { dataset: VisionDatasetDetail }) {
  const report = dataset.validation_report;
  return (
    <div className="space-y-4 max-h-[65vh] overflow-y-auto pr-1">
      {report.warnings.length > 0 ? (
        <div className="rounded-lg border border-warning/20 bg-warning/10 p-3 space-y-1.5">
          {report.warnings.map((w) => (
            <p key={w} className="text-sm text-warning flex items-start gap-2">
              <AlertTriangle size={14} className="flex-shrink-0 mt-0.5" />
              {w}
            </p>
          ))}
        </div>
      ) : (
        <p className="text-sm text-muted-foreground">Aucune alerte de qualité détectée sur ce dataset.</p>
      )}

      <QualityFileList title={`Images corrompues ou illisibles (${report.n_corrupted})`} files={report.corrupted_files} />
      <QualityFileList title={`Images trop petites (${report.n_undersized})`} files={report.undersized_files} />

      {(report.duplicate_removed_files?.length ?? 0) > 0 && (
        <QualityFileList
          title={`Doublons exclus (${report.n_duplicates_removed ?? 0}) — une seule copie conservée par doublon`}
          files={report.duplicate_removed_files ?? []}
        />
      )}

      {(report.label_conflicts?.length ?? 0) > 0 && (
        <div>
          <p className="text-xs uppercase tracking-wide text-muted-foreground mb-1.5">
            Conflits d'étiquette — mêmes images trouvées dans des classes différentes, toutes exclues
          </p>
          <div className="max-h-32 overflow-y-auto rounded-lg border border-border divide-y divide-border/60">
            {report.label_conflicts!.map((conflict, i) => (
              <p key={i} className="text-xs text-foreground/80 font-mono px-2 py-1 truncate">
                {conflict.categories.join(" ↔ ")} : {conflict.paths.join(" = ")}
              </p>
            ))}
          </div>
        </div>
      )}

      {report.duplicate_detection_note && (
        <p className="text-xs text-muted-foreground italic">{report.duplicate_detection_note}</p>
      )}
    </div>
  );
}

function QualityFileList({ title, files }: { title: string; files: string[] }) {
  if (files.length === 0) return null;
  return (
    <div>
      <p className="text-xs uppercase tracking-wide text-muted-foreground mb-1.5">{title}</p>
      <div className="max-h-32 overflow-y-auto rounded-lg border border-border divide-y divide-border/60">
        {files.map((f) => (
          <p key={f} className="text-xs text-foreground/80 font-mono px-2 py-1 truncate">
            {f}
          </p>
        ))}
      </div>
    </div>
  );
}
