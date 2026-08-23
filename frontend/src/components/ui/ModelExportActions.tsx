import { useState } from "react";
import { Download, FileJson } from "lucide-react";
import { ApiError } from "../../api/client";
import { Button } from "./Button";

/** Actions d'export partagées par tous les résultats de modèle non
 * supervisé/vision (Lot 10, retour utilisateur direct — parité avec le
 * registre de modèles supervisé, `ModelResultModal.tsx`) : le bundle
 * artefact réel (`GET .../model/export`, déjà persisté par chaque worker,
 * voir JOURNAL.md) et un export JSON de la configuration/des métriques
 * déjà chargées par la page — jamais un second appel réseau pour ce
 * second export, uniquement les données déjà en mémoire. */
export function ModelExportActions({
  onExportArtifact,
  exportConfig,
  configFilename,
}: {
  onExportArtifact: () => Promise<void>;
  exportConfig: Record<string, unknown>;
  configFilename: string;
}) {
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleExportArtifact() {
    setExporting(true);
    setError(null);
    try {
      await onExportArtifact();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Impossible d'exporter l'artefact");
    } finally {
      setExporting(false);
    }
  }

  function handleExportConfig() {
    const blob = new Blob([JSON.stringify(exportConfig, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = configFilename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="flex items-center gap-2 flex-wrap">
      <Button variant="secondary" size="sm" onClick={handleExportArtifact} loading={exporting}>
        <Download size={14} />
        Exporter l'artefact
      </Button>
      <Button variant="secondary" size="sm" onClick={handleExportConfig}>
        <FileJson size={14} />
        Exporter la configuration (JSON)
      </Button>
      {error && <p className="text-xs text-destructive">{error}</p>}
    </div>
  );
}
