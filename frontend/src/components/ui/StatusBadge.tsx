import type { DatasetSummary, TrainingJobSummary } from "../../api/client";
import { Badge } from "./Badge";

/** Badges de statut unifiés — extraits de trois implémentations identiques
 * (Dashboard.tsx, TrainingHistory.tsx, Datasets.tsx) — AUDIT_ROADMAP.md, H14. */

export function JobStatusBadge({ status }: { status: TrainingJobSummary["status"] }) {
  switch (status) {
    case "completed":
      return (
        <Badge variant="success" dot>
          Terminé
        </Badge>
      );
    case "failed":
      return (
        <Badge variant="danger" dot>
          Échec
        </Badge>
      );
    case "running":
      return (
        <Badge variant="primary" dot pulse>
          En cours
        </Badge>
      );
    default:
      return (
        <Badge variant="neutral" dot>
          En file
        </Badge>
      );
  }
}

export function DatasetStatusBadge({ status }: { status: DatasetSummary["status"] }) {
  if (status === "ready")
    return (
      <Badge variant="success" dot>
        Prêt
      </Badge>
    );
  if (status === "error")
    return (
      <Badge variant="danger" dot>
        Erreur
      </Badge>
    );
  return (
    <Badge variant="primary" dot pulse>
      Analyse…
    </Badge>
  );
}
