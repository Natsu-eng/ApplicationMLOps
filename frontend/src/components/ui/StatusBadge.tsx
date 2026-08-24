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
    case "cancelled":
      // Bug réel trouvé en revue (Lot bulk-select/cohérence) — ce cas
      // n'était jamais géré, un job "cancelled" retombait dans le `default`
      // ci-dessous et s'affichait "En file", identique à un job réellement
      // en attente : trompeur pour un utilisateur qui a explicitement
      // annulé un job (rien n'indique que l'annulation a bien eu lieu).
      return (
        <Badge variant="neutral" dot>
          Annulé
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
