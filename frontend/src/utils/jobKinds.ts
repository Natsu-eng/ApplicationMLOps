import { api } from "../api/client";

/** Les 6 types de job d'analyse ML de l'app — un seul point de vérité,
 * réutilisé par `Dashboard.tsx`, `AllHistory.tsx` et toute future page qui
 * doit distinguer/agréger les 6 piliers (avant ce fichier, chaque page
 * redéfinissait son propre union type identique — `ActivityKind` dans
 * `Dashboard.tsx`, `JobKind` dans `AllHistory.tsx` — sans jamais diverger,
 * mais sans garde-fou empêchant qu'elles le fassent un jour). */
export type JobKind = "supervised" | "clustering" | "dimensionality" | "anomalies" | "vision_classification" | "vision_anomalies";

/** Appel de suppression correspondant à chaque type — un seul endroit à
 * mettre à jour si un 7ᵉ pilier est ajouté un jour, plutôt qu'un `switch
 * (kind)` recopié dans chaque page qui a besoin de supprimer un job sans
 * connaître son type à l'avance (Dashboard "activité récente",
 * AllHistory). */
export const JOB_KIND_REMOVE: Record<JobKind, (id: number) => Promise<void>> = {
  supervised: api.training.remove,
  clustering: api.clustering.remove,
  dimensionality: api.dimensionality.remove,
  anomalies: api.anomalies.remove,
  vision_classification: api.visionClassification.remove,
  vision_anomalies: api.visionAnomalies.remove,
};
