import type { AccentColor } from "../components/ui/ColorIconBadge";

/** Vocabulaire de verdict qualité PARTAGÉ entre les 3 piliers ML non
 * supervisé (clustering, réduction de dimension, détection d'anomalies) —
 * Lot 6B, §F.2. Chaque pilier a sa propre fonction d'évaluation
 * (`clusterQuality.ts`/`dimensionalityQuality.ts`/`anomalyQuality.ts`,
 * seuils et libellés propres à sa métrique), mais toutes partagent CETTE
 * même forme et CETTE même palette tone→couleur — pour qu'un badge de
 * qualité se lise de la même façon d'une page à l'autre, jamais une
 * variante de tons/couleurs réinventée à chaque nouvelle page. */

export type QualityTone = "low" | "moderate" | "good";

export interface QualityAssessment {
  tone: QualityTone;
  label: string;
  caveat: string;
}

export const QUALITY_TONE_ACCENT: Record<QualityTone, AccentColor> = {
  low: "amber",
  moderate: "blue",
  good: "teal",
};
