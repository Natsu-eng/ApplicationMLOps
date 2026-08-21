import type { QualityAssessment } from "./qualityAssessment";

/** Lecture, interprétation et présentation du résultat d'une détection
 * d'anomalies — même esprit que `clusterQuality.ts` (Lot 6B, §F.2) :
 * fonction pure, ne fait que recouper un taux déjà calculé côté backend
 * (`anomaly_rate_consensus`), jamais de nouveau calcul ML. */

/** Un taux de consensus élevé (les deux méthodes signalent une part
 * importante du dataset comme atypique) n'est PAS forcément une "bonne"
 * détection — le plus souvent, c'est le signe d'un réglage de contamination
 * trop large, ou d'un dataset couvrant plusieurs populations distinctes. Pas
 * un seuil universel, mêmes précautions de formulation que
 * `assessSilhouetteQuality`. */
export function assessConsensusQuality(consensusRate: number): QualityAssessment {
  if (consensusRate > 0.3) {
    return {
      tone: "low",
      label: "Taux de consensus élevé",
      caveat:
        "Une part importante de vos données est jugée atypique par les deux méthodes à la fois — vérifiez le réglage de la proportion attendue d'anomalies, ou si vos données mélangent plusieurs populations très différentes.",
    };
  }
  if (consensusRate > 0.1) {
    return {
      tone: "moderate",
      label: "Taux de consensus modéré",
      caveat: "Une part notable de vos données est jugée atypique par les deux méthodes — à confirmer par les observations listées ci-dessous.",
    };
  }
  return {
    tone: "good",
    label: "Détection ciblée",
    caveat: "Seule une faible part de vos données est jugée atypique par les deux méthodes à la fois.",
  };
}
