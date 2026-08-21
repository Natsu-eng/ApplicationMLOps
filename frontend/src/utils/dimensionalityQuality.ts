import type { QualityAssessment } from "./qualityAssessment";

/** Lecture, interprétation et présentation du résultat d'une réduction de
 * dimension — même esprit que `clusterQuality.ts` (Lot 6B, §F.2) : fonction
 * pure, ne fait que recouper un nombre déjà calculé côté backend
 * (`trustworthiness`, sklearn.manifold.trustworthiness), jamais de nouveau
 * calcul ML ni de statistique inventée. */

/** Repères indicatifs sur l'échelle usuelle de la conservation des
 * voisinages (trustworthiness, bornée [0, 1]) — PAS un seuil universel,
 * mêmes précautions de formulation que `assessSilhouetteQuality`. */
export function assessTrustworthinessQuality(trustworthiness: number): QualityAssessment {
  if (trustworthiness < 0.7) {
    return {
      tone: "low",
      label: "Projection peu fidèle",
      caveat:
        "Les points proches sur cette projection ne correspondent pas toujours à des observations proches dans vos données d'origine — repère indicatif, à interpréter avec prudence avant de tirer des conclusions des regroupements visuels.",
    };
  }
  if (trustworthiness < 0.9) {
    return {
      tone: "moderate",
      label: "Fidélité modérée",
      caveat:
        "Cette projection conserve raisonnablement les voisinages d'origine, avec quelques distorsions — repère indicatif, à recouper avec la variance expliquée.",
    };
  }
  return {
    tone: "good",
    label: "Projection fidèle",
    caveat: "Les points proches sur cette projection étaient déjà proches dans vos données d'origine.",
  };
}
