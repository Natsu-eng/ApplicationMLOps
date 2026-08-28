import type { ClusterCandidate, ClusterProfile } from "../api/client";
import type { QualityAssessment, QualityTone } from "./qualityAssessment";

/** Lecture, interprétation et présentation du résultat d'un clustering —
 * fonctions pures, aucun appel réseau, testées indépendamment de
 * `pages/Clustering.tsx`. N'invente jamais de statistique : ne fait que
 * recouper/reformater des nombres déjà calculés côté backend (silhouette,
 * tailles de segments...), jamais de nouveau calcul ML (voir skill
 * senior-ai-saas-engineer, data-science.md : "jamais un texte inventé").
 *
 * Le vocabulaire de verdict (`QualityTone`/`QualityAssessment`) vit dans
 * `qualityAssessment.ts`, PARTAGÉ avec `dimensionalityQuality.ts`/
 * `anomalyQuality.ts` (Lot 6B, §F.2) — réexporté ici pour ne pas casser les
 * imports existants (`pages/Clustering.tsx`). */

export type { QualityTone, QualityAssessment };

/** Repère indicatif (échelle usuelle du score de silhouette, proche de
 * Kaufman & Rousseeuw) — PAS un seuil universel : présenté avec une
 * formulation prudente, toujours accompagné d'un rappel à recouper avec les
 * profils de segments et la connaissance métier. */
export function assessSilhouetteQuality(silhouette: number | null): QualityAssessment {
  if (silhouette === null) {
    return {
      tone: "low",
      label: "Structure non évaluable",
      caveat: "Le score de silhouette n'a pas pu être calculé pour cette configuration.",
    };
  }
  if (silhouette < 0.25) {
    return {
      tone: "low",
      label: "Structure plutôt faible",
      caveat:
        "Les groupes se chevauchent nettement selon ce score — repère indicatif, à confirmer par les profils de segments avant d'en tirer une conclusion.",
    };
  }
  if (silhouette < 0.5) {
    return {
      tone: "moderate",
      label: "Structure modérée",
      caveat:
        "Des groupes se distinguent mais avec un recouvrement notable — repère indicatif, pas un seuil universel, à recouper avec les deux autres métriques.",
    };
  }
  return {
    tone: "good",
    label: "Structure plutôt bonne",
    caveat:
      "Les groupes sont bien séparés selon ce score — repère indicatif, à confirmer par la cohérence des profils de segments et la connaissance métier.",
  };
}

/** Stabilité par sous-échantillonnage (Lot 6B, §F.2) — mesure à quel point
 * le regroupement retenu dépend des données EXACTEMENT utilisées (ARI moyen
 * entre sous-échantillons, calculé côté backend, voir
 * `services/clustering_training.py::_compute_cluster_stability`). Seuils
 * repris de la lecture usuelle de l'Adjusted Rand Index (>0.75 = accord
 * quasi parfait, 0.5-0.75 = accord modéré, <0.5 = fragile) — mêmes
 * précautions de formulation que `assessSilhouetteQuality` : un repère
 * indicatif, jamais une affirmation absolue. */
export function assessStabilityQuality(stabilityAri: number | null): QualityAssessment {
  if (stabilityAri === null) {
    return {
      tone: "low",
      label: "Stabilité non évaluable",
      caveat: "Pas assez d'observations pour estimer la sensibilité du regroupement à l'échantillon utilisé.",
    };
  }
  if (stabilityAri < 0.5) {
    return {
      tone: "low",
      label: "Regroupement peu stable",
      caveat:
        "Refaire le calcul sur un sous-ensemble légèrement différent de vos données change sensiblement les groupes obtenus — à interpréter avec prudence, même si les métriques de qualité ci-dessus sont bonnes.",
    };
  }
  if (stabilityAri < 0.75) {
    return {
      tone: "moderate",
      label: "Stabilité modérée",
      caveat: "Les groupes obtenus varient un peu selon les données exactement utilisées — repère indicatif.",
    };
  }
  return {
    tone: "good",
    label: "Regroupement stable",
    caveat: "Refaire le calcul sur un sous-ensemble légèrement différent de vos données change peu les groupes obtenus.",
  };
}

/** Explication de la configuration recommandée — retour utilisateur direct
 * (deux cas réels observés : la sélection au seul silhouette élisait une
 * configuration nettement pire sur les 2 autres métriques pour un gain de
 * silhouette marginal) : la sélection se fait désormais sur un RANG
 * COMPOSITE (moyenne des rangs sur silhouette/Davies-Bouldin/Calinski-
 * Harabasz, voir `domains/clustering/services/engine.py::
 * _attach_composite_rank`), jamais la silhouette seule. Ce texte décrit
 * fidèlement ce critère — jamais une affirmation non vérifiée, tout est
 * recalculé à partir des candidats réellement évalués. */
export function buildRecommendationExplanation(winner: ClusterCandidate, allCandidates: ClusterCandidate[]): string {
  const valid = allCandidates.filter((c) => c.silhouette !== null);
  const n = valid.length;
  if (n === 0 || winner.silhouette === null) {
    return "Configuration retenue faute d'autre candidat exploitable dans cette comparaison.";
  }

  if (n === 1 || winner.composite_rank === null) {
    return `Seule configuration exploitable parmi ${n} testée${n > 1 ? "s" : ""} (silhouette ${winner.silhouette.toFixed(3)}).`;
  }

  const parts: string[] = [];
  if (winner.rank_silhouette !== null) parts.push(`silhouette rang ${winner.rank_silhouette}/${n}`);
  if (winner.rank_davies_bouldin !== null) parts.push(`Davies-Bouldin rang ${winner.rank_davies_bouldin}/${n}`);
  if (winner.rank_calinski_harabasz !== null) parts.push(`Calinski-Harabasz rang ${winner.rank_calinski_harabasz}/${n}`);

  const base = `Sélectionnée pour le meilleur compromis sur les 3 métriques de qualité (rang composite ${winner.composite_rank.toFixed(2)}, silhouette ${winner.silhouette.toFixed(3)}) parmi ${n} configurations testées — ${parts.join(", ")}.`;

  // 1ère du classement sur les 3 métriques à la fois : compromis "propre",
  // rien à nuancer.
  if (winner.rank_silhouette === 1 && winner.rank_davies_bouldin === 1 && winner.rank_calinski_harabasz === 1) {
    return `${base} Elle arrive également en tête sur chacune des 3 métriques prise isolément — aucun compromis à faire ici.`;
  }
  // Le meilleur silhouette isolé N'EST PAS le gagnant du rang composite —
  // cas explicitement signalé (c'est exactement le comportement que le
  // rang composite corrige) pour que l'utilisateur voie l'arbitrage fait.
  const bestSilhouette = valid.reduce((best, c) => (c.silhouette! > best.silhouette! ? c : best), valid[0]);
  if (bestSilhouette.rank !== winner.rank) {
    return `${base} Le meilleur score de silhouette isolé (${bestSilhouette.algorithm}, ${bestSilhouette.silhouette?.toFixed(3)}) a été écarté : il se classe nettement moins bien sur Davies-Bouldin et/ou Calinski-Harabasz — comparez les 3 meilleures configurations en détail avant de vous arrêter à ce choix.`;
  }
  return `${base} Elle reste aussi la meilleure sur le seul critère de silhouette.`;
}

export interface DistributionEntry {
  id: string;
  label: string;
  count: number;
  pct: number;
  isNoise: boolean;
}

/** Répartition clusters + observations atypiques, pourcentages arrondis par
 * la méthode du plus grand reste pour garantir un total exact de 100 %
 * (des pourcentages indépendamment arrondis à 1 décimale ne totalisent pas
 * toujours 100 exactement). Le nombre d'observations atypiques est déduit
 * par complément (n_samples - somme des tailles de segments) : c'est un
 * nombre exact, pas une nouvelle estimation. */
export function computeClusterDistribution(profiles: ClusterProfile[], totalSamples: number): DistributionEntry[] {
  if (totalSamples <= 0) return [];
  const clusterTotal = profiles.reduce((sum, p) => sum + p.size, 0);
  const noiseCount = Math.max(0, totalSamples - clusterTotal);

  const raw: { id: string; label: string; count: number; isNoise: boolean }[] = profiles.map((p) => ({
    id: `cluster-${p.cluster_id}`,
    label: `Segment ${p.cluster_id + 1}`,
    count: p.size,
    isNoise: false,
  }));
  if (noiseCount > 0) {
    raw.push({ id: "noise", label: "Observations atypiques", count: noiseCount, isNoise: true });
  }

  return applyLargestRemainderPercentages(raw, totalSamples);
}

function applyLargestRemainderPercentages<T extends { count: number }>(entries: T[], total: number): (T & { pct: number })[] {
  if (entries.length === 0) return [];
  // Précision au dixième de pourcent — le reste est réparti sur les entrées
  // dont la partie décimale tronquée est la plus grande jusqu'à ce que la
  // somme des parts entières (en dixièmes) atteigne exactement 1000.
  const shares = entries.map((e) => (e.count / total) * 1000);
  const floors = shares.map(Math.floor);
  const remainder = 1000 - floors.reduce((a, b) => a + b, 0);
  const order = shares
    .map((v, i) => ({ i, frac: v - floors[i] }))
    .sort((a, b) => b.frac - a.frac);
  const tenths = [...floors];
  for (let k = 0; k < remainder && order.length > 0; k++) {
    tenths[order[k % order.length].i] += 1;
  }
  return entries.map((e, i) => ({ ...e, pct: tenths[i] / 10 }));
}
