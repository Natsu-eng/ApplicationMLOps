import type { ClusterCandidate, ClusterProfile } from "../api/client";

/** Lecture, interprétation et présentation du résultat d'un clustering —
 * fonctions pures, aucun appel réseau, testées indépendamment de
 * `pages/Clustering.tsx`. N'invente jamais de statistique : ne fait que
 * recouper/reformater des nombres déjà calculés côté backend (silhouette,
 * tailles de segments...), jamais de nouveau calcul ML (voir skill
 * senior-ai-saas-engineer, data-science.md : "jamais un texte inventé"). */

export type QualityTone = "low" | "moderate" | "good";

export interface QualityAssessment {
  tone: QualityTone;
  label: string;
  caveat: string;
}

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

/** Explication de la configuration recommandée : indique le critère de
 * sélection (silhouette) et recoupe avec le classement de la même
 * configuration sur les deux autres métriques parmi les candidats
 * réellement évalués — jamais une affirmation non vérifiée. */
export function buildRecommendationExplanation(winner: ClusterCandidate, allCandidates: ClusterCandidate[]): string {
  const valid = allCandidates.filter((c) => c.silhouette !== null);
  const n = valid.length;
  if (n === 0 || winner.silhouette === null) {
    return "Configuration retenue faute d'autre candidat exploitable dans cette comparaison.";
  }

  const base = `Sélectionnée pour le meilleur score de silhouette (${winner.silhouette.toFixed(3)}) parmi ${n} configuration${n > 1 ? "s" : ""} testée${n > 1 ? "s" : ""}.`;
  if (n === 1) return base;

  const dbRanked = valid.filter((c) => c.davies_bouldin !== null).sort((a, b) => a.davies_bouldin! - b.davies_bouldin!);
  const chRanked = valid
    .filter((c) => c.calinski_harabasz !== null)
    .sort((a, b) => b.calinski_harabasz! - a.calinski_harabasz!);
  const dbRank = dbRanked.findIndex((c) => c.rank === winner.rank) + 1;
  const chRank = chRanked.findIndex((c) => c.rank === winner.rank) + 1;

  if (dbRank === 0 || chRank === 0) return base;

  const topThird = Math.max(1, Math.ceil(Math.max(dbRanked.length, chRanked.length) * 0.3));
  const corroborated = dbRank <= topThird && chRank <= topThird;

  if (corroborated) {
    return `${base} Le Davies-Bouldin (rang ${dbRank}/${dbRanked.length}) et le Calinski-Harabasz (rang ${chRank}/${chRanked.length}) confirment ce choix.`;
  }
  return `${base} Le Davies-Bouldin (rang ${dbRank}/${dbRanked.length}) et le Calinski-Harabasz (rang ${chRank}/${chRanked.length}) ne placent pas cette configuration en tête sur ces deux critères — à interpréter avec prudence, le score de silhouette reste le critère de sélection retenu.`;
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
