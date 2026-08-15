import { describe, expect, it } from "vitest";
import type { ClusterCandidate, ClusterProfile } from "../api/client";
import { assessSilhouetteQuality, buildRecommendationExplanation, computeClusterDistribution } from "./clusterQuality";

function candidate(overrides: Partial<ClusterCandidate>): ClusterCandidate {
  return {
    algorithm: "K-Means (k=3)",
    family: "partitionnement",
    params: { n_clusters: 3 },
    n_clusters: 3,
    silhouette: 0.5,
    davies_bouldin: 0.5,
    calinski_harabasz: 100,
    noise_ratio: 0,
    is_winner: false,
    rank: 1,
    ...overrides,
  };
}

function profile(overrides: Partial<ClusterProfile>): ClusterProfile {
  return {
    cluster_id: 0,
    size: 10,
    size_pct: 10,
    numeric_summary: {},
    categorical_summary: {},
    differentiating_variables: [],
    ...overrides,
  };
}

describe("assessSilhouetteQuality", () => {
  it("classe un score bas en structure faible", () => {
    expect(assessSilhouetteQuality(0.1).tone).toBe("low");
  });
  it("classe un score intermédiaire en structure modérée", () => {
    expect(assessSilhouetteQuality(0.35).tone).toBe("moderate");
  });
  it("classe un score élevé en structure bonne", () => {
    expect(assessSilhouetteQuality(0.85).tone).toBe("good");
  });
  it("gère l'absence de score sans planter", () => {
    const result = assessSilhouetteQuality(null);
    expect(result.tone).toBe("low");
    expect(result.label).toMatch(/non évaluable/i);
  });
});

describe("buildRecommendationExplanation", () => {
  it("confirme le choix quand les 3 métriques s'accordent", () => {
    const winner = candidate({ rank: 1, silhouette: 0.9, davies_bouldin: 0.1, calinski_harabasz: 500, is_winner: true });
    const loser = candidate({ rank: 2, silhouette: 0.4, davies_bouldin: 0.8, calinski_harabasz: 50 });
    const text = buildRecommendationExplanation(winner, [winner, loser]);
    expect(text).toMatch(/confirment ce choix/);
    expect(text).toContain("0.900");
  });

  it("signale la prudence quand les métriques divergent", () => {
    // Le gagnant a la meilleure silhouette mais le pire Davies-Bouldin/Calinski-Harabasz.
    const winner = candidate({ rank: 1, silhouette: 0.9, davies_bouldin: 5.0, calinski_harabasz: 1, is_winner: true });
    const others = [2, 3, 4, 5].map((rank) => candidate({ rank, silhouette: 0.5, davies_bouldin: 0.2, calinski_harabasz: 200 }));
    const text = buildRecommendationExplanation(winner, [winner, ...others]);
    expect(text).toMatch(/à interpréter avec prudence/);
  });

  it("reste correct avec un seul candidat valide", () => {
    const winner = candidate({ rank: 1, silhouette: 0.6, is_winner: true });
    const text = buildRecommendationExplanation(winner, [winner]);
    expect(text).toContain("0.600");
    expect(text).not.toMatch(/confirment|prudence/);
  });
});

describe("computeClusterDistribution", () => {
  it("déduit exactement le nombre d'observations atypiques par complément", () => {
    const profiles = [profile({ cluster_id: 0, size: 60 }), profile({ cluster_id: 1, size: 30 })];
    const dist = computeClusterDistribution(profiles, 100);
    const noise = dist.find((d) => d.isNoise);
    expect(noise?.count).toBe(10);
  });

  it("omet l'entrée atypique quand toutes les observations sont rattachées", () => {
    const profiles = [profile({ cluster_id: 0, size: 100 })];
    const dist = computeClusterDistribution(profiles, 100);
    expect(dist.some((d) => d.isNoise)).toBe(false);
  });

  it("les pourcentages totalisent toujours exactement 100", () => {
    // Répartition volontairement disgracieuse (tiers) pour stresser
    // l'arrondi au plus grand reste.
    const profiles = [
      profile({ cluster_id: 0, size: 1 }),
      profile({ cluster_id: 1, size: 1 }),
      profile({ cluster_id: 2, size: 1 }),
    ];
    const dist = computeClusterDistribution(profiles, 3);
    const total = dist.reduce((sum, d) => sum + d.pct, 0);
    expect(total).toBeCloseTo(100, 5);
  });

  it("totalise 100 avec un mélange clusters + bruit sur un grand échantillon", () => {
    const profiles = [
      profile({ cluster_id: 0, size: 4123 }),
      profile({ cluster_id: 1, size: 2877 }),
      profile({ cluster_id: 2, size: 1500 }),
    ];
    const dist = computeClusterDistribution(profiles, 10000);
    const total = dist.reduce((sum, d) => sum + d.pct, 0);
    expect(total).toBeCloseTo(100, 5);
    expect(dist.find((d) => d.isNoise)?.count).toBe(1500);
  });

  it("retourne un tableau vide sans échantillon", () => {
    expect(computeClusterDistribution([], 0)).toEqual([]);
  });
});
