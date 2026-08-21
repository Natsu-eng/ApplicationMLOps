import { describe, expect, it } from "vitest";
import { assessConsensusQuality } from "./anomalyQuality";

describe("assessConsensusQuality", () => {
  it("classe un taux de consensus élevé comme suspect", () => {
    expect(assessConsensusQuality(0.4).tone).toBe("low");
  });
  it("classe un taux de consensus modéré", () => {
    expect(assessConsensusQuality(0.2).tone).toBe("moderate");
  });
  it("classe un taux de consensus faible comme détection ciblée", () => {
    expect(assessConsensusQuality(0.02).tone).toBe("good");
  });
  it("classe l'absence totale d'anomalie comme détection ciblée", () => {
    expect(assessConsensusQuality(0).tone).toBe("good");
  });
});
