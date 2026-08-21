import { describe, expect, it } from "vitest";
import { assessTrustworthinessQuality } from "./dimensionalityQuality";

describe("assessTrustworthinessQuality", () => {
  it("classe une conservation des voisinages basse en projection peu fidèle", () => {
    expect(assessTrustworthinessQuality(0.5).tone).toBe("low");
  });
  it("classe une conservation intermédiaire en fidélité modérée", () => {
    expect(assessTrustworthinessQuality(0.8).tone).toBe("moderate");
  });
  it("classe une conservation élevée en projection fidèle", () => {
    expect(assessTrustworthinessQuality(0.95).tone).toBe("good");
  });
});
