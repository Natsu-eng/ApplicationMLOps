import { describe, expect, it } from "vitest";
import { clampUnitScore, foldScoresToBoxPlotDatum } from "./cvScore";

describe("clampUnitScore", () => {
  it("laisse un score déjà dans [0, 1] inchangé", () => {
    expect(clampUnitScore(0.87)).toBeCloseTo(0.87);
    expect(clampUnitScore(0)).toBe(0);
    expect(clampUnitScore(1)).toBe(1);
  });

  it("borne un R² effondré sur un fold à variance quasi nulle (GroupKFold, petit dataset)", () => {
    expect(clampUnitScore(-651626728)).toBe(0);
    expect(clampUnitScore(-1)).toBe(0);
  });

  it("borne un score au-dessus de 1", () => {
    expect(clampUnitScore(1.2)).toBe(1);
  });
});

describe("foldScoresToBoxPlotDatum", () => {
  it("garde toutes les valeurs de la boîte dans [0, 1] même avec un fold dégénéré", () => {
    // Reproduit le bug réel Lot D : un candidat perdant dont UN fold s'est
    // effondré à cause d'une variance quasi nulle sur ce sous-ensemble
    // (R² mathématiquement non borné en dessous), les autres folds étant
    // corrects — voir docstring de `clampUnitScore`.
    const datum = foldScoresToBoxPlotDatum("RandomForest", [0.85, 0.91, -651626728.34, 0.88]);
    for (const value of [datum.min, datum.q1, datum.median, datum.q3, datum.max]) {
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThanOrEqual(1);
    }
    expect(datum.min).toBe(0);
    expect(datum.name).toBe("RandomForest");
  });

  it("reste correcte sur des scores normaux (classification, AUC bornée)", () => {
    const datum = foldScoresToBoxPlotDatum("LightGBM", [0.9, 0.92, 0.88, 0.95]);
    expect(datum.min).toBeCloseTo(0.88);
    expect(datum.max).toBeCloseTo(0.95);
    expect(datum.median).toBeGreaterThanOrEqual(datum.q1);
    expect(datum.q3).toBeGreaterThanOrEqual(datum.median);
  });
});
