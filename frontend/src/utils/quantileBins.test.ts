import { describe, expect, it } from "vitest";
import { binIndexForValue, computeQuantileEdges, formatBinLabel } from "./quantileBins";

describe("computeQuantileEdges", () => {
  it("retourne bins + 1 bornes couvrant min et max", () => {
    const values = Array.from({ length: 100 }, (_, i) => i);
    const edges = computeQuantileEdges(values, 5);
    expect(edges[0]).toBe(0);
    expect(edges[edges.length - 1]).toBe(99);
  });

  it("dédoublonne les bornes identiques sans planter", () => {
    const values = Array(50).fill(7);
    const edges = computeQuantileEdges(values, 5);
    expect(edges).toEqual([7]);
  });

  it("retourne un tableau vide sans données", () => {
    expect(computeQuantileEdges([], 5)).toEqual([]);
  });
});

describe("binIndexForValue", () => {
  const edges = [0, 10, 20, 30, 40];

  it("place chaque valeur dans le bon intervalle", () => {
    expect(binIndexForValue(5, edges)).toBe(0);
    expect(binIndexForValue(15, edges)).toBe(1);
    expect(binIndexForValue(25, edges)).toBe(2);
  });

  it("inclut la valeur maximale dans le dernier bucket", () => {
    expect(binIndexForValue(40, edges)).toBe(3);
  });

  it("ne plante jamais avec un seul bord (toutes valeurs identiques)", () => {
    expect(binIndexForValue(7, [7])).toBe(0);
  });
});

describe("formatBinLabel", () => {
  it("formate une plage lisible", () => {
    expect(formatBinLabel([0, 10.456], 0)).toBe("0.00 – 10.46");
  });

  it("retourne un tiret pour un index hors bornes", () => {
    expect(formatBinLabel([0, 10], 5)).toBe("—");
  });
});
