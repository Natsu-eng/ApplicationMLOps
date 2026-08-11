import { describe, expect, it } from "vitest";
import { beeswarmColor, CHART_BEESWARM_HIGH, CHART_BEESWARM_LOW } from "./charts";

describe("beeswarmColor", () => {
  it("retourne la couleur basse à t=0 et la couleur haute à t=1", () => {
    expect(beeswarmColor(0)).toBe(hexToRgb(CHART_BEESWARM_LOW));
    expect(beeswarmColor(1)).toBe(hexToRgb(CHART_BEESWARM_HIGH));
  });

  it("borne les valeurs hors de [0, 1] (variable constante ou point atypique)", () => {
    expect(beeswarmColor(-5)).toBe(hexToRgb(CHART_BEESWARM_LOW));
    expect(beeswarmColor(5)).toBe(hexToRgb(CHART_BEESWARM_HIGH));
  });

  it("interpole à mi-chemin sans sortir de la plage [basse, haute]", () => {
    const mid = beeswarmColor(0.5);
    const [r, g, b] = mid.match(/\d+/g)!.map(Number);
    const [rLo, gLo, bLo] = hexToRgb(CHART_BEESWARM_LOW).match(/\d+/g)!.map(Number);
    const [rHi, gHi, bHi] = hexToRgb(CHART_BEESWARM_HIGH).match(/\d+/g)!.map(Number);
    expect(r).toBeGreaterThanOrEqual(Math.min(rLo, rHi));
    expect(r).toBeLessThanOrEqual(Math.max(rLo, rHi));
    expect(g).toBeGreaterThanOrEqual(Math.min(gLo, gHi));
    expect(g).toBeLessThanOrEqual(Math.max(gLo, gHi));
    expect(b).toBeGreaterThanOrEqual(Math.min(bLo, bHi));
    expect(b).toBeLessThanOrEqual(Math.max(bLo, bHi));
  });
});

function hexToRgb(hex: string): string {
  const n = parseInt(hex.slice(1), 16);
  return `rgb(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255})`;
}
