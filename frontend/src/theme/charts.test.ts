import { describe, expect, it } from "vitest";
import { lerpRgb } from "./charts";

// `beeswarmColor` résout les couleurs du thème actif via le DOM (var(--info)/
// var(--danger), voir charts.ts) — pas testable en environnement Node sans
// jsdom (non configuré dans ce projet). `lerpRgb` porte la seule logique
// vraiment testable ici : l'interpolation elle-même, indépendante du thème.
const LOW: [number, number, number] = [37, 99, 235]; // bleu (ancre "basse" arbitraire pour le test)
const HIGH: [number, number, number] = [220, 38, 38]; // rouge (ancre "haute" arbitraire pour le test)

describe("lerpRgb", () => {
  it("retourne la couleur basse à t=0 et la couleur haute à t=1", () => {
    expect(lerpRgb(LOW, HIGH, 0)).toBe(`rgb(${LOW[0]}, ${LOW[1]}, ${LOW[2]})`);
    expect(lerpRgb(LOW, HIGH, 1)).toBe(`rgb(${HIGH[0]}, ${HIGH[1]}, ${HIGH[2]})`);
  });

  it("borne les valeurs hors de [0, 1] (variable constante ou point atypique)", () => {
    expect(lerpRgb(LOW, HIGH, -5)).toBe(`rgb(${LOW[0]}, ${LOW[1]}, ${LOW[2]})`);
    expect(lerpRgb(LOW, HIGH, 5)).toBe(`rgb(${HIGH[0]}, ${HIGH[1]}, ${HIGH[2]})`);
  });

  it("interpole à mi-chemin sans sortir de la plage [basse, haute]", () => {
    const mid = lerpRgb(LOW, HIGH, 0.5);
    const [r, g, b] = mid.match(/\d+/g)!.map(Number);
    expect(r).toBeGreaterThanOrEqual(Math.min(LOW[0], HIGH[0]));
    expect(r).toBeLessThanOrEqual(Math.max(LOW[0], HIGH[0]));
    expect(g).toBeGreaterThanOrEqual(Math.min(LOW[1], HIGH[1]));
    expect(g).toBeLessThanOrEqual(Math.max(LOW[1], HIGH[1]));
    expect(b).toBeGreaterThanOrEqual(Math.min(LOW[2], HIGH[2]));
    expect(b).toBeLessThanOrEqual(Math.max(LOW[2], HIGH[2]));
  });
});
