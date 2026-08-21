import { describe, expect, it } from "vitest";
import { formatDuration } from "./format";

describe("formatDuration", () => {
  it("affiche des secondes en dessous d'une minute", () => {
    expect(formatDuration(45)).toBe("≈ 45 s");
  });
  it("arrondit au moins à 1 seconde", () => {
    expect(formatDuration(0.2)).toBe("≈ 1 s");
  });
  it("affiche des minutes entre 1 minute et 1 heure", () => {
    expect(formatDuration(150)).toBe("≈ 3 min");
  });
  it("affiche heures et minutes au-delà d'une heure", () => {
    expect(formatDuration(5400)).toBe("≈ 1 h 30 min");
  });
  it("omet les minutes quand la durée tombe sur une heure ronde", () => {
    expect(formatDuration(7200)).toBe("≈ 2 h");
  });
});
