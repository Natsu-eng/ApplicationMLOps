import { describe, expect, it } from "vitest";
import { nextSortState, sortRows } from "./tableSort";

describe("nextSortState", () => {
  it("passe de aucun tri à asc sur la colonne cliquée", () => {
    expect(nextSortState(null, "score")).toEqual({ key: "score", direction: "asc" });
  });

  it("passe de asc à desc sur la même colonne", () => {
    expect(nextSortState({ key: "score", direction: "asc" }, "score")).toEqual({ key: "score", direction: "desc" });
  });

  it("passe de desc à aucun tri sur la même colonne (cycle complet)", () => {
    expect(nextSortState({ key: "score", direction: "desc" }, "score")).toBeNull();
  });

  it("recommence à asc si on clique une AUTRE colonne", () => {
    expect(nextSortState({ key: "score", direction: "desc" }, "algorithme")).toEqual({
      key: "algorithme",
      direction: "asc",
    });
  });
});

describe("sortRows", () => {
  const rows = [
    { id: 1, score: 0.9 },
    { id: 2, score: 0.5 },
    { id: 3, score: null },
    { id: 4, score: 0.7 },
  ];
  const getValue = (r: (typeof rows)[number]) => r.score;

  it("retourne les lignes inchangées sans tri actif", () => {
    expect(sortRows(rows, null, getValue)).toEqual(rows);
  });

  it("trie en ordre croissant, valeurs nulles toujours en fin", () => {
    const sorted = sortRows(rows, { key: "score", direction: "asc" }, getValue);
    expect(sorted.map((r) => r.id)).toEqual([2, 4, 1, 3]);
  });

  it("trie en ordre décroissant, valeurs nulles TOUJOURS en fin (pas au début)", () => {
    const sorted = sortRows(rows, { key: "score", direction: "desc" }, getValue);
    expect(sorted.map((r) => r.id)).toEqual([1, 4, 2, 3]);
  });
});
