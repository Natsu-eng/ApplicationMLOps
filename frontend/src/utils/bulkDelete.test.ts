import { describe, expect, it } from "vitest";
import { runBulkDelete } from "./bulkDelete";

// Aucun endpoint de suppression groupée côté backend (voir bulkDelete.ts) —
// ces tests prouvent que N appels individuels en parallèle se comportent
// correctement même quand certains échouent, ce qu'un simple `Promise.all`
// ne garantirait pas (il rejetterait tout au premier échec).
describe("runBulkDelete", () => {
  it("compte tous les succès quand rien n'échoue", async () => {
    const result = await runBulkDelete([1, 2, 3], async () => {});
    expect(result).toEqual({ succeeded: 3, failed: 0 });
  });

  it("compte séparément succès et échecs — jamais d'abandon global comme Promise.all", async () => {
    const result = await runBulkDelete([1, 2, 3, 4], async (id: number) => {
      if (id % 2 === 0) throw new Error("échec simulé");
    });
    expect(result).toEqual({ succeeded: 2, failed: 2 });
  });

  it("exécute réellement chaque suppression, pas seulement les comptabilise", async () => {
    const deleted: number[] = [];
    await runBulkDelete([10, 20, 30], async (id: number) => {
      deleted.push(id);
    });
    expect(deleted.sort()).toEqual([10, 20, 30]);
  });

  it("liste vide : aucun appel, résultat neutre", async () => {
    let calls = 0;
    const result = await runBulkDelete([], async () => {
      calls += 1;
    });
    expect(result).toEqual({ succeeded: 0, failed: 0 });
    expect(calls).toBe(0);
  });
});
