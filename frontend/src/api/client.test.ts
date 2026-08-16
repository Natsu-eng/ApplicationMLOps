import { describe, expect, it } from "vitest";
import { shouldRedirectToLogin } from "./client";

// Lot 0.3 (correctif C5, AUDIT_DATALAB_2026-08-16.md) — logique de décision
// extraite de handleUnauthorized() pour rester testable sans `window`
// (pas d'environnement jsdom dans ce dépôt, voir vite.config.ts).
describe("shouldRedirectToLogin", () => {
  it("redirige depuis une page protégée quelconque", () => {
    expect(shouldRedirectToLogin("/")).toBe(true);
    expect(shouldRedirectToLogin("/datasets")).toBe(true);
    expect(shouldRedirectToLogin("/training/jobs/42")).toBe(true);
  });

  it("ne redirige jamais si on est déjà sur l'écran de connexion (évite une boucle)", () => {
    expect(shouldRedirectToLogin("/login")).toBe(false);
    expect(shouldRedirectToLogin("/login?expired=1")).toBe(false);
  });
});
