import { describe, expect, it } from "vitest";
import { ApiError, apiErrorReference, shouldRedirectToLogin } from "./client";

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

// Phase 6 (AUDIT_BACKEND_2026-08-23.md, Axe I) — le backend inclut
// désormais `request_id` dans chaque réponse d'erreur (Phase 1) ; ce
// helper décide QUAND l'afficher côté frontend (jamais pour une erreur
// métier déjà explicite, un 404/quota/validation).
describe("apiErrorReference", () => {
  it("affiche une référence pour une erreur serveur (5xx) avec request_id", () => {
    const err = new ApiError(500, "Une erreur inattendue est survenue.", "ERREUR_INTERNE", "abc-123");
    expect(apiErrorReference(err)).toBe("réf. abc-123");
  });

  it("n'affiche jamais de référence pour une erreur métier (4xx)", () => {
    const err = new ApiError(404, "Dataset introuvable", "DATASET_INTROUVABLE", "abc-123");
    expect(apiErrorReference(err)).toBeUndefined();
  });

  it("n'affiche rien si le backend n'a pas fourni de request_id", () => {
    const err = new ApiError(500, "Erreur", "ERREUR_INTERNE", undefined);
    expect(apiErrorReference(err)).toBeUndefined();
  });

  it("n'affiche rien pour une erreur qui n'est pas une ApiError", () => {
    expect(apiErrorReference(new Error("réseau indisponible"))).toBeUndefined();
    expect(apiErrorReference("chaîne brute")).toBeUndefined();
  });
});
