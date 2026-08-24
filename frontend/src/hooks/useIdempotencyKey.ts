import { useRef } from "react";

/** Clé d'idempotence pour une création de job (Phase 2,
 * AUDIT_BACKEND_2026-08-23.md §F4) — un double-clic sur "Lancer" avant que
 * le bouton ne se désactive (`isSubmitting`), ou une requête retentée après
 * un timeout réseau, ne doivent jamais créer deux jobs identiques. La
 * même clé est réutilisée tant qu'une tentative de soumission n'a pas
 * abouti (succès OU échec définitif) ; `reset()` en génère une nouvelle
 * pour la PROCHAINE tentative distincte — appelé après un succès, jamais
 * après un échec récupérable (l'utilisateur qui corrige un champ et
 * resoumet doit rester couvert par la même protection).
 *
 * @example
 * const idempotencyKey = useIdempotencyKey();
 * await api.training.createJob(payload, idempotencyKey.current);
 * idempotencyKey.reset(); // après un succès, avant de permettre une nouvelle soumission
 */
export function useIdempotencyKey() {
  const keyRef = useRef<string>(crypto.randomUUID());

  function reset(): void {
    keyRef.current = crypto.randomUUID();
  }

  // `keyRef` lui-même (pas `keyRef.current` déjà lu) : `reset()` ne
  // déclenche pas de nouveau rendu (mutation d'un ref, pas un state) — un
  // appelant qui aurait déstructuré `current` au moment du rendu garderait
  // l'ancienne valeur figée dans sa closure. Toujours lire
  // `idempotencyKey.current` au moment de l'appel API, jamais avant.
  return Object.assign(keyRef, { reset });
}
