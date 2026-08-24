/** Bannière d'erreur inline minimale — extraite de Dashboard.tsx pour être
 * réutilisée par Profile.tsx (les deux affichent des listes chargées en
 * parallèle, chacune avec son propre état d'erreur distinct du "vide").
 *
 * `reference` (Phase 6, AUDIT_BACKEND_2026-08-23.md, Axe I) — optionnel,
 * voir `api/client.ts::apiErrorReference` : affiché uniquement pour une
 * erreur serveur (5xx), jamais pour une erreur métier déjà explicite. */
export function ErrorNote({ message, reference }: { message: string; reference?: string }) {
  return (
    <p className="text-sm text-destructive bg-destructive/10 border border-destructive/20 rounded-lg px-3 py-2 mb-3">
      {message}
      {reference && <span className="block text-xs opacity-70 mt-0.5">{reference}</span>}
    </p>
  );
}
