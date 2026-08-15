/** Bannière d'erreur inline minimale — extraite de Dashboard.tsx pour être
 * réutilisée par Profile.tsx (les deux affichent des listes chargées en
 * parallèle, chacune avec son propre état d'erreur distinct du "vide"). */
export function ErrorNote({ message }: { message: string }) {
  return (
    <p className="text-sm text-destructive bg-destructive/10 border border-destructive/20 rounded-lg px-3 py-2 mb-3">
      {message}
    </p>
  );
}
