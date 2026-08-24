/** Exécute `deleteOne` sur chaque item en parallèle (`Promise.allSettled` —
 * jamais `Promise.all`, qui abandonnerait au premier échec et laisserait
 * les suppressions déjà en vol dans un état inconnu côté appelant) et
 * rapporte combien ont réussi/échoué.
 *
 * Aucun endpoint de suppression groupée n'existe côté backend (vérifié :
 * chaque domaine n'expose que `DELETE /{ressource}/{id}`) — N requêtes
 * individuelles en parallèle reproduisent le même résultat qu'un vrai
 * endpoint groupé pour l'utilisateur, sans ajouter de code serveur pour
 * une opération qui n'a pas besoin d'atomicité transactionnelle (chaque
 * suppression est déjà indépendante et auditée individuellement, voir
 * `domains/shared/audit.py::log_action`, appelé par chaque route DELETE). */
export interface BulkDeleteResult {
  succeeded: number;
  failed: number;
}

export async function runBulkDelete<T>(items: T[], deleteOne: (item: T) => Promise<void>): Promise<BulkDeleteResult> {
  const results = await Promise.allSettled(items.map(deleteOne));
  const succeeded = results.filter((r) => r.status === "fulfilled").length;
  return { succeeded, failed: results.length - succeeded };
}
