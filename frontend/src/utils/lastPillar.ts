import { PILLARS, type PillarId } from "../config/pillars";

/** Mémoire du dernier pilier utilisé (Lot 7, §J.1 — "revenir sur `/` force
 * aujourd'hui à rechoisir"). Ne redirige JAMAIS automatiquement — l'écran
 * d'orientation reste un vrai choix, pas un piège de navigation — mais
 * permet d'y afficher un raccourci "Reprendre" mis en avant. */
const STORAGE_KEY = "datalab_last_pillar";

export function setLastPillar(id: PillarId): void {
  try {
    localStorage.setItem(STORAGE_KEY, id);
  } catch {
    // localStorage indisponible (navigation privée stricte...) — dégrade
    // silencieusement, ce n'est qu'un confort de navigation.
  }
}

export function getLastPillar(): PillarId | null {
  try {
    const value = localStorage.getItem(STORAGE_KEY);
    return PILLARS.some((p) => p.id === value) ? (value as PillarId) : null;
  } catch {
    return null;
  }
}
