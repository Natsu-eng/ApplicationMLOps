/** Analyse d'un flux Server-Sent Events (Lot 7, §J.2) — fonction pure,
 * testable indépendamment de `fetch`/`ReadableStream` (voir `useJobEvents`,
 * qui l'utilise sur les morceaux reçus au fil de l'eau). Un "événement" SSE
 * est un bloc terminé par une ligne vide (`\n\n`) ; ce qui suit la dernière
 * ligne vide n'est pas encore complet et doit être conservé pour le
 * prochain appel (`remainder`). */
export interface ParsedSseChunk {
  events: string[];
  remainder: string;
}

export function parseSseBuffer(buffer: string): ParsedSseChunk {
  const parts = buffer.split("\n\n");
  const remainder = parts.pop() ?? "";
  return { events: parts, remainder };
}

/** Extrait le JSON porté par un événement `data: {...}` — `null` si
 * l'événement n'a pas de ligne `data:` (ex. `event: error` sans donnée
 * associée) ou si le JSON est malformé (jamais une exception qui
 * interromprait le flux pour un seul événement mal formé). */
export function parseSseData<T>(event: string): T | null {
  const dataLine = event.split("\n").find((line) => line.startsWith("data: "));
  if (!dataLine) return null;
  try {
    return JSON.parse(dataLine.slice("data: ".length)) as T;
  } catch {
    return null;
  }
}
