import { useEffect, useRef } from "react";
import { apiUrl, getToken, type JobStatus } from "../api/client";
import { parseSseBuffer, parseSseData } from "../utils/sse";

export interface JobEventSnapshot {
  status: JobStatus;
  progress_percent: number;
  progress_step: string | null;
  error_message: string | null;
}

/** Notifications de fin de job par SSE (Lot 7, §J.2) — remplace le
 * `setInterval(3000)` dupliqué sur 6 pages par un flux serveur qui pousse
 * une mise à jour seulement quand elle change (voir
 * `services/job_events.py`).
 *
 * `fetch` + `ReadableStream`, PAS `EventSource` natif : `EventSource` ne
 * permet pas d'en-tête `Authorization` personnalisé — la seule alternative
 * native aurait été de passer le token en paramètre d'URL, qui finit alors
 * dans les logs serveur/l'historique du navigateur. `fetch` garde le même
 * en-tête `Authorization: Bearer` que le reste du client API.
 *
 * `path` à `null` désactive l'écoute (ex. hors de la phase "progress") —
 * mêmes conventions que les hooks conditionnels existants du projet. */
export function useJobEvents(path: string | null, onUpdate: (snapshot: JobEventSnapshot) => void): void {
  const onUpdateRef = useRef(onUpdate);
  onUpdateRef.current = onUpdate;

  useEffect(() => {
    if (!path) return;
    const controller = new AbortController();
    const token = getToken();

    async function run() {
      try {
        const res = await fetch(apiUrl(path as string), {
          headers: token ? { Authorization: `Bearer ${token}` } : undefined,
          signal: controller.signal,
        });
        if (!res.ok || !res.body) return;
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        for (;;) {
          const { done, value } = await reader.read();
          if (done) return;
          buffer += decoder.decode(value, { stream: true });
          const { events, remainder } = parseSseBuffer(buffer);
          buffer = remainder;
          for (const event of events) {
            const snapshot = parseSseData<JobEventSnapshot>(event);
            if (snapshot) onUpdateRef.current(snapshot);
          }
        }
      } catch {
        // AbortError attendu au démontage (cleanup ci-dessous) ; toute
        // autre erreur réseau reste silencieuse aussi — fonctionnalité de
        // confort, la page garde son dernier état connu plutôt que
        // d'afficher une erreur, et reste consultable/rafraîchissable.
      }
    }

    run();
    return () => controller.abort();
  }, [path]);
}
