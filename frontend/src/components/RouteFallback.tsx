import { Loader2 } from "lucide-react";

/** État de chargement affiché pendant qu'une page découpée en chunk
 * (`React.lazy`, `App.tsx`) télécharge son code — jamais visible pour un
 * utilisateur sur un réseau normal une fois le chunk mis en cache par le
 * navigateur, mais nécessaire au tout premier accès à chaque route. Même
 * spinner (`Loader2` + `animate-spin`) que `Button.tsx`, seul motif de
 * chargement déjà établi dans l'app — pas un second motif inventé ici. */
export function RouteFallback() {
  return (
    <div role="status" aria-label="Chargement de la page" className="flex items-center justify-center min-h-screen">
      <Loader2 size={28} className="animate-spin text-muted-foreground motion-reduce:animate-none" aria-hidden="true" />
    </div>
  );
}
