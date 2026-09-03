import type { ReactNode } from "react";
import { Navigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import ForcePasswordChange from "./auth/ForcePasswordChange";

/** Silhouette de l'application pendant la vérification de session (revue
 * design). Remplace un « Chargement… » nu au centre d'un écran vide, qui
 * restait affiché 7 à 8 secondes en usage réel : sans repère de mise en
 * page, l'attente se lit comme une panne plutôt que comme un chargement.
 *
 * Reproduit la structure d'`AppShell` (barre latérale, en-tête, contenu)
 * sans en dépendre — `AppShell` exige un `user`, qui est précisément ce
 * qu'on est en train d'aller chercher. Volontairement statique et sans
 * texte : rien ici ne doit laisser croire qu'une donnée réelle est déjà
 * affichée.
 *
 * `motion-reduce:animate-none` : le battement est une aide, jamais une
 * contrainte imposée à qui a demandé moins d'animations. `role="status"`
 * sur le conteneur et `aria-hidden` sur les blocs : un lecteur d'écran
 * annonce le chargement une fois, au lieu d'énumérer une vingtaine de
 * blocs vides.
 *
 * Ceci corrige la PERCEPTION de l'attente, pas sa durée — la cause des
 * 7 à 8 secondes reste à diagnostiquer séparément. */
function AppSkeleton() {
  return (
    <div className="min-h-screen bg-background flex" role="status" aria-label="Chargement de l'application">
      <aside
        aria-hidden="true"
        className="hidden w-64 flex-col border-r border-sidebar-border bg-sidebar lg:flex"
      >
        <div className="h-16 flex items-center gap-2.5 border-b border-sidebar-border px-4">
          <div className="h-8 w-8 rounded-lg bg-sidebar-accent/60 animate-pulse motion-reduce:animate-none" />
          <div className="h-3.5 w-28 rounded bg-sidebar-accent/60 animate-pulse motion-reduce:animate-none" />
        </div>
        <div className="flex-1 px-3 py-3 space-y-1.5">
          {Array.from({ length: 5 }).map((_, i) => (
            <div
              key={i}
              className="h-9 rounded-lg bg-sidebar-accent/40 animate-pulse motion-reduce:animate-none"
              style={{ width: `${88 - i * 7}%` }}
            />
          ))}
        </div>
        <div className="border-t border-sidebar-border p-3 flex items-center gap-2.5">
          <div className="h-8 w-8 rounded-full bg-sidebar-accent/60 animate-pulse motion-reduce:animate-none" />
          <div className="flex-1 space-y-1.5">
            <div className="h-3 w-24 rounded bg-sidebar-accent/60 animate-pulse motion-reduce:animate-none" />
            <div className="h-2.5 w-16 rounded bg-sidebar-accent/40 animate-pulse motion-reduce:animate-none" />
          </div>
        </div>
      </aside>

      <div aria-hidden="true" className="flex min-h-screen flex-1 flex-col">
        <header className="h-16 border-b border-border" />
        <main className="flex-1 px-4 py-8 sm:px-6 lg:px-8">
          <div className="max-w-6xl mx-auto space-y-6">
            <div className="h-36 rounded-card bg-card border border-border animate-pulse motion-reduce:animate-none" />
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              {Array.from({ length: 4 }).map((_, i) => (
                <div
                  key={i}
                  className="h-32 rounded-card bg-card border border-border animate-pulse motion-reduce:animate-none"
                />
              ))}
            </div>
            <div className="grid gap-4 lg:grid-cols-2">
              <div className="h-56 rounded-card bg-card border border-border animate-pulse motion-reduce:animate-none" />
              <div className="h-56 rounded-card bg-card border border-border animate-pulse motion-reduce:animate-none" />
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}

/** Bloque l'accès aux pages qui exigent une session valide — redirige vers /login sinon. */
export default function ProtectedRoute({ children }: { children: ReactNode }) {
  const { isAuth, isLoading, user } = useAuth();

  if (isLoading) {
    return <AppSkeleton />;
  }

  if (!isAuth) {
    return <Navigate to="/login" replace />;
  }

  // Mot de passe provisoire non remplacé : le serveur refuse déjà tout le
  // reste de l'API (AUTH_MDP_PROVISOIRE). Intercepté ici, au point de
  // passage unique des pages authentifiées, pour expliquer la situation au
  // lieu de laisser l'utilisateur enchaîner des 403 incompréhensibles.
  if (user?.must_change_password) {
    return <ForcePasswordChange />;
  }

  return <>{children}</>;
}
