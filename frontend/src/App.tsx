import { useEffect, useState } from "react";
import { api, type HealthStatus } from "./api/client";

type ConnexionState =
  | { phase: "chargement" }
  | { phase: "ok"; sante: HealthStatus }
  | { phase: "erreur"; message: string };

// Page d'accueil du Lot 0 : vérifie simplement que le frontend parle au
// backend. Les vraies pages (auth, datasets, entraînement...) arrivent lot
// par lot — voir ../backend/workflow.md pour l'avancement.
export default function App() {
  const [etat, setEtat] = useState<ConnexionState>({ phase: "chargement" });

  useEffect(() => {
    api
      .health()
      .then((sante) => setEtat({ phase: "ok", sante }))
      .catch((err) =>
        setEtat({
          phase: "erreur",
          message: err instanceof Error ? err.message : "Erreur inconnue",
        }),
      );
  }, []);

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 flex items-center justify-center px-6">
      <div className="max-w-md w-full space-y-6 text-center">
        <div>
          <p className="text-xs uppercase tracking-widest text-teal-400 font-semibold mb-2">
            Lot 0 — squelette
          </p>
          <h1 className="text-3xl font-serif">DataLab Pro</h1>
        </div>

        <div className="rounded-lg border border-slate-800 bg-slate-900 px-5 py-4 text-sm">
          {etat.phase === "chargement" && (
            <p className="text-slate-400">Connexion au backend…</p>
          )}
          {etat.phase === "ok" && (
            <div className="space-y-1 text-left">
              <p className="flex items-center gap-2 text-emerald-400 font-medium">
                <span className="h-2 w-2 rounded-full bg-emerald-400" />
                Backend connecté
              </p>
              <p className="text-slate-400">
                {etat.sante.app} v{etat.sante.version} ·{" "}
                {etat.sante.environment}
              </p>
              <p className="text-slate-400">
                Base de données :{" "}
                <span
                  className={
                    etat.sante.database === "up"
                      ? "text-emerald-400"
                      : "text-amber-400"
                  }
                >
                  {etat.sante.database === "up" ? "connectée" : "indisponible"}
                </span>
              </p>
            </div>
          )}
          {etat.phase === "erreur" && (
            <div className="space-y-1 text-left">
              <p className="flex items-center gap-2 text-rose-400 font-medium">
                <span className="h-2 w-2 rounded-full bg-rose-400" />
                Backend injoignable
              </p>
              <p className="text-slate-500 text-xs">{etat.message}</p>
              <p className="text-slate-500 text-xs">
                Vérifier que <code>uvicorn api.main:app --reload</code> tourne
                sur le port 8000.
              </p>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
