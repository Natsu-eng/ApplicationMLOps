import type { ReactNode } from "react";
import { Link, useLocation } from "react-router-dom";
import { ArrowLeft, LogOut } from "lucide-react";
import { useAuth } from "../contexts/AuthContext";
import { Avatar } from "./ui/Avatar";
import { PILLARS, type PillarId } from "../config/pillars";

/** Coquille commune à toutes les pages protégées : en-tête, navigation, session.
 *
 * `pillarId` détermine les items de nav affichés (lus depuis le registre des
 * piliers) — omis sur l'écran d'orientation, qui est pilier-agnostique.
 * Ajouter un futur pilier ne touche pas ce composant : une entrée dans
 * `config/pillars.ts` suffit. */
export default function AppShell({ children, pillarId }: { children: ReactNode; pillarId?: PillarId }) {
  const { user, logout } = useAuth();
  const location = useLocation();
  const pillar = pillarId ? PILLARS.find((p) => p.id === pillarId) : undefined;
  const navItems = pillar?.navItems ?? [];

  if (!user) return null;

  return (
    <div className="min-h-screen text-slate-100">
      <header className="sticky top-0 z-10 border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-md">
        <div className="max-w-6xl mx-auto px-6 py-3.5 flex items-center justify-between">
          <div className="flex items-center gap-8">
            <div className="flex items-center gap-2.5">
              <div className="h-7 w-7 rounded-lg bg-brand-gradient flex items-center justify-center text-white font-bold text-sm">
                D
              </div>
              <div className="leading-tight">
                <p className="text-[10px] uppercase tracking-widest text-teal-400 font-semibold">
                  DataLab Pro
                </p>
                <p className="text-sm text-slate-300 -mt-0.5">{user.organization_name}</p>
              </div>
            </div>

            <nav className="flex gap-1 items-center">
              {pillar && (
                <Link
                  to="/"
                  className="flex items-center gap-1.5 text-sm px-3 py-1.5 rounded-lg text-slate-500 hover:text-slate-300 hover:bg-slate-800/50 transition-colors mr-1"
                >
                  <ArrowLeft size={15} strokeWidth={2} />
                  Objectifs
                </Link>
              )}
              {navItems.map(({ to, label, icon: Icon }) => {
                const active = location.pathname === to;
                return (
                  <Link
                    key={to}
                    to={to}
                    className={`flex items-center gap-1.5 text-sm px-3 py-1.5 rounded-lg transition-colors ${
                      active
                        ? "bg-slate-800 text-slate-100"
                        : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
                    }`}
                  >
                    <Icon size={15} strokeWidth={2} />
                    {label}
                  </Link>
                );
              })}
            </nav>
          </div>

          <div className="flex items-center gap-3">
            <Avatar name={user.nom} size="sm" />
            <button
              onClick={logout}
              className="flex items-center gap-1.5 text-sm text-slate-400 hover:text-slate-200 border border-slate-800 hover:border-slate-700 rounded-lg px-3 py-1.5 transition-colors"
            >
              <LogOut size={14} />
              Déconnexion
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-10">{children}</main>
    </div>
  );
}
