import { type ReactNode, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { Bell, LogOut, Menu, Search, Target, X } from "lucide-react";
import { useAuth } from "../contexts/AuthContext";
import { Avatar } from "./ui/Avatar";
import { Badge } from "./ui/Badge";
import { PILLARS, type PillarId } from "../config/pillars";

/** Coquille commune à toutes les pages protégées — sidebar fixe façon SaaS
 * moderne (refonte UI, calquée sur la maquette de référence) : logo en
 * tête, navigation groupée par pilier (ML supervisé actif, non
 * supervisé/vision "Bientôt"), profil utilisateur en pied de sidebar.
 * Barre du haut réduite à une recherche + notifications (visuel seul pour
 * l'instant, aucun des deux n'est câblé — hors périmètre de cette refonte).
 *
 * `pillarId` détermine les items de nav affichés (lus depuis le registre des
 * piliers) — omis sur l'écran d'orientation, qui est pilier-agnostique.
 * Ajouter un futur pilier ne touche pas ce composant : une entrée dans
 * `config/pillars.ts` suffit. */
export default function AppShell({ children, pillarId }: { children: ReactNode; pillarId?: PillarId }) {
  const { user, logout } = useAuth();
  const location = useLocation();
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  if (!user) return null;

  const sidebarContent = (
    <>
      <div className="flex h-16 items-center gap-2.5 border-b border-sidebar-border px-4 flex-shrink-0">
        <Link to="/" className="flex items-center gap-2.5 min-w-0" onClick={() => setMobileNavOpen(false)}>
          <div className="h-7 w-[37px] overflow-hidden flex-shrink-0">
            <img
              src="/logo.png"
              alt="DataLab Pro"
              className="h-[49px] w-[49px] max-w-none -translate-x-[5.5px] -translate-y-[1.5px]"
            />
          </div>
          <div className="leading-tight min-w-0">
            <p className="text-[10px] uppercase tracking-widest text-primary font-semibold truncate">DataLab Pro</p>
            <p className="text-sm text-sidebar-foreground/70 truncate">{user.organization_name}</p>
          </div>
        </Link>
      </div>

      <nav className="flex-1 overflow-y-auto px-3 py-3">
        <Link
          to="/"
          onClick={() => setMobileNavOpen(false)}
          className={`flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
            location.pathname === "/"
              ? "bg-accent text-accent-foreground"
              : "text-muted-foreground hover:bg-muted hover:text-foreground"
          }`}
        >
          <Target size={16} strokeWidth={2} className="flex-shrink-0" />
          Objectifs
        </Link>

        {PILLARS.map((pillar) => (
          <div key={pillar.id}>
            <div className="flex items-center justify-between px-3 pt-5 pb-1.5">
              <span
                className={`text-[11px] font-semibold tracking-wide uppercase ${
                  pillar.id === pillarId ? "text-primary" : "text-muted-foreground/80"
                }`}
              >
                {pillar.id === "supervised" ? "ML supervisé" : pillar.id === "unsupervised" ? "ML non supervisé" : "Vision"}
              </span>
              {pillar.status === "soon" && (
                <Badge variant="neutral">Bientôt</Badge>
              )}
            </div>
            <div className="flex flex-col gap-0.5">
              {pillar.navItems.length > 0 ? (
                pillar.navItems.map(({ to, label, icon: Icon }) => {
                  const active = location.pathname === to;
                  return (
                    <Link
                      key={to}
                      to={to}
                      onClick={() => setMobileNavOpen(false)}
                      className={`flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                        active
                          ? "bg-accent text-accent-foreground"
                          : "text-muted-foreground hover:bg-muted hover:text-foreground"
                      }`}
                    >
                      <Icon size={16} strokeWidth={2} className="flex-shrink-0" />
                      {label}
                    </Link>
                  );
                })
              ) : (
                <Link
                  to={pillar.route}
                  onClick={() => setMobileNavOpen(false)}
                  className="flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm font-medium text-muted-foreground/60 hover:bg-muted hover:text-muted-foreground transition-colors"
                >
                  <pillar.icon size={16} strokeWidth={2} className="flex-shrink-0" />
                  {pillar.title}
                </Link>
              )}
            </div>
          </div>
        ))}
      </nav>

      <div className="border-t border-sidebar-border p-3 flex-shrink-0">
        <div className="flex items-center gap-2.5 rounded-lg px-2 py-1.5">
          <Avatar name={user.nom} size="sm" />
          <div className="flex min-w-0 flex-1 flex-col leading-tight">
            <span className="truncate text-sm font-medium text-sidebar-foreground">{user.nom}</span>
            <span className="truncate text-xs text-muted-foreground">{user.organization_name}</span>
          </div>
          <button
            onClick={logout}
            aria-label="Déconnexion"
            title="Déconnexion"
            className="flex-shrink-0 flex items-center justify-center h-7 w-7 rounded-md text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
          >
            <LogOut size={14} />
          </button>
        </div>
      </div>
    </>
  );

  return (
    <div className="flex min-h-screen bg-background">
      {/* Sidebar desktop — fixe, toujours visible dès lg */}
      <aside className="fixed inset-y-0 left-0 hidden w-64 flex-col border-r border-sidebar-border bg-sidebar lg:flex">
        {sidebarContent}
      </aside>

      {/* Sidebar mobile — panneau glissant, masqué par défaut */}
      {mobileNavOpen && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div className="absolute inset-0 bg-slate-900/40" onClick={() => setMobileNavOpen(false)} />
          <aside className="absolute inset-y-0 left-0 w-64 flex flex-col bg-sidebar shadow-xl">
            {sidebarContent}
          </aside>
        </div>
      )}

      <div className="flex min-h-screen flex-1 flex-col lg:pl-64">
        <header className="sticky top-0 z-30 flex h-16 items-center gap-3 border-b border-border bg-background/80 px-4 backdrop-blur-md sm:px-6">
          <button
            onClick={() => setMobileNavOpen((v) => !v)}
            aria-label="Menu"
            className="lg:hidden flex items-center justify-center h-9 w-9 rounded-lg text-muted-foreground hover:bg-muted transition-colors flex-shrink-0"
          >
            {mobileNavOpen ? <X size={18} /> : <Menu size={18} />}
          </button>

          <div className="ml-auto flex items-center gap-2">
            <div className="hidden sm:flex items-center gap-2 w-64 rounded-lg border border-input bg-card px-3 py-1.5 text-sm text-muted-foreground focus-within:ring-2 focus-within:ring-ring/30 transition-shadow">
              <Search size={14} className="flex-shrink-0" />
              <input
                type="search"
                placeholder="Rechercher…"
                className="w-full bg-transparent outline-none placeholder:text-muted-foreground"
              />
            </div>
            <button
              aria-label="Notifications"
              className="flex items-center justify-center h-9 w-9 rounded-lg text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
            >
              <Bell size={16} />
            </button>
          </div>
        </header>

        <main className="flex-1 px-4 py-8 sm:px-6 lg:px-8">
          <div className="max-w-6xl mx-auto">{children}</div>
        </main>
      </div>
    </div>
  );
}
