import { type ReactNode, useEffect, useRef, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { HelpCircle, History, LayoutDashboard, LogOut, Menu, Palette, ShieldCheck, Target, X } from "lucide-react";
import { useAuth } from "../contexts/AuthContext";
import { NotificationBell } from "./NotificationBell";
import { Avatar } from "./ui/Avatar";
import { Badge } from "./ui/Badge";
import { ThemePickerCompact } from "./ui/ThemePicker";
import { PILLARS, type PillarId } from "../config/pillars";
import { setLastPillar } from "../utils/lastPillar";

/** Coquille commune à toutes les pages protégées — sidebar fixe façon SaaS
 * moderne (refonte UI, calquée sur la maquette de référence) : logo en
 * tête, navigation groupée par pilier (ML supervisé actif, non
 * supervisé/vision "Bientôt"), profil utilisateur en pied de sidebar.
 * Barre du haut réduite à un point d'entrée d'aide (lien vers `/aide`,
 * Lot 10 — remplace l'ancienne modale `HelpModal`, même contenu, une seule
 * source désormais) — la
 * recherche et les notifications, visuelles mais jamais câblées, ont été
 * retirées (AUDIT_ROADMAP.md, H16 : une UI qui a l'air fonctionnelle sans
 * l'être casse la confiance, contrairement au traitement honnête déjà
 * appliqué aux piliers "Bientôt"). À réintroduire seulement quand une vraie
 * fonctionnalité existe derrière.
 *
 * `pillarId` détermine QUEL pilier affiche ses modules — "Objectifs",
 * "Tableau de bord" et "Historique" restent épinglés en haut, toujours
 * accessibles quel que soit le pilier actif ; en dessous, seule la section
 * du pilier courant est
 * affichée (pas les 3 en permanence) — retour utilisateur direct : mieux
 * vaut choisir son objectif et voir SES modules que tout empiler à chaque
 * page, un choix qui s'aggraverait avec un 3ᵉ pilier (Vision) actif. Omis
 * sur l'écran d'orientation, qui est pilier-agnostique. Ajouter un futur
 * pilier ne touche pas ce composant : une entrée dans `config/pillars.ts`
 * suffit. */
export default function AppShell({ children, pillarId }: { children: ReactNode; pillarId?: PillarId }) {
  const { user, logout } = useAuth();
  const location = useLocation();
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [themeMenuOpen, setThemeMenuOpen] = useState(false);
  const themeMenuRef = useRef<HTMLDivElement>(null);

  // Menu de thème de l'avatar — ferme au clic extérieur ou à Échap (même
  // convention que le menu "Nouvelle analyse" du dashboard).
  useEffect(() => {
    if (!themeMenuOpen) return;
    function onPointerDown(e: MouseEvent) {
      if (themeMenuRef.current && !themeMenuRef.current.contains(e.target as Node)) setThemeMenuOpen(false);
    }
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") setThemeMenuOpen(false);
    }
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [themeMenuOpen]);

  useEffect(() => {
    if (pillarId) setLastPillar(pillarId);
  }, [pillarId]);

  if (!user) return null;

  const sidebarContent = (
    <>
      <div className="flex h-16 items-center gap-2.5 border-b border-sidebar-border px-4 flex-shrink-0">
        <Link to="/" className="flex items-center gap-2.5 min-w-0" onClick={() => setMobileNavOpen(false)}>
          <img src="/icon.svg" alt="" className="h-8 w-8 flex-shrink-0 rounded-lg" />
          <span className="text-sm font-semibold text-sidebar-foreground truncate">DataLab Pro</span>
        </Link>
      </div>

      <nav className="flex-1 overflow-y-auto px-3 py-3">
        <Link
          to="/"
          onClick={() => setMobileNavOpen(false)}
          className={`flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
            location.pathname === "/"
              ? "bg-sidebar-accent text-sidebar-accent-foreground"
              : "text-sidebar-muted-foreground hover:bg-sidebar-accent/50 hover:text-sidebar-foreground"
          }`}
        >
          <Target size={16} strokeWidth={2} className="flex-shrink-0" />
          Objectifs
        </Link>

        <Link
          to="/dashboard"
          onClick={() => setMobileNavOpen(false)}
          className={`flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
            location.pathname === "/dashboard"
              ? "bg-sidebar-accent text-sidebar-accent-foreground"
              : "text-sidebar-muted-foreground hover:bg-sidebar-accent/50 hover:text-sidebar-foreground"
          }`}
        >
          <LayoutDashboard size={16} strokeWidth={2} className="flex-shrink-0" />
          Tableau de bord
        </Link>

        <Link
          to="/historique"
          onClick={() => setMobileNavOpen(false)}
          className={`flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
            location.pathname === "/historique"
              ? "bg-sidebar-accent text-sidebar-accent-foreground"
              : "text-sidebar-muted-foreground hover:bg-sidebar-accent/50 hover:text-sidebar-foreground"
          }`}
        >
          <History size={16} strokeWidth={2} className="flex-shrink-0" />
          Historique
        </Link>

        {PILLARS.filter((pillar) => pillar.id === pillarId).map((pillar) => (
          <div key={pillar.id}>
            <div className="flex items-center justify-between px-3 pt-5 pb-1.5">
              <span
                className={`text-overline uppercase ${
                  pillar.id === pillarId ? "text-sidebar-highlight" : "text-sidebar-muted-foreground/80"
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
                          ? "bg-sidebar-accent text-sidebar-accent-foreground"
                          : "text-sidebar-muted-foreground hover:bg-sidebar-accent/50 hover:text-sidebar-foreground"
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
                  className="flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm font-medium text-sidebar-muted-foreground/50 hover:bg-sidebar-accent/30 hover:text-sidebar-muted-foreground transition-colors"
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
        {/* Administration de la plateforme — hors des piliers, qui décrivent
            les usages métier : superviser toutes les organisations n'en est
            pas un. Masquée aux comptes ordinaires, pour qui elle mènerait à
            un écran vide. */}
        {user.is_platform_admin && (
          <Link
            to="/admin"
            onClick={() => setMobileNavOpen(false)}
            className="mb-2 flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm font-medium text-sidebar-muted-foreground hover:bg-sidebar-accent/50 hover:text-sidebar-foreground transition-colors"
          >
            <ShieldCheck size={16} strokeWidth={2} className="flex-shrink-0" />
            Administration
          </Link>
        )}
        <div className="flex items-center gap-2.5 rounded-lg px-2 py-1.5">
          <Link
            to="/profile"
            onClick={() => setMobileNavOpen(false)}
            className="flex items-center gap-2.5 min-w-0 flex-1 rounded-lg -mx-1 px-1 py-0.5 hover:bg-sidebar-accent/50 transition-colors"
          >
            <Avatar name={user.nom} size="sm" />
            <div className="flex min-w-0 flex-1 flex-col leading-tight">
              <span className="truncate text-sm font-medium text-sidebar-foreground">{user.nom}</span>
              <span className="truncate text-xs text-sidebar-muted-foreground">{user.organization_name}</span>
            </div>
          </Link>
          <div className="relative flex-shrink-0" ref={themeMenuRef}>
            <button
              onClick={() => setThemeMenuOpen((o) => !o)}
              aria-label="Changer de thème"
              title="Changer de thème"
              aria-haspopup="menu"
              aria-expanded={themeMenuOpen}
              className="flex items-center justify-center h-7 w-7 rounded-md text-sidebar-muted-foreground hover:bg-sidebar-accent/50 hover:text-sidebar-foreground transition-colors"
            >
              <Palette size={14} />
            </button>
            {themeMenuOpen && (
              <div
                role="menu"
                aria-label="Thème d'interface"
                className="absolute bottom-full left-0 mb-2 w-56 rounded-xl border border-border bg-card shadow-lg p-1.5 z-20"
              >
                <ThemePickerCompact />
              </div>
            )}
          </div>
          <button
            onClick={logout}
            aria-label="Déconnexion"
            title="Déconnexion"
            className="flex-shrink-0 flex items-center justify-center h-7 w-7 rounded-md text-sidebar-muted-foreground hover:bg-sidebar-accent/50 hover:text-sidebar-foreground transition-colors"
          >
            <LogOut size={14} />
          </button>
        </div>
      </div>
    </>
  );

  return (
    <div className="relative flex min-h-screen bg-background">
      {/* Atmosphère — deux halos + grille très faible, uniquement pour que le
          rail et la barre haute en verre (.glass) aient quelque chose à
          flouter (SPEC-UI.md §4 : « sans halo, pas de verre »). Fixe, hors
          du flux, jamais au-dessus du contenu (z-0) ni cliquable. */}
      <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none" aria-hidden="true">
        <div
          className="absolute -left-40 -top-64 h-[560px] w-[560px] rounded-full opacity-60"
          style={{ background: "radial-gradient(circle, color-mix(in oklch, var(--accent) 16%, transparent), transparent 68%)" }}
        />
        <div
          className="absolute -right-40 -bottom-56 h-[480px] w-[480px] rounded-full opacity-60"
          style={{ background: "radial-gradient(circle, color-mix(in oklch, var(--info) 14%, transparent), transparent 68%)" }}
        />
      </div>

      {/* Rail — flottant en verre (SPEC-UI.md §5) : navigation complète par
          étiquettes conservée (pas la version 60px icônes seules de la
          maquette de référence, qui suppose une IA à 5 entrées fixes — cette
          app a des sous-listes de longueur variable par pilier ; réduire à
          des icônes nues aurait exigé une refonte de la navigation elle-même,
          hors périmètre de ce lot — voir _design/JOURNAL.md). */}
      <aside className="fixed left-[18px] top-[18px] bottom-[18px] z-20 hidden w-64 flex-col rounded-sheet glass lg:flex">
        {sidebarContent}
      </aside>

      {/* Sidebar mobile — panneau glissant, masqué par défaut */}
      {mobileNavOpen && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div className="absolute inset-0 bg-foreground/40" onClick={() => setMobileNavOpen(false)} />
          <aside className="absolute inset-y-0 left-0 w-64 flex flex-col bg-sidebar shadow-xl">
            {sidebarContent}
          </aside>
        </div>
      )}

      <div className="relative z-10 flex min-h-screen min-w-0 flex-1 flex-col lg:pl-[292px]">
        <header className="sticky top-0 z-30 mx-4 mt-4 flex h-14 items-center gap-3 rounded-card glass px-4 sm:mx-6 sm:mt-[22px] sm:px-5 lg:mr-[4px]">
          <button
            onClick={() => setMobileNavOpen((v) => !v)}
            aria-label="Menu"
            className="lg:hidden flex items-center justify-center h-9 w-9 rounded-lg text-muted-foreground hover:bg-muted/60 transition-colors flex-shrink-0"
          >
            {mobileNavOpen ? <X size={18} /> : <Menu size={18} />}
          </button>

          <div className="ml-auto flex items-center gap-2">
            <NotificationBell />
            <Link
              to="/aide"
              className="flex items-center gap-1.5 h-9 rounded-lg px-3 text-sm text-muted-foreground hover:bg-muted/60 hover:text-foreground transition-colors"
            >
              <HelpCircle size={16} />
              <span className="hidden sm:inline">Aide</span>
            </Link>
          </div>
        </header>

        <main className="flex-1 min-w-0 px-4 py-6 sm:px-6 lg:px-8">
          <div className="max-w-6xl mx-auto">{children}</div>
        </main>
      </div>

    </div>
  );
}
