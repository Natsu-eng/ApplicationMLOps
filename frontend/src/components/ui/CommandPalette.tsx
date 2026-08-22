import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { History, LayoutDashboard, Search, Target, User } from "lucide-react";
import { PILLARS } from "../../config/pillars";

interface Command {
  id: string;
  label: string;
  group: string;
  icon: typeof Target;
  to: string;
}

function buildCommands(): Command[] {
  const pinned: Command[] = [
    { id: "objectifs", label: "Objectifs", group: "Épinglé", icon: Target, to: "/" },
    { id: "dashboard", label: "Tableau de bord", group: "Épinglé", icon: LayoutDashboard, to: "/dashboard" },
    { id: "historique", label: "Historique", group: "Épinglé", icon: History, to: "/historique" },
    { id: "profil", label: "Profil & Organisation", group: "Épinglé", icon: User, to: "/profile" },
  ];
  const fromPillars: Command[] = PILLARS.flatMap((pillar) =>
    pillar.navItems.map((item) => ({
      id: `${pillar.id}-${item.to}`,
      label: item.label,
      group: pillar.title,
      icon: item.icon,
      to: item.to,
    }))
  );
  return [...pinned, ...fromPillars];
}

/** Palette de commandes (⌘K / Ctrl+K) — navigation clavier globale, montée
 * une seule fois (`App.tsx`) sur toutes les routes authentifiées. Liste
 * construite depuis `config/pillars.ts` : ajouter un écran au registre des
 * piliers l'ajoute automatiquement ici, aucune liste à maintenir en double. */
export function CommandPalette() {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  const commands = useMemo(buildCommands, []);
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return commands;
    return commands.filter((c) => c.label.toLowerCase().includes(q) || c.group.toLowerCase().includes(q));
  }, [commands, query]);

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen((o) => !o);
      } else if (e.key === "Escape" && open) {
        setOpen(false);
      }
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [open]);

  useEffect(() => {
    if (open) {
      setQuery("");
      setActiveIndex(0);
      // Le focus doit atterrir dans le champ après le rendu de la boîte de dialogue.
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }, [open]);

  useEffect(() => {
    setActiveIndex(0);
  }, [query]);

  function select(command: Command) {
    setOpen(false);
    navigate(command.to);
  }

  function onListKeyDown(e: React.KeyboardEvent) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveIndex((i) => Math.min(filtered.length - 1, i + 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveIndex((i) => Math.max(0, i - 1));
    } else if (e.key === "Enter") {
      e.preventDefault();
      const command = filtered[activeIndex];
      if (command) select(command);
    }
  }

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-[70] flex items-start justify-center pt-[15vh] px-4">
      <div className="absolute inset-0 bg-foreground/40 backdrop-blur-sm" onClick={() => setOpen(false)} aria-hidden="true" />
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Palette de commandes"
        className="relative w-full max-w-lg rounded-2xl border border-border glass shadow-xl overflow-hidden"
      >
        <div className="flex items-center gap-2.5 px-4 py-3 border-b border-border/60">
          <Search size={15} className="text-muted-foreground flex-shrink-0" aria-hidden="true" />
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={onListKeyDown}
            placeholder="Rechercher un écran…"
            aria-label="Rechercher un écran"
            aria-activedescendant={filtered[activeIndex] ? `cmdk-${filtered[activeIndex].id}` : undefined}
            role="combobox"
            aria-expanded={true}
            aria-controls="cmdk-listbox"
            className="flex-1 bg-transparent text-body text-foreground placeholder:text-muted-foreground outline-none"
          />
          <span className="num flex-shrink-0 rounded border border-border px-1.5 py-0.5 text-[10.5px] text-muted-foreground">
            Échap
          </span>
        </div>
        <ul id="cmdk-listbox" role="listbox" className="max-h-80 overflow-y-auto py-2">
          {filtered.length === 0 && <li className="px-4 py-6 text-center text-caption text-muted-foreground">Aucun écran ne correspond.</li>}
          {filtered.map((command, i) => {
            const Icon = command.icon;
            const active = i === activeIndex;
            return (
              <li key={command.id} id={`cmdk-${command.id}`} role="option" aria-selected={active}>
                <button
                  type="button"
                  onMouseEnter={() => setActiveIndex(i)}
                  onClick={() => select(command)}
                  className={`w-full flex items-center gap-3 px-4 py-2.5 text-left text-body transition-colors ${
                    active ? "bg-primary/10 text-primary" : "text-foreground hover:bg-muted"
                  }`}
                >
                  <Icon size={15} className="flex-shrink-0" aria-hidden="true" />
                  <span className="flex-1 min-w-0 truncate">{command.label}</span>
                  <span className="text-caption text-muted-foreground flex-shrink-0">{command.group}</span>
                </button>
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}
