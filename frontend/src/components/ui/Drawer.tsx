import { useEffect, useRef, type ReactNode } from "react";
import { X } from "lucide-react";

let drawerIdCounter = 0;

/** Panneau latéral — même a11y que `Modal` (piège de focus, `Échap`,
 * restauration du focus au ferme) mais glisse depuis un bord au lieu de se
 * centrer : pour un contenu secondaire consulté PENDANT que l'écran
 * principal reste visible en arrière-plan (ex. détail d'une ligne sans
 * perdre le contexte du tableau). */
export function Drawer({
  title,
  onClose,
  children,
  side = "right",
  width = "max-w-md",
}: {
  title: string;
  onClose: () => void;
  children: ReactNode;
  side?: "left" | "right";
  width?: string;
}) {
  const panelRef = useRef<HTMLDivElement>(null);
  const titleId = useRef(`drawer-title-${++drawerIdCounter}`).current;

  useEffect(() => {
    const previouslyFocused = document.activeElement as HTMLElement | null;
    panelRef.current?.focus();

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        onClose();
        return;
      }
      if (event.key !== "Tab" || !panelRef.current) return;
      const focusable = panelRef.current.querySelectorAll<HTMLElement>(
        'a[href], button:not([disabled]), textarea, input, select, [tabindex]:not([tabindex="-1"])'
      );
      if (focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }

    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      previouslyFocused?.focus();
    };
  }, [onClose]);

  const sideClass = side === "right" ? "ml-auto" : "mr-auto";
  const enterClass = side === "right" ? "drawer-enter-right" : "drawer-enter-left";

  return (
    <div className="fixed inset-0 z-50 flex">
      <div className="absolute inset-0 bg-foreground/40 backdrop-blur-sm" onClick={onClose} aria-hidden="true" />
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
        className={`${enterClass} relative ${sideClass} h-full w-full ${width} bg-card border-border ${side === "right" ? "border-l" : "border-r"} shadow-xl flex flex-col outline-none flex-shrink-0`}
      >
        <div className="flex items-center justify-between gap-4 px-5 py-4 border-b border-border flex-shrink-0">
          <h3 id={titleId} className="text-h3 text-foreground truncate">
            {title}
          </h3>
          <button
            onClick={onClose}
            aria-label="Fermer"
            className="text-muted-foreground hover:text-foreground transition-colors flex-shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)] rounded"
          >
            <X size={18} />
          </button>
        </div>
        <div className="overflow-auto p-5 flex-1">{children}</div>
      </div>
    </div>
  );
}
