import { useEffect, useRef, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { X } from "lucide-react";

const SIZE_CLASSES = {
  md: "max-w-4xl",
  // Contenu riche (ex. exploration de données à plusieurs sections) — évite
  // de compresser des graphiques Recharts dans une largeur trop étroite.
  xl: "max-w-6xl",
  // Contenu le plus riche de l'app (résultats d'entraînement, exploration de
  // données) — retour utilisateur direct : xl restait perçu comme "petit"
  // pour des grilles de graphiques côte à côte. Largeur quasi pleine
  // fenêtre, plafonnée pour rester lisible sur très grand écran.
  "2xl": "max-w-[96rem] w-[95vw]",
} as const;

let modalIdCounter = 0;

export function Modal({
  title,
  onClose,
  children,
  size = "md",
  headerExtra,
}: {
  title: string;
  onClose: () => void;
  children: ReactNode;
  size?: keyof typeof SIZE_CLASSES;
  /** Contenu additionnel dans l'en-tête, entre le titre et le bouton fermer
   * (ex. onglets de navigation interne — voir EdaModal). */
  headerExtra?: ReactNode;
}) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const titleId = useRef(`modal-title-${++modalIdCounter}`).current;

  // Accessibilité (AUDIT_ROADMAP.md, D4/H10) : Échap ferme la modale, focus
  // posé sur la boîte de dialogue à l'ouverture, piège de focus au Tab pour
  // qu'un utilisateur clavier ne puisse jamais "sortir" de la modale vers le
  // contenu masqué derrière — utilisé par tous les écrans de résultats/
  // exploration de l'app (ModelResultModal, EdaModal...).
  useEffect(() => {
    const previouslyFocused = document.activeElement as HTMLElement | null;
    dialogRef.current?.focus();

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        onClose();
        return;
      }
      if (event.key !== "Tab" || !dialogRef.current) return;
      const focusable = dialogRef.current.querySelectorAll<HTMLElement>(
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

  // Portail vers document.body : AppShell place la barre latérale fixe
  // (aside, z-20) en dehors du wrapper <main> qui, lui, crée sa propre
  // pile d'empilement (relative z-10). Sans portail, le z-50 de la modale
  // ne s'applique qu'à l'intérieur de cette pile et perd face à l'aside —
  // la modale reste visible mais devient non cliquable/non focusable sous
  // la bande occupée par la barre latérale.
  return createPortal(
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-foreground/40 backdrop-blur-sm" onClick={onClose} aria-hidden="true" />
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
        className={`relative w-full ${SIZE_CLASSES[size]} max-h-[90vh] overflow-hidden rounded-2xl border border-border bg-card shadow-xl flex flex-col outline-none`}
      >
        <div className="flex items-center justify-between gap-4 px-5 py-4 border-b border-border flex-shrink-0">
          <h3 id={titleId} className="text-h3 text-foreground truncate">
            {title}
          </h3>
          <div className="flex items-center gap-4 flex-shrink-0">
            {headerExtra}
            <button
              onClick={onClose}
              aria-label="Fermer"
              className="text-muted-foreground hover:text-foreground transition-colors flex-shrink-0"
            >
              <X size={18} />
            </button>
          </div>
        </div>
        {/* Fond légèrement teinté (pas la couleur des cartes) : les cartes
            internes (Card, bg-card + bordure) ressortent nettement plutôt
            que de se fondre dans un modal uniforme — même logique
            canevas/carte que le reste de l'app (index.css --color-background). */}
        <div className="overflow-auto p-5 bg-muted/40">{children}</div>
      </div>
    </div>,
    document.body,
  );
}
