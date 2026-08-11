import type { ReactNode } from "react";
import { X } from "lucide-react";

const SIZE_CLASSES = {
  md: "max-w-4xl",
  // Contenu riche (ex. exploration de données à plusieurs sections) — évite
  // de compresser des graphiques Recharts dans une largeur trop étroite.
  xl: "max-w-6xl",
} as const;

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
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-slate-950/50 backdrop-blur-sm" onClick={onClose} />
      <div
        className={`relative w-full ${SIZE_CLASSES[size]} max-h-[85vh] overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-xl flex flex-col`}
      >
        <div className="flex items-center justify-between gap-4 px-5 py-4 border-b border-slate-200 flex-shrink-0">
          <h3 className="text-sm font-medium text-slate-900 truncate">{title}</h3>
          <div className="flex items-center gap-4 flex-shrink-0">
            {headerExtra}
            <button
              onClick={onClose}
              aria-label="Fermer"
              className="text-slate-400 hover:text-slate-600 transition-colors flex-shrink-0"
            >
              <X size={18} />
            </button>
          </div>
        </div>
        {/* Fond gris pâle (pas blanc) : les cartes internes (Card, blanches
            + bordure) ressortent nettement plutôt que de se fondre dans un
            modal tout blanc — même contraste canevas/carte que le reste de
            l'app (index.css --color-canvas). */}
        <div className="overflow-auto p-5 bg-slate-50">{children}</div>
      </div>
    </div>
  );
}
