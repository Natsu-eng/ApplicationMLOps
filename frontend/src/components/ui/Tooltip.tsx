import { useState, type ReactNode } from "react";
import { Info } from "lucide-react";

/** Info-bulle discrète pour expliquer un terme technique en langage clair —
 * destinée aux utilisateurs qui ne sont pas data scientists de métier. */
export function Tooltip({ text }: { text: string }) {
  const [open, setOpen] = useState(false);

  return (
    <span
      className="relative inline-flex items-center"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
    >
      <button
        type="button"
        aria-label="Aide"
        className="text-muted-foreground hover:text-primary transition-colors"
      >
        <Info size={12} />
      </button>
      {open && (
        <span
          role="tooltip"
          className="absolute z-20 bottom-full left-1/2 -translate-x-1/2 mb-2 w-56 rounded-lg border border-foreground bg-foreground px-3 py-2 text-xs text-card shadow-xl"
        >
          {text}
        </span>
      )}
    </span>
  );
}

/** Libellé + info-bulle, pour ne pas répéter le motif dans chaque écran. */
export function LabelWithHelp({ label, help }: { label: ReactNode; help: string }) {
  return (
    <span className="inline-flex items-center gap-1">
      {label}
      <Tooltip text={help} />
    </span>
  );
}

/** Info-bulle générique — enveloppe N'IMPORTE QUEL élément (contrairement à
 * `Tooltip` ci-dessus, toujours une icône "?" isolée) : utile pour expliquer
 * *pourquoi* un contrôle est désactivé, par exemple. `children` doit être un
 * seul élément focusable (bouton, lien...) — le survol ET le focus clavier
 * déclenchent l'ouverture, jamais l'un sans l'autre. */
export function TooltipWrapper({ content, children }: { content: ReactNode; children: ReactNode }) {
  const [open, setOpen] = useState(false);

  return (
    <span
      className="relative inline-flex"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
    >
      {children}
      {open && (
        <span
          role="tooltip"
          className="absolute z-20 bottom-full left-1/2 -translate-x-1/2 mb-2 w-max max-w-64 rounded-lg border border-foreground bg-foreground px-3 py-2 text-xs text-card shadow-xl"
        >
          {content}
        </span>
      )}
    </span>
  );
}
