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
        className="text-slate-400 hover:text-primary transition-colors"
      >
        <Info size={12} />
      </button>
      {open && (
        <span
          role="tooltip"
          className="absolute z-20 bottom-full left-1/2 -translate-x-1/2 mb-2 w-56 rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-xs text-slate-200 shadow-xl"
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
