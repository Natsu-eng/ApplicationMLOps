import { useEffect, useRef, useState, type ReactNode } from "react";

/** Ouverture/fermeture au clic extérieur + `Échap` — même convention que le
 * menu "Nouvelle analyse" du dashboard et le menu de thème de l'avatar
 * (AppShell.tsx), factorisée ici pour ne pas la réécrire une 3ᵉ fois. */
export function usePopover() {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function onPointerDown(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  return { open, setOpen, ref };
}

/** Panneau flottant générique — déclenché par un élément quelconque
 * (`trigger`, render-prop qui reçoit `open`/`toggle`/`triggerProps` à poser
 * sur le déclencheur). Contenu posé sur `--raised` (SPEC-UI.md §1 : menu
 * déroulant/popover), jamais `--surface`. */
export function Popover({
  trigger,
  children,
  align = "start",
  className = "",
}: {
  trigger: (ctx: { open: boolean; toggle: () => void; triggerProps: { "aria-haspopup": "dialog"; "aria-expanded": boolean } }) => ReactNode;
  children: ReactNode;
  align?: "start" | "end";
  className?: string;
}) {
  const { open, setOpen, ref } = usePopover();
  const toggle = () => setOpen((o) => !o);

  return (
    <div className="relative inline-block" ref={ref}>
      {trigger({ open, toggle, triggerProps: { "aria-haspopup": "dialog", "aria-expanded": open } })}
      {open && (
        <div
          role="dialog"
          className={`absolute z-30 mt-2 min-w-[14rem] rounded-xl border border-border bg-popover shadow-lg p-3 ${
            align === "end" ? "right-0" : "left-0"
          } ${className}`}
        >
          {children}
        </div>
      )}
    </div>
  );
}
