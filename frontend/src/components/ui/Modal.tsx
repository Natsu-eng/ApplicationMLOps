import type { ReactNode } from "react";
import { X } from "lucide-react";

export function Modal({
  title,
  onClose,
  children,
}: {
  title: string;
  onClose: () => void;
  children: ReactNode;
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-slate-950/70 backdrop-blur-sm" onClick={onClose} />
      <div className="relative w-full max-w-4xl max-h-[85vh] overflow-hidden rounded-2xl border border-slate-800 bg-slate-900 shadow-2xl flex flex-col">
        <div className="flex items-center justify-between px-5 py-4 border-b border-slate-800 flex-shrink-0">
          <h3 className="text-sm font-medium text-slate-100 truncate pr-4">{title}</h3>
          <button
            onClick={onClose}
            aria-label="Fermer"
            className="text-slate-500 hover:text-slate-300 transition-colors flex-shrink-0"
          >
            <X size={18} />
          </button>
        </div>
        <div className="overflow-auto p-5">{children}</div>
      </div>
    </div>
  );
}
