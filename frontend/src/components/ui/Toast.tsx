import { createContext, useCallback, useContext, useState, type ReactNode } from "react";
import { AlertTriangle, CheckCircle2, Info, OctagonAlert, X, type LucideIcon } from "lucide-react";

export type ToastVariant = "info" | "success" | "warning" | "danger";

interface ToastItem {
  id: string;
  variant: ToastVariant;
  title: string;
  description?: string;
}

const ICONS: Record<ToastVariant, LucideIcon> = {
  info: Info,
  success: CheckCircle2,
  warning: AlertTriangle,
  danger: OctagonAlert,
};

const ICON_CLASSES: Record<ToastVariant, string> = {
  info: "text-info",
  success: "text-success",
  warning: "text-warning",
  danger: "text-destructive",
};

interface ToastContextValue {
  push: (toast: { variant?: ToastVariant; title: string; description?: string; durationMs?: number }) => void;
}

const ToastContext = createContext<ToastContextValue | null>(null);

/** Notifications empilées, coin bas-droit — auto-disparition (5 s par
 * défaut), jamais la seule façon d'apprendre un résultat important (un
 * toast peut être manqué ; les résultats critiques vivent aussi dans
 * l'écran/la fiche correspondante). `aria-live="polite"` : annoncé sans
 * interrompre le lecteur d'écran en cours de lecture. */
export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  const dismiss = useCallback((id: string) => {
    setToasts((ts) => ts.filter((t) => t.id !== id));
  }, []);

  const push = useCallback<ToastContextValue["push"]>(
    ({ variant = "info", title, description, durationMs = 5000 }) => {
      const id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      setToasts((ts) => [...ts, { id, variant, title, description }]);
      if (durationMs > 0) {
        setTimeout(() => dismiss(id), durationMs);
      }
    },
    [dismiss]
  );

  return (
    <ToastContext.Provider value={{ push }}>
      {children}
      <div
        aria-live="polite"
        role="status"
        className="fixed bottom-4 right-4 z-[60] flex flex-col gap-2 w-80 max-w-[calc(100vw-2rem)]"
      >
        {toasts.map((toast) => {
          const Icon = ICONS[toast.variant];
          return (
            <div
              key={toast.id}
              className="toast-enter flex items-start gap-2.5 rounded-card border border-border bg-popover shadow-overlay p-3.5"
            >
              <Icon size={16} className={`flex-shrink-0 mt-0.5 ${ICON_CLASSES[toast.variant]}`} aria-hidden="true" />
              <div className="flex-1 min-w-0">
                <p className="text-body font-medium text-foreground">{toast.title}</p>
                {toast.description && <p className="text-caption text-muted-foreground mt-0.5">{toast.description}</p>}
              </div>
              <button
                type="button"
                onClick={() => dismiss(toast.id)}
                aria-label="Fermer la notification"
                className="flex-shrink-0 text-muted-foreground hover:text-foreground transition-colors"
              >
                <X size={14} />
              </button>
            </div>
          );
        })}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast(): ToastContextValue {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error("useToast() doit être appelé à l'intérieur de <ToastProvider>");
  return ctx;
}
