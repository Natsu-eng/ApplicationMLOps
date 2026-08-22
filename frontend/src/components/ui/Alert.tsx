import { AlertTriangle, CheckCircle2, Info, OctagonAlert, X, type LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

export type AlertVariant = "info" | "warning" | "danger" | "success";

const ICONS: Record<AlertVariant, LucideIcon> = {
  info: Info,
  warning: AlertTriangle,
  danger: OctagonAlert,
  success: CheckCircle2,
};

const CLASSES: Record<AlertVariant, string> = {
  info: "border-info/25 bg-info/8",
  warning: "border-warning/25 bg-warning/8",
  danger: "border-destructive/25 bg-destructive/8",
  success: "border-success/25 bg-success/8",
};

const ICON_CLASSES: Record<AlertVariant, string> = {
  info: "text-info",
  warning: "text-warning",
  danger: "text-destructive",
  success: "text-success",
};

/** Alerte générique — les 4 sémantiques (SPEC-UI.md §1 : jamais une teinte
 * décorative). Règle de fond n°3 : « un avertissement dit quoi faire » —
 * `actions` accepte un ou plusieurs `Button`, et « ne rien changer » est
 * toujours une action valable (à fournir par l'appelant, pas devinée ici).
 * `role="alert"` uniquement pour `danger` (interrompt un lecteur d'écran) —
 * les 3 autres sont `role="status"` (annoncé sans interrompre). */
export function Alert({
  variant = "info",
  title,
  children,
  actions,
  onDismiss,
  className = "",
}: {
  variant?: AlertVariant;
  title?: ReactNode;
  children?: ReactNode;
  actions?: ReactNode;
  onDismiss?: () => void;
  className?: string;
}) {
  const Icon = ICONS[variant];
  return (
    <div
      role={variant === "danger" ? "alert" : "status"}
      className={`rounded-card border p-4 flex gap-3 ${CLASSES[variant]} ${className}`}
    >
      <Icon size={18} className={`flex-shrink-0 mt-0.5 ${ICON_CLASSES[variant]}`} aria-hidden="true" />
      <div className="flex-1 min-w-0">
        {title && <p className="text-body font-semibold text-foreground">{title}</p>}
        {children && <div className="text-caption text-foreground/85 mt-1 leading-relaxed">{children}</div>}
        {actions && <div className="flex flex-wrap gap-2 mt-3">{actions}</div>}
      </div>
      {onDismiss && (
        <button
          type="button"
          onClick={onDismiss}
          aria-label="Fermer l'alerte"
          className="flex-shrink-0 text-muted-foreground hover:text-foreground transition-colors rounded focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)]"
        >
          <X size={16} />
        </button>
      )}
    </div>
  );
}
