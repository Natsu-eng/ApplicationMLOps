import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";
import { accentSurfaceClass, type AccentColor } from "./ColorIconBadge";

/** En-tête de page unifié — ancre visuellement chaque page avec une vraie
 * couleur (retour utilisateur direct sur captures, 2026-08-14 — deux allers-
 * retours : un lavis à 5% d'opacité était invisible, puis un badge pastel
 * plat restait trop discret face aux références fournies, qui utilisent des
 * icônes en DÉGRADÉ vif sur une ambiance de fond légèrement teintée). Icône
 * en dégradé de marque (`.bg-brand-gradient`, réservé aux moments vitrine —
 * un en-tête de page en est un) + fond teinté dans la couleur du pilier. */
export function PageHeader({
  eyebrow,
  title,
  description,
  icon: Icon,
  color,
  action,
}: {
  eyebrow: string;
  title: ReactNode;
  description?: ReactNode;
  icon: LucideIcon;
  color: AccentColor;
  action?: ReactNode;
}) {
  return (
    <div
      className={`relative overflow-hidden rounded-2xl border shadow-md shadow-foreground/[0.04] mb-8 ${accentSurfaceClass(color)}`}
    >
      <div className="p-6 sm:p-7 flex items-start justify-between gap-4 flex-wrap">
        <div className="flex items-start gap-4 min-w-0">
          <div className="h-12 w-12 rounded-2xl bg-brand-gradient flex items-center justify-center flex-shrink-0 shadow-sm shadow-primary/20">
            <Icon size={22} className="text-white" strokeWidth={2} />
          </div>
          <div className="min-w-0">
            <p className="text-xs uppercase tracking-widest text-primary font-semibold mb-1.5">{eyebrow}</p>
            <h1 className="text-2xl font-serif text-foreground">{title}</h1>
            {description && <p className="text-sm text-muted-foreground mt-1.5 max-w-xl">{description}</p>}
          </div>
        </div>
        {action && <div className="flex-shrink-0">{action}</div>}
      </div>
    </div>
  );
}
