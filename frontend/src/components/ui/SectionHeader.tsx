import type { LucideIcon } from "lucide-react";
import { ColorIconBadge, type AccentColor } from "./ColorIconBadge";
import { LabelWithHelp } from "./Tooltip";

/** En-tête de section commun — icône colorée + titre (+ info-bulle
 * optionnelle), pour que chaque bloc d'une page à plusieurs sections reste
 * identifiable au premier coup d'œil plutôt qu'une liste uniforme de
 * libellés gris en majuscules. Partagé entre EdaModal et ModelResultModal
 * (Lot refonte Résultats/Datasets) — avant, chacun dupliquait le même motif
 * `<p className="text-xs uppercase ...">`. */
export function SectionHeader({
  icon,
  color,
  label,
  help,
}: {
  icon: LucideIcon;
  color: AccentColor;
  label: string;
  help?: string;
}) {
  return (
    <div className="flex items-center gap-2.5 mb-3">
      <ColorIconBadge icon={icon} color={color} size="sm" />
      <span className="text-sm font-medium text-foreground">
        {help ? <LabelWithHelp label={label} help={help} /> : label}
      </span>
    </div>
  );
}
