import type { LucideIcon } from "lucide-react";

/** Palette d'accent partagée pour les badges d'icône colorés (dashboard,
 * cartes dataset, listes) — une teinte par identité (Lot E1-ter), jamais
 * une magnitude : familles Tailwind -600 déjà utilisées ailleurs dans
 * l'app (règle E1-bis, lisible sur fond clair), fond -50/bordure -200
 * assortis comme les badges de garde-fous (DataQualityWarnings). Le bleu
 * n'est PAS le dégradé de marque (réservé aux moments de marque, voir
 * index.css) — c'est un bleu Tailwind ordinaire au même titre que les
 * trois autres teintes. */
export type AccentColor = "blue" | "teal" | "amber" | "violet";

const ACCENT_CLASSES: Record<AccentColor, string> = {
  blue: "bg-blue-50 border-blue-200 text-blue-600",
  teal: "bg-teal-50 border-teal-200 text-teal-600",
  amber: "bg-amber-50 border-amber-200 text-amber-600",
  violet: "bg-violet-50 border-violet-200 text-violet-600",
};

// Fond de CARTE (pas de l'icône) — lavage très léger de la même teinte, pour
// que la carte entière porte l'identité de couleur, pas seulement le badge
// d'icône (retour explicite : des cartes toutes blanches manquaient de vie).
// Reste largement en dessous du contraste du texte slate-900/700 dessus.
const ACCENT_SURFACE_CLASSES: Record<AccentColor, string> = {
  blue: "bg-blue-50/70 border-blue-100",
  teal: "bg-teal-50/70 border-teal-100",
  amber: "bg-amber-50/70 border-amber-100",
  violet: "bg-violet-50/70 border-violet-100",
};

export function accentSurfaceClass(color: AccentColor): string {
  return ACCENT_SURFACE_CLASSES[color];
}

// Barre d'accent pleine couleur (liseré haut de carte) — un ton plus vif
// que le badge d'icône, réservé à un usage en trait fin (jamais en aplat de
// texte, contraste insuffisant).
const ACCENT_BAR_CLASSES: Record<AccentColor, string> = {
  blue: "bg-blue-400",
  teal: "bg-teal-400",
  amber: "bg-amber-400",
  violet: "bg-violet-400",
};

export function accentBarClass(color: AccentColor): string {
  return ACCENT_BAR_CLASSES[color];
}

const ACCENT_ROTATION: AccentColor[] = ["blue", "teal", "amber", "violet"];

/** Teinte déterministe à partir d'un identifiant (ex. id de dataset) — pour
 * qu'une liste de cartes ne soit pas monochrome sans dépendre d'un champ
 * "type" que le backend ne fournit pas. Même id = même teinte à chaque
 * rendu (pas un random visuel qui change au rafraîchissement). */
export function accentColorForId(id: number): AccentColor {
  return ACCENT_ROTATION[((id % ACCENT_ROTATION.length) + ACCENT_ROTATION.length) % ACCENT_ROTATION.length];
}

export function ColorIconBadge({
  icon: Icon,
  color,
  size = "md",
}: {
  icon: LucideIcon;
  color: AccentColor;
  size?: "sm" | "md";
}) {
  const boxClass = size === "sm" ? "h-8 w-8 rounded-lg" : "h-10 w-10 rounded-xl";
  const iconSize = size === "sm" ? 15 : 18;
  return (
    <div
      className={`${boxClass} border flex items-center justify-center flex-shrink-0 transition-transform duration-200 group-hover:scale-110 group-hover:-rotate-3 ${ACCENT_CLASSES[color]}`}
    >
      <Icon size={iconSize} />
    </div>
  );
}
