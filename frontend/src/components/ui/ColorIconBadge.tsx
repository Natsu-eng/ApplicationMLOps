import type { LucideIcon } from "lucide-react";

/** Palette d'accent partagée pour les badges d'icône colorés (dashboard,
 * cartes dataset, listes) — une teinte par identité (Lot E1-ter), jamais
 * une magnitude. Refonte visuelle (Lot UI, Fondations) : les anciennes
 * classes Tailwind par teinte/magnitude fixe sont interdites partout
 * (SPEC-UI.md §1 — « un composant lit un jeton, point ») — chaque identité
 * pointe maintenant vers l'une des 4 premières couleurs de série
 * (--s1…--s4, alias Tailwind accent-1…4 posés dans index.css), calculées
 * pour rester distinguables entre elles dans les 5 thèmes ET en
 * deutéranopie simulée (_design/tune.py). "rose" reste réservé aux états
 * d'échec/danger réels (jamais dans la rotation déterministe). */
export type AccentColor = "blue" | "teal" | "amber" | "violet" | "rose" | "neutral";

const ACCENT_CLASSES: Record<AccentColor, string> = {
  blue: "bg-accent-1/12 border-accent-1/35 text-accent-1",
  teal: "bg-accent-3/12 border-accent-3/35 text-accent-3",
  amber: "bg-accent-2/12 border-accent-2/35 text-accent-2",
  violet: "bg-accent-4/12 border-accent-4/35 text-accent-4",
  // Réservé aux états d'échec/danger (ex. entraînement en échec) — jamais
  // dans la rotation déterministe par id (ACCENT_ROTATION ci-dessous),
  // seulement quand la couleur doit refléter un statut réel.
  rose: "bg-destructive/12 border-destructive/35 text-destructive",
  // Pour une identité qui n'appartient à AUCUN pilier (ex. StatTile
  // "Datasets"/"Membres" du dashboard, Lot 2A correctif 3) — jamais dans
  // ACCENT_ROTATION : une teinte de pilier mal assignée est pire qu'une
  // absence de teinte.
  neutral: "bg-muted border-border text-muted-foreground",
};

// Fond de CARTE (pas de l'icône) — lavage très léger de la même teinte, pour
// que la carte entière porte l'identité de couleur, pas seulement le badge
// d'icône (retour explicite : des cartes toutes blanches manquaient de vie).
// Opacité de fond volontairement faible : --s1…--s4 sont des couleurs de
// SÉRIE, pas des pastels calibrés comme --canvas/--surface — certaines sont
// sombres même en thème clair (ex. --s1 en ivoire est un bleu profond), et
// une teinte trop appuyée assombrit assez le fond pour faire tomber
// `text-muted-foreground` sous 4,5:1. /4 mesurait 4,47:1 en ivoire
// (_design/JOURNAL.md Lot 1), /4 encore 4,48-4,49:1 mesuré par axe-core en
// conditions réelles sur `bg-destructive/4`/`bg-accent-1/4` (Lot 11,
// vérification finale) — sous le seuil dans les deux cas, l'écart n'étant
// que de quelques centièmes. /3 repasse au-dessus avec une marge réelle,
// vérifié par axe-core sur les 5 thèmes.
const ACCENT_SURFACE_CLASSES: Record<AccentColor, string> = {
  blue: "bg-accent-1/2 border-accent-1/20",
  teal: "bg-accent-3/2 border-accent-3/20",
  amber: "bg-accent-2/2 border-accent-2/20",
  violet: "bg-accent-4/2 border-accent-4/20",
  rose: "bg-destructive/2 border-destructive/20",
  neutral: "bg-muted/70 border-border",
};

export function accentSurfaceClass(color: AccentColor): string {
  return ACCENT_SURFACE_CLASSES[color];
}

// Encre forte — couleur de TEXTE portant une valeur chiffrée ou un libellé.
//
// CORRECTIF (revue finale) : ce bloc pointait vers --s1…--s4, les couleurs de
// SÉRIE. Elles sont calculées pour des MARQUES DE GRAPHIQUE (seuil 3:1,
// WCAG 1.4.11), pas pour du texte (4,5:1) — le commentaire précédent
// affirmait l'inverse, c'était faux. Mesuré sur themes.css :
//     --s2 : 3,50:1 en ivoire · 3,41:1 en porcelaine
//     --s3 : 4,06:1 en ivoire · 3,96:1 en porcelaine
// Un `text-h2` (18px/600) n'est PAS du « texte large » au sens WCAG
// (il faut ≥24px, ou ≥18,66px ET gras 700) : il exige donc 4,5:1 plein.
// La tuile « Analyses ML » du tableau de bord affichait ainsi le compte
// Vision (pilier teal → --s3) à 4,18-4,27:1 dans les deux thèmes clairs.
//
// Les jetons --sN-ink gardent exactement la même identité de teinte et
// n'ajustent que la clarté jusqu'à ≥4,5:1 sur les trois fonds (écart de
// couleur ΔE2000 ≤ 8,3, invisible en usage ; séparation entre encres
// préservée : ΔE ≥ 24, ≥ 14 en deutéranopie). En thème sombre, les séries
// passaient déjà : les encres y sont identiques, le changement est neutre.
// Les graphiques continuent d'utiliser --s1…--s4 — ne pas les y remplacer.
const ACCENT_VALUE_TEXT_CLASSES: Record<AccentColor, string> = {
  blue: "text-accent-1-ink",
  teal: "text-accent-3-ink",
  amber: "text-accent-2-ink",
  violet: "text-accent-4-ink",
  rose: "text-destructive",
  neutral: "text-foreground",
};

export function accentValueTextClass(color: AccentColor): string {
  return ACCENT_VALUE_TEXT_CLASSES[color];
}

// Bordure pleine teinte (liseré fin, ex. carte de métrique) — un cran plus
// affirmé que la bordure /20 des surfaces.
const ACCENT_BORDER_CLASSES: Record<AccentColor, string> = {
  blue: "border-accent-1/35",
  teal: "border-accent-3/35",
  amber: "border-accent-2/35",
  violet: "border-accent-4/35",
  rose: "border-destructive/35",
  neutral: "border-border",
};

export function accentBorderClass(color: AccentColor): string {
  return ACCENT_BORDER_CLASSES[color];
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
  size?: "sm" | "md" | "lg";
}) {
  const boxClass =
    size === "sm" ? "h-8 w-8 rounded-lg" : size === "lg" ? "h-12 w-12 rounded-2xl" : "h-10 w-10 rounded-xl";
  const iconSize = size === "sm" ? 15 : size === "lg" ? 22 : 18;
  return (
    <div
      className={`${boxClass} border flex items-center justify-center flex-shrink-0 transition-transform duration-200 group-hover:scale-110 group-hover:-rotate-3 ${ACCENT_CLASSES[color]}`}
    >
      <Icon size={iconSize} />
    </div>
  );
}
