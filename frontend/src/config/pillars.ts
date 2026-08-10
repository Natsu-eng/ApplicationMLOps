import { BrainCircuit, Database, LayoutDashboard, ScanEye, Shapes, Target, type LucideIcon } from "lucide-react";

/** Registre unique des piliers du produit — l'écran d'orientation, la
 * navigation (AppShell) et les routes réservées lisent tous ce fichier.
 * Ajouter un futur pilier réel (non-supervisé, vision) = compléter cette
 * liste (title/description/navItems/status), pas refondre l'architecture. */

export type PillarId = "supervised" | "unsupervised" | "vision";

export interface PillarNavItem {
  to: string;
  label: string;
  icon: LucideIcon;
}

export interface Pillar {
  id: PillarId;
  title: string;
  description: string;
  icon: LucideIcon;
  status: "active" | "soon";
  /** Route vers laquelle la carte d'orientation navigue. */
  route: string;
  /** Items de nav affichés dans AppShell quand ce pilier est actif — vide tant que le pilier n'a pas d'écrans réels. */
  navItems: PillarNavItem[];
}

export const PILLARS: Pillar[] = [
  {
    id: "supervised",
    title: "Prédire une valeur ou une catégorie",
    description:
      "À partir de vos données passées, estimez une valeur (un prix, une résistance, une durée) ou prédisez une catégorie (conforme ou non, à risque ou non). Idéal quand vous connaissez déjà le résultat sur d'anciens cas et voulez l'anticiper sur de nouveaux.",
    icon: Target,
    status: "active",
    route: "/dashboard",
    navItems: [
      { to: "/dashboard", label: "Tableau de bord", icon: LayoutDashboard },
      { to: "/datasets", label: "Mes données", icon: Database },
      { to: "/training", label: "Entraînement", icon: BrainCircuit },
    ],
  },
  {
    id: "unsupervised",
    title: "Découvrir des groupes dans vos données",
    description:
      "Repérez des profils similaires ou des cas atypiques, sans savoir à l'avance ce que vous cherchez.",
    icon: Shapes,
    status: "soon",
    route: "/unsupervised",
    navItems: [],
  },
  {
    id: "vision",
    title: "Analyser des images",
    description: "Classez des images ou détectez des défauts à partir de photos ou de scans.",
    icon: ScanEye,
    status: "soon",
    route: "/vision",
    navItems: [],
  },
];
