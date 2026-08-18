import {
  AlertTriangle,
  BrainCircuit,
  Boxes,
  Database,
  History,
  ScanEye,
  ScatterChart,
  Shapes,
  Sparkles,
  Target,
  type LucideIcon,
} from "lucide-react";
import type { AccentColor } from "../components/ui/ColorIconBadge";

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
  /** 3 cas d'usage concrets affichés en puces sur la carte d'orientation
   * (Lot 16) — rend l'objectif tangible pour un utilisateur non expert,
   * plutôt que la seule description générique. */
  examples: string[];
  /** Items de nav affichés dans AppShell quand ce pilier est actif — vide tant que le pilier n'a pas d'écrans réels. */
  navItems: PillarNavItem[];
  /** Couleur d'identité du pilier (Lot 2A correctif 3) — SOURCE UNIQUE :
   * toute UI qui affiche une donnée appartenant à un pilier précis (tuile,
   * badge, section) doit lire cette couleur via `pillarColor()` plutôt que
   * choisir une teinte au hasard localement. Une donnée qui n'appartient à
   * AUCUN pilier (ex. total de datasets tous piliers confondus) reste
   * `"neutral"`, jamais une couleur de pilier empruntée arbitrairement.
   * Distinctes des 3 couleurs entre elles ; alignées sur l'usage déjà
   * dominant de chaque pilier sur ses propres pages (`Training.tsx`,
   * `Clustering.tsx`, `VisionDatasets.tsx`) pour minimiser la dissonance
   * quand le Lot 2B appliquera cette même couleur aux 16 autres pages. */
  color: AccentColor;
}

export const PILLARS: Pillar[] = [
  {
    id: "supervised",
    title: "Prédire une valeur ou une catégorie",
    description:
      "À partir de vos données passées, estimez une valeur (un prix, une résistance, une durée) ou prédisez une catégorie (conforme ou non, à risque ou non). Idéal quand vous connaissez déjà le résultat sur d'anciens cas et voulez l'anticiper sur de nouveaux.",
    icon: Target,
    examples: ["Prédiction de défaillance", "Estimation de durée de vie", "Classification de pièces conformes"],
    status: "active",
    // Route vers le module d'action du pilier (Entraînement), PAS vers
    // "/dashboard" — choisir un objectif depuis l'écran d'orientation doit
    // ouvrir directement ce pilier, jamais rediriger vers la vue d'ensemble
    // générale (bug réel signalé : "Objectifs sert à ça"). Même principe que
    // le pilier non supervisé ci-dessous, dont la route pointe déjà sur son
    // propre module ("/clustering"), pas sur un hub intermédiaire.
    route: "/training",
    // "Tableau de bord" n'est PAS ici : épinglé en haut de la sidebar
    // (AppShell.tsx), toujours accessible quel que soit le pilier actif —
    // seul ce qui suit reste filtré sur l'objectif choisi.
    navItems: [
      { to: "/datasets", label: "Mes données", icon: Database },
      { to: "/training", label: "Entraînement", icon: BrainCircuit },
      { to: "/training/history", label: "Historique", icon: History },
    ],
    color: "violet",
  },
  {
    id: "unsupervised",
    title: "Découvrir des groupes dans vos données",
    description:
      "Repérez des profils similaires ou des cas atypiques, sans savoir à l'avance ce que vous cherchez.",
    icon: Shapes,
    examples: ["Segmentation d'équipements similaires", "Visualisation 2D d'un jeu de données", "Détection de mesures atypiques"],
    // Lot 11+ : clustering + profils de segments. Lot 13 : réduction de
    // dimension. Lot 14 : détection d'anomalies. Les 3 modules du pilier
    // sont désormais actifs.
    status: "active",
    route: "/clustering",
    navItems: [
      { to: "/clustering", label: "Clustering", icon: Shapes },
      { to: "/reduction-dimension", label: "Réduction de dimension", icon: ScatterChart },
      { to: "/anomalies", label: "Détection d'anomalies", icon: AlertTriangle },
      { to: "/non-supervise/historique", label: "Historique", icon: History },
    ],
    color: "rose",
  },
  {
    id: "vision",
    title: "Analyser des images",
    description: "Classez des images ou détectez des défauts à partir de photos ou de scans.",
    icon: ScanEye,
    examples: ["Contrôle qualité visuel", "Tri automatique de pièces", "Détection de défauts de fabrication"],
    // Lot 15 : classification d'images (transfer learning, Grad-CAM) et
    // détection d'anomalies visuelles (structure normal/défaut) actives.
    // Détection d'objets et annotation assistée hors périmètre (Lot 16+, non cadré).
    status: "active",
    route: "/vision/classification",
    navItems: [
      { to: "/vision/datasets", label: "Mes données", icon: Database },
      { to: "/vision/classification", label: "Classification d'images", icon: Boxes },
      { to: "/vision/anomalies", label: "Anomalies visuelles", icon: Sparkles },
      { to: "/vision/historique", label: "Historique", icon: History },
    ],
    color: "teal",
  },
];

/** Couleur d'identité d'un pilier — jamais un lookup local dans une page ou
 * un composant (voir `Pillar.color`). Lève si `id` est absent du registre :
 * un pilier qui n'existe pas ne doit jamais silencieusement retomber sur
 * une couleur par défaut. */
export function pillarColor(id: PillarId): AccentColor {
  const pillar = PILLARS.find((p) => p.id === id);
  if (!pillar) throw new Error(`Pilier inconnu : ${id}`);
  return pillar.color;
}
