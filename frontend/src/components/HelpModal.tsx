import {
  AlertTriangle,
  BrainCircuit,
  ChartColumn,
  Database,
  History,
  Images,
  LineChart,
  ScatterChart,
  Shapes,
} from "lucide-react";
import { Modal } from "./ui/Modal";
import { Card } from "./ui/Card";
import { ColorIconBadge, type AccentColor } from "./ui/ColorIconBadge";

interface HelpStep {
  icon: typeof Database;
  color: AccentColor;
  title: string;
  description: string;
}

const SUPERVISED_STEPS: HelpStep[] = [
  {
    icon: Database,
    color: "blue",
    title: "1. Importez vos données",
    description:
      "Déposez un fichier CSV, Parquet, Excel ou JSON depuis « Mes données ». Explorez-le (statistiques, corrélations, valeurs manquantes) avant même de lancer un entraînement.",
  },
  {
    icon: BrainCircuit,
    color: "teal",
    title: "2. Entraînez un modèle",
    description:
      "Choisissez un dataset et la colonne à prédire. DataLab compare automatiquement plusieurs algorithmes et retient le meilleur — aucune expertise technique requise en mode guidé.",
  },
  {
    icon: LineChart,
    color: "violet",
    title: "3. Comprenez le résultat",
    description:
      "Chaque résultat s'accompagne d'une explication en langage clair : pourquoi ce modèle a gagné, quelles variables comptent le plus, à quel point on peut lui faire confiance.",
  },
  {
    icon: ChartColumn,
    color: "amber",
    title: "4. Testez une prédiction",
    description:
      "Saisissez un nouveau cas dans le formulaire généré automatiquement et obtenez une prédiction immédiate, avec sa fourchette de confiance ou ses probabilités par classe.",
  },
  {
    icon: History,
    color: "rose",
    title: "5. Suivez vos expériences",
    description:
      "L'historique garde chaque entraînement passé — comparez plusieurs modèles entre eux et promouvez celui sur lequel vous pouvez compter en « production ».",
  },
];

const UNSUPERVISED_STEPS: HelpStep[] = [
  {
    icon: Shapes,
    color: "rose",
    title: "1. Découvrez des groupes (Clustering)",
    description:
      "Choisissez un dataset et les variables à analyser. Plusieurs algorithmes et nombres de groupes sont comparés automatiquement, jamais un seul essai — chaque groupe est décrit par ce qui le distingue le plus des autres.",
  },
  {
    icon: ScatterChart,
    color: "blue",
    title: "2. Visualisez en 2D (Réduction de dimension)",
    description:
      "PCA, t-SNE ou UMAP ramènent vos données à 2 dimensions pour repérer visuellement des groupes ou des points isolés. Attention : t-SNE/UMAP préservent les voisinages locaux, pas les distances globales — jamais une carte fidèle des distances réelles.",
  },
  {
    icon: AlertTriangle,
    color: "amber",
    title: "3. Repérez les atypiques (Détection d'anomalies)",
    description:
      "Isolation Forest et LOF tournent toujours ensemble — le score de consensus recoupe les deux méthodes, jamais un seul algorithme élu à l'aveugle. Chaque observation classée porte une explication réelle (variables qui s'écartent le plus).",
  },
];

const VISION_STEPS: HelpStep[] = [
  {
    icon: Images,
    color: "teal",
    title: "1. Classifiez vos images",
    description:
      "Importez un ZIP organisé en dossiers de classes, explorez les images avant d'entraîner (galerie, rapport de qualité), puis laissez DataLab comparer les architectures pré-entraînées disponibles. Chaque prédiction peut être expliquée visuellement (Grad-CAM) : les zones rouges sont celles qui ont le plus influencé la classe prédite.",
  },
  {
    icon: AlertTriangle,
    color: "violet",
    title: "2. Détectez les défauts visuels",
    description:
      "Structure MVTec AD (images normales pour l'entraînement, normales + défectueuses pour le test) : le modèle apprend à reconstruire une image normale, un écart important signale un défaut. Le seuil de détection est calibré automatiquement, jamais deviné.",
  },
];

/** Premier point d'aide/onboarding du produit (AUDIT_ROADMAP.md, refonte UI)
 * — accessible à tout moment depuis l'AppShell, pas seulement à la première
 * connexion : un rappel du workflow doit rester disponible, pas seulement
 * montré une fois puis perdu. Étendu (AUDIT_PILIER2_ET_REFONTE_UX.md, D4) au
 * pilier ML non supervisé, puis (Lot 16) au pilier Vision — chacun muet sur
 * l'Aide dès son activation si non ajouté ici en même temps que la page. */
export function HelpModal({ onClose }: { onClose: () => void }) {
  return (
    <Modal title="Comment utiliser DataLab Pro" onClose={onClose} size="xl">
      <div className="mb-6">
        <p className="text-sm text-muted-foreground mb-3">
          Le parcours du ML supervisé, en 5 étapes — de vos données à une prédiction exploitable.
        </p>
        <div className="grid gap-3 sm:grid-cols-2">
          {SUPERVISED_STEPS.map((step) => (
            <Card key={step.title} className="p-4 flex items-start gap-3">
              <ColorIconBadge icon={step.icon} color={step.color} size="sm" />
              <div className="min-w-0">
                <p className="text-sm font-medium text-foreground mb-1">{step.title}</p>
                <p className="text-xs text-muted-foreground leading-relaxed">{step.description}</p>
              </div>
            </Card>
          ))}
        </div>
      </div>

      <div className="mb-6">
        <p className="text-sm text-muted-foreground mb-3">
          Le parcours du ML non supervisé, en 3 modules — sans cible à prédire, pour explorer vos données autrement.
        </p>
        <div className="grid gap-3 sm:grid-cols-2">
          {UNSUPERVISED_STEPS.map((step) => (
            <Card key={step.title} className="p-4 flex items-start gap-3">
              <ColorIconBadge icon={step.icon} color={step.color} size="sm" />
              <div className="min-w-0">
                <p className="text-sm font-medium text-foreground mb-1">{step.title}</p>
                <p className="text-xs text-muted-foreground leading-relaxed">{step.description}</p>
              </div>
            </Card>
          ))}
        </div>
      </div>

      <div>
        <p className="text-sm text-muted-foreground mb-3">
          Le parcours Vision, en 2 modules — pour analyser des images plutôt que des tableaux de données.
        </p>
        <div className="grid gap-3 sm:grid-cols-2">
          {VISION_STEPS.map((step) => (
            <Card key={step.title} className="p-4 flex items-start gap-3">
              <ColorIconBadge icon={step.icon} color={step.color} size="sm" />
              <div className="min-w-0">
                <p className="text-sm font-medium text-foreground mb-1">{step.title}</p>
                <p className="text-xs text-muted-foreground leading-relaxed">{step.description}</p>
              </div>
            </Card>
          ))}
        </div>
      </div>
    </Modal>
  );
}
