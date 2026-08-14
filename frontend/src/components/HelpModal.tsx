import { BrainCircuit, ChartColumn, Database, History, LineChart } from "lucide-react";
import { Modal } from "./ui/Modal";
import { Card } from "./ui/Card";
import { ColorIconBadge, type AccentColor } from "./ui/ColorIconBadge";

interface HelpStep {
  icon: typeof Database;
  color: AccentColor;
  title: string;
  description: string;
}

const STEPS: HelpStep[] = [
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

/** Premier point d'aide/onboarding du produit (AUDIT_ROADMAP.md, refonte UI)
 * — accessible à tout moment depuis l'AppShell, pas seulement à la première
 * connexion : un rappel du workflow doit rester disponible, pas seulement
 * montré une fois puis perdu. */
export function HelpModal({ onClose }: { onClose: () => void }) {
  return (
    <Modal title="Comment utiliser DataLab Pro" onClose={onClose}>
      <p className="text-sm text-muted-foreground mb-5">
        Le parcours du ML supervisé, en 5 étapes — de vos données à une prédiction exploitable.
      </p>
      <div className="grid gap-3 sm:grid-cols-2">
        {STEPS.map((step) => (
          <Card key={step.title} className="p-4 flex items-start gap-3">
            <ColorIconBadge icon={step.icon} color={step.color} size="sm" />
            <div className="min-w-0">
              <p className="text-sm font-medium text-foreground mb-1">{step.title}</p>
              <p className="text-xs text-muted-foreground leading-relaxed">{step.description}</p>
            </div>
          </Card>
        ))}
      </div>
    </Modal>
  );
}
