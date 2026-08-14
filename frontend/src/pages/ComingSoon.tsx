import {
  Boxes,
  Crosshair,
  Layers,
  Link as LinkIcon,
  PenLine,
  ScanSearch,
  Sparkles,
  type LucideIcon,
} from "lucide-react";
import { Link } from "react-router-dom";
import AppShell from "../components/AppShell";
import { Badge } from "../components/ui/Badge";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { ColorIconBadge, accentColorForId } from "../components/ui/ColorIconBadge";
import { PILLARS, type PillarId } from "../config/pillars";

interface PreviewFeature {
  icon: LucideIcon;
  title: string;
  description: string;
}

/** Aperçu des fonctionnalités prévues par pilier réservé — donne une
 * trajectoire produit concrète plutôt qu'un simple "bientôt disponible"
 * vide, sans en promettre le calendrier. Contenu volontairement propre à
 * cette page (pas dans le registre des piliers, qui reste la source pour
 * la navigation, pas pour le contenu marketing d'une page de présentation). */
const PREVIEW_FEATURES: Record<PillarId, PreviewFeature[]> = {
  supervised: [],
  unsupervised: [
    {
      icon: Boxes,
      title: "Clustering automatique",
      description: "K-Means, DBSCAN et clustering hiérarchique avec choix du nombre de groupes assisté.",
    },
    {
      icon: ScanSearch,
      title: "Détection d'anomalies",
      description: "Isolez les points atypiques et les comportements inhabituels dans vos jeux de données.",
    },
    {
      icon: Layers,
      title: "Réduction de dimension",
      description: "Visualisez vos données en 2D grâce à PCA, t-SNE et UMAP intégrés.",
    },
    {
      icon: Sparkles,
      title: "Profils de segments",
      description: "Comprenez chaque groupe avec des descriptions statistiques générées automatiquement.",
    },
  ],
  vision: [
    {
      icon: Boxes,
      title: "Classification d'images",
      description: "Catégorisez automatiquement vos images à partir d'exemples annotés.",
    },
    {
      icon: Crosshair,
      title: "Détection d'objets",
      description: "Localisez et identifiez plusieurs objets dans une même image.",
    },
    {
      icon: PenLine,
      title: "Annotation assistée",
      description: "Un outil d'annotation intégré accéléré par des pré-annotations intelligentes.",
    },
    {
      icon: LinkIcon,
      title: "Transfer learning",
      description: "Partez de modèles pré-entraînés pour des résultats de qualité avec peu de données.",
    },
  ],
};

/** Page de présentation d'un pilier réservé (non-supervisé, vision) —
 * réutilisable, alimentée par le registre des piliers. Démontre le
 * branchement concret d'un futur pilier : une route + cette page
 * générique, pas une refonte. */
export default function ComingSoon({ pillarId }: { pillarId: PillarId }) {
  const pillar = PILLARS.find((p) => p.id === pillarId);
  if (!pillar) return null;
  const Icon = pillar.icon;
  const features = PREVIEW_FEATURES[pillarId];

  return (
    <AppShell pillarId={pillarId}>
      <div className="max-w-3xl mx-auto text-center py-16">
        <div className="h-16 w-16 mx-auto mb-5 rounded-2xl bg-brand-gradient text-white flex items-center justify-center shadow-sm shadow-primary/20">
          <Icon size={28} strokeWidth={1.75} />
        </div>
        <Badge variant="neutral">
          {pillarId === "unsupervised" ? "ML non supervisé" : "Vision"} · Bientôt disponible
        </Badge>
        <h1 className="text-2xl font-serif text-foreground mt-4 mb-2">{pillar.title}</h1>
        <p className="text-sm text-muted-foreground leading-relaxed mb-8 max-w-lg mx-auto">{pillar.description}</p>

        {features.length > 0 && (
          <div className="grid sm:grid-cols-2 gap-4 mb-8 text-left">
            {features.map((feature, i) => (
              <Card key={feature.title} className="p-4 flex items-start gap-3">
                <ColorIconBadge icon={feature.icon} color={accentColorForId(i)} size="sm" />
                <div className="min-w-0">
                  <p className="text-sm font-medium text-foreground mb-1">{feature.title}</p>
                  <p className="text-xs text-muted-foreground leading-relaxed">{feature.description}</p>
                </div>
              </Card>
            ))}
          </div>
        )}

        <p className="text-sm text-muted-foreground mb-6">
          Ce pilier n'est pas encore actif. La prédiction (valeur ou catégorie) est disponible
          dès maintenant depuis l'écran des objectifs.
        </p>
        <Link to="/" className="inline-block">
          <Button variant="secondary">Retour aux objectifs</Button>
        </Link>
      </div>
    </AppShell>
  );
}
