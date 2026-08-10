import { Link } from "react-router-dom";
import AppShell from "../components/AppShell";
import { Badge } from "../components/ui/Badge";
import { Button } from "../components/ui/Button";
import { PILLARS, type PillarId } from "../config/pillars";

/** Page de présentation d'un pilier réservé (non-supervisé, vision) —
 * réutilisable, alimentée par le registre des piliers. Démontre le
 * branchement concret d'un futur pilier : une route + cette page
 * générique, pas une refonte. */
export default function ComingSoon({ pillarId }: { pillarId: PillarId }) {
  const pillar = PILLARS.find((p) => p.id === pillarId);
  if (!pillar) return null;
  const Icon = pillar.icon;

  return (
    <AppShell pillarId={pillarId}>
      <div className="max-w-lg mx-auto text-center py-16">
        <div className="h-14 w-14 mx-auto mb-5 rounded-2xl bg-slate-800 flex items-center justify-center text-slate-400">
          <Icon size={26} strokeWidth={1.75} />
        </div>
        <h1 className="text-xl font-serif text-slate-100 mb-2">{pillar.title}</h1>
        <p className="text-sm text-slate-400 mb-6">{pillar.description}</p>
        <Badge variant="neutral">Bientôt disponible</Badge>
        <p className="text-sm text-slate-500 mt-6">
          Ce pilier n'est pas encore actif. La prédiction (valeur ou catégorie) est disponible
          dès maintenant depuis l'écran des objectifs.
        </p>
        <Link to="/" className="inline-block mt-5">
          <Button variant="secondary">Retour aux objectifs</Button>
        </Link>
      </div>
    </AppShell>
  );
}
