import AppShell from "../components/AppShell";
import { PillarCard } from "../components/pillars/PillarCard";
import { PILLARS } from "../config/pillars";
import { useAuth } from "../contexts/AuthContext";

/** Écran d'entrée post-connexion — l'utilisateur choisit son OBJECTIF
 * métier, pas la méthode technique. Point d'orientation entre les piliers
 * du produit (supervisé actif, non-supervisé/vision réservés). */
export default function Orientation() {
  const { user } = useAuth();
  if (!user) return null;

  return (
    <AppShell>
      <div className="mb-10 text-center">
        <p className="text-xs uppercase tracking-widest text-primary font-semibold mb-2">
          Bonjour, {user.nom.split(" ")[0]}
        </p>
        <h1 className="text-title text-foreground mb-2">Que voulez-vous faire aujourd'hui ?</h1>
        <p className="text-sm text-muted-foreground">Choisissez votre objectif — la méthode s'adapte, pas l'inverse.</p>
      </div>

      <div className="grid gap-6 sm:grid-cols-3">
        {PILLARS.map((pillar) => (
          <PillarCard key={pillar.id} pillar={pillar} />
        ))}
      </div>
    </AppShell>
  );
}
