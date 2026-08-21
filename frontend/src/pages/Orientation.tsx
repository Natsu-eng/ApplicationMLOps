import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import AppShell from "../components/AppShell";
import { PillarCard } from "../components/pillars/PillarCard";
import { PILLARS, type Pillar } from "../config/pillars";
import { useAuth } from "../contexts/AuthContext";
import { getLastPillar } from "../utils/lastPillar";

/** Écran d'entrée post-connexion — l'utilisateur choisit son OBJECTIF
 * métier, pas la méthode technique. Point d'orientation entre les piliers
 * du produit (supervisé actif, non-supervisé/vision réservés). */
export default function Orientation() {
  const { user } = useAuth();
  // Lecture après montage (pas d'état initial dérivé de localStorage) —
  // évite tout écart d'hydratation, et cet écran n'a aucune raison de
  // dépendre du rendu serveur ici de toute façon ; simple prudence.
  const [lastPillar, setLastPillarState] = useState<Pillar | null>(null);
  useEffect(() => {
    const id = getLastPillar();
    setLastPillarState(id ? PILLARS.find((p) => p.id === id) ?? null : null);
  }, []);

  if (!user) return null;

  return (
    <AppShell>
      <div className="mb-10 text-center">
        <p className="text-xs uppercase tracking-widest text-primary font-semibold mb-2">
          Bonjour, {user.nom.split(" ")[0]}
        </p>
        <h1 className="text-title text-foreground mb-2">Que voulez-vous faire aujourd'hui ?</h1>
        <p className="text-sm text-muted-foreground">Choisissez votre objectif — la méthode s'adapte, pas l'inverse.</p>
        {/* Raccourci, jamais une redirection automatique (Lot 7, §J.1) —
            l'écran d'orientation reste un vrai choix à chaque visite. */}
        {lastPillar && (
          <Link
            to={lastPillar.route}
            className="inline-flex items-center gap-1.5 text-sm text-primary hover:text-primary/80 mt-3"
          >
            Reprendre « {lastPillar.title} »
            <ArrowRight size={14} />
          </Link>
        )}
      </div>

      <div className="grid gap-6 sm:grid-cols-3">
        {PILLARS.map((pillar) => (
          <PillarCard key={pillar.id} pillar={pillar} />
        ))}
      </div>
    </AppShell>
  );
}
