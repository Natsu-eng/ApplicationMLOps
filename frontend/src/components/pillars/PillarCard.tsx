import { ArrowRight } from "lucide-react";
import { Link } from "react-router-dom";
import type { Pillar } from "../../config/pillars";
import { Badge } from "../ui/Badge";
import { Card } from "../ui/Card";

/** Carte objectif de l'écran d'orientation. Le pilier actif porte le
 * dégradé de marque sur son badge d'icône (même traitement que le mark du
 * logo) — les piliers "bientôt" restent volontairement en retrait
 * (icône neutre, pas de survol) tout en restant cliquables vers leur
 * page de présentation, pour montrer la trajectoire produit. */
export function PillarCard({ pillar }: { pillar: Pillar }) {
  const Icon = pillar.icon;
  const isActive = pillar.status === "active";

  return (
    <Link
      to={pillar.route}
      className="block h-full rounded-2xl focus:outline-none focus-visible:ring-2 focus-visible:ring-teal-500/50"
    >
      <Card interactive={isActive} className={`p-6 h-full flex flex-col ${isActive ? "" : "opacity-70"}`}>
        <div
          className={`h-11 w-11 rounded-xl flex items-center justify-center mb-4 ${
            isActive ? "bg-brand-gradient text-white" : "bg-slate-100 text-slate-400"
          }`}
        >
          <Icon size={20} strokeWidth={2} />
        </div>
        <h2 className="text-base font-medium text-slate-900 mb-2">{pillar.title}</h2>
        <p className="text-sm text-slate-600 leading-relaxed flex-1">{pillar.description}</p>
        <div className="mt-5">
          {isActive ? (
            <span className="inline-flex items-center gap-1.5 text-sm font-medium text-teal-600">
              Commencer <ArrowRight size={15} />
            </span>
          ) : (
            <Badge variant="neutral">Bientôt disponible</Badge>
          )}
        </div>
      </Card>
    </Link>
  );
}
