import { Boxes } from "lucide-react";
import type { ClusterProfile } from "../../api/client";
import { Card } from "../ui/Card";
import { ColorIconBadge, accentSurfaceClass, type AccentColor } from "../ui/ColorIconBadge";

/** Grille de cartes "profil de segment" — extraite de `Clustering.tsx`
 * (jusqu'ici dupliquée en germe, la seule utilisation vivait dans l'onglet
 * "Profils de segments" du gagnant) pour être réutilisée par la
 * comparaison détaillée du top 3 (retour utilisateur direct : "propose
 * les 3 meilleurs modèles, résultats propres pour chaque, laisse le choix
 * à l'utilisateur") — même rendu exact pour le gagnant ET pour chaque
 * candidat du top 3, jamais deux présentations différentes du même genre
 * d'information. */

const CANDIDATE_COLORS: AccentColor[] = ["blue", "violet", "teal", "amber", "rose"];

export function ClusterProfileGrid({ profiles }: { profiles: ClusterProfile[] }) {
  return (
    <div className="grid sm:grid-cols-2 gap-4">
      {profiles.map((profile, i) => {
        const color = CANDIDATE_COLORS[i % CANDIDATE_COLORS.length];
        return (
          <Card key={profile.cluster_id} className={`p-4 ${accentSurfaceClass(color)}`}>
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <ColorIconBadge icon={Boxes} color={color} size="sm" />
                <span className="text-sm font-medium text-foreground">Segment {profile.cluster_id + 1}</span>
              </div>
              <span className="text-sm font-semibold tabular-nums text-foreground">{profile.size_pct.toFixed(1)} %</span>
            </div>
            <p className="text-xs text-muted-foreground mb-2">
              {profile.size} observation{profile.size > 1 ? "s" : ""}
            </p>
            {profile.differentiating_variables.length > 0 && (
              <div className="space-y-1 mb-2">
                <p className="text-overline uppercase text-muted-foreground">Variables différenciantes</p>
                {profile.differentiating_variables.slice(0, 3).map((varName) => {
                  const stat = profile.numeric_summary[varName];
                  if (!stat) return null;
                  return (
                    <p key={varName} className="text-xs text-foreground/90">
                      <span className="font-medium">{varName}</span> : moyenne {stat.mean.toFixed(2)}{" "}
                      <span className="text-muted-foreground">
                        ({stat.z_score > 0 ? "+" : ""}
                        {stat.z_score.toFixed(1)}σ vs population)
                      </span>
                    </p>
                  );
                })}
              </div>
            )}
            {Object.entries(profile.categorical_summary).length > 0 && (
              <div className="space-y-1">
                {Object.entries(profile.categorical_summary)
                  .slice(0, 2)
                  .map(([col, cat]) => (
                    <p key={col} className="text-xs text-foreground/90">
                      <span className="font-medium">{col}</span> dominant : {cat.top_category} ({cat.top_pct.toFixed(0)} %
                      {cat.lift !== null && (cat.lift >= 1.5 || cat.lift <= 0.67) && (
                        <span className="text-muted-foreground">
                          {" "}
                          — vs {cat.population_pct.toFixed(0)} % sur l'ensemble, ×{cat.lift.toFixed(1)}
                        </span>
                      )}
                      )
                    </p>
                  ))}
              </div>
            )}
          </Card>
        );
      })}
    </div>
  );
}
