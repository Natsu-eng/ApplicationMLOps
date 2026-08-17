import { useState } from "react";
import { AlertTriangle, ChevronDown, Info, Lightbulb, ShieldAlert } from "lucide-react";
import type { ModelVerdictData, VerdictLevel } from "../../api/client";
import { Badge } from "../ui/Badge";
import { Card } from "../ui/Card";
import { SectionHeader } from "../ui/SectionHeader";

/** Verdict post-entraînement (Lot 3, correctif I1, AUDIT_DATALAB_2026-08-16.md
 * §E.3) — jusqu'ici, un modèle affichait des graphiques et des nombres bruts
 * sans jamais répondre à « ce modèle surapprend-il ? », « est-il fiable ? »,
 * « et maintenant ? ». Toujours en tête de la vue Résultats (jamais dans un
 * onglet secondaire) : c'est la promesse produit elle-même.
 *
 * Règles calculées côté serveur (`services/model_verdict.py`), jamais
 * recalculées ici — ce composant affiche, il ne juge pas. Même vocabulaire
 * de niveau ("critique"/"attention"/"info") et même motif visuel (carte
 * dépliable par affirmation) que `DataQualityWarnings.tsx`, pour que
 * l'utilisateur reconnaisse tout de suite le même type de garde-fou. */

const LEVEL_CONFIG: Record<
  VerdictLevel,
  { badge: "danger" | "warning" | "accent"; icon: typeof ShieldAlert; border: string; iconColor: string }
> = {
  critique: { badge: "danger", icon: ShieldAlert, border: "border-destructive/20", iconColor: "text-destructive" },
  attention: { badge: "warning", icon: AlertTriangle, border: "border-amber-200", iconColor: "text-amber-600" },
  info: { badge: "accent", icon: Info, border: "border-primary/20", iconColor: "text-primary" },
};

const LEVEL_LABEL: Record<VerdictLevel, string> = {
  critique: "Critique",
  attention: "Attention",
  info: "Info",
};

export function ModelVerdict({ verdict }: { verdict: ModelVerdictData }) {
  const [expanded, setExpanded] = useState<Set<number>>(new Set());

  function toggle(index: number) {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  }

  return (
    <Card className="p-5">
      <SectionHeader icon={Lightbulb} color="amber" label="Verdict" />

      <div className="flex items-start gap-2.5 rounded-lg border border-primary/20 bg-primary/5 px-3 py-2.5 mb-3">
        <Lightbulb size={15} className="flex-shrink-0 mt-0.5 text-primary" />
        <p className="text-sm text-foreground">{verdict.next_action}</p>
      </div>

      <div className="space-y-2">
        {verdict.claims.map((claim, index) => {
          const config = LEVEL_CONFIG[claim.level];
          const Icon = config.icon;
          const isOpen = expanded.has(index);
          return (
            <div key={claim.code} className={`rounded-lg border ${config.border} bg-muted px-3 py-2.5`}>
              <button type="button" onClick={() => toggle(index)} className="w-full flex items-start gap-2 text-left">
                <Icon size={15} className={`flex-shrink-0 mt-0.5 ${config.iconColor}`} />
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2 flex-wrap">
                    <Badge variant={config.badge}>{LEVEL_LABEL[claim.level]}</Badge>
                    <p className="text-sm text-foreground font-medium">{claim.title}</p>
                  </div>
                </div>
                <ChevronDown
                  size={14}
                  className={`flex-shrink-0 text-muted-foreground transition-transform mt-1 ${isOpen ? "rotate-180" : ""}`}
                />
              </button>
              {isOpen && <p className="text-xs text-muted-foreground mt-2 pl-6">{claim.explanation}</p>}
            </div>
          );
        })}
      </div>
    </Card>
  );
}
