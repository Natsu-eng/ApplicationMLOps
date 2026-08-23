import { useState } from "react";
import { AlertTriangle, ChevronDown, Info, Lightbulb, ShieldAlert } from "lucide-react";
import type { ModelVerdictData, VerdictLevel } from "../../api/client";
import { Badge } from "../ui/Badge";
import { Card } from "../ui/Card";
import { SectionHeader } from "../ui/SectionHeader";
import { formatMetricValue, formatPercent } from "../../utils/format";

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
  attention: { badge: "warning", icon: AlertTriangle, border: "border-warning/20", iconColor: "text-warning" },
  info: { badge: "accent", icon: Info, border: "border-primary/20", iconColor: "text-primary" },
};

const LEVEL_LABEL: Record<VerdictLevel, string> = {
  critique: "Critique",
  attention: "Attention",
  info: "Info",
};

// Libellés courts des clés de `claim.details` (services/verdict.py) — le
// backend envoie déjà les nombres qui fondent chaque affirmation, jusqu'ici
// ignorés côté frontend. Purement un dictionnaire d'affichage : aucune
// valeur n'est recalculée ici, seulement mise en forme (Lot 6, Verdict.html
// — ligne de preuve monospace sous chaque affirmation).
const DETAIL_LABELS: Record<string, string> = {
  delta: "écart",
  train: "train",
  test: "test",
  ci_low: "IC bas",
  ci_high: "IC haut",
  width: "largeur",
  majority_class: "classe majoritaire",
  majority_fraction: "part",
  winner: "gagnant",
  runner_up: "2e",
  gap: "écart",
  fold_std: "écart-type plis",
  mean_deviation: "écart moyen",
  last_gain: "dernier gain",
  total_range: "amplitude",
  target_coverage: "couverture visée",
  empirical_coverage: "couverture observée",
};

const PERCENT_DETAIL_KEYS = new Set(["majority_fraction", "target_coverage", "empirical_coverage"]);

function formatDetailValue(key: string, value: unknown): string {
  if (typeof value === "number") return PERCENT_DETAIL_KEYS.has(key) ? formatPercent(value) : formatMetricValue(value);
  return String(value);
}

function EvidenceLine({ details }: { details: Record<string, unknown> }) {
  const entries = Object.entries(details);
  if (entries.length === 0) return null;
  return (
    <p className="num text-caption text-muted-foreground mt-1.5 pl-6">
      {entries.map(([key, value], i) => (
        <span key={key}>
          {i > 0 && " · "}
          {DETAIL_LABELS[key] ?? key} {formatDetailValue(key, value)}
        </span>
      ))}
    </p>
  );
}

/** Une affirmation du Verdict — repliée par défaut (retour utilisateur
 * direct, Lot 10 : la section devenait longue une fois la ligne de preuve
 * ajoutée au Lot 6) : seuls le niveau et le titre restent toujours
 * visibles, l'explication et la ligne de preuve se déplient au clic — même
 * motif que la FAQ/le lexique du centre d'aide. */
function VerdictClaimItem({ claim }: { claim: ModelVerdictData["claims"][number] }) {
  const [open, setOpen] = useState(false);
  const config = LEVEL_CONFIG[claim.level];
  const Icon = config.icon;
  return (
    <div className={`rounded-lg border ${config.border} bg-muted px-3 py-2.5`}>
      <button type="button" onClick={() => setOpen((v) => !v)} aria-expanded={open} className="w-full flex items-start gap-2 text-left">
        <Icon size={15} className={`flex-shrink-0 mt-0.5 ${config.iconColor}`} />
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 flex-wrap">
            <Badge variant={config.badge}>{LEVEL_LABEL[claim.level]}</Badge>
            <p className="text-sm text-foreground font-medium">{claim.title}</p>
          </div>
        </div>
        <ChevronDown
          size={14}
          className={`flex-shrink-0 text-muted-foreground transition-transform mt-1 ${open ? "rotate-180" : ""}`}
        />
      </button>
      {open && (
        <>
          <p className="text-xs text-muted-foreground mt-2 pl-6">{claim.explanation}</p>
          <EvidenceLine details={claim.details} />
        </>
      )}
    </div>
  );
}

export function ModelVerdict({ verdict }: { verdict: ModelVerdictData }) {
  return (
    <Card className="p-5">
      <SectionHeader icon={Lightbulb} color="amber" label="Verdict" />

      <div className="flex items-start gap-2.5 rounded-lg border border-primary/20 bg-primary/5 px-3 py-2.5 mb-3">
        <Lightbulb size={15} className="flex-shrink-0 mt-0.5 text-primary" />
        <p className="text-sm text-foreground">{verdict.next_action}</p>
      </div>

      <div className="space-y-2">
        {verdict.claims.map((claim) => (
          <VerdictClaimItem key={claim.code} claim={claim} />
        ))}
      </div>
    </Card>
  );
}
