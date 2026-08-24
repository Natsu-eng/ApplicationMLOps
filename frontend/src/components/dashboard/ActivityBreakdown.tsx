import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip as RechartsTooltip } from "recharts";
import { CHART_HEIGHT_SM, CHART_TOOLTIP_STYLE } from "../../theme/charts";
import type { JobStatus } from "../../api/client";

/** Répartition visuelle du tableau de bord (Lot dashboard-dynamique) —
 * avant ce composant, `Dashboard.tsx` n'affichait que des compteurs
 * statiques (`StatTile`) et des listes, sans aucun graphique malgré une
 * bibliothèque de graphes déjà construite (`components/charts/`,
 * `Sparkline`/`Gauge` compris) et recharts déjà une dépendance du projet.
 *
 * Données HONNÊTES uniquement — jamais de tendance temporelle inventée :
 * `GET /dashboard/summary` ne renvoie pas de comptage jour par jour
 * (seulement les 6 activités les plus récentes par pilier + des totaux
 * globaux), donc ce composant visualise ce qui est RÉELLEMENT disponible
 * (répartition de statut sur l'activité récente réellement chargée,
 * répartition par pilier sur les totaux réels) plutôt que d'extrapoler
 * une tendance sur des données qui ne la contiennent pas — un graphique
 * qui suggère "évolution dans le temps" sans données temporelles réelles
 * induirait l'utilisateur en erreur. */

const STATUS_LABELS: Record<JobStatus, string> = {
  queued: "En file",
  running: "En cours",
  completed: "Terminé",
  failed: "Échec",
  cancelled: "Annulé",
};

// Alignées sur les couleurs sémantiques déjà utilisées par JobStatusBadge
// (Badge variant="success"/"danger"/"primary"/"neutral") — un même statut
// porte la même couleur partout dans l'app, jamais une seconde palette
// inventée pour ce seul graphique.
const STATUS_COLORS: Record<JobStatus, string> = {
  completed: "var(--success)",
  failed: "var(--destructive)",
  running: "var(--primary)",
  queued: "var(--muted-foreground)",
  cancelled: "var(--muted-foreground)",
};

export function StatusBreakdownChart({ statusCounts }: { statusCounts: Partial<Record<JobStatus, number>> }) {
  const entries = (Object.entries(statusCounts) as [JobStatus, number][]).filter(([, count]) => count > 0);
  const total = entries.reduce((sum, [, count]) => sum + count, 0);

  if (total === 0) {
    return <p className="text-sm text-muted-foreground py-6 text-center">Aucune activité récente à représenter.</p>;
  }

  return (
    <div className="flex items-center gap-4">
      <div style={{ width: CHART_HEIGHT_SM, height: CHART_HEIGHT_SM }} className="flex-shrink-0">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={entries.map(([status, count]) => ({ status, count }))}
              dataKey="count"
              nameKey="status"
              innerRadius="60%"
              outerRadius="90%"
              paddingAngle={2}
              isAnimationActive={false}
            >
              {entries.map(([status]) => (
                <Cell key={status} fill={STATUS_COLORS[status]} stroke="var(--card)" strokeWidth={2} />
              ))}
            </Pie>
            <RechartsTooltip
              {...CHART_TOOLTIP_STYLE}
              formatter={(value, _name, entry) => {
                const count = Number(value);
                const status = (entry.payload as { status: JobStatus }).status;
                return [`${count} (${Math.round((count / total) * 100)}%)`, STATUS_LABELS[status]];
              }}
            />
          </PieChart>
        </ResponsiveContainer>
      </div>
      <ul className="flex-1 min-w-0 space-y-1.5">
        {entries
          .sort(([, a], [, b]) => b - a)
          .map(([status, count]) => (
            <li key={status} className="flex items-center gap-2 text-xs">
              <span className="h-2.5 w-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: STATUS_COLORS[status] }} aria-hidden="true" />
              <span className="text-foreground/80 flex-1">{STATUS_LABELS[status]}</span>
              <span className="text-muted-foreground tabular-nums">{count}</span>
            </li>
          ))}
      </ul>
    </div>
  );
}

export function PillarDistributionBars({
  supervised,
  unsupervised,
  vision,
}: {
  supervised: number;
  unsupervised: number;
  vision: number;
}) {
  const total = supervised + unsupervised + vision;
  const rows: { label: string; value: number; color: string }[] = [
    { label: "Supervisé", value: supervised, color: "var(--s1)" },
    { label: "Non supervisé", value: unsupervised, color: "var(--s2)" },
    { label: "Vision", value: vision, color: "var(--s3)" },
  ];

  if (total === 0) {
    return <p className="text-sm text-muted-foreground py-6 text-center">Aucune analyse pour l'instant.</p>;
  }

  return (
    <ul className="space-y-3">
      {rows.map((row) => {
        const pct = total > 0 ? (row.value / total) * 100 : 0;
        return (
          <li key={row.label}>
            <div className="flex items-center justify-between text-xs mb-1">
              <span className="text-foreground/80">{row.label}</span>
              <span className="text-muted-foreground tabular-nums">{row.value}</span>
            </div>
            <div className="h-2 rounded-full bg-muted overflow-hidden">
              <div
                className="h-full rounded-full transition-[width] duration-500"
                style={{ width: `${pct}%`, backgroundColor: row.color }}
              />
            </div>
          </li>
        );
      })}
    </ul>
  );
}
