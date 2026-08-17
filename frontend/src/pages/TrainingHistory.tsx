import { useCallback, useEffect, useMemo, useState } from "react";
import { Award, BrainCircuit, GitCompareArrows, History } from "lucide-react";
import {
  ApiError,
  api,
  type JobComparisonResponse,
  type ModelRegistryEntry,
  type TrainingJobSummary,
} from "../api/client";
import AppShell from "../components/AppShell";
import { pillarColor } from "../config/pillars";
import { Badge } from "../components/ui/Badge";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { ColorIconBadge, accentColorForId } from "../components/ui/ColorIconBadge";
import { PageHeader } from "../components/ui/PageHeader";
import { SectionHeader } from "../components/ui/SectionHeader";
import { JobStatusBadge } from "../components/ui/StatusBadge";
import { formatDateTime, formatMetricLabel, formatMetricValue } from "../utils/format";

const CONFIG_ROWS: { key: string; label: string }[] = [
  { key: "test_size", label: "Part du jeu de test" },
  { key: "optuna_trials", label: "Essais Optuna" },
  { key: "cv_folds", label: "Blocs de validation croisée" },
  { key: "seed", label: "Graine aléatoire" },
  { key: "cqr_alpha", label: "Alpha CQR (régression)" },
  { key: "class_rebalancing", label: "Rééquilibrage des classes" },
  { key: "model_ids", label: "Modèles comparés (mode expert)" },
];

function formatConfigValue(key: string, value: unknown): string {
  if (value === null || value === undefined) return "Par défaut";
  if (key === "model_ids" && Array.isArray(value)) return value.length > 0 ? value.join(", ") : "Catalogue par défaut";
  if (key === "class_rebalancing") return value ? "Activé" : "Désactivé";
  if (typeof value === "number") return formatMetricValue(value);
  return String(value);
}

/** Historique complet des entraînements — sélection multiple + comparaison
 * inter-jobs (Lot D-bis). Complète le Lot D (leaderboard intra-job, déjà
 * sur la page Résultats) : ici on compare plusieurs ENTRAÎNEMENTS entre
 * eux (config, métriques), pas seulement les modèles d'un même job. Devient
 * aussi la page "Voir tout" du dashboard, qui pointait jusqu'ici vers le
 * formulaire d'entraînement (aucun historique n'y était consultable). */
export default function TrainingHistory() {
  const [jobs, setJobs] = useState<TrainingJobSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [comparison, setComparison] = useState<JobComparisonResponse | null>(null);
  const [comparing, setComparing] = useState(false);
  const [compareError, setCompareError] = useState<string | null>(null);
  const [registry, setRegistry] = useState<ModelRegistryEntry[] | null>(null);
  const [registryError, setRegistryError] = useState<string | null>(null);

  const load = useCallback(() => {
    api.training
      .listJobs()
      .then(setJobs)
      .catch((err) => setError(err instanceof ApiError ? err.message : "Impossible de charger l'historique"));
    api.training
      .registry()
      .then((r) => {
        setRegistry(r.entries);
        setRegistryError(null);
      })
      // AUDIT_ROADMAP.md, H4/D3 : un échec ici faisait disparaître le
      // panneau "Registre de modèles" sans aucun indice — indiscernable de
      // "rien n'a encore été promu".
      .catch((err) => setRegistryError(err instanceof ApiError ? err.message : "Registre indisponible"));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  function toggle(id: number) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
    setComparison(null);
  }

  async function handleCompare() {
    setComparing(true);
    setCompareError(null);
    try {
      setComparison(await api.training.compareJobs(Array.from(selected)));
    } catch (err) {
      setCompareError(err instanceof ApiError ? err.message : "Comparaison impossible");
    } finally {
      setComparing(false);
    }
  }

  const metricKeys = useMemo(() => {
    if (!comparison) return [];
    const keys = new Set<string>();
    comparison.entries.forEach((e) =>
      Object.entries(e.metrics).forEach(([k, v]) => {
        if (typeof v === "number") keys.add(k);
      }),
    );
    // cv_score/headline en tête, puis le reste dans un ordre stable.
    return Array.from(keys).sort((a, b) => (a === "cv_score" ? -1 : b === "cv_score" ? 1 : a.localeCompare(b)));
  }, [comparison]);

  return (
    <AppShell pillarId="supervised">
      <PageHeader
        eyebrow="Historique"
        title="Entraînements"
        description="Sélectionnez au moins deux entraînements pour les comparer côte à côte."
        icon={History}
        color={pillarColor("supervised")}
        action={
          <Button onClick={handleCompare} disabled={selected.size < 2 || comparing}>
            <GitCompareArrows size={15} />
            {comparing ? "Comparaison…" : `Comparer (${selected.size})`}
          </Button>
        }
      />

      {error && <p className="text-sm text-destructive mb-4">{error}</p>}
      {registryError && <p className="text-sm text-destructive mb-4">{registryError}</p>}

      {registry && registry.length > 0 && (
        <Card className="p-5 mb-6">
          <SectionHeader
            icon={Award}
            color="violet"
            label="Registre de modèles"
            help="Les modèles explicitement promus (validation ou production) depuis la page Résultats d'un entraînement — pas un doublon de l'historique complet ci-dessous, seulement ce qui a été retenu."
          />
          <ul className="divide-y divide-border">
            {registry.map((entry) => (
              <li key={entry.model_id} className="py-2.5 flex items-center gap-3">
                <ColorIconBadge icon={Award} color={accentColorForId(entry.model_id)} size="sm" />
                <div className="min-w-0 flex-1">
                  <p className="text-sm text-foreground truncate">
                    {entry.dataset_name ?? "Dataset"} <span className="text-muted-foreground">→</span> {entry.target_column}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {entry.algorithm}
                    {entry.promoted_at ? ` · promu le ${formatDateTime(entry.promoted_at)}` : ""}
                  </p>
                </div>
                <div className="flex items-center gap-2 flex-shrink-0">
                  {entry.headline_metric?.value !== null && entry.headline_metric?.value !== undefined && (
                    <span className="text-xs text-muted-foreground tabular-nums">
                      {entry.headline_metric.name} = {entry.headline_metric.value.toFixed(3)}
                    </span>
                  )}
                  <Badge variant={entry.stage === "production" ? "success" : "warning"} dot>
                    {entry.stage === "production" ? "Production" : "Staging"}
                  </Badge>
                </div>
              </li>
            ))}
          </ul>
        </Card>
      )}

      <Card className="p-5 mb-6">
        <SectionHeader icon={History} color="blue" label="Tous les entraînements" />
        {jobs === null ? (
          error ? null : <p className="text-sm text-muted-foreground">Chargement…</p>
        ) : jobs.length === 0 ? (
          <p className="text-sm text-muted-foreground">Aucun entraînement pour l'instant.</p>
        ) : (
          <ul className="divide-y divide-border">
            {jobs.map((job) => (
              <li key={job.id} className="py-2.5 flex items-center gap-3">
                <input
                  type="checkbox"
                  className="accent-primary flex-shrink-0"
                  checked={selected.has(job.id)}
                  onChange={() => toggle(job.id)}
                  aria-label={`Sélectionner l'entraînement ${job.id}`}
                />
                <ColorIconBadge icon={BrainCircuit} color={accentColorForId(job.id)} size="sm" />
                <div className="min-w-0 flex-1">
                  <p className="text-sm text-foreground truncate">
                    {job.dataset_name ?? "Dataset"} <span className="text-muted-foreground">→</span> {job.target_column}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {formatDateTime(job.created_at)}
                    {job.algorithm ? ` · ${job.algorithm}` : ""}
                  </p>
                </div>
                <div className="flex items-center gap-2 flex-shrink-0">
                  {job.headline_metric?.value !== null && job.headline_metric?.value !== undefined && (
                    <span className="text-xs text-muted-foreground tabular-nums">
                      {job.headline_metric.name} = {job.headline_metric.value.toFixed(3)}
                    </span>
                  )}
                  <JobStatusBadge status={job.status} />
                </div>
              </li>
            ))}
          </ul>
        )}
      </Card>

      {compareError && <p className="text-sm text-destructive mb-4">{compareError}</p>}

      {comparison && (
        <Card className="p-5 overflow-x-auto">
          <SectionHeader
            icon={GitCompareArrows}
            color="violet"
            label="Comparaison"
            help="Les lignes surlignées correspondent à des réglages qui diffèrent entre au moins deux des entraînements sélectionnés."
          />
          <table className="min-w-full text-sm border-separate border-spacing-0">
            <thead>
              <tr>
                <th className="text-left px-3 py-2 text-xs font-medium text-muted-foreground sticky left-0 bg-card">
                  &nbsp;
                </th>
                {comparison.entries.map((e) => (
                  <th key={e.job_id} className="text-left px-3 py-2 min-w-[160px]">
                    <p className="text-sm font-medium text-foreground truncate">{e.dataset_name ?? "Dataset"}</p>
                    <p className="text-xs text-muted-foreground truncate">→ {e.target_column}</p>
                    <div className="mt-1 flex items-center gap-1.5">
                      <JobStatusBadge status={e.status} />
                      {e.algorithm && <span className="text-xs text-muted-foreground">{e.algorithm}</span>}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              <tr className="bg-muted">
                <td className="px-3 py-2 text-xs font-semibold text-muted-foreground sticky left-0 bg-muted" colSpan={comparison.entries.length + 1}>
                  Métriques
                </td>
              </tr>
              {metricKeys.map((key) => (
                <tr key={key} className="border-t border-border/60">
                  <td className="px-3 py-2 text-xs text-muted-foreground sticky left-0 bg-card">{formatMetricLabel(key)}</td>
                  {comparison.entries.map((e) => {
                    const value = e.metrics[key];
                    return (
                      <td key={e.job_id} className="px-3 py-2 text-sm tabular-nums text-foreground">
                        {typeof value === "number" ? formatMetricValue(value) : "—"}
                      </td>
                    );
                  })}
                </tr>
              ))}

              <tr className="bg-muted">
                <td className="px-3 py-2 text-xs font-semibold text-muted-foreground sticky left-0 bg-muted" colSpan={comparison.entries.length + 1}>
                  Configuration
                </td>
              </tr>
              {CONFIG_ROWS.map((row) => {
                const differs = comparison.differing_config_fields.includes(row.key);
                return (
                  <tr key={row.key} className={`border-t border-border/60 ${differs ? "bg-warning/10" : ""}`}>
                    <td className={`px-3 py-2 text-xs text-muted-foreground sticky left-0 ${differs ? "bg-warning/15" : "bg-card"}`}>
                      {row.label}
                      {differs && <span className="ml-1 text-warning">●</span>}
                    </td>
                    {comparison.entries.map((e) => (
                      <td key={e.job_id} className="px-3 py-2 text-sm text-foreground">
                        {formatConfigValue(row.key, e.config[row.key])}
                      </td>
                    ))}
                  </tr>
                );
              })}
              <tr className="border-t border-border/60">
                <td className="px-3 py-2 text-xs text-muted-foreground sticky left-0 bg-card">Ingénierie de variables</td>
                {comparison.entries.map((e) => (
                  <td key={e.job_id} className="px-3 py-2 text-sm text-foreground">
                    {e.feature_engineering_active ? "Activée" : "Non appliquée"}
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </Card>
      )}
    </AppShell>
  );
}
