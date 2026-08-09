import { useEffect, useState } from "react";
import { ShieldCheck, Sparkles } from "lucide-react";
import { ApiError, api, type BootstrapCI, type MLModelDetail, type TrainingJobSummary } from "../../api/client";
import { Badge } from "../ui/Badge";
import { Modal } from "../ui/Modal";
import { LabelWithHelp } from "../ui/Tooltip";
import { formatMetricValue, formatPercent } from "../../utils/format";
import EvaluationCharts from "./EvaluationCharts";
import PredictionForm from "./PredictionForm";

function isBootstrapCI(value: unknown): value is BootstrapCI {
  return typeof value === "object" && value !== null && "ci_low" in value && "ci_high" in value;
}

/** Statut d'explicabilité (Lot 5) — "ok" ou "degraded" + message clair.
 * Doit être AFFICHÉ quand dégradé, jamais une section qui disparaît sans
 * explication (ex. SVM/KNN sur un jeu de données avec beaucoup de variables). */
interface ExplainabilityStatus {
  status: "ok" | "degraded";
  message: string | null;
}

function isExplainabilityStatus(value: unknown): value is ExplainabilityStatus {
  return typeof value === "object" && value !== null && "status" in value;
}

/** Cartes de métriques principales — l'ensemble affiché dépend du type de tâche. */
function MetricCard({
  label,
  help,
  value,
  ci,
}: {
  label: string;
  help?: string;
  value: number | null | undefined;
  ci?: BootstrapCI;
}) {
  if (value === null || value === undefined) return null;
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950/40 px-4 py-3">
      <p className="text-xs text-slate-500 mb-1">
        {help ? <LabelWithHelp label={label} help={help} /> : label}
      </p>
      <p className="text-xl font-semibold text-slate-100 tabular-nums">{formatMetricValue(value)}</p>
      {ci && (
        <p className="text-[11px] text-slate-600 mt-0.5 tabular-nums">
          IC 95 % [{formatMetricValue(ci.ci_low)} – {formatMetricValue(ci.ci_high)}]
        </p>
      )}
    </div>
  );
}

/** Résumé lisible d'une transformation approuvée (Lot 4c) — transparence sur
 * le résultat, pas une re-configuration : l'utilisateur ne peut plus la
 * modifier depuis cet écran. */
function describeTransformation(t: Record<string, unknown>): string {
  if (t.type === "datetime_decompose") return `Décomposition de date : ${t.source_column}`;
  if (t.type === "ratio") return `Ratio : ${t.numerator} / ${t.denominator}`;
  return JSON.stringify(t);
}

function FeatureEngineeringSummary({ spec }: { spec: MLModelDetail["feature_engineering"] }) {
  if (!spec) return null;
  const items = [
    ...spec.upstream.map(describeTransformation),
    ...((spec.pipeline.frequency_encoding as string[] | undefined) ?? []).map(
      (col) => `Regroupement des rares + fréquence : ${col}`,
    ),
    ...Object.entries((spec.pipeline.imputation as Record<string, { strategy: string }> | undefined) ?? {}).map(
      ([col, cfg]) => `Imputation (${cfg.strategy}) : ${col}`,
    ),
  ];
  if (items.length === 0) return null;
  return (
    <section>
      <p className="text-xs uppercase tracking-wide text-slate-500 mb-2">Ingénierie de variables appliquée</p>
      <ul className="text-xs text-slate-400 space-y-1 list-disc list-inside">
        {items.map((item, i) => (
          <li key={i}>{item}</li>
        ))}
      </ul>
    </section>
  );
}

function ShapBars({ features }: { features: MLModelDetail["shap_summary"] }) {
  if (features.length === 0) return null;
  const max = Math.max(...features.map((f) => f.importance));
  return (
    <div className="space-y-2">
      {features.slice(0, 8).map((f) => (
        <div key={f.feature} className="flex items-center gap-3">
          <span className="text-xs text-slate-400 w-40 truncate flex-shrink-0" title={f.feature}>
            {f.feature}
          </span>
          <div className="flex-1 h-2 rounded-full bg-slate-800 overflow-hidden">
            <div
              className="h-full rounded-full bg-gradient-to-r from-teal-500/70 to-teal-400"
              style={{ width: `${(f.importance / max) * 100}%` }}
            />
          </div>
          <span className="text-xs text-slate-500 w-12 text-right tabular-nums">
            {f.importance.toFixed(2)}
          </span>
        </div>
      ))}
    </div>
  );
}

export default function ModelResultModal({
  job,
  onClose,
}: {
  job: TrainingJobSummary;
  onClose: () => void;
}) {
  const [model, setModel] = useState<MLModelDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.training
      .getModel(job.id)
      .then(setModel)
      .catch((err) => setError(err instanceof ApiError ? err.message : "Résultat indisponible"));
  }, [job.id]);

  const explainability =
    model && isExplainabilityStatus(model.model_card.explainability) ? model.model_card.explainability : undefined;

  return (
    <Modal title={`${job.dataset_name ?? "Dataset"} — ${job.target_column}`} onClose={onClose}>
      {error && <p className="text-sm text-rose-400">{error}</p>}
      {!model && !error && <p className="text-sm text-slate-500">Chargement…</p>}

      {model && (
        <div className="space-y-6">
          <div className="flex items-center gap-2 flex-wrap">
            <Badge variant="accent">{model.algorithm}</Badge>
            <Badge variant="neutral">
              {model.task_type === "regression" ? "Régression" : "Classification"}
            </Badge>
            {Boolean(model.model_card.anti_leak_grouping) && (
              <Badge variant="success">
                <ShieldCheck size={11} className="mr-1 inline" />
                Split anti-fuite
              </Badge>
            )}
          </div>

          <section>
            <p className="text-xs uppercase tracking-wide text-slate-500 mb-2">Performance</p>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {model.task_type === "regression" ? (
                <>
                  <MetricCard
                    label="R² (test)"
                    help="Part de la variation de la cible expliquée par le modèle, de 0 à 1. 0,90 = le modèle explique 90 % des écarts observés — plus c'est proche de 1, mieux c'est."
                    value={model.metrics.r2_test as number}
                    ci={isBootstrapCI(model.metrics.r2_bootstrap) ? model.metrics.r2_bootstrap : undefined}
                  />
                  <MetricCard
                    label="RMSE"
                    help="Erreur moyenne de prédiction, dans l'unité de la cible. Plus c'est bas, mieux c'est — à comparer à l'échelle typique de vos valeurs."
                    value={model.metrics.rmse as number}
                    ci={isBootstrapCI(model.metrics.rmse_bootstrap) ? model.metrics.rmse_bootstrap : undefined}
                  />
                  <MetricCard label="MAE" help="Erreur absolue moyenne — comme le RMSE mais moins sensible aux grosses erreurs isolées." value={model.metrics.mae as number} />
                  <MetricCard label="Score CV" help="Performance moyenne sur plusieurs découpages des données d'entraînement — plus fiable qu'un seul test, c'est ce score qui a servi à choisir ce modèle." value={model.metrics.cv_score as number} />
                </>
              ) : (
                <>
                  <MetricCard
                    label="Précision globale"
                    help="Pourcentage de prédictions correctes sur le jeu de test."
                    value={model.metrics.accuracy as number}
                    ci={isBootstrapCI(model.metrics.accuracy_bootstrap) ? model.metrics.accuracy_bootstrap : undefined}
                  />
                  <MetricCard label="F1-score" help="Équilibre entre précision et rappel — utile quand les classes sont déséquilibrées, où la précision seule peut être trompeuse." value={model.metrics.f1 as number} />
                  <MetricCard label="AUC-ROC" help="Capacité du modèle à distinguer les classes, de 0,5 (hasard) à 1 (parfait)." value={model.metrics.roc_auc as number} />
                  <MetricCard label="Score CV" help="Performance moyenne sur plusieurs découpages des données d'entraînement — plus fiable qu'un seul test, c'est ce score qui a servi à choisir ce modèle." value={model.metrics.cv_score as number} />
                </>
              )}
            </div>
            {model.task_type === "regression" && typeof model.metrics.delta_r2 === "number" && (
              <p className="text-xs text-slate-500 mt-2">
                Écart train/test (R²) : <span className="tabular-nums">{formatMetricValue(model.metrics.delta_r2)}</span>
                {" — "}
                {model.metrics.delta_r2 < 0.08 ? "pas de surapprentissage notable" : "surapprentissage à surveiller"}
              </p>
            )}
          </section>

          <EvaluationCharts taskType={model.task_type} evaluation={model.evaluation} />

          <section>
            <p className="text-xs uppercase tracking-wide text-slate-500 mb-2 flex items-center gap-1.5">
              <Sparkles size={12} className="text-teal-400" />
              <LabelWithHelp
                label="Variables les plus influentes"
                help="Plus une variable a une barre longue, plus elle pèse dans les décisions du modèle — calculé par la méthode SHAP, standard en explicabilité de modèles ML."
              />
            </p>
            {explainability?.status === "degraded" ? (
              <p className="text-xs text-slate-500 italic">{explainability.message}</p>
            ) : (
              <ShapBars features={model.shap_summary} />
            )}
          </section>

          {model.cqr && (
            <section>
              <p className="text-xs uppercase tracking-wide text-slate-500 mb-2">
                <LabelWithHelp
                  label="Fiabilité des prédictions"
                  help="Plutôt qu'une seule valeur, le modèle peut donner une fourchette dans laquelle la vraie valeur tombe la plupart du temps — utile pour savoir jusqu'où faire confiance à une prédiction."
                />
              </p>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                <MetricCard label="Couverture visée" value={model.cqr.target_coverage} />
                <MetricCard label="Couverture observée" value={model.cqr.empirical_coverage} />
                <MetricCard label="Largeur moyenne" value={model.cqr.mean_interval_width} />
              </div>
              <p className="text-xs text-slate-500 mt-2">
                {formatPercent(model.cqr.empirical_coverage)} des valeurs test tombent dans l'intervalle prédit,
                pour une cible de {formatPercent(model.cqr.target_coverage)} — calibré par {model.cqr.n_strata} strates.
              </p>
            </section>
          )}

          <FeatureEngineeringSummary spec={model.feature_engineering} />

          <PredictionForm jobId={job.id} taskType={model.task_type} featureSchema={model.feature_schema} />

          <section>
            <p className="text-xs uppercase tracking-wide text-slate-500 mb-2">Fiche modèle</p>
            <dl className="grid grid-cols-2 sm:grid-cols-4 gap-x-4 gap-y-2 text-xs">
              <Fact label="Échantillons train" value={String(model.model_card.n_train ?? "—")} />
              <Fact label="Échantillons test" value={String(model.model_card.n_test ?? "—")} />
              <Fact label="Doublons retirés" value={String(model.model_card.duplicates_removed ?? "—")} />
              <Fact label="Essais Optuna" value={String(model.model_card.optuna_trials ?? "—")} />
              <Fact label="Folds de CV" value={String(model.model_card.cv_folds ?? "—")} />
              <Fact label="Variables" value={String(model.feature_columns.length)} />
            </dl>
          </section>
        </div>
      )}
    </Modal>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-slate-600">{label}</dt>
      <dd className="text-slate-300 tabular-nums">{value}</dd>
    </div>
  );
}
