import { type ReactNode, useEffect, useState } from "react";
import { Activity, AlertTriangle, Award, Calculator, ClipboardList, Download, FileCode, FileJson, Gauge, History, Scale, ShieldCheck, Sparkles, Trophy, UploadCloud, Wand2, Waves } from "lucide-react";
import {
  ApiError,
  api,
  type BootstrapCI,
  type LeaderboardResponse,
  type MLModelDetail,
  type ModelCandidate,
  type ModelStage,
  type ModelVersionEntry,
  type TrainingJobSummary,
} from "../../api/client";
import { Badge } from "../ui/Badge";
import { Button } from "../ui/Button";
import { Card } from "../ui/Card";
import { accentSurfaceClass, accentValueTextClass, type AccentColor } from "../ui/ColorIconBadge";
import { Modal } from "../ui/Modal";
import { LabelWithHelp } from "../ui/Tooltip";
import { SectionHeader } from "../ui/SectionHeader";
import { Tabs } from "../ui/Tabs";
import { formatDateTime, formatMetricValue, formatPercent } from "../../utils/format";
import { clampUnitScore } from "../../utils/cvScore";
import { buildModelCard } from "../../utils/modelCard";
import EvaluationCharts from "./EvaluationCharts";
import { ShapBeeswarmChart, PermutationImportanceChart } from "./GlobalExplainability";
import { DriftPanel } from "../shared/DriftPanel";
import { ModelVerdict } from "./ModelVerdict";
import { CalibrationChart, LearningCurveChart } from "./ReliabilityDiagnostics";
import PredictionForm from "./PredictionForm";
import BatchPredictionForm from "./BatchPredictionForm";

const RESULT_TABS = [
  { id: "performance", label: "Performance", icon: Gauge },
  { id: "comparaison", label: "Comparaison", icon: Scale },
  { id: "explicabilite", label: "Explicabilité", icon: Sparkles },
  { id: "fiabilite", label: "Fiabilité", icon: Activity },
  { id: "derive", label: "Dérive", icon: Waves },
  { id: "predire", label: "Prédire", icon: Calculator },
  { id: "predire-lot", label: "Prédiction en lot", icon: UploadCloud },
  { id: "details", label: "Détails", icon: ClipboardList },
] as const;
type ResultTabId = (typeof RESULT_TABS)[number]["id"];

function isBootstrapCI(value: unknown): value is BootstrapCI {
  return typeof value === "object" && value !== null && "ci_low" in value && "ci_high" in value;
}

/** Statut d'un diagnostic (Lot 5 : explicabilité ; Lot Explicabilité globale :
 * beeswarm/permutation/calibration/learning curve) — "ok" ou "degraded" +
 * message clair. Doit être AFFICHÉ quand dégradé, jamais une section qui
 * disparaît sans explication (ex. SVM/KNN sur un jeu de données avec
 * beaucoup de variables). */
interface DiagnosticStatus {
  status: "ok" | "degraded";
  message: string | null;
}

function isDiagnosticStatus(value: unknown): value is DiagnosticStatus {
  return typeof value === "object" && value !== null && "status" in value;
}

/** Bloc de dégradation partagé par les 4 diagnostics du Lot Explicabilité
 * globale — 3 cas : statut absent (job antérieur au lot, ou intermédiaire
 * pour le beeswarm qui partage le statut SHAP du Lot 5 sans données propres
 * → `isEmpty`) : "réentraînez pour l'obtenir" ; statut "degraded" : message
 * backend ; sinon : le graphe. Jamais de section vide muette. */
function DiagnosticBlock({ status, isEmpty, children }: { status: unknown; isEmpty: boolean; children: ReactNode }) {
  if (status === undefined || (isDiagnosticStatus(status) && status.status === "ok" && isEmpty)) {
    return (
      <p className="text-xs text-muted-foreground italic">Non disponible pour ce modèle — réentraînez-le pour l'obtenir.</p>
    );
  }
  if (isDiagnosticStatus(status) && status.status === "degraded") {
    return <p className="text-xs text-muted-foreground italic">{status.message}</p>;
  }
  return <>{children}</>;
}

/** Cartes de métriques principales — l'ensemble affiché dépend du type de
 * tâche. Une teinte par métrique (comme les tuiles du dashboard) plutôt
 * qu'un fond gris uniforme — retour explicite : "des cartes colorées" pour
 * la page Résultats. La couleur est purement décorative/identitaire (quelle
 * métrique, pas si elle est bonne ou mauvaise) : jamais rouge/vert, qui
 * lirait à tort comme un jugement de qualité sur la valeur. */
function MetricCard({
  label,
  help,
  value,
  ci,
  color = "blue",
}: {
  label: string;
  help?: string;
  value: number | null | undefined;
  ci?: BootstrapCI;
  color?: AccentColor;
}) {
  if (value === null || value === undefined) return null;
  return (
    <div className={`rounded-xl border px-4 py-3 ${accentSurfaceClass(color)}`}>
      <p className="text-xs text-muted-foreground mb-1">
        {help ? <LabelWithHelp label={label} help={help} /> : label}
      </p>
      <p className={`text-xl font-bold tabular-nums ${accentValueTextClass(color)}`}>{formatMetricValue(value)}</p>
      {ci && (
        <p className="text-caption text-muted-foreground mt-0.5 tabular-nums">
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
    <Card className={`p-5 ${accentSurfaceClass("teal")}`}>
      <SectionHeader icon={Wand2} color="teal" label="Ingénierie de variables appliquée" />
      <ul className="text-xs text-muted-foreground space-y-1 list-disc list-inside">
        {items.map((item, i) => (
          <li key={i}>{item}</li>
        ))}
      </ul>
    </Card>
  );
}

function ShapBars({ features }: { features: MLModelDetail["shap_summary"] }) {
  if (features.length === 0) return null;
  const max = Math.max(...features.map((f) => f.importance));
  return (
    <div className="space-y-2">
      {features.slice(0, 8).map((f) => (
        <div key={f.feature} className="flex items-center gap-3">
          <span className="text-xs text-muted-foreground w-40 truncate flex-shrink-0" title={f.feature}>
            {f.feature}
          </span>
          <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
            <div
              className="h-full rounded-full bg-gradient-to-r from-primary/80 to-primary"
              style={{ width: `${(f.importance / max) * 100}%` }}
            />
          </div>
          <span className="text-xs text-muted-foreground w-12 text-right tabular-nums">
            {f.importance.toFixed(2)}
          </span>
        </div>
      ))}
    </div>
  );
}

/** Bloc "Interprétation du modèle" (Lot E1-ter) — langage clair, thèse
 * transparence : pourquoi ce modèle a été retenu (écart au 2e sur la
 * métrique de sélection) et ce que dit l'analyse SHAP, sans jargon.
 * Réutilise uniquement des données déjà calculées ailleurs (leaderboard,
 * shap_summary) — aucun nouveau calcul côté modèle. */
function ModelInterpretation({
  model,
  explainabilityDegraded,
}: {
  model: MLModelDetail;
  explainabilityDegraded: boolean;
}) {
  // La comparaison au 2ᵉ candidat (pourquoi CE modèle a été retenu) vivait
  // ici sous forme d'un texte ad hoc (seuil arbitraire 0,01, sans rapport à
  // la variance inter-plis) — supprimée (Lot 3, correctif I1) : ModelVerdict,
  // en tête de page, répond maintenant à la même question avec la vraie
  // significativité statistique (écart vs écart-type des `fold_scores`).
  // Ne reste ici que ce que ModelVerdict ne couvre pas : l'importance des
  // variables.
  const topFeatures = model.shap_summary.slice(0, 3).map((f) => f.feature);
  if (explainabilityDegraded || topFeatures.length === 0) return null;
  const featuresText =
    topFeatures.length === 1
      ? `La variable qui pèse le plus dans ses décisions est ${topFeatures[0]}.`
      : `Les variables qui pèsent le plus dans ses décisions sont, par ordre d'importance : ${topFeatures.join(", ")}.`;

  return (
    <section className="rounded-xl border border-primary/20 bg-primary/5 p-4">
      <p className="text-xs uppercase tracking-wide text-primary mb-2 flex items-center gap-1.5">
        <Sparkles size={12} />
        Interprétation du modèle
      </p>
      <p className="text-sm text-foreground/90 leading-relaxed">
        {featuresText}{" "}
        <span className="text-muted-foreground">
          Une variable en tête de liste influence davantage la prédiction que les autres — cela ne prouve pas un
          lien de cause à effet, seulement que le modèle s'appuie fortement dessus.
        </span>
      </p>
    </section>
  );
}

/** Leaderboard (Lot D) — TOUS les modèles comparés par ce job, pas
 * seulement le gagnant déjà mis en avant plus haut. Rétrocompatible par
 * absence : un job antérieur à ce lot n'a aucun candidat persisté, la
 * section ne s'affiche alors simplement pas (le gagnant reste visible via
 * les sections existantes). Reçoit `data` en props (Lot E1-ter) — le fetch
 * est fait une fois par `ModelResultView`, partagé avec `ModelInterpretation`. */
function Leaderboard({ data }: { data: LeaderboardResponse | null }) {
  if (!data || data.candidates.length === 0) return null;

  const winner = data.candidates.find((c) => c.is_winner);
  const runnerUp = data.candidates.find((c) => !c.is_winner);
  const metricShortName = data.selection_metric_label.split(" (")[0];

  return (
    <Card className={`p-5 h-full ${accentSurfaceClass("amber")}`}>
      <SectionHeader
        icon={Trophy}
        color="amber"
        label="Modèles évalués"
        help={`Classement sur ${data.selection_metric_label} — la métrique qui a réellement départagé les candidats pendant l'entraînement, jamais une exactitude brute qui peut être trompeuse sur un dataset déséquilibré.`}
      />

      {winner && runnerUp && (
        <p className="text-xs text-muted-foreground mb-3">
          <span className="text-foreground font-medium">{winner.algorithm}</span> retenu : meilleur {metricShortName}{" "}
          en validation croisée, devant {runnerUp.algorithm} de{" "}
          <span className="tabular-nums text-foreground/90">
            {(winner.selection_score - runnerUp.selection_score).toFixed(3)}
          </span>{" "}
          points.
        </p>
      )}

      <div className="space-y-1.5">
        {data.candidates.map((c) => {
          // Podium : le gagnant en dégradé de marque, #2/#3 en liseré
          // or/argent discret — au-delà, une carte neutre. Purement une
          // hiérarchie de classement (jamais "bon/mauvais modèle" au sens
          // qualité, tous les candidats sont des choix valides).
          const podiumBorder =
            c.rank === 1 ? "border-primary/30 bg-primary/10" : c.rank === 2 ? "border-input bg-muted" : c.rank === 3 ? "border-warning/30 bg-warning/10" : "border-border bg-muted";
          return (
            <div
              key={c.algorithm}
              className={`flex items-center justify-between gap-3 rounded-lg border px-3 py-2 ${podiumBorder}`}
            >
              <div className="flex items-center gap-2 min-w-0">
                <Badge variant={c.is_winner ? "accent" : "neutral"}>#{c.rank}</Badge>
                <span className={`text-sm truncate ${c.is_winner ? "text-foreground font-medium" : "text-foreground"}`}>
                  {c.algorithm}
                </span>
                {c.is_winner && <Trophy size={13} className="text-warning flex-shrink-0" />}
              </div>
              <div className="flex items-center gap-3 flex-shrink-0 text-xs">
                {c.secondary_metric !== null && (
                  <span className="text-muted-foreground">
                    {c.secondary_metric_label} :{" "}
                    <span className="tabular-nums text-foreground/90">{c.secondary_metric.toFixed(2)}</span>
                  </span>
                )}
                <span className={`tabular-nums font-medium ${c.is_winner ? "text-primary" : "text-foreground"}`}>
                  {clampUnitScore(c.selection_score).toFixed(3)}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </Card>
  );
}

/** Trait de stabilité entre plis (Lot 6, Leaderboard.html) — `fold_scores`
 * existe déjà par candidat (`ModelCandidate.fold_scores_json`, calculé
 * pendant la validation croisée) mais n'était affiché nulle part avant ce
 * lot. Un simple polyline SVG, mis à l'échelle sur le min/max RÉEL des plis
 * de CE candidat — pas un graphique Recharts, juste assez pour montrer si
 * le score varie beaucoup ou peu d'un pli à l'autre. */
function FoldStabilitySparkline({ scores }: { scores: number[] }) {
  if (scores.length < 2) return null;
  const min = Math.min(...scores);
  const max = Math.max(...scores);
  const range = max - min || 1;
  const w = 110;
  const h = 24;
  const points = scores
    .map((s, i) => {
      const x = (i / (scores.length - 1)) * w;
      const y = h - ((s - min) / range) * h;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} className="text-primary/70 flex-shrink-0" aria-hidden="true">
      <polyline points={points} fill="none" stroke="currentColor" strokeWidth={1.5} />
    </svg>
  );
}

function whyThisRank(candidate: ModelCandidate, winner: ModelCandidate | undefined, metricLabel: string): string {
  if (candidate.is_winner) return `Meilleur ${metricLabel} en validation croisée.`;
  if (!winner) return "";
  const gap = winner.selection_score - candidate.selection_score;
  return `${gap.toFixed(3)} point${gap >= 0.01 ? "s" : ""} derrière ${winner.algorithm}.`;
}

/** Onglet "Comparaison" (Lot 6, Leaderboard.html) — vue dédiée pleine page
 * des candidats d'un job, complémentaire à la carte compacte `Leaderboard`
 * déjà affichée dans l'onglet Performance (gardée telle quelle, elle sert de
 * résumé rapide). Construite UNIQUEMENT à partir de données déjà persistées
 * par `ModelCandidate` (selection_score, secondary_metric, fold_scores,
 * rank) — jamais de durée d'entraînement ni de temps d'inférence par
 * candidat : cette donnée n'existe nulle part côté backend (seule la durée
 * TOTALE du job est connue, jamais répartie par modèle). L'ajouter
 * demanderait d'instrumenter le moteur d'entraînement ET une migration de
 * schéma (nouvelle colonne sur `ModelCandidate`) — un vrai chantier
 * fonctionnel, hors périmètre d'une refonte visuelle. Les colonnes
 * "Durée"/"Prédiction" et le graphe "coût de la précision" de la maquette
 * sont donc omis plutôt qu'inventés (voir JOURNAL.md, Lot 6). */
function ModelComparison({ data, verdict }: { data: LeaderboardResponse | null; verdict: MLModelDetail["verdict"] }) {
  if (!data || data.candidates.length === 0) {
    return <p className="text-sm text-muted-foreground">Aucun candidat détaillé disponible pour cet entraînement.</p>;
  }
  const winner = data.candidates.find((c) => c.is_winner);
  const metricShortName = data.selection_metric_label.split(" (")[0];
  const maxScore = Math.max(...data.candidates.map((c) => clampUnitScore(c.selection_score))) || 1;
  // Réutilise le même verdict que la carte "Verdict" (services/verdict.py,
  // `_assess_winner_margin`) — jamais une seconde comparaison recalculée ici,
  // pour ne jamais risquer de dire deux choses différentes sur le même écart.
  const marginClaim = verdict.claims.find((c) => c.code.startsWith("ecart_gagnant"));

  return (
    <div className="space-y-5">
      {winner && (
        <Card className={`p-5 ${accentSurfaceClass("amber")}`}>
          <div className="flex items-center gap-3 flex-wrap mb-3">
            <Badge variant="accent">#1</Badge>
            <h3 className="text-h3 text-foreground">{winner.algorithm}</h3>
            <Badge variant="success">Retenu</Badge>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div>
              <p className="text-xs text-muted-foreground mb-1">{metricShortName} (validation)</p>
              <p className="num text-xl font-semibold text-foreground">
                {clampUnitScore(winner.selection_score).toFixed(3)}
              </p>
            </div>
            {winner.secondary_metric !== null && (
              <div>
                <p className="text-xs text-muted-foreground mb-1">{winner.secondary_metric_label}</p>
                <p className="num text-xl font-semibold text-foreground">{winner.secondary_metric.toFixed(2)}</p>
              </div>
            )}
            {winner.fold_scores && winner.fold_scores.length > 1 && (
              <div>
                <p className="text-xs text-muted-foreground mb-1">Stabilité entre plis</p>
                <FoldStabilitySparkline scores={winner.fold_scores} />
              </div>
            )}
          </div>
        </Card>
      )}

      <Card className="p-5 overflow-x-auto">
        <SectionHeader
          icon={Scale}
          color="blue"
          label="Tous les modèles comparés"
          help={`Triés par ${data.selection_metric_label} — la métrique qui a réellement départagé les candidats pendant l'entraînement.`}
        />
        <table className="w-full text-sm mt-2">
          <thead>
            <tr className="text-xs text-muted-foreground uppercase tracking-wide text-left">
              <th className="pb-2 pr-3 font-medium">Rang</th>
              <th className="pb-2 pr-3 font-medium">Modèle</th>
              <th className="pb-2 pr-3 font-medium">{metricShortName} (validation)</th>
              <th className="pb-2 pr-3 font-medium">Stabilité entre plis</th>
              <th className="pb-2 font-medium">Pourquoi ce rang</th>
            </tr>
          </thead>
          <tbody>
            {data.candidates.map((c) => (
              <tr key={c.algorithm} className="border-t border-border/60">
                <td className="py-2.5 pr-3">
                  <Badge variant={c.is_winner ? "accent" : "neutral"}>#{c.rank}</Badge>
                </td>
                <td className="py-2.5 pr-3 font-medium text-foreground whitespace-nowrap">{c.algorithm}</td>
                <td className="py-2.5 pr-3">
                  <div className="flex items-center gap-2">
                    <div className="h-1.5 w-24 rounded-full bg-muted overflow-hidden flex-shrink-0">
                      <div
                        className={`h-full rounded-full ${c.is_winner ? "bg-primary" : "bg-primary/40"}`}
                        style={{ width: `${(clampUnitScore(c.selection_score) / maxScore) * 100}%` }}
                      />
                    </div>
                    <span className="num text-xs text-foreground">{clampUnitScore(c.selection_score).toFixed(3)}</span>
                  </div>
                </td>
                <td className="py-2.5 pr-3">
                  {c.fold_scores && c.fold_scores.length > 1 ? (
                    <FoldStabilitySparkline scores={c.fold_scores} />
                  ) : (
                    <span className="text-xs text-muted-foreground">—</span>
                  )}
                </td>
                <td className="py-2.5 text-xs text-muted-foreground max-w-[240px]">
                  {whyThisRank(c, winner, metricShortName)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {marginClaim && (
        <Card className="p-5 border-warning/20 bg-warning/5">
          <SectionHeader icon={AlertTriangle} color="amber" label="Ce que le classement ne dit pas" />
          <p className="text-sm text-foreground/90">{marginClaim.explanation}</p>
        </Card>
      )}
    </div>
  );
}

/** Registre de modèles (Lot 9) — promotion (staging/production) et export
 * de l'artefact. Un seul modèle "production" à la fois par dataset+cible,
 * la démotion de l'ancien est gérée côté serveur (voir `promote_model`,
 * `api/routers/training.py`) : ce composant se contente d'appeler l'API et
 * de refléter le nouvel état renvoyé, jamais de logique de démotion ici. */
function ModelRegistryControls({
  model,
  job,
  onUpdate,
}: {
  model: MLModelDetail;
  job: TrainingJobSummary;
  onUpdate: (model: MLModelDetail) => void;
}) {
  const jobId = job.id;
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function setStage(stage: Exclude<ModelStage, null> | "none") {
    setBusy(true);
    setError(null);
    try {
      onUpdate(await api.training.promoteModel(jobId, stage));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Action impossible");
    } finally {
      setBusy(false);
    }
  }

  function handleExportModelCard() {
    const card = buildModelCard(job, model);
    const filename = `fiche_modele_${(job.dataset_name ?? "dataset").replace(/[^a-zA-Z0-9_-]/g, "_")}_job${jobId}.json`;
    const blob = new Blob([JSON.stringify(card, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  const stageLabel =
    model.stage === "production" ? "En production" : model.stage === "staging" ? "En validation (staging)" : "Non promu";
  const stageBadgeVariant = model.stage === "production" ? "success" : model.stage === "staging" ? "warning" : "neutral";

  return (
    <Card className={`p-5 ${accentSurfaceClass("violet")}`}>
      <SectionHeader
        icon={Award}
        color="violet"
        label="Registre de modèles"
        help="Marquez ce modèle comme référence pour ce problème (production), ou en validation (staging) avant de le promouvoir. Un seul modèle en production à la fois par jeu de données et cible — promouvoir un nouveau modèle démet automatiquement l'ancien (repasse en staging, jamais supprimé)."
      />
      <div className="flex items-center justify-between flex-wrap gap-3">
        <Badge variant={stageBadgeVariant} dot>{stageLabel}</Badge>
        <div className="flex items-center gap-2 flex-wrap">
          {model.stage !== "staging" && (
            <Button variant="secondary" size="sm" disabled={busy} onClick={() => setStage("staging")}>
              Mettre en validation
            </Button>
          )}
          {model.stage !== "production" && (
            <Button size="sm" disabled={busy} onClick={() => setStage("production")}>
              Promouvoir en production
            </Button>
          )}
          {model.stage && (
            <Button variant="ghost" size="sm" disabled={busy} onClick={() => setStage("none")}>
              Retirer
            </Button>
          )}
          <Button variant="secondary" size="sm" onClick={() => api.training.exportModel(jobId)}>
            <Download size={14} />
            Exporter l'artefact
          </Button>
          <Button variant="secondary" size="sm" onClick={handleExportModelCard}>
            <FileJson size={14} />
            Fiche modèle (JSON)
          </Button>
          <Button variant="secondary" size="sm" onClick={() => api.training.exportDeploymentScript(jobId)}>
            <FileCode size={14} />
            Script de déploiement (.py)
          </Button>
        </div>
      </div>
      <p className="text-xs text-muted-foreground mt-3">
        Pour déployer ce modèle en dehors de DataLab Pro : téléchargez l'artefact ET le script de déploiement,
        placez-les dans le même dossier — le script recharge l'artefact et prédit, sans dépendre de cette
        plateforme (voir l'en-tête du script pour l'installation des bibliothèques nécessaires).
      </p>
      {error && <p className="text-xs text-destructive mt-2">{error}</p>}
    </Card>
  );
}

/** Historique des versions du "problème" (même dataset + même cible) avec
 * rollback (Lot lignage/versions, `GET /training/jobs/{id}/model/versions`)
 * — endpoint backend construit et testé depuis le Lot 5 (correctif P1),
 * sans AUCUN appelant frontend jusqu'ici : un utilisateur ne pouvait ni
 * voir qu'un problème avait plusieurs versions entraînées, ni revenir à
 * une version antérieure sans en connaître le `job_id` par cœur.
 *
 * "Rollback" = repromouvoir une version antérieure en production — RÉUTILISE
 * `api.training.promoteModel` (déjà appelé par `ModelRegistryControls`
 * ci-dessus), jamais un second mécanisme : le backend documente
 * explicitement qu'aucun endpoint dédié n'existe, la promotion normale
 * suffit (elle démet automatiquement la version alors en production). */
function ModelVersionHistory({ jobId }: { jobId: number }) {
  const [versions, setVersions] = useState<ModelVersionEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busyJobId, setBusyJobId] = useState<number | null>(null);

  useEffect(() => {
    api.training
      .modelVersions(jobId)
      .then((r) => setVersions(r.entries))
      .catch((err) => setError(err instanceof ApiError ? err.message : "Historique des versions indisponible"));
  }, [jobId]);

  async function handlePromote(targetJobId: number) {
    setBusyJobId(targetJobId);
    setError(null);
    try {
      await api.training.promoteModel(targetJobId, "production");
      const { entries } = await api.training.modelVersions(jobId);
      setVersions(entries);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Promotion impossible");
    } finally {
      setBusyJobId(null);
    }
  }

  // Une seule version (ce job) : rien à comparer, pas de rollback possible
  // — la carte n'ajoute aucune information par rapport à "Registre de
  // modèles" ci-dessus dans ce cas, jamais affichée pour rester silencieuse
  // plutôt que de montrer un tableau à une seule ligne sans action utile.
  if (versions !== null && versions.length <= 1) return null;

  return (
    <Card className={`p-5 ${accentSurfaceClass("blue")}`}>
      <SectionHeader
        icon={History}
        color="blue"
        label="Versions de ce problème"
        help="Chaque nouvel entraînement sur le même dataset et la même cible crée une nouvelle version — jamais un remplacement silencieux. Promouvoir une version antérieure en production revient en arrière (rollback) sans perdre les autres."
      />
      {error && <p className="text-xs text-destructive mb-3">{error}</p>}
      {versions === null ? (
        <p className="text-sm text-muted-foreground">Chargement…</p>
      ) : (
        <ul className="divide-y divide-border">
          {versions.map((v) => {
            const isCurrent = v.job_id === jobId;
            const stageLabel = v.stage === "production" ? "Production" : v.stage === "staging" ? "Validation" : null;
            const stageVariant = v.stage === "production" ? "success" : "warning";
            return (
              <li key={v.model_id} className="py-2.5 flex items-center gap-3 flex-wrap">
                <span className="text-sm font-medium text-foreground w-14 flex-shrink-0">v{v.version}</span>
                <span className="text-xs text-muted-foreground flex-1 min-w-0 truncate">
                  {v.algorithm} · {formatDateTime(v.created_at)}
                  {isCurrent && " · ce résultat"}
                </span>
                {v.headline_metric && v.headline_metric.value !== null && (
                  <span className="text-xs text-muted-foreground tabular-nums">
                    {v.headline_metric.name} = {v.headline_metric.value.toFixed(3)}
                  </span>
                )}
                {stageLabel && (
                  <Badge variant={stageVariant} dot>
                    {stageLabel}
                  </Badge>
                )}
                {v.stage !== "production" && (
                  <Button
                    variant="secondary"
                    size="sm"
                    disabled={busyJobId !== null}
                    loading={busyJobId === v.job_id}
                    onClick={() => handlePromote(v.job_id)}
                  >
                    Promouvoir en production
                  </Button>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </Card>
  );
}

/** Traçabilité (Lot 9, Traçabilité.html) — versions des librairies ML au
 * moment de l'entraînement, déjà calculées et persistées côté serveur
 * (`services/engine.py::_training_environment_versions`, commenté "Lot
 * 9/10 — traçabilité de l'environnement") mais jusqu'ici jamais affichées
 * nulle part côté frontend. Utile pour l'audit d'un modèle promu en
 * production : reproduire exactement l'environnement qui l'a entraîné.
 * Absent sur un modèle entraîné avant ce champ — carte simplement omise,
 * jamais une erreur. */
function TraceabilityCard({ environment }: { environment: unknown }) {
  if (!environment || typeof environment !== "object") return null;
  const entries = Object.entries(environment as Record<string, string>);
  if (entries.length === 0) return null;
  return (
    <Card className={`p-5 ${accentSurfaceClass("violet")}`}>
      <SectionHeader
        icon={History}
        color="violet"
        label="Environnement d'entraînement"
        help="Versions des librairies ML au moment de l'entraînement — une librairie mise à jour peut changer le résultat d'un même code sur les mêmes données ; ces versions font partie de ce qui définit ce modèle, au même titre que la graine aléatoire."
      />
      <dl className="grid grid-cols-2 sm:grid-cols-3 gap-x-4 gap-y-2 text-xs">
        {entries.map(([lib, version]) => (
          <Fact key={lib} label={lib} value={String(version)} />
        ))}
      </dl>
    </Card>
  );
}

/** Contenu du résultat d'un entraînement — sans chrome de modale, pour être
 * réutilisable tel quel dans deux contextes (Lot E1-ter) : la modale
 * (`ModelResultModal`, ouverte depuis le tableau de bord) et la page
 * Entraînement dédiée, qui l'affiche en place, pleine largeur, une fois le
 * job terminé. */
export function ModelResultView({ job }: { job: TrainingJobSummary }) {
  const [model, setModel] = useState<MLModelDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [leaderboard, setLeaderboard] = useState<LeaderboardResponse | null>(null);
  const [activeTab, setActiveTab] = useState<ResultTabId>("performance");

  useEffect(() => {
    api.training
      .getModel(job.id)
      .then(setModel)
      .catch((err) => setError(err instanceof ApiError ? err.message : "Résultat indisponible"));
    api.training
      .getCandidates(job.id)
      .then(setLeaderboard)
      .catch(() => setLeaderboard(null));
  }, [job.id]);

  const explainability =
    model && isDiagnosticStatus(model.model_card.explainability) ? model.model_card.explainability : undefined;
  const permutationStatus = model?.model_card.permutation_importance_status;
  const calibrationStatus = model?.model_card.calibration_status;
  const learningCurveStatus = model?.model_card.learning_curve_status;

  return (
    <>
      {error && <p className="text-sm text-destructive">{error}</p>}
      {!model && !error && <p className="text-sm text-muted-foreground">Chargement…</p>}

      {model && (
        <div className="space-y-5">
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

          {/* En tête de la vue Résultats (Lot 3, correctif I1) — jamais dans
              un onglet secondaire, c'est la promesse produit elle-même. */}
          <ModelVerdict verdict={model.verdict} />

          {/* Onglets — remplace l'ancien empilement vertical de 10+ cartes
              identiques (défilement long, aucune hiérarchie) par une
              navigation groupée par intention, même motif que EdaModal.tsx
              (Lot refonte Résultats/Datasets). */}
          <Tabs items={RESULT_TABS} active={activeTab} onChange={setActiveTab} />

          {activeTab === "performance" && (
            <div className="space-y-5">
              {/* Dashboard en grille (pas un empilement) — modèles comparés à
                  gauche, métriques détaillées du gagnant à droite. */}
              <div className="grid gap-5 lg:grid-cols-5">
                <div className="lg:col-span-2">
                  <Leaderboard data={leaderboard} />
                </div>
                <Card className="p-5 lg:col-span-3">
                  <SectionHeader icon={Gauge} color="blue" label="Performance" />
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {model.task_type === "regression" ? (
                      <>
                        <MetricCard
                          label="R² (test)"
                          help="Part de la variation de la cible expliquée par le modèle, de 0 à 1. 0,90 = le modèle explique 90 % des écarts observés — plus c'est proche de 1, mieux c'est."
                          value={model.metrics.r2_test as number}
                          ci={isBootstrapCI(model.metrics.r2_bootstrap) ? model.metrics.r2_bootstrap : undefined}
                          color="blue"
                        />
                        <MetricCard
                          label="RMSE"
                          help="Erreur moyenne de prédiction, dans l'unité de la cible. Plus c'est bas, mieux c'est — à comparer à l'échelle typique de vos valeurs."
                          value={model.metrics.rmse as number}
                          ci={isBootstrapCI(model.metrics.rmse_bootstrap) ? model.metrics.rmse_bootstrap : undefined}
                          color="teal"
                        />
                        <MetricCard label="MAE" help="Erreur absolue moyenne — comme le RMSE mais moins sensible aux grosses erreurs isolées." value={model.metrics.mae as number} color="amber" />
                        <MetricCard label="Score CV" help="Performance moyenne sur plusieurs découpages des données d'entraînement — plus fiable qu'un seul test, c'est ce score qui a servi à choisir ce modèle." value={model.metrics.cv_score as number} color="violet" />
                      </>
                    ) : (
                      <>
                        <MetricCard
                          label="Précision globale"
                          help="Pourcentage de prédictions correctes sur le jeu de test."
                          value={model.metrics.accuracy as number}
                          ci={isBootstrapCI(model.metrics.accuracy_bootstrap) ? model.metrics.accuracy_bootstrap : undefined}
                          color="blue"
                        />
                        <MetricCard label="F1-score" help="Équilibre entre précision et rappel — utile quand les classes sont déséquilibrées, où la précision seule peut être trompeuse." value={model.metrics.f1 as number} color="teal" />
                        <MetricCard label="AUC-ROC" help="Capacité du modèle à distinguer les classes, de 0,5 (hasard) à 1 (parfait)." value={model.metrics.roc_auc as number} color="amber" />
                        <MetricCard label="Score CV" help="Performance moyenne sur plusieurs découpages des données d'entraînement — plus fiable qu'un seul test, c'est ce score qui a servi à choisir ce modèle." value={model.metrics.cv_score as number} color="violet" />
                      </>
                    )}
                  </div>
                </Card>
              </div>

              <div className="grid gap-5 lg:grid-cols-2">
                <EvaluationCharts taskType={model.task_type} evaluation={model.evaluation} />
              </div>

              <ModelInterpretation
                model={model}
                explainabilityDegraded={explainability?.status === "degraded"}
              />
            </div>
          )}

          {activeTab === "comparaison" && <ModelComparison data={leaderboard} verdict={model.verdict} />}

          {activeTab === "explicabilite" && (
            <Card className={`p-5 ${accentSurfaceClass("violet")}`}>
              <SectionHeader icon={Sparkles} color="violet" label="Explicabilité SHAP" />
              <div className="space-y-5">
                <div>
                  <p className="text-xs text-muted-foreground mb-2">
                    <LabelWithHelp
                      label="Importance moyenne des variables"
                      help="Plus une variable a une barre longue, plus elle pèse en moyenne dans les décisions du modèle — calculé par la méthode SHAP, standard en explicabilité de modèles ML."
                    />
                  </p>
                  <DiagnosticBlock status={explainability} isEmpty={model.shap_summary.length === 0}>
                    <ShapBars features={model.shap_summary} />
                  </DiagnosticBlock>
                </div>

                <div>
                  <p className="text-xs text-muted-foreground mb-1">
                    <LabelWithHelp
                      label="Distribution des effets (beeswarm)"
                      help="Chaque point est une observation du jeu de test. Sa position horizontale montre si, pour ce cas précis, la variable a poussé la prédiction vers le haut ou vers le bas — la barre ci-dessus ne montrait que la moyenne, sans direction."
                    />
                  </p>
                  <DiagnosticBlock
                    status={explainability}
                    isEmpty={Object.keys(model.shap_beeswarm).length === 0}
                  >
                    <ShapBeeswarmChart beeswarm={model.shap_beeswarm} />
                  </DiagnosticBlock>
                  <p className="text-xs text-muted-foreground mt-2">
                    Les variables en haut ont le plus d'influence. Une couleur chaude (rouge) à droite signifie qu'une
                    valeur élevée de cette variable augmente la prédiction ; une couleur froide (bleu) à droite
                    signifierait l'inverse — une valeur basse qui l'augmente.
                  </p>
                </div>

                <div>
                  <p className="text-xs text-muted-foreground mb-1">
                    <LabelWithHelp
                      label="Importance par permutation"
                      help="Mesure indépendante du SHAP : on mélange aléatoirement les valeurs d'une variable et on regarde de combien le score du modèle chute — plus la chute est grande, plus la variable compte."
                    />
                  </p>
                  <DiagnosticBlock status={permutationStatus} isEmpty={model.permutation_importance.length === 0}>
                    <PermutationImportanceChart features={model.permutation_importance} />
                  </DiagnosticBlock>
                  <p className="text-xs text-muted-foreground mt-2">
                    Une variable qui arrive en tête ici ET dans les graphes SHAP ci-dessus est un signal fort qu'elle
                    compte vraiment pour ce modèle — les deux méthodes se recoupent.
                  </p>
                </div>
              </div>
            </Card>
          )}

          {activeTab === "fiabilite" && (
            <div className="space-y-5">
              <Card className={`p-5 ${accentSurfaceClass("teal")}`}>
                <SectionHeader icon={Activity} color="teal" label="Diagnostics de fiabilité" />
                <div className="space-y-5">
                  {model.task_type === "classification" && (
                    <div>
                      <p className="text-xs text-muted-foreground mb-1">
                        <LabelWithHelp
                          label="Courbe de calibration"
                          help="Compare la probabilité annoncée par le modèle à la fréquence réelle observée — le modèle est-il « sûr à raison » ?"
                        />
                      </p>
                      <DiagnosticBlock
                        status={calibrationStatus}
                        isEmpty={!model.calibration || Object.keys(model.calibration).length === 0}
                      >
                        <CalibrationChart calibration={model.calibration ?? {}} />
                      </DiagnosticBlock>
                      <p className="text-xs text-muted-foreground mt-2">
                        Une courbe proche de la diagonale grise signifie que le modèle est fiable : quand il annonce 80 %
                        de chances, c'est vrai environ 80 % du temps. Une courbe au-dessus de la diagonale : le modèle est
                        trop prudent. En dessous : trop confiant.
                      </p>
                    </div>
                  )}

                  <div>
                    <p className="text-xs text-muted-foreground mb-1">
                      <LabelWithHelp
                        label="Courbe d'apprentissage"
                        help="Montre comment le score évoluerait avec plus ou moins de données d'entraînement — aide à savoir si collecter davantage de données aiderait le modèle."
                      />
                    </p>
                    <DiagnosticBlock status={learningCurveStatus} isEmpty={model.learning_curve === null}>
                      {model.learning_curve && <LearningCurveChart data={model.learning_curve} />}
                    </DiagnosticBlock>
                    <p className="text-xs text-muted-foreground mt-2">
                      Si les deux courbes se rapprochent et restent hautes à droite, ajouter des données n'apporterait
                      probablement pas grand-chose. Un écart persistant entre les deux est un signe de surapprentissage —
                      le modèle a appris le train par cœur plus qu'il n'a généralisé.
                    </p>
                  </div>
                </div>
              </Card>

              {model.cqr && (
                <Card className="p-5">
                  <SectionHeader
                    icon={Gauge}
                    color="amber"
                    label="Fiabilité des prédictions"
                    help="Plutôt qu'une seule valeur, le modèle peut donner une fourchette dans laquelle la vraie valeur tombe la plupart du temps — utile pour savoir jusqu'où faire confiance à une prédiction."
                  />
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                    <MetricCard label="Couverture visée" value={model.cqr.target_coverage} color="blue" />
                    <MetricCard label="Couverture observée" value={model.cqr.empirical_coverage} color="teal" />
                    <MetricCard label="Largeur moyenne" value={model.cqr.mean_interval_width} color="amber" />
                  </div>
                  <p className="text-xs text-muted-foreground mt-2">
                    {formatPercent(model.cqr.empirical_coverage)} des valeurs test tombent dans l'intervalle prédit,
                    pour une cible de {formatPercent(model.cqr.target_coverage)} — calibré par {model.cqr.n_strata} strates.
                  </p>
                </Card>
              )}
            </div>
          )}

          {activeTab === "derive" && <DriftPanel pillar="training" jobId={job.id} />}

          {activeTab === "predire" && (
            <PredictionForm jobId={job.id} taskType={model.task_type} featureSchema={model.feature_schema} />
          )}

          {activeTab === "predire-lot" && <BatchPredictionForm jobId={job.id} model={model} />}

          {activeTab === "details" && (
            <div className="space-y-5">
              <ModelRegistryControls model={model} job={job} onUpdate={setModel} />

              <ModelVersionHistory jobId={job.id} />

              <FeatureEngineeringSummary spec={model.feature_engineering} />

              <Card className={`p-5 ${accentSurfaceClass("blue")}`}>
                <SectionHeader icon={ClipboardList} color="blue" label="Fiche modèle" />
                <dl className="grid grid-cols-2 sm:grid-cols-4 gap-x-4 gap-y-2 text-xs">
                  <Fact label="Jeu de données" value={job.dataset_name ?? "—"} />
                  <Fact label="Cible" value={model.target_column} />
                  <Fact label="Échantillons train" value={String(model.model_card.n_train ?? "—")} />
                  <Fact label="Échantillons test" value={String(model.model_card.n_test ?? "—")} />
                  <Fact label="Doublons retirés" value={String(model.model_card.duplicates_removed ?? "—")} />
                  <Fact label="Essais Optuna" value={String(model.model_card.optuna_trials ?? "—")} />
                  <Fact label="Folds de CV" value={String(model.model_card.cv_folds ?? "—")} />
                  <Fact label="Variables" value={String(model.feature_columns.length)} />
                  <Fact label="Graine aléatoire" value={String(model.model_card.seed ?? "—")} />
                  <Fact
                    label="Découpe anti-fuite par groupe"
                    value={model.model_card.anti_leak_grouping ? "Oui" : "Non"}
                  />
                  <Fact
                    label="Rééquilibrage des classes"
                    value={model.model_card.class_rebalancing_applied ? "Appliqué" : "Non appliqué"}
                  />
                </dl>
              </Card>

              <TraceabilityCard environment={model.model_card.environment} />
            </div>
          )}
        </div>
      )}
    </>
  );
}

export default function ModelResultModal({
  job,
  onClose,
}: {
  job: TrainingJobSummary;
  onClose: () => void;
}) {
  return (
    <Modal title={`${job.dataset_name ?? "Dataset"} — ${job.target_column}`} onClose={onClose} size="2xl">
      <ModelResultView job={job} />
    </Modal>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="text-foreground/90 tabular-nums">{value}</dd>
    </div>
  );
}
