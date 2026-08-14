import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  BarChart3,
  Columns3,
  GitCompareArrows,
  Hash,
  Rows3,
  ShieldCheck,
  Table2,
  Tags,
  Target as TargetIcon,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  ApiError,
  api,
  type DatasetSummary,
  type EdaResponse,
  type FeatureByTargetResponse,
  type HistogramResponse,
} from "../../api/client";
import { Card } from "../ui/Card";
import { Modal } from "../ui/Modal";
import { Heatmap } from "../ui/Heatmap";
import { BoxPlotChart, type BoxPlotDatum } from "../ui/BoxPlot";
import { SectionHeader } from "../ui/SectionHeader";
import { Tabs } from "../ui/Tabs";
import { StatTile, StatTileRow } from "../dashboard/StatTile";
import { DataQualityWarnings } from "../training/DataQualityWarnings";
import {
  CHART_COLOR_PRIMARY,
  CHART_COLOR_SECONDARY,
  CHART_COLOR_TERTIARY,
  CHART_GRID_STROKE,
  CHART_TICK_STYLE,
  CHART_TICK_STYLE_MUTED,
  CHART_TICK_STYLE_SM,
  CHART_TOOLTIP_STYLE,
} from "../../theme/charts";

/** Bornes de bin lisibles : pas de décimale inutile sur de grandes valeurs
 * (ex. "5001" plutôt que "5001.0"), une décimale seulement pour les petites
 * valeurs où elle reste informative — évite les libellés à rallonge qui se
 * chevauchaient en axe pivoté (bug réel constaté sur un histogramme à
 * grandes valeurs, ex. UDI 1–10000). */
function formatBinEdge(value: number): string {
  return Math.abs(value) >= 10 ? Math.round(value).toLocaleString("fr-FR") : value.toFixed(1);
}

/** Convertit une réponse histogramme (bins numériques OU comptage
 * catégoriel) en données Recharts — factorisé car réutilisé pour la
 * distribution d'une variable choisie ET pour la distribution de la cible. */
function histogramToChartData(histogram: HistogramResponse | null) {
  if (!histogram) return [];
  return histogram.kind === "numeric"
    ? histogram.counts.map((count, i) => ({
        name: `${formatBinEdge(histogram.bin_edges![i])}–${formatBinEdge(histogram.bin_edges![i + 1])}`,
        count,
      }))
    : (histogram.categories ?? []).map((cat, i) => ({ name: cat, count: histogram.counts[i] }));
}

function boxplotStatsToDatum(stats: { min: number | null; q1: number | null; median: number | null; q3: number | null; max: number | null; outliers: number[] }, name: string): BoxPlotDatum | null {
  if (stats.min === null || stats.q1 === null || stats.median === null || stats.q3 === null || stats.max === null) {
    return null;
  }
  return { name, min: stats.min, q1: stats.q1, median: stats.median, q3: stats.q3, max: stats.max, outliers: stats.outliers };
}

const TABS = [
  { id: "overview", label: "Vue d'ensemble", icon: Rows3 },
  { id: "quality", label: "Qualité des données", icon: ShieldCheck },
  { id: "correlations", label: "Corrélations", icon: GitCompareArrows },
  { id: "distributions", label: "Distributions", icon: BarChart3 },
  { id: "target", label: "Relation à la cible", icon: TargetIcon },
] as const;
type TabId = (typeof TABS)[number]["id"];

export default function EdaModal({ dataset, onClose }: { dataset: DatasetSummary; onClose: () => void }) {
  const [eda, setEda] = useState<EdaResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedColumn, setSelectedColumn] = useState<string>("");
  const [histogram, setHistogram] = useState<HistogramResponse | null>(null);
  const [targetColumn, setTargetColumn] = useState<string>("");
  const [featureForTarget, setFeatureForTarget] = useState<string>("");
  const [featureByTarget, setFeatureByTarget] = useState<FeatureByTargetResponse | null>(null);
  const [activeTab, setActiveTab] = useState<TabId>("overview");

  useEffect(() => {
    api.datasets
      .eda(dataset.id, targetColumn || undefined)
      .then((data) => {
        setEda(data);
        if (data.column_stats.length > 0 && !selectedColumn) setSelectedColumn(data.column_stats[0].name);
      })
      .catch((err) => setError(err instanceof ApiError ? err.message : "Exploration indisponible"));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataset.id, targetColumn]);

  useEffect(() => {
    if (!selectedColumn) return;
    api.datasets.histogram(dataset.id, selectedColumn).then(setHistogram).catch(() => setHistogram(null));
  }, [dataset.id, selectedColumn]);

  useEffect(() => {
    if (!targetColumn || !featureForTarget) {
      setFeatureByTarget(null);
      return;
    }
    api.datasets
      .featureByTarget(dataset.id, featureForTarget, targetColumn)
      .then(setFeatureByTarget)
      .catch(() => setFeatureByTarget(null));
  }, [dataset.id, targetColumn, featureForTarget]);

  const missingData = eda?.missing_summary.map((m) => ({ name: m.column, pct: m.missing_pct })) ?? [];
  const histogramData = histogramToChartData(histogram);
  const targetDistributionData = histogramToChartData(eda?.target_distribution ?? null);

  const numericFeatureOptions = (eda?.column_stats ?? []).filter(
    (c) => c.kind === "numeric" && c.name !== targetColumn,
  );

  const outlierBoxData: BoxPlotDatum[] =
    eda?.outlier_summary
      .map((b) => boxplotStatsToDatum(b, b.column))
      .filter((d): d is BoxPlotDatum => d !== null) ?? [];

  const featureByTargetBoxData: BoxPlotDatum[] =
    featureByTarget?.groups
      .map((g) => boxplotStatsToDatum(g, g.class_name))
      .filter((d): d is BoxPlotDatum => d !== null) ?? [];

  const numericCount = eda?.column_stats.filter((c) => c.kind === "numeric").length;
  const categoricalCount = eda?.column_stats.filter((c) => c.kind === "categorical").length;
  const avgMissingPct = useMemo(() => {
    if (!eda || eda.column_stats.length === 0) return null;
    const total = eda.column_stats.reduce((sum, c) => sum + c.missing_pct, 0);
    return total / eda.column_stats.length;
  }, [eda]);

  return (
    <Modal title={`${dataset.name} — Exploration`} onClose={onClose} size="xl">
      {error && <p className="text-sm text-rose-600">{error}</p>}
      {!eda && !error && <p className="text-sm text-slate-500">Chargement…</p>}

      {eda && (
        <div className="space-y-5">
          <StatTileRow>
            <StatTile icon={Rows3} color="blue" label="Lignes analysées" value={eda.row_count} />
            <StatTile icon={Columns3} color="teal" label="Colonnes" value={eda.column_stats.length} />
            <StatTile icon={Hash} color="amber" label="Variables numériques" value={numericCount} />
            <StatTile icon={Tags} color="violet" label="Variables catégorielles" value={categoricalCount} />
          </StatTileRow>

          {avgMissingPct !== null && avgMissingPct > 0 && (
            <p className="text-xs text-slate-500 -mt-3">
              Taux de valeurs manquantes moyen : <span className="tabular-nums font-medium text-slate-700">{avgMissingPct.toFixed(1)}%</span>
            </p>
          )}

          {/* Onglets — remplace l'ancien empilement vertical de 9 cartes
              identiques (défilement long, aucune hiérarchie) par une
              navigation groupée par intention (vue d'ensemble, qualité,
              corrélations, distributions, cible). */}
          <Tabs items={TABS} active={activeTab} onChange={setActiveTab} />

          {activeTab === "overview" && (
            <Card className="p-4">
              <SectionHeader
                icon={Table2}
                color="blue"
                label="Résumé par colonne"
                help="Un coup d'œil sur chaque variable : type détecté, part de valeurs manquantes, et un résumé adapté (moyenne/écart-type pour le numérique, cardinalité/valeur la plus fréquente pour le catégoriel)."
              />
              <div className="overflow-x-auto rounded-lg border border-slate-200">
                <table className="min-w-full text-xs">
                  <thead>
                    <tr className="border-b border-slate-200 bg-slate-50">
                      <th className="text-left px-3 py-2 font-medium text-slate-500">Colonne</th>
                      <th className="text-left px-3 py-2 font-medium text-slate-500">Type</th>
                      <th className="text-right px-3 py-2 font-medium text-slate-500">Manquant</th>
                      <th className="text-left px-3 py-2 font-medium text-slate-500">Résumé</th>
                    </tr>
                  </thead>
                  <tbody>
                    {eda.column_stats.map((c) => (
                      <tr key={c.name} className="border-b border-slate-100">
                        <td className="px-3 py-1.5 text-slate-800">{c.name}</td>
                        <td className="px-3 py-1.5 text-slate-500">{c.dtype}</td>
                        <td className="px-3 py-1.5 text-right tabular-nums">
                          {c.missing_pct > 30 && (
                            <AlertTriangle size={11} className="inline mr-1 text-amber-600" />
                          )}
                          <span className={c.missing_pct > 30 ? "text-amber-600" : "text-slate-500"}>
                            {c.missing_pct.toFixed(0)}%
                          </span>
                        </td>
                        <td className="px-3 py-1.5 text-slate-500">
                          {c.kind === "numeric"
                            ? `moyenne ${c.mean?.toFixed(2)} · écart-type ${c.std?.toFixed(2)}`
                            : `${c.n_unique} valeurs · fréquente : ${c.top_values?.[0]?.value ?? "—"}`}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}

          {activeTab === "quality" && (
            <Card className="p-4">
              <SectionHeader
                icon={ShieldCheck}
                color="teal"
                label="Contrôle qualité"
                help="Détection automatique des colonnes sans valeur prédictive (identifiants, constantes, doublons), des variables numériques mal typées, et d'autres signaux à connaître avant d'entraîner un modèle. Choisissez une cible ci-dessous pour affiner l'analyse (fuite de données, déséquilibre des classes)."
              />
              <div className="mb-3">
                <label className="block text-xs text-slate-500 mb-1">
                  Cible envisagée <span className="text-slate-400">(optionnel — affine l'analyse)</span>
                </label>
                <select
                  value={targetColumn}
                  onChange={(e) => {
                    setTargetColumn(e.target.value);
                    setFeatureForTarget("");
                  }}
                  className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-primary/40"
                >
                  <option value="">Aucune — analyse structurelle uniquement</option>
                  {eda.column_stats.map((c) => (
                    <option key={c.name} value={c.name}>
                      {c.name} ({c.dtype})
                    </option>
                  ))}
                </select>
              </div>
              <DataQualityWarnings datasetId={dataset.id} targetColumn={targetColumn || undefined} />
            </Card>
          )}

          {activeTab === "correlations" && (
            <div className="space-y-5">
              {eda.correlation_matrix.columns.length >= 2 && (
                <Card className="p-4">
                  <SectionHeader
                    icon={GitCompareArrows}
                    color="blue"
                    label="Corrélations numériques"
                    help="Deux variables très corrélées (proche de 1 ou -1) portent souvent une information redondante — utile à savoir avant de choisir les variables d'un entraînement."
                  />
                  <Heatmap
                    xLabels={eda.correlation_matrix.columns}
                    yLabels={eda.correlation_matrix.columns}
                    matrix={eda.correlation_matrix.matrix}
                    variant="diverging"
                  />
                </Card>
              )}

              {eda.categorical_correlation_matrix.columns.length >= 2 && (
                <Card className="p-4">
                  <SectionHeader
                    icon={GitCompareArrows}
                    color="violet"
                    label="Corrélations catégorielles"
                    help="Association entre variables catégorielles (V de Cramér, corrigé pour ne pas surestimer à cause du nombre de catégories) — de 0 (indépendantes) à 1 (l'une détermine complètement l'autre)."
                  />
                  <Heatmap
                    xLabels={eda.categorical_correlation_matrix.columns}
                    yLabels={eda.categorical_correlation_matrix.columns}
                    matrix={eda.categorical_correlation_matrix.matrix}
                    variant="sequential"
                  />
                </Card>
              )}

              {eda.top_correlated_pairs.length > 0 && (
                <Card className="p-4">
                  <SectionHeader
                    icon={GitCompareArrows}
                    color="amber"
                    label="Paires de variables les plus corrélées"
                    help="Nuage de points des paires numériques les plus liées entre elles — permet de visualiser directement la relation derrière un chiffre de corrélation."
                  />
                  <div className="grid gap-4 sm:grid-cols-2">
                    {eda.top_correlated_pairs.map((pair) => {
                      // Points à valeur manquante (null) exclus AVANT de les
                      // passer à Recharts — bug réel constaté : un axe "number"
                      // avec domaine auto traite un null comme 0, ce qui fausse
                      // le calcul du domaine (nuage écrasé dans un coin, échelle
                      // aberrante) même quand la grande majorité des points sont
                      // valides.
                      const points = pair.points.filter(
                        (p): p is { x: number; y: number } => p.x !== null && p.y !== null,
                      );
                      if (points.length === 0) return null;
                      return (
                        <div key={`${pair.x_column}-${pair.y_column}`}>
                          <p className="text-xs text-slate-500 mb-1">
                            {pair.x_column} × {pair.y_column}{" "}
                            <span className="text-slate-600">(r = {pair.correlation?.toFixed(2) ?? "—"})</span>
                          </p>
                          <ResponsiveContainer width="100%" height={180}>
                            <ScatterChart margin={{ left: 0, right: 12, bottom: 8 }}>
                              <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID_STROKE} />
                              <XAxis
                                type="number"
                                dataKey="x"
                                domain={["auto", "auto"]}
                                tick={CHART_TICK_STYLE_SM}
                                name={pair.x_column}
                              />
                              <YAxis
                                type="number"
                                dataKey="y"
                                domain={["auto", "auto"]}
                                tick={CHART_TICK_STYLE_SM}
                                name={pair.y_column}
                              />
                              <RechartsTooltip {...CHART_TOOLTIP_STYLE} />
                              <Scatter data={points} fill={CHART_COLOR_PRIMARY} fillOpacity={0.6} isAnimationActive={false} />
                            </ScatterChart>
                          </ResponsiveContainer>
                        </div>
                      );
                    })}
                  </div>
                </Card>
              )}
            </div>
          )}

          {activeTab === "distributions" && (
            <div className="space-y-5">
              {missingData.length > 0 && (
                <Card className="p-4">
                  <SectionHeader
                    icon={AlertTriangle}
                    color="amber"
                    label="Valeurs manquantes"
                    help="Colonnes avec au moins une valeur absente — au-delà de 30-40%, la colonne devient souvent peu fiable à utiliser telle quelle."
                  />
                  <ResponsiveContainer width="100%" height={Math.max(80, missingData.length * 28)}>
                    <BarChart data={missingData} layout="vertical" margin={{ left: 8 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID_STROKE} horizontal={false} />
                      <XAxis type="number" domain={[0, 100]} tick={CHART_TICK_STYLE} unit="%" />
                      <YAxis type="category" dataKey="name" width={100} tick={CHART_TICK_STYLE_MUTED} />
                      <RechartsTooltip {...CHART_TOOLTIP_STYLE} formatter={(v) => `${Number(v).toFixed(1)} %`} />
                      <Bar dataKey="pct" fill={CHART_COLOR_SECONDARY} radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </Card>
              )}

              {outlierBoxData.length > 0 && (
                <Card className="p-4">
                  <SectionHeader
                    icon={BarChart3}
                    color="violet"
                    label="Détection d'outliers"
                    help="Boîte à moustaches par variable numérique — les points isolés au-delà des moustaches sont des valeurs atypiques (règle IQR), à vérifier avant de les considérer comme des erreurs ou des cas réels rares."
                  />
                  <BoxPlotChart data={outlierBoxData} height={240} />
                </Card>
              )}

              <Card className="p-4">
                <SectionHeader
                  icon={BarChart3}
                  color="blue"
                  label="Distribution d'une variable"
                  help="Histogramme d'une colonne au choix — bins réguliers pour le numérique, comptage des modalités les plus fréquentes pour le catégoriel."
                />
                <select
                  value={selectedColumn}
                  onChange={(e) => setSelectedColumn(e.target.value)}
                  className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 mb-3 focus:outline-none focus:ring-2 focus:ring-primary/40"
                >
                  {eda.column_stats.map((c) => (
                    <option key={c.name} value={c.name}>
                      {c.name} ({c.dtype})
                    </option>
                  ))}
                </select>
                {histogramData.length > 0 && (
                  <ResponsiveContainer width="100%" height={220}>
                    <BarChart data={histogramData} margin={{ left: 0, bottom: 8 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID_STROKE} vertical={false} />
                      <XAxis
                        dataKey="name"
                        tick={CHART_TICK_STYLE_SM}
                        angle={-45}
                        textAnchor="end"
                        height={62}
                        interval="preserveStartEnd"
                        minTickGap={12}
                      />
                      <YAxis tick={CHART_TICK_STYLE} allowDecimals={false} />
                      <RechartsTooltip {...CHART_TOOLTIP_STYLE} />
                      <Bar dataKey="count" fill={CHART_COLOR_PRIMARY} radius={[4, 4, 0, 0]} isAnimationActive={false} />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </Card>
            </div>
          )}

          {activeTab === "target" && (
            <div className="space-y-5">
              <Card className="p-4">
                <SectionHeader
                  icon={TargetIcon}
                  color="teal"
                  label="Analyser par rapport à une cible"
                  help="Choisir une colonne cible débloque sa distribution et le pouvoir discriminant des autres variables par rapport à elle."
                />
                <select
                  value={targetColumn}
                  onChange={(e) => {
                    setTargetColumn(e.target.value);
                    setFeatureForTarget("");
                  }}
                  className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-primary/40"
                >
                  <option value="">Aucune — choisir une cible</option>
                  {eda.column_stats.map((c) => (
                    <option key={c.name} value={c.name}>
                      {c.name} ({c.dtype})
                    </option>
                  ))}
                </select>
              </Card>

              {!targetColumn && (
                <p className="text-xs text-slate-400 text-center py-6">
                  Choisissez une cible ci-dessus pour voir sa distribution et le pouvoir discriminant des autres
                  variables.
                </p>
              )}

              {targetColumn && targetDistributionData.length > 0 && (
                <Card className="p-4">
                  <SectionHeader
                    icon={BarChart3}
                    color="teal"
                    label={`Distribution de « ${targetColumn} »`}
                    help="Pour une cible numérique : forme de la distribution (symétrique, étalée, avec des valeurs extrêmes...). Pour une cible catégorielle : équilibre entre les classes — un fort déséquilibre est signalé séparément dans les garde-fous."
                  />
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={targetDistributionData} margin={{ left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID_STROKE} vertical={false} />
                      <XAxis dataKey="name" tick={CHART_TICK_STYLE_SM} angle={-30} textAnchor="end" height={50} />
                      <YAxis tick={CHART_TICK_STYLE} allowDecimals={false} />
                      <RechartsTooltip {...CHART_TOOLTIP_STYLE} />
                      <Bar dataKey="count" fill={CHART_COLOR_TERTIARY} radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </Card>
              )}

              {targetColumn && numericFeatureOptions.length > 0 && (
                <Card className="p-4">
                  <SectionHeader
                    icon={GitCompareArrows}
                    color="amber"
                    label="Pouvoir discriminant d'une variable"
                    help="Boîtes à moustaches d'une variable numérique, une par valeur de la cible — si les boîtes sont nettement séparées, cette variable aide probablement à distinguer les cas."
                  />
                  <select
                    value={featureForTarget}
                    onChange={(e) => setFeatureForTarget(e.target.value)}
                    className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 mb-3 focus:outline-none focus:ring-2 focus:ring-primary/40"
                  >
                    <option value="">Choisir une variable numérique…</option>
                    {numericFeatureOptions.map((c) => (
                      <option key={c.name} value={c.name}>
                        {c.name}
                      </option>
                    ))}
                  </select>
                  {featureByTargetBoxData.length > 0 && <BoxPlotChart data={featureByTargetBoxData} height={220} />}
                </Card>
              )}
            </div>
          )}
        </div>
      )}
    </Modal>
  );
}
