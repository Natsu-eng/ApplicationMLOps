import { useEffect, useState } from "react";
import { AlertTriangle } from "lucide-react";
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
import { Modal } from "../ui/Modal";
import { Heatmap } from "../ui/Heatmap";
import { LabelWithHelp } from "../ui/Tooltip";
import { BoxPlotChart, type BoxPlotDatum } from "../ui/BoxPlot";
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

/** Convertit une réponse histogramme (bins numériques OU comptage
 * catégoriel) en données Recharts — factorisé car réutilisé pour la
 * distribution d'une variable choisie ET pour la distribution de la cible. */
function histogramToChartData(histogram: HistogramResponse | null) {
  if (!histogram) return [];
  return histogram.kind === "numeric"
    ? histogram.counts.map((count, i) => ({
        name: `${histogram.bin_edges![i].toFixed(1)}–${histogram.bin_edges![i + 1].toFixed(1)}`,
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

export default function EdaModal({ dataset, onClose }: { dataset: DatasetSummary; onClose: () => void }) {
  const [eda, setEda] = useState<EdaResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedColumn, setSelectedColumn] = useState<string>("");
  const [histogram, setHistogram] = useState<HistogramResponse | null>(null);
  const [targetColumn, setTargetColumn] = useState<string>("");
  const [featureForTarget, setFeatureForTarget] = useState<string>("");
  const [featureByTarget, setFeatureByTarget] = useState<FeatureByTargetResponse | null>(null);

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

  return (
    <Modal title={`${dataset.name} — Exploration`} onClose={onClose}>
      {error && <p className="text-sm text-rose-600">{error}</p>}
      {!eda && !error && <p className="text-sm text-slate-500">Chargement…</p>}

      {eda && (
        <div className="space-y-6">
          <p className="text-xs text-slate-500">{eda.row_count} lignes analysées</p>

          <section>
            <p className="text-xs uppercase tracking-wide text-slate-500 mb-2">
              <LabelWithHelp
                label="Analyser par rapport à une cible"
                help="Optionnel — choisir une colonne cible débloque sa distribution et le pouvoir discriminant des autres variables par rapport à elle."
              />
            </p>
            <select
              value={targetColumn}
              onChange={(e) => {
                setTargetColumn(e.target.value);
                setFeatureForTarget("");
              }}
              className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-teal-500/40"
            >
              <option value="">Aucune — exploration générale</option>
              {eda.column_stats.map((c) => (
                <option key={c.name} value={c.name}>
                  {c.name} ({c.dtype})
                </option>
              ))}
            </select>
          </section>

          {targetColumn && targetDistributionData.length > 0 && (
            <section>
              <p className="text-xs uppercase tracking-wide text-slate-500 mb-2">
                <LabelWithHelp
                  label={`Distribution de « ${targetColumn} »`}
                  help="Pour une cible numérique : forme de la distribution (symétrique, étalée, avec des valeurs extrêmes...). Pour une cible catégorielle : équilibre entre les classes — un fort déséquilibre est signalé séparément dans les garde-fous."
                />
              </p>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={targetDistributionData} margin={{ left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID_STROKE} vertical={false} />
                  <XAxis dataKey="name" tick={CHART_TICK_STYLE_SM} angle={-30} textAnchor="end" height={50} />
                  <YAxis tick={CHART_TICK_STYLE} allowDecimals={false} />
                  <RechartsTooltip {...CHART_TOOLTIP_STYLE} />
                  <Bar dataKey="count" fill={CHART_COLOR_TERTIARY} radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </section>
          )}

          {targetColumn && numericFeatureOptions.length > 0 && (
            <section>
              <p className="text-xs uppercase tracking-wide text-slate-500 mb-2">
                <LabelWithHelp
                  label="Pouvoir discriminant d'une variable"
                  help="Boîtes à moustaches d'une variable numérique, une par valeur de la cible — si les boîtes sont nettement séparées, cette variable aide probablement à distinguer les cas."
                />
              </p>
              <select
                value={featureForTarget}
                onChange={(e) => setFeatureForTarget(e.target.value)}
                className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 mb-3 focus:outline-none focus:ring-2 focus:ring-teal-500/40"
              >
                <option value="">Choisir une variable numérique…</option>
                {numericFeatureOptions.map((c) => (
                  <option key={c.name} value={c.name}>
                    {c.name}
                  </option>
                ))}
              </select>
              {featureByTargetBoxData.length > 0 && <BoxPlotChart data={featureByTargetBoxData} height={220} />}
            </section>
          )}

          {missingData.length > 0 && (
            <section>
              <p className="text-xs uppercase tracking-wide text-slate-500 mb-2">
                <LabelWithHelp
                  label="Valeurs manquantes"
                  help="Colonnes avec au moins une valeur absente — au-delà de 30-40%, la colonne devient souvent peu fiable à utiliser telle quelle."
                />
              </p>
              <ResponsiveContainer width="100%" height={Math.max(80, missingData.length * 28)}>
                <BarChart data={missingData} layout="vertical" margin={{ left: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID_STROKE} horizontal={false} />
                  <XAxis type="number" domain={[0, 100]} tick={CHART_TICK_STYLE} unit="%" />
                  <YAxis
                    type="category"
                    dataKey="name"
                    width={100}
                    tick={CHART_TICK_STYLE_MUTED}
                  />
                  <RechartsTooltip {...CHART_TOOLTIP_STYLE} formatter={(v) => `${Number(v).toFixed(1)} %`} />
                  <Bar dataKey="pct" fill={CHART_COLOR_SECONDARY} radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </section>
          )}

          {eda.correlation_matrix.columns.length >= 2 && (
            <section>
              <p className="text-xs uppercase tracking-wide text-slate-500 mb-2">
                <LabelWithHelp
                  label="Corrélations numériques"
                  help="Deux variables très corrélées (proche de 1 ou -1) portent souvent une information redondante — utile à savoir avant de choisir les variables d'un entraînement."
                />
              </p>
              <Heatmap
                xLabels={eda.correlation_matrix.columns}
                yLabels={eda.correlation_matrix.columns}
                matrix={eda.correlation_matrix.matrix}
                variant="diverging"
              />
            </section>
          )}

          {eda.categorical_correlation_matrix.columns.length >= 2 && (
            <section>
              <p className="text-xs uppercase tracking-wide text-slate-500 mb-2">
                <LabelWithHelp
                  label="Corrélations catégorielles"
                  help="Association entre variables catégorielles (V de Cramér, corrigé pour ne pas surestimer à cause du nombre de catégories) — de 0 (indépendantes) à 1 (l'une détermine complètement l'autre)."
                />
              </p>
              <Heatmap
                xLabels={eda.categorical_correlation_matrix.columns}
                yLabels={eda.categorical_correlation_matrix.columns}
                matrix={eda.categorical_correlation_matrix.matrix}
                variant="sequential"
              />
            </section>
          )}

          {outlierBoxData.length > 0 && (
            <section>
              <p className="text-xs uppercase tracking-wide text-slate-500 mb-2">
                <LabelWithHelp
                  label="Détection d'outliers"
                  help="Boîte à moustaches par variable numérique — les points isolés au-delà des moustaches sont des valeurs atypiques (règle IQR), à vérifier avant de les considérer comme des erreurs ou des cas réels rares."
                />
              </p>
              <BoxPlotChart data={outlierBoxData} height={240} />
            </section>
          )}

          {eda.top_correlated_pairs.length > 0 && (
            <section>
              <p className="text-xs uppercase tracking-wide text-slate-500 mb-2">
                <LabelWithHelp
                  label="Paires de variables les plus corrélées"
                  help="Nuage de points des paires numériques les plus liées entre elles — permet de visualiser directement la relation derrière un chiffre de corrélation."
                />
              </p>
              <div className="space-y-4">
                {eda.top_correlated_pairs.map((pair) => (
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
                          tick={CHART_TICK_STYLE_SM}
                          name={pair.x_column}
                        />
                        <YAxis
                          type="number"
                          dataKey="y"
                          tick={CHART_TICK_STYLE_SM}
                          name={pair.y_column}
                        />
                        <RechartsTooltip {...CHART_TOOLTIP_STYLE} />
                        <Scatter data={pair.points} fill={CHART_COLOR_PRIMARY} fillOpacity={0.6} isAnimationActive={false} />
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>
                ))}
              </div>
            </section>
          )}

          <section>
            <p className="text-xs uppercase tracking-wide text-slate-500 mb-2">Distribution d'une variable</p>
            <select
              value={selectedColumn}
              onChange={(e) => setSelectedColumn(e.target.value)}
              className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 mb-3 focus:outline-none focus:ring-2 focus:ring-teal-500/40"
            >
              {eda.column_stats.map((c) => (
                <option key={c.name} value={c.name}>
                  {c.name} ({c.dtype})
                </option>
              ))}
            </select>
            {histogramData.length > 0 && (
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={histogramData} margin={{ left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID_STROKE} vertical={false} />
                  <XAxis
                    dataKey="name"
                    tick={CHART_TICK_STYLE_SM}
                    angle={-30}
                    textAnchor="end"
                    height={50}
                  />
                  <YAxis tick={CHART_TICK_STYLE} allowDecimals={false} />
                  <RechartsTooltip {...CHART_TOOLTIP_STYLE} />
                  <Bar dataKey="count" fill={CHART_COLOR_PRIMARY} radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            )}
          </section>

          <section>
            <p className="text-xs uppercase tracking-wide text-slate-500 mb-2">Résumé par colonne</p>
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
          </section>
        </div>
      )}
    </Modal>
  );
}
