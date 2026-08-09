import { useEffect, useState } from "react";
import { AlertTriangle } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ApiError, api, type DatasetSummary, type EdaResponse, type HistogramResponse } from "../../api/client";
import { Modal } from "../ui/Modal";
import { Heatmap } from "../ui/Heatmap";
import { LabelWithHelp } from "../ui/Tooltip";

const CHART_TOOLTIP_STYLE = {
  contentStyle: {
    backgroundColor: "#0f172a",
    border: "1px solid #1e293b",
    borderRadius: 8,
    fontSize: 12,
  },
  labelStyle: { color: "#cbd5e1" },
};

export default function EdaModal({ dataset, onClose }: { dataset: DatasetSummary; onClose: () => void }) {
  const [eda, setEda] = useState<EdaResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedColumn, setSelectedColumn] = useState<string>("");
  const [histogram, setHistogram] = useState<HistogramResponse | null>(null);

  useEffect(() => {
    api.datasets
      .eda(dataset.id)
      .then((data) => {
        setEda(data);
        if (data.column_stats.length > 0) setSelectedColumn(data.column_stats[0].name);
      })
      .catch((err) => setError(err instanceof ApiError ? err.message : "Exploration indisponible"));
  }, [dataset.id]);

  useEffect(() => {
    if (!selectedColumn) return;
    api.datasets.histogram(dataset.id, selectedColumn).then(setHistogram).catch(() => setHistogram(null));
  }, [dataset.id, selectedColumn]);

  const missingData = eda?.missing_summary.map((m) => ({ name: m.column, pct: m.missing_pct })) ?? [];
  const histogramData =
    histogram?.kind === "numeric"
      ? histogram.counts.map((count, i) => ({
          name: `${histogram.bin_edges![i].toFixed(1)}–${histogram.bin_edges![i + 1].toFixed(1)}`,
          count,
        }))
      : histogram?.categories?.map((cat, i) => ({ name: cat, count: histogram.counts[i] })) ?? [];

  return (
    <Modal title={`${dataset.name} — Exploration`} onClose={onClose}>
      {error && <p className="text-sm text-rose-400">{error}</p>}
      {!eda && !error && <p className="text-sm text-slate-500">Chargement…</p>}

      {eda && (
        <div className="space-y-6">
          <p className="text-xs text-slate-500">{eda.row_count} lignes analysées</p>

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
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                  <XAxis type="number" domain={[0, 100]} tick={{ fill: "#64748b", fontSize: 11 }} unit="%" />
                  <YAxis
                    type="category"
                    dataKey="name"
                    width={100}
                    tick={{ fill: "#94a3b8", fontSize: 11 }}
                  />
                  <RechartsTooltip {...CHART_TOOLTIP_STYLE} formatter={(v) => `${Number(v).toFixed(1)} %`} />
                  <Bar dataKey="pct" fill="#f472b6" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </section>
          )}

          {eda.correlation_matrix.columns.length >= 2 && (
            <section>
              <p className="text-xs uppercase tracking-wide text-slate-500 mb-2">
                <LabelWithHelp
                  label="Corrélations"
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

          <section>
            <p className="text-xs uppercase tracking-wide text-slate-500 mb-2">Distribution d'une variable</p>
            <select
              value={selectedColumn}
              onChange={(e) => setSelectedColumn(e.target.value)}
              className="w-full rounded-lg border border-slate-700 bg-slate-950/60 px-3 py-2 text-sm text-slate-100 mb-3 focus:outline-none focus:ring-2 focus:ring-teal-500/50"
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
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                  <XAxis
                    dataKey="name"
                    tick={{ fill: "#64748b", fontSize: 10 }}
                    angle={-30}
                    textAnchor="end"
                    height={50}
                  />
                  <YAxis tick={{ fill: "#64748b", fontSize: 11 }} allowDecimals={false} />
                  <RechartsTooltip {...CHART_TOOLTIP_STYLE} />
                  <Bar dataKey="count" fill="#2dd4bf" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            )}
          </section>

          <section>
            <p className="text-xs uppercase tracking-wide text-slate-500 mb-2">Résumé par colonne</p>
            <div className="overflow-x-auto rounded-lg border border-slate-800">
              <table className="min-w-full text-xs">
                <thead>
                  <tr className="border-b border-slate-800 bg-slate-900/60">
                    <th className="text-left px-3 py-2 font-medium text-slate-400">Colonne</th>
                    <th className="text-left px-3 py-2 font-medium text-slate-400">Type</th>
                    <th className="text-right px-3 py-2 font-medium text-slate-400">Manquant</th>
                    <th className="text-left px-3 py-2 font-medium text-slate-400">Résumé</th>
                  </tr>
                </thead>
                <tbody>
                  {eda.column_stats.map((c) => (
                    <tr key={c.name} className="border-b border-slate-800/50">
                      <td className="px-3 py-1.5 text-slate-200">{c.name}</td>
                      <td className="px-3 py-1.5 text-slate-500">{c.dtype}</td>
                      <td className="px-3 py-1.5 text-right tabular-nums">
                        {c.missing_pct > 30 && (
                          <AlertTriangle size={11} className="inline mr-1 text-amber-400" />
                        )}
                        <span className={c.missing_pct > 30 ? "text-amber-400" : "text-slate-500"}>
                          {c.missing_pct.toFixed(0)}%
                        </span>
                      </td>
                      <td className="px-3 py-1.5 text-slate-400">
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
