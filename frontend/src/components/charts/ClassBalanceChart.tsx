import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip as RechartsTooltip, XAxis, YAxis } from "recharts";
import {
  CHART_COLOR_PRIMARY,
  CHART_COLOR_WARNING,
  CHART_GRID_STROKE,
  CHART_TICK_STYLE_SM,
  CHART_TOOLTIP_STYLE,
} from "../../theme/charts";
import { ChartFallbackTable, ChartFrame } from "./ChartFrame";

/** Répartition des classes + effet du rééquilibrage (retour utilisateur
 * direct : "on détecte bien le déséquilibre mais le problème lors des
 * entraînements on ne montre pas par un graphique adéquat... ce que ça
 * donnera [avec le rééquilibrage]") — partagé par le ML tabulaire
 * (`ClassRebalancingSuggestion.tsx`) et la classification d'images
 * (`ClassImbalanceBanner`, `VisionWizard.tsx`) : même formule "balanced"
 * des deux côtés (`sklearn.utils.class_weight.compute_sample_weight` côté
 * tabulaire, `_class_weights` côté vision), donc le même graphique a du
 * sens pour les deux.
 *
 * `classWeights` optionnel — si fourni (tabulaire : calculé côté backend,
 * voir `data_quality.py::_detect_class_imbalance`), utilisé tel quel
 * (source unique de vérité, jamais recalculé côté client). Absent (vision :
 * aucun calcul backend équivalent aujourd'hui), dérivé ici avec EXACTEMENT
 * la même formule que le poids réellement appliqué à l'entraînement — pas
 * une approximation inventée, un miroir direct du calcul serveur. */
export function ClassBalanceChart({
  classCounts,
  classWeights,
}: {
  classCounts: Record<string, number>;
  classWeights?: Record<string, number>;
}) {
  const total = Object.values(classCounts).reduce((a, b) => a + b, 0);
  const nClasses = Object.keys(classCounts).length;
  const effectiveWeights =
    classWeights ??
    Object.fromEntries(Object.entries(classCounts).map(([name, count]) => [name, total / (nClasses * count)]));

  const data = Object.entries(classCounts)
    .sort((a, b) => b[1] - a[1])
    .map(([name, count]) => ({
      name,
      "Nombre d'exemples": count,
      "Poids appliqué (rééquilibrage)": Number((effectiveWeights[name] ?? 0).toFixed(2)),
    }));

  return (
    <ChartFrame
      title="Répartition des classes et poids appliqué en cas de rééquilibrage"
      reading="Barres bleues : nombre d'exemples réels par classe (échelle de gauche). Barres orange : le poids que le modèle donnerait à chaque exemple de cette classe SI le rééquilibrage est activé (échelle de droite) — une classe rare reçoit un poids plus élevé pour compenser son faible effectif, sans dupliquer ni supprimer aucune ligne."
      ariaLabel={`Répartition de ${nClasses} classes sur ${total} exemples au total. Sans rééquilibrage, le modèle voit la classe majoritaire ${Math.round(Math.max(...Object.values(classCounts)) / Math.min(...Object.values(classCounts)))} fois plus souvent que la classe la plus rare.`}
      fallbackTable={
        <ChartFallbackTable
          columns={["Classe", "Nombre d'exemples", "Poids appliqué (rééquilibrage)"]}
          rows={data.map((d) => [d.name, d["Nombre d'exemples"], d["Poids appliqué (rééquilibrage)"]])}
        />
      }
    >
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} margin={{ top: 8, right: 8, bottom: 8, left: 0 }} accessibilityLayer={false}>
          <CartesianGrid stroke={CHART_GRID_STROKE} vertical={false} />
          <XAxis dataKey="name" tick={CHART_TICK_STYLE_SM} />
          <YAxis
            yAxisId="left"
            tick={CHART_TICK_STYLE_SM}
            label={{ value: "Exemples", angle: -90, position: "insideLeft", ...CHART_TICK_STYLE_SM }}
          />
          <YAxis
            yAxisId="right"
            orientation="right"
            tick={CHART_TICK_STYLE_SM}
            label={{ value: "Poids", angle: 90, position: "insideRight", ...CHART_TICK_STYLE_SM }}
          />
          <RechartsTooltip {...CHART_TOOLTIP_STYLE} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Bar yAxisId="left" dataKey="Nombre d'exemples" fill={CHART_COLOR_PRIMARY} isAnimationActive={false} />
          <Bar
            yAxisId="right"
            dataKey="Poids appliqué (rééquilibrage)"
            fill={CHART_COLOR_WARNING}
            isAnimationActive={false}
          />
        </BarChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}
