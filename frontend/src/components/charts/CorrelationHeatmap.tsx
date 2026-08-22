import { Heatmap } from "../ui/Heatmap";
import { ChartFallbackTable, ChartFrame } from "./ChartFrame";

/** Corrélations entre variables numériques — diverge autour de 0 (bleu =
 * positive, rouge = négative, gris = quasi nulle). Une forte corrélation
 * n'est PAS une causalité — le rappeler dans la légende, pas seulement en
 * commentaire de code. */
export function CorrelationHeatmap({ variables, matrix }: { variables: string[]; matrix: number[][] }) {
  return (
    <ChartFrame
      title="Corrélations entre variables numériques"
      reading="Bleu = les deux variables varient ensemble ; rouge = en sens opposé ; gris = pas de lien linéaire détecté. Une forte corrélation ne prouve jamais qu'une variable cause l'autre."
      ariaLabel={`Matrice de corrélation entre ${variables.length} variables numériques.`}
      fallbackTable={<ChartFallbackTable columns={["Variable", ...variables]} rows={matrix.map((row, i) => [variables[i], ...row.map((v) => Number(v.toFixed(2)))])} />}
    >
      <Heatmap xLabels={variables} yLabels={variables} matrix={matrix} variant="diverging" formatValue={(v) => v.toFixed(2)} />
    </ChartFrame>
  );
}
