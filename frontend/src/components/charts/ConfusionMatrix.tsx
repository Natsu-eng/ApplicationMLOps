import { Heatmap } from "../ui/Heatmap";
import { ChartFallbackTable, ChartFrame } from "./ChartFrame";

/** Matrice de confusion — chaque ligne est la vraie classe, chaque colonne
 * la classe prédite. La diagonale est ce que le modèle a bien classé ; tout
 * le reste est une erreur, et sa position dit CE AVEC QUOI le modèle
 * confond quoi. */
export function ConfusionMatrix({ classNames, matrix }: { classNames: string[]; matrix: number[][] }) {
  const total = matrix.flat().reduce((s, v) => s + v, 0);
  const correct = matrix.reduce((s, row, i) => s + (row[i] ?? 0), 0);
  const accuracy = total > 0 ? correct / total : 0;

  return (
    <ChartFrame
      title="Matrice de confusion — vraie classe en ligne, classe prédite en colonne"
      reading="La diagonale montre les prédictions correctes. Une case hors diagonale dit précisément avec quelle classe le modèle confond quoi — pas seulement qu'il se trompe."
      ariaLabel={`Matrice de confusion sur ${classNames.length} classes, ${(accuracy * 100).toFixed(1)}% de prédictions correctes sur la diagonale.`}
      fallbackTable={
        <ChartFallbackTable
          columns={["Vraie \\ Prédite", ...classNames]}
          rows={matrix.map((row, i) => [classNames[i], ...row])}
        />
      }
    >
      <Heatmap xLabels={classNames} yLabels={classNames} matrix={matrix} variant="sequential" />
    </ChartFrame>
  );
}
