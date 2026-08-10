import type { BoxPlotDatum } from "../components/ui/BoxPlot";

/** Borne un score de validation croisée à [0, 1] pour l'AFFICHAGE uniquement.
 * Le R² par fold (régression) est mathématiquement non borné en dessous : sur
 * un fold dont la cible a une variance quasi nulle (petit sous-ensemble issu
 * d'un GroupKFold anti-fuite, ex. 4 folds sur un dataset modeste), il peut
 * s'effondrer à des valeurs comme -8×10^8 — valide statistiquement, mais
 * illisible sur un graphe censé montrer "un score entre 0 et 1" à un
 * utilisateur non-DS, et cet unique fold dégénéré écrase l'échelle partagée
 * de TOUS les candidats du graphe de variance. Jamais utilisé pour la
 * sélection du modèle (qui reste sur la valeur brute côté backend,
 * `ml_training.py`) — uniquement pour ce que l'écran affiche.
 * L'AUC/accuracy (classification) est déjà naturellement dans [0, 1] : le
 * clamp y est un simple filet de sécurité, sans effet en pratique. */
export function clampUnitScore(value: number): number {
  return Math.min(1, Math.max(0, value));
}

/** Résumé statistique à 5 chiffres (min/Q1/médiane/Q3/max) à partir des
 * scores par fold de CV — interpolation linéaire (méthode par défaut de
 * numpy.percentile côté backend), pour alimenter BoxPlotChart. */
function quantile(sorted: number[], q: number): number {
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  return sorted[base + 1] !== undefined ? sorted[base] + rest * (sorted[base + 1] - sorted[base]) : sorted[base];
}

/** Convertit les scores par fold d'un candidat en boîte à moustaches pour
 * le graphe "Variance entre les découpages de validation croisée" — chaque
 * valeur est bornée à [0, 1] avant calcul des quantiles (voir
 * `clampUnitScore`), pour que le graphe reste lisible même si un fold est
 * mathématiquement dégénéré. */
export function foldScoresToBoxPlotDatum(name: string, scores: number[]): BoxPlotDatum {
  const sorted = scores.map(clampUnitScore).sort((a, b) => a - b);
  return {
    name,
    min: sorted[0],
    q1: quantile(sorted, 0.25),
    median: quantile(sorted, 0.5),
    q3: quantile(sorted, 0.75),
    max: sorted[sorted.length - 1],
  };
}
