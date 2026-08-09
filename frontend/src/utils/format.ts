/** Taille de fichier lisible (o/Ko/Mo/Go) — évite une dépendance externe pour ça. */
export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} o`;
  const units = ["Ko", "Mo", "Go"];
  let value = bytes / 1024;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(value < 10 ? 1 : 0)} ${units[unitIndex]}`;
}

export function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString("fr-FR", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  });
}

export function formatDateTime(iso: string): string {
  return new Date(iso).toLocaleString("fr-FR", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** Libellés lisibles pour les clés de métriques renvoyées par le backend
 * (services/ml_training.py) — évite d'afficher des noms de variables Python. */
const METRIC_LABELS: Record<string, string> = {
  r2_test: "R² (test)",
  r2_train: "R² (train)",
  delta_r2: "Écart train/test (R²)",
  rmse: "RMSE",
  mae: "MAE",
  cv_score: "Score de validation croisée",
  accuracy: "Précision globale",
  f1: "F1-score",
  precision: "Précision",
  recall: "Rappel",
  roc_auc: "AUC-ROC",
};

export function formatMetricLabel(key: string): string {
  return METRIC_LABELS[key] ?? key;
}

export function formatMetricValue(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(3);
}

export function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)} %`;
}
