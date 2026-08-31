import type { AnomalyResult } from "../api/client";

/** Fiche modèle de la détection d'anomalies — même principe que
 * `utils/clusteringModelCard.ts` (retour utilisateur direct : "on peut
 * télécharger le modèle mais pas un json... qui suit le modèle") :
 * construite ENTIÈREMENT à partir de `AnomalyResult` déjà chargé en
 * mémoire, jamais un second appel réseau ni une statistique inventée.
 *
 * Adaptée à ce pilier : Isolation Forest + LOF tournent TOUJOURS ensemble
 * (jamais de "gagnant" élu, voir engine.py) — le "modèle" est le
 * consensus des deux, jamais un seul des deux pris isolément. */
export function buildAnomalyModelCard(
  featureColumns: string[],
  datasetName: string | null,
  result: AnomalyResult,
): Record<string, unknown> {
  const card = result.model_card;
  const warnings: string[] = [];

  if (result.n_anomalies_isolation_forest === 0 && result.n_anomalies_lof === 0) {
    warnings.push(
      "Aucune observation atypique détectée par l'une ou l'autre méthode — soit vos données sont réellement " +
        "homogènes, soit la proportion attendue d'anomalies (contamination) est mal réglée pour ce jeu de données.",
    );
  }
  if (result.n_anomalies_consensus === 0 && (result.n_anomalies_isolation_forest > 0 || result.n_anomalies_lof > 0)) {
    warnings.push(
      "Aucun accord entre Isolation Forest et LOF sur les mêmes observations — les deux méthodes détectent des " +
        "signaux différents ici, à examiner individuellement plutôt que de se fier au seul consensus.",
    );
  }
  if (result.sampled) {
    warnings.push(
      `Entraîné sur un échantillon de ${result.n_samples_used} lignes sur ${result.n_samples_total} au total ` +
        "(plafond mémoire) — tirage aléatoire déterministe, statistiquement fiable, mais à garder en tête. " +
        "Exportez les scores pour couvrir la totalité du dataset.",
    );
  }

  return {
    plateforme: "DataLab Pro",
    genere_le: new Date().toISOString(),
    probleme: {
      dataset: datasetName,
      variables_entree: featureColumns,
      n_variables: featureColumns.length,
    },
    modele: {
      methodes: ["Isolation Forest", "Local Outlier Factor (LOF)"],
      principe: "Les deux méthodes tournent systématiquement ensemble — jamais un seul essai lancé à l'aveugle. " +
        "Le score de consensus (moyenne des rangs percentiles) est le signal le plus fiable ; l'accord/désaccord " +
        "entre les deux reste toujours visible (voir `agreement`).",
      proportion_anomalies_attendue: card.contamination,
      etat: "entraîné",
    },
    validation: {
      n_lignes_utilisees: card.n_samples_used,
      n_lignes_totales: card.n_samples_total,
      echantillonnage_applique: Boolean(card.sampled),
      graine_aleatoire: card.seed,
    },
    resultats: {
      taux_anomalies_isolation_forest: result.anomaly_rate_isolation_forest,
      taux_anomalies_lof: result.anomaly_rate_lof,
      taux_anomalies_consensus: result.anomaly_rate_consensus,
      n_anomalies_isolation_forest: result.n_anomalies_isolation_forest,
      n_anomalies_lof: result.n_anomalies_lof,
      n_anomalies_consensus: result.n_anomalies_consensus,
    },
    notation_nouvelles_observations: {
      disponible: true,
      description:
        "Isolation Forest est nativement inductif. LOF utilise une instance dédiée entraînée en mode " +
        "'novelty' sur les mêmes données. Le score de consensus d'une nouvelle observation est calculé au même " +
        "rang percentile que celui de l'entraînement (jamais recalculé sans distribution de référence).",
    },
    avertissements: warnings,
    deploiement: {
      artefact: "Bundle joblib (préprocesseur + Isolation Forest + LOF) — bouton \"Exporter l'artefact\".",
      script_autonome:
        "Script Python autonome (aucune dépendance à DataLab Pro) — bouton \"Script de déploiement (.py)\", " +
        "reproduit exactement la notation d'anomalies pour de nouvelles observations.",
    },
  };
}
