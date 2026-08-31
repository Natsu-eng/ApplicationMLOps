import type { DimensionalityResult } from "../api/client";

/** Fiche modèle de la réduction de dimension — même principe que
 * `utils/clusteringModelCard.ts` (retour utilisateur direct : "on peut
 * télécharger le modèle mais pas un json... qui suit le modèle") :
 * construite ENTIÈREMENT à partir de `DimensionalityResult` déjà chargé en
 * mémoire, jamais un second appel réseau ni une statistique inventée.
 *
 * Adaptée à ce pilier : PCA est TOUJOURS calculée en plus (référence de
 * fidélité), même quand t-SNE/UMAP est la méthode principale — la fiche le
 * documente. La capacité de projeter une NOUVELLE observation dépend de
 * l'algorithme (native pour PCA/UMAP, absente pour t-SNE — modèle
 * transductif, jamais d'approximation inventée). */
export function buildDimensionalityModelCard(
  datasetName: string | null,
  result: DimensionalityResult,
): Record<string, unknown> {
  const card = result.model_card;
  const warnings: string[] = [];
  const supportsNewPoints = result.algorithm_id !== "tsne";

  if (!supportsNewPoints) {
    warnings.push(
      "t-SNE est un modèle transductif : il n'existe littéralement aucune façon de projeter une nouvelle " +
        "observation dans cet embedding sans ré-entraîner sur l'ensemble des données. Utilisez PCA ou UMAP si " +
        "vous avez besoin de projeter de nouvelles observations.",
    );
  }
  if (result.sampled) {
    warnings.push(
      `Entraîné sur un échantillon de ${result.n_samples_used} lignes sur ${result.n_samples_total} au total ` +
        "(plafond mémoire) — tirage aléatoire déterministe, statistiquement fiable, mais à garder en tête.",
    );
  }
  if (result.trustworthiness_primary < 0.8) {
    warnings.push(
      `Fidélité des voisinages modérée (${result.trustworthiness_primary.toFixed(2)} sur 1) — cette projection ` +
        "2D déforme sensiblement la structure réelle de vos données, à interpréter avec prudence.",
    );
  }

  return {
    plateforme: "DataLab Pro",
    genere_le: new Date().toISOString(),
    probleme: {
      dataset: datasetName,
      variables_entree: result.feature_columns,
      n_variables: result.feature_columns.length,
    },
    modele: {
      methode_principale: result.algorithm,
      methode_principale_id: result.algorithm_id,
      note_fidelite_distances: result.distance_fidelity_note,
      reference_pca_toujours_calculee: true,
      etat: "entraîné",
    },
    validation: {
      n_lignes_utilisees: card.n_samples_used,
      n_lignes_totales: card.n_samples_total,
      echantillonnage_applique: Boolean(card.sampled),
      graine_aleatoire: card.seed,
    },
    qualite_projection: {
      fidelite_voisinages_methode_principale: result.trustworthiness_primary,
      fidelite_voisinages_pca_reference: result.trustworthiness_pca,
      variance_expliquee_totale_pca: result.total_variance_explained,
      variance_expliquee_par_axe_pca: result.variance_explained,
    },
    projection_nouvelles_observations: {
      disponible: supportsNewPoints,
      description: supportsNewPoints
        ? "Projection EXACTE via la méthode `.transform()` native de l'algorithme (PCA ou UMAP)."
        : "Non disponible pour t-SNE (modèle transductif) — voir les avertissements.",
    },
    verdict: result.trustworthiness_primary >= 0.9
      ? "projection très fidèle localement"
      : result.trustworthiness_primary >= 0.8
        ? "projection fidèle localement"
        : "projection à interpréter avec prudence",
    avertissements: warnings,
    deploiement: supportsNewPoints
      ? {
          artefact: "Bundle joblib (préprocesseur + modèle de projection) — bouton \"Exporter l'artefact\".",
          script_autonome:
            "Script Python autonome (aucune dépendance à DataLab Pro) — bouton \"Script de déploiement (.py)\", " +
            "reproduit exactement la projection pour de nouvelles observations.",
        }
      : {
          artefact: "Bundle joblib exportable, mais SANS capacité de projeter de nouvelles observations (t-SNE).",
        },
  };
}
