import type { ClusteringJobSummary, ClusteringResult } from "../api/client";

/** Fiche modèle du clustering — même principe que `utils/modelCard.ts`
 * (pilier supervisé, retour utilisateur direct : "on peut télécharger le
 * modèle mais pas un json... qui suit le modèle") : construite ENTIÈREMENT
 * à partir de données déjà chargées en mémoire (`ClusteringResult`, dont
 * `model_card` — voir `engine.py::train_and_evaluate_clustering`), jamais
 * un second appel réseau ni une statistique inventée.
 *
 * Adaptée au non supervisé : pas de cible ni de performance sur un jeu de
 * test (le clustering n'a rien à prédire au sens supervisé) — remplacé par
 * la qualité de la structure trouvée (silhouette/Davies-Bouldin/
 * Calinski-Harabasz), sa stabilité (ré-échantillonnage), et la méthode
 * d'assignation disponible pour de nouvelles observations. */
export function buildClusteringModelCard(
  job: ClusteringJobSummary,
  result: ClusteringResult,
): Record<string, unknown> {
  const card = result.model_card;
  const warnings: string[] = [];

  if (card.noise_budget_exceeded_for_all) {
    warnings.push(
      "Aucune configuration testée ne structure une part suffisante des données (plus de la moitié classée " +
        "atypique/bruit dans chaque cas) — le modèle retenu reste le moins mauvais essai, à interpréter avec prudence.",
    );
  }
  if (typeof card.stability_ari !== "number") {
    warnings.push(
      "Stabilité non calculable (pas assez de points pour une estimation fiable par ré-échantillonnage) — la " +
        "reproductibilité de ce découpage sur de nouvelles données n'est pas garantie.",
    );
  } else if (card.stability_ari < 0.5) {
    warnings.push(
      `Stabilité faible (indice de Rand ajusté = ${(card.stability_ari as number).toFixed(2)}) — un ` +
        "ré-échantillonnage des mêmes données produit des groupes sensiblement différents, ce découpage peut ne " +
        "pas se reproduire sur de nouvelles données.",
    );
  }
  if (card.sampled) {
    warnings.push(
      `Entraîné sur un échantillon de ${card.n_samples_used} lignes sur ${card.n_samples_total} au total ` +
        "(plafond mémoire) — tirage aléatoire déterministe, statistiquement fiable, mais à garder en tête.",
    );
  }
  const assignmentMethod =
    card.algorithm_id === "kmeans" || card.algorithm_id === "minibatch_kmeans"
      ? "exacte"
      : card.algorithm_id === "hierarchical" || card.algorithm_id === "dbscan"
        ? "approximative"
        : "non prise en charge";
  if (assignmentMethod === "approximative") {
    warnings.push(
      `Ce modèle (${card.family}) n'a pas de règle d'assignation native pour de nouvelles observations — ` +
        "l'assignation utilisée (par ce produit et par le script de déploiement) est une approximation standard " +
        "de la littérature, jamais un chiffre affiché sans préciser comment il a été obtenu.",
    );
  }

  return {
    plateforme: "DataLab Pro",
    genere_le: new Date().toISOString(),
    probleme: {
      dataset: job.dataset_name,
      variables_entree: job.feature_columns,
      n_variables: job.feature_columns.length,
    },
    modele: {
      algorithme: result.algorithm,
      famille: card.family,
      n_clusters: result.n_clusters,
      etat: "entraîné",
    },
    validation: {
      n_configurations_comparees: card.n_candidates_evaluated,
      n_lignes_clusterisees: card.n_samples_used,
      n_lignes_totales: card.n_samples_total,
      echantillonnage_applique: Boolean(card.sampled),
      graine_aleatoire: card.seed,
    },
    qualite_structure: {
      silhouette: result.metrics.silhouette,
      davies_bouldin: result.metrics.davies_bouldin,
      calinski_harabasz: result.metrics.calinski_harabasz,
      taux_bruit_atypique: result.metrics.noise_ratio,
    },
    stabilite: {
      indice_rand_ajuste: card.stability_ari,
      interpretation:
        typeof card.stability_ari === "number"
          ? card.stability_ari >= 0.75
            ? "élevée"
            : card.stability_ari >= 0.5
              ? "moyenne"
              : "faible"
          : "non calculable",
    },
    assignation_nouvelles_observations: {
      methode: assignmentMethod,
      description:
        assignmentMethod === "exacte"
          ? "Distance au centroïde le plus proche — identique au critère utilisé à l'entraînement."
          : assignmentMethod === "approximative"
            ? "Approximation standard (centroïde ou point cœur le plus proche selon l'algorithme) — voir les avertissements."
            : "Ce modèle ne permet pas d'assigner un cluster à une nouvelle observation.",
    },
    verdict: card.noise_budget_exceeded_for_all
      ? "structure faible — à interpréter avec prudence"
      : (result.metrics.silhouette ?? 0) >= 0.5
        ? "structure bien séparée"
        : "structure modérée",
    avertissements: warnings,
    deploiement: {
      artefact: "Bundle joblib (préprocesseur + modèle) — bouton \"Exporter l'artefact\".",
      script_autonome:
        "Script Python autonome (aucune dépendance à DataLab Pro) — bouton \"Script de déploiement (.py)\", " +
        "reproduit exactement l'assignation de cluster pour de nouvelles observations.",
    },
  };
}
