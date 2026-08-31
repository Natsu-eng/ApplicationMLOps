import type { VisionAnomalyResult } from "../api/client";

/** Fiche modèle de la détection d'anomalies visuelles — même principe que
 * `utils/visionClassificationModelCard.ts` : construite ENTIÈREMENT à
 * partir de `VisionAnomalyResult` déjà chargé en mémoire, jamais un second
 * appel réseau ni une statistique inventée. */
export function buildVisionAnomalyModelCard(
  datasetName: string | null,
  result: VisionAnomalyResult,
): Record<string, unknown> {
  const card = result.model_card;
  const warnings: string[] = [];

  if (card.threshold_calibration_status === "degraded") {
    warnings.push(String(card.threshold_calibration_message ?? "Calibration du seuil dégradée."));
  }
  if (card.time_capped) {
    warnings.push(
      "Entraînement arrêté par le garde-fou de temps CPU avant la fin des époques demandées — le modèle a pu " +
        "converger partiellement seulement.",
    );
  }
  if (result.roc_auc < 0.7) {
    warnings.push(
      `ROC-AUC modeste (${result.roc_auc.toFixed(3)}) — ce modèle distingue mal les images normales des ` +
        "défauts sur ce jeu de test, à interpréter avec prudence avant un déploiement.",
    );
  }

  return {
    plateforme: "DataLab Pro",
    genere_le: new Date().toISOString(),
    probleme: {
      dataset: datasetName,
      categories_defaut: card.defect_categories ?? null,
      n_entrainement_normal: result.n_train,
      n_validation: result.n_val,
      n_test: result.n_test,
    },
    modele: {
      architecture: card.model_id,
      resolution_entree: card.image_size,
      seuil_detection: result.threshold,
      seuil_calibre_par: "Point de Youden sur la courbe ROC (calibration séparée de l'évaluation)",
      epoques_demandees: card.num_epochs_requested,
      epoques_effectuees: card.num_epochs_run,
      arrete_par_garde_fou_temps: Boolean(card.time_capped),
      etat: "entraîné",
    },
    performance_test: {
      roc_auc: result.roc_auc,
      exactitude: result.test_accuracy,
      precision: result.test_precision,
      rappel: result.test_recall,
      f1: result.test_f1,
      matrice_confusion: result.confusion_matrix,
      repartition_par_categorie: result.category_breakdown ?? null,
    },
    notation_nouvelles_observations: {
      disponible: true,
      description:
        "Un autoencodeur est nativement inductif — l'erreur de reconstruction se calcule identiquement sur une " +
        "image jamais vue. Comparée au seuil déjà calibré (ci-dessus), jamais recalculé sur une seule image.",
    },
    verdict:
      result.roc_auc >= 0.9 ? "détection fiable" : result.roc_auc >= 0.7 ? "détection correcte" : "détection à améliorer",
    avertissements: warnings,
    deploiement: {
      artefact: "Poids du réseau (state_dict PyTorch) + seuil calibré — bouton \"Exporter l'artefact\".",
      script_autonome:
        "Script Python autonome (aucune dépendance à DataLab Pro) — bouton \"Script de déploiement (.py)\", " +
        "reproduit exactement la notation d'anomalies pour de nouvelles images (score numérique, sans la carte " +
        "de chaleur visuelle — disponible dans l'application via le bouton \"Noter une image\").",
    },
  };
}
