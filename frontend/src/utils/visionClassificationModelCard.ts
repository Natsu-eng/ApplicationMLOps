import type { VisionClassificationResult } from "../api/client";

/** Fiche modèle de la classification d'images — même principe que
 * `utils/modelCard.ts` (pilier supervisé tabulaire, retour utilisateur
 * direct : "on peut télécharger le modèle mais pas un json... qui suit le
 * modèle") : construite ENTIÈREMENT à partir de `VisionClassificationResult`
 * déjà chargé en mémoire, jamais un second appel réseau ni une statistique
 * inventée. */
export function buildVisionClassificationModelCard(
  datasetName: string | null,
  result: VisionClassificationResult,
): Record<string, unknown> {
  const card = result.model_card;
  const warnings: string[] = [];

  if (card.time_capped) {
    warnings.push(
      "Entraînement arrêté par le garde-fou de temps CPU avant la fin des époques demandées — le modèle a pu " +
        "converger partiellement seulement, voir le nombre d'époques réellement effectuées.",
    );
  }
  if (result.test_accuracy < 0.7) {
    warnings.push(
      `Exactitude de test modeste (${(result.test_accuracy * 100).toFixed(1)} %) — vérifiez la quantité et la ` +
        "qualité des images d'entraînement avant un déploiement en production.",
    );
  }

  return {
    plateforme: "DataLab Pro",
    genere_le: new Date().toISOString(),
    probleme: {
      dataset: datasetName,
      classes: result.class_names,
      n_classes: result.class_names.length,
      n_entrainement: result.n_train,
      n_validation: result.n_val,
      n_test: result.n_test,
    },
    modele: {
      backbone: card.backbone_id,
      resolution_entree: card.image_size,
      epoques_demandees: card.num_epochs_requested,
      epoques_effectuees: card.num_epochs_run,
      arrete_par_garde_fou_temps: Boolean(card.time_capped),
      etat: "entraîné",
    },
    performance_test: {
      exactitude: result.test_accuracy,
      precision_macro: result.test_precision_macro,
      rappel_macro: result.test_recall_macro,
      f1_macro: result.test_f1_macro,
      roc_auc: result.test_roc_auc ?? null,
      matrice_confusion: result.confusion_matrix,
    },
    fiabilite_calibration: result.calibration ?? null,
    explicabilite: {
      methode: "Grad-CAM (carte de chaleur visuelle par image, superposée sur l'image d'origine)",
      disponible_sur: "Onglet \"Grad-CAM\" de cette page — une image à la fois ou en lot sur les exemples du dataset.",
    },
    verdict:
      result.test_accuracy >= 0.9
        ? "performance élevée"
        : result.test_accuracy >= 0.7
          ? "performance correcte"
          : "performance à améliorer",
    avertissements: warnings,
    deploiement: {
      artefact: "Poids du réseau (state_dict PyTorch) — bouton \"Exporter l'artefact\".",
      script_autonome:
        "Script Python autonome (aucune dépendance à DataLab Pro) — bouton \"Script de déploiement (.py)\", " +
        "reconstruit l'architecture du backbone et reproduit exactement la prédiction pour de nouvelles images.",
    },
  };
}
