import type { MLModelDetail, TrainingJobSummary } from "../api/client";

/** Fiche modèle exportable (retour utilisateur direct : "on peut
 * télécharger le modèle mais pas un JSON... qui suit le modèle également")
 * — un document JSON autoportant qui accompagne l'artefact exporté, pensé
 * pour être lu par un humain (documentation) OU consommé par une autre
 * plateforme (déploiement). Construit ENTIÈREMENT à partir de données déjà
 * chargées en mémoire (`MLModelDetail`, `TrainingJobSummary`) — jamais un
 * second appel réseau, jamais une statistique recalculée côté client :
 * chaque champ ci-dessous existe déjà tel quel côté serveur
 * (`services/engine.py::model_card`), cette fonction ne fait que
 * RÉORGANISER l'existant en un document cohérent, jamais inventer une
 * valeur absente. */
export function buildModelCard(job: TrainingJobSummary, model: MLModelDetail): Record<string, unknown> {
  const diag = model.model_card; // diagnostics internes déjà calculés à l'entraînement (voir engine.py)

  // Avertissements — chaque diagnostic "*_status" dégradé (Lot Explicabilité
  // globale, calibration, courbe d'apprentissage...) devient une ligne en
  // langage clair ici, jamais silencieux dans un JSON qu'on regarde une
  // seule fois puis qu'on oublie.
  const warnings: string[] = [];
  if (diag.explainability === "degraded") warnings.push("Explicabilité locale (SHAP) indisponible pour ce modèle.");
  if (diag.permutation_importance_status === "degraded") warnings.push("Importance par permutation indisponible.");
  if (diag.calibration_status === "degraded") warnings.push("Diagnostic de calibration indisponible.");
  if (diag.learning_curve_status === "degraded") warnings.push("Courbe d'apprentissage indisponible.");
  if (job.task_type === "classification" && diag.class_rebalancing_requested && !diag.class_rebalancing_applied) {
    warnings.push("Rééquilibrage des classes demandé mais non appliqué (modèle retenu sans ce support).");
  }

  return {
    plateforme: "DataLab Pro",
    genere_le: new Date().toISOString(),
    probleme: {
      dataset: job.dataset_name ?? null,
      cible: model.target_column,
      type_tache: model.task_type === "classification" ? "classification" : "régression",
      n_variables: model.feature_columns.length,
      n_lignes_entrainement: diag.n_train ?? null,
      n_lignes_test: diag.n_test ?? null,
      lignes_dupliquees_retirees: diag.duplicates_removed ?? null,
    },
    modele: {
      algorithme: model.algorithm,
      version: model.version,
      etat: model.stage ?? "non promu",
      entraine_le: model.created_at,
      promu_le: model.promoted_at,
    },
    variables_entree: model.feature_columns,
    validation: {
      validation_croisee: diag.cv_folds != null ? `${diag.cv_folds} blocs (k-fold)` : null,
      score_validation_croisee: diag.cv_score ?? null,
      recherche_hyperparametres: diag.optuna_trials != null ? `Optuna, ${diag.optuna_trials} essais` : null,
      graine_aleatoire: diag.seed ?? null,
      anti_fuite_par_groupe: Boolean(diag.anti_leak_grouping),
    },
    performance_test: model.metrics,
    incertitude:
      model.cqr != null
        ? {
            methode: "Régression conforme (CQR — Conformalized Quantile Regression)",
            alpha: model.cqr.alpha,
            couverture_visee: model.cqr.target_coverage,
            couverture_empirique: model.cqr.empirical_coverage,
            largeur_intervalle_moyenne: model.cqr.mean_interval_width,
          }
        : null,
    fiabilite_calibration: model.calibration,
    explicabilite: {
      methodes: [
        "SHAP (contributions locales et globales)",
        "Importance par permutation",
        ...(model.learning_curve ? ["Courbe d'apprentissage"] : []),
      ],
      variables_les_plus_influentes: model.shap_summary.slice(0, 10),
    },
    reequilibrage_classes:
      model.task_type === "classification"
        ? { demande: Boolean(diag.class_rebalancing_requested), applique: Boolean(diag.class_rebalancing_applied) }
        : null,
    ingenierie_variables: model.feature_engineering,
    verdict: model.verdict,
    environnement_entrainement: diag.environment ?? null,
    avertissements: warnings,
    deploiement: {
      artefact: "Bouton « Exporter l'artefact » — bundle .joblib (préprocesseur entraîné + modèle)",
      script_autonome: "Bouton « Script de déploiement (.py) » — aucune dépendance à DataLab Pro",
      format: model.cqr != null
        ? "joblib : préprocesseur scikit-learn + modèle + régresseurs d'intervalle (CQR)"
        : "joblib : préprocesseur scikit-learn + modèle",
    },
  };
}
