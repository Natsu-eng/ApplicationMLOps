import type { FeatureEngineeringSpec, HyperparameterOverrides, TrainingJobCreatePayload } from "../api/client";

/** État du formulaire d'entraînement nécessaire à la construction du payload
 * — extrait de `TrainingForm` (Lot E2) pour rester testable sans dépendre du
 * rendu React (pas d'infra de test de composants dans ce dépôt à ce jour). */
export interface TrainingFormState {
  datasetId: number;
  targetColumn: string;
  featureColumns: string[];
  groupColumn: string;
  optunaTrials: number;
  cvFolds: number;
  testSize: number;
  seed: number;
  cqrAlpha: number;
  featureEngineering: Pick<FeatureEngineeringSpec, "upstream" | "pipeline"> | null;
  classRebalancing: boolean;
  expertMode: boolean;
  selectedModelIds: Set<string>;
  // Mode expert hyperparamètres (retour utilisateur direct : "laisser le
  // choix sur les hyperparamètres, profondeur des arbres etc.").
  hyperparameterOverrides: HyperparameterOverrides;
}

/** Construit le payload envoyé à `POST /training/jobs` — mode expert OFF
 * (`expertMode: false`, valeur par défaut du formulaire) : `model_ids` n'est
 * jamais envoyé, comportement strictement identique à avant le Lot E2. Mode
 * expert ON sans sélection de modèles (catalogue jamais ouvert) : même
 * résultat, `model_ids` toujours omis — c'est le serveur qui retombe alors
 * sur son sous-ensemble par défaut (`services/ml_registry.models_for_task`). */
export function buildTrainingJobPayload(state: TrainingFormState): TrainingJobCreatePayload {
  const activeOverrides =
    state.expertMode && Object.keys(state.hyperparameterOverrides).length > 0
      ? Object.fromEntries(
          Object.entries(state.hyperparameterOverrides).filter(([modelId]) => state.selectedModelIds.has(modelId)),
        )
      : {};

  return {
    dataset_id: state.datasetId,
    target_column: state.targetColumn,
    feature_columns: state.featureColumns,
    group_column: state.groupColumn || undefined,
    optuna_trials: state.optunaTrials,
    cv_folds: state.cvFolds,
    test_size: state.testSize,
    seed: state.seed,
    cqr_alpha: state.cqrAlpha,
    feature_engineering: state.featureEngineering ?? undefined,
    class_rebalancing: state.classRebalancing,
    model_ids:
      state.expertMode && state.selectedModelIds.size > 0 ? Array.from(state.selectedModelIds) : undefined,
    // Même garde que model_ids ci-dessus : mode expert OFF (ou jamais
    // ouvert) -> jamais envoyé, comportement strictement inchangé. Ne
    // garde que les modèles réellement sélectionnés — un override laissé
    // pour un modèle décoché entre-temps ne doit jamais partir au serveur
    // (rejeté de toute façon côté API, mais autant ne pas l'envoyer).
    hyperparameter_overrides: Object.keys(activeOverrides).length > 0 ? activeOverrides : undefined,
  };
}
