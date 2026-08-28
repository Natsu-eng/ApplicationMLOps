import { describe, expect, it } from "vitest";
import { DEFAULT_CQR_ALPHA, DEFAULT_CV_FOLDS, DEFAULT_SEED } from "../components/training/ExpertModePanel";
import { buildTrainingJobPayload, type TrainingFormState } from "./trainingPayload";

const DEFAULT_OPTUNA_TRIALS = 20;

function baseState(overrides: Partial<TrainingFormState> = {}): TrainingFormState {
  return {
    datasetId: 1,
    targetColumn: "cible",
    featureColumns: ["x1", "x2"],
    groupColumn: "",
    optunaTrials: DEFAULT_OPTUNA_TRIALS,
    cvFolds: DEFAULT_CV_FOLDS,
    testSize: 0.2,
    seed: DEFAULT_SEED,
    cqrAlpha: DEFAULT_CQR_ALPHA,
    featureEngineering: null,
    classRebalancing: false,
    expertMode: false,
    selectedModelIds: new Set(),
    hyperparameterOverrides: {},
    ...overrides,
  };
}

describe("buildTrainingJobPayload", () => {
  it("mode guidé (par défaut) : n'envoie jamais model_ids — comportement serveur inchangé (Lot E2)", () => {
    const payload = buildTrainingJobPayload(baseState());
    expect(payload.model_ids).toBeUndefined();
    expect(payload.cv_folds).toBe(DEFAULT_CV_FOLDS);
    expect(payload.seed).toBe(DEFAULT_SEED);
    expect(payload.cqr_alpha).toBe(DEFAULT_CQR_ALPHA);
    expect(payload.optuna_trials).toBe(DEFAULT_OPTUNA_TRIALS);
  });

  it("mode expert activé SANS rien changer produit le même payload que le mode guidé (exigence du cadrage)", () => {
    const guided = buildTrainingJobPayload(baseState({ expertMode: false }));
    const expertUntouched = buildTrainingJobPayload(baseState({ expertMode: true }));
    expect(expertUntouched).toEqual(guided);
  });

  it("mode expert avec une sélection de modèles envoie model_ids", () => {
    const payload = buildTrainingJobPayload(
      baseState({ expertMode: true, selectedModelIds: new Set(["lightgbm", "extra_trees"]) }),
    );
    expect(payload.model_ids).toEqual(["lightgbm", "extra_trees"]);
  });

  it("une sélection de modèles en mode guidé (jamais possible via l'UI, garde-fou) reste sans effet", () => {
    const payload = buildTrainingJobPayload(
      baseState({ expertMode: false, selectedModelIds: new Set(["lightgbm"]) }),
    );
    expect(payload.model_ids).toBeUndefined();
  });

  it("colonne de groupe vide envoyée comme absente, pas comme chaîne vide", () => {
    const payload = buildTrainingJobPayload(baseState({ groupColumn: "" }));
    expect(payload.group_column).toBeUndefined();
  });

  // ── Mode expert : hyperparamètres fixés (retour utilisateur direct :
  // "laisser le choix sur les hyperparamètres, profondeur des arbres etc.") ──

  it("mode expert avec un hyperparamètre fixé sur un modèle sélectionné l'envoie", () => {
    const payload = buildTrainingJobPayload(
      baseState({
        expertMode: true,
        selectedModelIds: new Set(["random_forest"]),
        hyperparameterOverrides: { random_forest: { max_depth: 6 } },
      }),
    );
    expect(payload.hyperparameter_overrides).toEqual({ random_forest: { max_depth: 6 } });
  });

  it("mode guidé (expert OFF) n'envoie jamais hyperparameter_overrides, même si l'état en contient", () => {
    const payload = buildTrainingJobPayload(
      baseState({ expertMode: false, hyperparameterOverrides: { random_forest: { max_depth: 6 } } }),
    );
    expect(payload.hyperparameter_overrides).toBeUndefined();
  });

  it("un override laissé pour un modèle décoché entre-temps n'est jamais envoyé", () => {
    const payload = buildTrainingJobPayload(
      baseState({
        expertMode: true,
        selectedModelIds: new Set(["lightgbm"]), // random_forest décoché
        hyperparameterOverrides: { random_forest: { max_depth: 6 }, lightgbm: { max_depth: 8 } },
      }),
    );
    expect(payload.hyperparameter_overrides).toEqual({ lightgbm: { max_depth: 8 } });
  });

  it("aucun hyperparamètre fixé envoie undefined, jamais un objet vide", () => {
    const payload = buildTrainingJobPayload(
      baseState({ expertMode: true, selectedModelIds: new Set(["lightgbm"]), hyperparameterOverrides: {} }),
    );
    expect(payload.hyperparameter_overrides).toBeUndefined();
  });

  it("mode expert activé SANS rien changer (hyperparamètres compris) reste identique au mode guidé", () => {
    const guided = buildTrainingJobPayload(baseState({ expertMode: false }));
    const expertUntouched = buildTrainingJobPayload(baseState({ expertMode: true, hyperparameterOverrides: {} }));
    expect(expertUntouched).toEqual(guided);
  });
});
