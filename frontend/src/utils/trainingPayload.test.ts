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
});
