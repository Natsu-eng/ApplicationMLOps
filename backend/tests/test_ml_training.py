"""Tests de services/ml_training.py (Lot 3) — cœur du pipeline d'entraînement.

Essais Optuna volontairement réduits (3) pour un temps d'exécution
raisonnable en test — la logique exercée est identique à la production
(voir TrainingConfig).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import services.ml_training as ml_training_module
from services.ml_preprocessing import split_dataset
from services.ml_registry import MODEL_REGISTRY
from services.ml_training import (
    TrainingConfig,
    _classification_selection_score,
    _compute_cqr,
    _cqr_fit_calib_indices,
    _make_cv,
    _optimize_one_model,
    build_preprocessor,
    train_and_evaluate,
)

_FAST_CONFIG = TrainingConfig(optuna_trials=3, cv_folds=3)


def _make_regression_df(n=200, seed=42):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(50, 10, n)
    x2 = rng.normal(20, 5, n)
    y = 2.5 * x1 - 1.2 * x2 + rng.normal(0, 3, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "cible": y})


def _make_multiclass_df(n_per_class=40, seed=0):
    """Mêmes proportions qu'un jeu type Iris (3 classes équilibrées, features
    numériques) — reproduit le bug réel trouvé en usage réel : SHAP renvoie
    un tableau 3D en multiclasse selon la version/le backend, ce qui faisait
    planter `_compute_shap_summary` avec `IndexError: only integer scalar
    arrays can be converted to a scalar index` avant correction."""
    rng = np.random.default_rng(seed)
    rows = []
    for cls, center in enumerate([(5.0, 3.4), (6.0, 2.8), (6.6, 3.0)]):
        for _ in range(n_per_class):
            rows.append([rng.normal(center[0], 0.3), rng.normal(center[1], 0.3), cls])
    return pd.DataFrame(rows, columns=["f1", "f2", "cible"])


def test_regression_pipeline_end_to_end():
    df = _make_regression_df()
    split = split_dataset(df, "cible", ["x1", "x2"], "regression", None, 0.2, 42)
    result = train_and_evaluate(split, "regression", _FAST_CONFIG, lambda s, p: None)

    assert result.algorithm in ("LightGBM", "XGBoost", "CatBoost")
    assert result.metrics["r2_test"] > 0.8  # signal fort et peu bruité, doit être bien capté
    assert result.cqr is not None
    assert 0 <= result.cqr["empirical_coverage"] <= 1
    assert len(result.shap_summary) == 2
    assert len(result.evaluation["actual"]) == len(result.evaluation["predicted"])
    assert len(result.evaluation["residuals"]) == len(result.evaluation["actual"])


def test_multiclass_classification_shap_does_not_crash():
    """Test de non-régression pour le bug trouvé en usage réel (dataset Iris)."""
    df = _make_multiclass_df()
    split = split_dataset(df, "cible", ["f1", "f2"], "classification", None, 0.2, 42)
    result = train_and_evaluate(split, "classification", _FAST_CONFIG, lambda s, p: None)

    assert result.metrics["accuracy"] > 0.7
    assert len(result.shap_summary) == 2
    for entry in result.shap_summary:
        assert isinstance(entry["importance"], float)
    assert result.cqr is None  # pas de CQR en classification

    matrix = result.evaluation["confusion_matrix"]
    assert len(matrix) == 3 and all(len(row) == 3 for row in matrix)  # 3 classes
    assert set(result.evaluation["roc_curves"].keys()) == {"0", "1", "2"}


def test_group_anti_leak_split_reflected_in_model_card():
    rng = np.random.default_rng(3)
    n = 150
    df = pd.DataFrame(
        {"x": rng.normal(size=n), "groupe": rng.integers(0, 30, n), "cible": rng.normal(size=n)}
    )
    split = split_dataset(df, "cible", ["x"], "regression", "groupe", 0.2, 42)
    result = train_and_evaluate(split, "regression", _FAST_CONFIG, lambda s, p: None)
    assert result.model_card["anti_leak_grouping"] is True


# ── Lot A — non-fuite préprocesseur/CV et calibration CQR groupée ──


def test_cv_estimator_is_pipeline_with_preprocessor_first_step(monkeypatch):
    """Preuve structurelle : l'estimateur passé à cross_val_score est un
    Pipeline dont la 1re étape est le préprocesseur — garantit qu'il est
    cloné et refit à l'intérieur de chaque fold, jamais fit en amont sur
    tout le train (Lot A, fuite #1)."""
    captured: dict = {}

    def fake_cross_val_score(estimator, X, y, cv=None, groups=None, scoring=None, n_jobs=None):
        captured["estimator"] = estimator
        return np.array([0.5, 0.5, 0.5])

    monkeypatch.setattr(ml_training_module, "cross_val_score", fake_cross_val_score)

    df = _make_regression_df()
    split = split_dataset(df, "cible", ["x1", "x2"], "regression", None, 0.2, 42)
    y_train = split.y_train.to_numpy(dtype=float)
    preprocessor_template = build_preprocessor(split.X_train)
    cv = _make_cv("regression", 3, None)
    config = TrainingConfig(optuna_trials=1, cv_folds=3)

    spec = MODEL_REGISTRY["lightgbm"]
    _optimize_one_model(
        spec, split.X_train, y_train, "regression", cv, split.groups_train,
        preprocessor_template, config, lambda s, p: None, 0, 10,
    )

    estimator = captured["estimator"]
    assert isinstance(estimator, Pipeline)
    assert estimator.steps[0][0] == "preprocess"
    assert isinstance(estimator.named_steps["preprocess"], ColumnTransformer)


def test_cqr_fit_calib_split_is_group_disjoint():
    """Quand une colonne de groupe est fournie, aucun groupe ne doit se
    retrouver à la fois dans la portion fit et la portion calibration du
    CQR (Lot A, fuite #2 — échangeabilité Mondrian)."""
    rng = np.random.default_rng(11)
    n = 400
    groups = rng.integers(0, 60, n)
    fit_idx, cal_idx = _cqr_fit_calib_indices(n, groups, test_size=0.30, seed=42)
    assert set(groups[fit_idx]).isdisjoint(set(groups[cal_idx]))


def test_cqr_preprocessor_fit_only_on_fit_portion():
    """Preuve comportementale (OHE) : une catégorie présente UNIQUEMENT dans
    la portion calibration du CQR doit mapper à du tout-zéro (inconnu) après
    le préprocesseur du CQR — preuve que celui-ci a été fit sur la seule
    portion fit, pas sur la calibration (Lot A, fuite #2, couplage avec la
    fuite #1)."""
    rng = np.random.default_rng(7)
    n = 300
    df = pd.DataFrame(
        {"x": rng.normal(size=n), "cat": rng.choice(["a", "b", "c"], size=n), "cible": 0.0}
    )
    df["cible"] = 3 * df["x"] + rng.normal(0, 0.5, n)
    split = split_dataset(df, "cible", ["x", "cat"], "regression", None, 0.2, 42)
    y_train = split.y_train.to_numpy(dtype=float)
    y_test = split.y_test.to_numpy(dtype=float)

    config = TrainingConfig(cqr_alpha=0.2, cqr_n_strata=3, seed=42)
    fit_idx, cal_idx = _cqr_fit_calib_indices(len(split.X_train), None, test_size=0.30, seed=config.seed)

    # Catégorie qui n'existe nulle part ailleurs dans le train, placée sur
    # une ligne qui tombera côté calibration.
    X_train_mod = split.X_train.copy()
    target_row = int(cal_idx[0])
    X_train_mod.loc[target_row, "cat"] = "ONLY_IN_CALIB"

    cqr_full = _compute_cqr(X_train_mod, y_train, None, split.X_test, y_test, config)
    cqr_preprocessor = cqr_full["_preprocessor"]

    row_df = X_train_mod.iloc[[target_row]][["x", "cat"]]
    transformed = cqr_preprocessor.transform(row_df)
    transformed = np.asarray(transformed.todense()) if hasattr(transformed, "todense") else np.asarray(transformed)

    feature_names = cqr_preprocessor.get_feature_names_out()
    cat_cols = [i for i, name in enumerate(feature_names) if name.startswith("cat__")]
    assert transformed[0, cat_cols].sum() == 0


def test_feature_engineering_frequency_encoding_survives_pipeline_wiring(monkeypatch):
    """Preuve que le Pipeline passé à cross_val_score (donc cloné/refit par
    fold, Lot A) contient bien l'encodeur de fréquence quand
    feature_engineering_config est actif — la fold-safety de l'encodeur
    lui-même est déjà prouvée isolément (test_ml_preprocessing.py) ; ce test
    prouve le dernier maillon : que ml_training.py le branche réellement au
    même Pipeline que Lot A garantit refit par fold."""
    rng = np.random.default_rng(9)
    n = 200
    df = pd.DataFrame({
        "x": rng.normal(size=n),
        "ville": rng.choice(["paris", "lyon", "marseille"], n),
        "cible": rng.normal(size=n),
    })
    split = split_dataset(df, "cible", ["x", "ville"], "regression", None, 0.2, 42)
    fe_config = {"frequency_encoding": ["ville"]}
    preprocessor_template = build_preprocessor(split.X_train, fe_config)
    cv = _make_cv("regression", 3, None)
    config = TrainingConfig(optuna_trials=1, cv_folds=3)

    captured: dict = {}

    def fake_cross_val_score(estimator, X, y, cv=None, groups=None, scoring=None, n_jobs=None):
        captured["estimator"] = estimator
        return np.array([0.5, 0.5, 0.5])

    monkeypatch.setattr(ml_training_module, "cross_val_score", fake_cross_val_score)

    spec = MODEL_REGISTRY["lightgbm"]
    _optimize_one_model(
        spec, split.X_train, split.y_train.to_numpy(dtype=float),
        "regression", cv, None, preprocessor_template, config, lambda s, p: None, 0, 10,
    )

    estimator = captured["estimator"]
    # `.transformers` (pas `.transformers_`, réservé au ColumnTransformer fitté) :
    # le Pipeline capturé n'a pas encore été fit, cross_val_score étant mocké.
    freq_step_names = [name for name, _, _ in estimator.named_steps["preprocess"].transformers if name.startswith("freq_")]
    assert freq_step_names, "l'encodeur de fréquence doit être un step du même ColumnTransformer que Lot A refit par fold"


def test_train_and_evaluate_with_feature_engineering_config_end_to_end():
    rng = np.random.default_rng(15)
    n = 250
    ville = rng.choice([f"v{i}" for i in range(40)], n)  # cardinalité excessive
    x = rng.normal(50, 10, n)
    df = pd.DataFrame({"x": x, "ville": ville, "cible": 2 * x + rng.normal(0, 3, n)})
    split = split_dataset(df, "cible", ["x", "ville"], "regression", None, 0.2, 42)

    result = train_and_evaluate(
        split, "regression", _FAST_CONFIG, lambda s, p: None,
        feature_engineering_config={"frequency_encoding": ["ville"]},
    )

    assert result.model_card["feature_engineering_active"] is True
    assert any(name.endswith("_frequence") for name in result.pipeline_bundle["feature_names"])


def test_train_and_evaluate_without_feature_engineering_config_flags_inactive():
    df = _make_regression_df()
    split = split_dataset(df, "cible", ["x1", "x2"], "regression", None, 0.2, 42)
    result = train_and_evaluate(split, "regression", _FAST_CONFIG, lambda s, p: None)
    assert result.model_card["feature_engineering_active"] is False


def test_cqr_coverage_non_regression_after_fix():
    """Non-régression : la couverture empirique du CQR reste proche de la
    cible sur un jeu de données réaliste, même après correction de la fuite
    (les scores de CV baissent, mais la couverture CQR — déjà correcte en
    théorie sur le principe Mondrian — doit rester au niveau attendu)."""
    rng = np.random.default_rng(123)
    n = 800
    x1 = rng.normal(50, 10, n)
    x2 = rng.normal(20, 5, n)
    y = 2.5 * x1 - 1.2 * x2 + rng.normal(0, 3, n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "cible": y})
    split = split_dataset(df, "cible", ["x1", "x2"], "regression", None, 0.2, 42)
    result = train_and_evaluate(split, "regression", _FAST_CONFIG, lambda s, p: None)

    target = result.cqr["target_coverage"]
    empirical = result.cqr["empirical_coverage"]
    assert empirical >= target - 0.12


# ── Lot 5 — score de sélection robuste (classification) ────────────────────
#
# `_classification_selection_score` doit rester calculable et comparable
# (même échelle AUC) quel que soit le type d'estimateur du catalogue — c'est
# la condition nécessaire pour que "meilleur score CV parmi N modèles
# hétérogènes" ait un sens (correction 1 du cadrage Lot 5).


def _make_binary_df(n=200, seed=1):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (2 * x1 - x2 + rng.normal(0, 0.5, n) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "cible": y})


def test_selection_score_uses_predict_proba_when_available():
    """Estimateur avec predict_proba (cas standard, ex. RandomForest) : le
    score de sélection doit être un AUC calculé depuis predict_proba, égal à
    l'appel manuel équivalent — non un score tronqué ou un repli silencieux."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    df = _make_binary_df()
    X, y = df[["x1", "x2"]].to_numpy(), df["cible"].to_numpy()
    model = RandomForestClassifier(random_state=42, n_estimators=50).fit(X, y)

    score = _classification_selection_score(model, X, y)
    expected = roc_auc_score(y, model.predict_proba(X)[:, 1])
    assert score == pytest.approx(expected)


def test_selection_score_falls_back_to_decision_function_without_predict_proba():
    """SVC construit SANS probability=True (recherche Optuna, Lot 5 —
    évite la calibration Platt coûteuse à chaque essai) : le score de
    sélection doit passer par decision_function et rester un AUC comparable
    à celui des autres candidats, pas planter ni dégénérer en accuracy."""
    from sklearn.metrics import roc_auc_score
    from sklearn.svm import SVC

    df = _make_binary_df()
    X, y = df[["x1", "x2"]].to_numpy(), df["cible"].to_numpy()
    model = SVC(probability=False, random_state=42).fit(X, y)
    assert not hasattr(model, "predict_proba") or not model.get_params().get("probability", False)

    score = _classification_selection_score(model, X, y)
    expected = roc_auc_score(y, model.decision_function(X))
    assert score == pytest.approx(expected)
    assert 0.0 <= score <= 1.0


def test_selection_score_falls_back_to_accuracy_without_proba_or_decision():
    """Estimateur sans predict_proba ni decision_function (cas limite,
    aucun modèle du catalogue actuel n'est dans ce cas — filet de sécurité
    générique) : repli sur l'accuracy plutôt qu'une exception."""

    class _PredictOnly:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    df = _make_binary_df()
    X, y = df[["x1", "x2"]].to_numpy(), (df["cible"].to_numpy() * 0)  # toutes les classes à 0 → accuracy = 1.0
    score = _classification_selection_score(_PredictOnly(), X, y)
    assert score == 1.0


def test_selection_score_comparable_across_registry_families():
    """Chaque modèle du registre, une fois fit, produit un score de
    sélection CALCULABLE (pas d'exception) et dans l'échelle AUC/accuracy
    [0, 1] — condition de comparabilité entre familles hétérogènes."""
    from services.ml_registry import models_for_task

    df = _make_binary_df(n=150)
    split = split_dataset(df, "cible", ["x1", "x2"], "classification", None, 0.2, 42)
    X_train, y_train = split.X_train.to_numpy(), split.y_train.to_numpy()

    for spec in models_for_task("classification", subset="all"):
        model = spec.build_estimator("classification", 42, {}, False)
        model.fit(X_train, y_train)
        score = _classification_selection_score(model, split.X_test.to_numpy(), split.y_test.to_numpy())
        assert 0.0 <= score <= 1.0, f"{spec.id} : score hors échelle ({score})"
