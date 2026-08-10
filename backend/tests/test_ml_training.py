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
    """Le gagnant peut désormais être n'importe quel modèle du registre
    supportant la régression (Lot 5 : catalogue élargi au-delà des 3
    boosters) — l'assertion sur `algorithm` reste dynamique plutôt que
    figée sur les 3 noms historiques, qui deviendrait fausse dès qu'un
    autre modèle gagne légitimement (ex. Ridge sur un signal linéaire)."""
    from services.ml_registry import models_for_task

    df = _make_regression_df()
    split = split_dataset(df, "cible", ["x1", "x2"], "regression", None, 0.2, 42)
    result = train_and_evaluate(split, "regression", _FAST_CONFIG, lambda s, p: None)

    valid_labels = {spec.label("regression") for spec in models_for_task("regression", "all")}
    assert result.algorithm in valid_labels
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

    def fake_cross_val_score(estimator, X, y, cv=None, groups=None, scoring=None, n_jobs=None, fit_params=None):
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

    def fake_cross_val_score(estimator, X, y, cv=None, groups=None, scoring=None, n_jobs=None, fit_params=None):
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


# ── Lot 5 — SHAP par famille (tree/linear/kernel) ───────────────────────────


def test_shap_linear_explainer_matches_coefficients_in_processed_space():
    """Sanity check numérique (correction 2 du cadrage Lot 5) : sur un modèle
    linéaire connu, les valeurs SHAP doivent égaler coef_ * (x - moyenne du
    fond) DANS L'ESPACE PRÉPROCESSÉ. Si l'explainer recevait les données
    brutes (mauvais espace — les coefficients ont été appris sur des données
    centrées-réduites), cette égalité échouerait silencieusement sans lever
    d'exception, ce qui est précisément le risque signalé par la correction :
    une explicabilité fausse mais sans crash. Fond ≤ 100 lignes (paramètre
    par défaut `max_samples` du masker Independent de SHAP) pour que la
    moyenne utilisée soit prévisible, sans sous-échantillonnage interne."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    from services.ml_training import _build_explainer

    rng = np.random.default_rng(5)
    n = 80
    X_raw = rng.normal(size=(n, 3))
    y = 5.0 * X_raw[:, 0] + 0.5 * X_raw[:, 1] - 2.0 * X_raw[:, 2] + rng.normal(0, 0.05, n)

    # Préprocessing minimal explicite (équivalent, pour l'essentiel, à ce que
    # fait build_preprocessor sur les colonnes numériques) — isole le
    # comportement de l'explainer du reste du pipeline.
    X_proc = StandardScaler().fit_transform(X_raw)
    model = Ridge(alpha=0.01, random_state=42).fit(X_proc, y)

    explainer = _build_explainer("linear", model, X_proc)
    shap_values = np.asarray(explainer.shap_values(X_proc[:20]))

    expected = model.coef_ * (X_proc[:20] - X_proc.mean(axis=0))
    assert np.allclose(shap_values, expected, atol=1e-6)


def test_shap_kernel_explainer_produces_result_for_small_feature_set():
    """KernelExplainer (routage 'kernel' — SVM/KNN/Naive Bayes) produit un
    résultat exploitable pour un petit nombre de variables — une entrée par
    variable, une importance numérique, pas seulement 'non vide'."""
    from sklearn.svm import SVC

    from services.ml_training import _compute_shap_summary

    rng = np.random.default_rng(2)
    n = 60
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = SVC(probability=True, random_state=42).fit(X, y)

    summary = _compute_shap_summary(model, "kernel", X, X[:10], ["f1", "f2", "f3"])
    assert len(summary) == 3
    assert all(isinstance(entry["importance"], float) for entry in summary)


def test_explainability_degrades_above_kernel_feature_threshold():
    """Au-delà de _KERNEL_SHAP_MAX_FEATURES variables, l'explicabilité kernel
    est désactivée avec un statut + message explicite plutôt que de lancer un
    calcul trop long — jamais une disparition silencieuse (Lot 5)."""
    from services.ml_training import _KERNEL_SHAP_MAX_FEATURES, _compute_explainability

    n_features = _KERNEL_SHAP_MAX_FEATURES + 5
    feature_names = [f"f{i}" for i in range(n_features)]

    class _DummyModel:
        def predict(self, X):
            return np.zeros(len(X))

    shap_summary, status = _compute_explainability(
        _DummyModel(), "kernel",
        np.zeros((10, n_features)), np.zeros((10, n_features)),
        feature_names, TrainingConfig(),
    )
    assert shap_summary == []
    assert status["status"] == "degraded"
    assert status["message"]  # message non vide, en langage clair (pas de jargon brut)


def test_explainability_degrades_on_unexpected_shap_failure(monkeypatch):
    """Filet de sécurité générique : si le calcul SHAP échoue pour une raison
    imprévue, l'entraînement ne doit pas planter — dégradation avec message,
    pas d'exception qui remonte à l'appelant."""

    def _boom(*args, **kwargs):
        raise RuntimeError("échec simulé")

    monkeypatch.setattr(ml_training_module, "_compute_shap_summary", _boom)

    shap_summary, status = ml_training_module._compute_explainability(
        object(), "tree", np.zeros((5, 2)), np.zeros((5, 2)), ["f1", "f2"], TrainingConfig(),
    )
    assert shap_summary == []
    assert status["status"] == "degraded"
    assert status["message"]


def test_model_card_carries_ok_explainability_status_for_tree_models():
    """Non-régression : les boosters (routage 'tree') restent au statut 'ok'
    — le chantier SHAP par famille ne dégrade pas ce qui marchait déjà."""
    df = _make_regression_df()
    split = split_dataset(df, "cible", ["x1", "x2"], "regression", None, 0.2, 42)
    result = train_and_evaluate(split, "regression", _FAST_CONFIG, lambda s, p: None)
    assert result.model_card["explainability"] == {"status": "ok", "message": None}


# ── Lot 5 — nouveaux modèles du registre (RandomForest/ExtraTrees/linéaire/SVM/KNN/NaiveBayes) ──


def test_every_registry_model_fits_and_predicts_end_to_end():
    """Chaque modèle du registre s'entraîne et prédit sans erreur, pour
    chacune de ses tâches supportées, via le Pipeline(préprocesseur, modèle)
    exact utilisé par le moteur — pas un raccourci (Lot 5 : catalogue élargi
    à 9 modèles, 3 familles)."""
    from services.ml_registry import models_for_task

    reg_df = _make_regression_df(n=120)
    reg_split = split_dataset(reg_df, "cible", ["x1", "x2"], "regression", None, 0.2, 42)

    clf_df = _make_binary_df(n=120)
    clf_split = split_dataset(clf_df, "cible", ["x1", "x2"], "classification", None, 0.2, 42)

    for task_type, split in (("regression", reg_split), ("classification", clf_split)):
        for spec in models_for_task(task_type, "all"):
            preprocessor = build_preprocessor(split.X_train)
            model = spec.build_estimator(task_type, 42, {}, False)
            pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
            pipeline.fit(split.X_train, split.y_train)
            preds = pipeline.predict(split.X_test)
            assert len(preds) == len(split.X_test), f"{spec.id}/{task_type} : taille de prédiction incorrecte"


def test_cv_estimator_is_pipeline_for_scaling_required_models(monkeypatch):
    """Même preuve structurelle que Lot A (`test_cv_estimator_is_pipeline_with_preprocessor_first_step`),
    mais pour des modèles qui EXIGENT le scaling (`requires_scaling=True` —
    SVM/KNN/régression linéaire) : confirme que le pattern
    `Pipeline(préprocesseur, modèle)` cloné/refit par fold par
    `cross_val_score` n'est pas spécifique aux arbres — le scaler de ces
    modèles est fit DANS le fold, jamais en amont (Lot 5, non-régression de
    l'anti-fuite Lot A pour les nouveaux modèles)."""
    captured: dict = {}

    def fake_cross_val_score(estimator, X, y, cv=None, groups=None, scoring=None, n_jobs=None, fit_params=None):
        captured["estimator"] = estimator
        return np.array([0.7, 0.7, 0.7])

    monkeypatch.setattr(ml_training_module, "cross_val_score", fake_cross_val_score)

    df = _make_regression_df()
    split = split_dataset(df, "cible", ["x1", "x2"], "regression", None, 0.2, 42)
    y_train = split.y_train.to_numpy(dtype=float)
    preprocessor_template = build_preprocessor(split.X_train)
    cv = _make_cv("regression", 3, None)
    config = TrainingConfig(optuna_trials=1, cv_folds=3)

    for spec_id in ("svm", "knn", "linear_reg"):
        spec = MODEL_REGISTRY[spec_id]
        assert spec.requires_scaling
        captured.clear()
        _optimize_one_model(
            spec, split.X_train, y_train, "regression", cv, split.groups_train,
            preprocessor_template, config, lambda s, p: None, 0, 10,
        )
        estimator = captured["estimator"]
        assert isinstance(estimator, Pipeline), spec_id
        assert estimator.steps[0][0] == "preprocess", spec_id
        assert isinstance(estimator.named_steps["preprocess"], ColumnTransformer), spec_id


# ── Lot 5 — sélection par défaut (stratégie produit "B") ────────────────────


def test_default_subset_is_boosters_plus_random_forest(monkeypatch):
    """Sans sélection explicite (mode expert pas encore exposé, Lot E), seul
    le sous-ensemble par défaut du registre tourne — boosters + RandomForest
    (`ModelSpec.is_default`) — pas le catalogue complet à chaque
    entraînement. Le reste du catalogue (ExtraTrees, linéaire, SVM, KNN,
    Naive Bayes) reste disponible dans le registre mais n'est pas lancé."""
    from services.ml_registry import models_for_task

    called_ids: list[str] = []
    original = ml_training_module._optimize_one_model

    def _tracking(spec, *args, **kwargs):
        called_ids.append(spec.id)
        return original(spec, *args, **kwargs)

    monkeypatch.setattr(ml_training_module, "_optimize_one_model", _tracking)

    df = _make_regression_df()
    split = split_dataset(df, "cible", ["x1", "x2"], "regression", None, 0.2, 42)
    train_and_evaluate(split, "regression", _FAST_CONFIG, lambda s, p: None)

    expected_ids = {spec.id for spec in models_for_task("regression", "default")}
    assert set(called_ids) == expected_ids


# ── Lot déséquilibre — rééquilibrage des classes par sample_weight ──────────


def _make_imbalanced_classification_df(n=500, minority_frac=0.08, seed=7):
    """~92/8, minorité décalée mais avec chevauchement — conçu pour qu'un
    entraînement non pondéré la néglige nettement (voir
    `test_class_rebalancing_improves_minority_recall_on_imbalanced_dataset`)."""
    rng = np.random.default_rng(seed)
    n_min = max(int(n * minority_frac), 20)
    n_maj = n - n_min
    x1 = np.concatenate([rng.normal(0, 1.5, n_maj), rng.normal(1.5, 1.5, n_min)])
    x2 = np.concatenate([rng.normal(0, 1.5, n_maj), rng.normal(1.5, 1.5, n_min)])
    y = np.array([0] * n_maj + [1] * n_min)
    return pd.DataFrame({"x1": x1, "x2": x2, "cible": y}).sample(frac=1, random_state=seed).reset_index(drop=True)


def _minority_class_recall(evaluation) -> float:
    cm = np.array(evaluation["confusion_matrix"])
    minority_idx = int(np.argmin(cm.sum(axis=1)))
    return float(cm[minority_idx, minority_idx] / cm[minority_idx].sum())


def test_supports_rebalancing_declared_for_every_model_except_knn():
    """KNN est le seul modèle du catalogue sans notion de pondération
    d'échantillon (`KNeighborsClassifier.fit` n'a pas de paramètre
    `sample_weight`) — tous les autres le supportent nativement, y compris
    GaussianNB (contrairement à l'hypothèse initiale du cadrage du lot)."""
    for model_id, spec in MODEL_REGISTRY.items():
        assert spec.supports_rebalancing is (model_id != "knn"), model_id


def test_class_rebalancing_disabled_by_default_leaves_model_card_flags_false():
    """Rétrocompatibilité totale : sans activation explicite, le comportement
    (et la fiche modèle) reste strictement identique à avant ce lot."""
    df = _make_multiclass_df()
    split = split_dataset(df, "cible", ["f1", "f2"], "classification", None, 0.2, 42)
    result = train_and_evaluate(split, "classification", _FAST_CONFIG, lambda s, p: None)
    assert result.model_card["class_rebalancing_requested"] is False
    assert result.model_card["class_rebalancing_applied"] is False


def test_optimize_one_model_routes_sample_weight_for_supporting_model(monkeypatch):
    """Preuve structurelle : le poids par échantillon est bien routé vers
    `model__sample_weight` de `cross_val_score` pour un modèle qui le
    supporte (LightGBM)."""
    captured: dict = {}

    def fake_cross_val_score(estimator, X, y, cv=None, groups=None, scoring=None, n_jobs=None, fit_params=None):
        captured["fit_params"] = fit_params
        return np.array([0.5, 0.5, 0.5])

    monkeypatch.setattr(ml_training_module, "cross_val_score", fake_cross_val_score)

    df = _make_multiclass_df()
    split = split_dataset(df, "cible", ["f1", "f2"], "classification", None, 0.2, 42)
    n_train = len(split.X_train)
    rng = np.random.default_rng(0)
    y_train = rng.integers(0, 3, n_train)
    sample_weight = rng.uniform(0.5, 1.5, n_train)
    preprocessor_template = build_preprocessor(split.X_train)
    cv = _make_cv("classification", 3, None)
    config = TrainingConfig(optuna_trials=1, cv_folds=3, class_rebalancing=True)

    spec = MODEL_REGISTRY["lightgbm"]
    _optimize_one_model(
        spec, split.X_train, y_train, "classification", cv, split.groups_train,
        preprocessor_template, config, lambda s, p: None, 0, 10,
        sample_weight=sample_weight,
    )

    assert captured["fit_params"] is not None
    np.testing.assert_array_equal(captured["fit_params"]["model__sample_weight"], sample_weight)


def test_optimize_one_model_skips_sample_weight_for_unsupported_model(monkeypatch):
    """KNN (`supports_rebalancing=False`) ne reçoit jamais de sample_weight,
    même quand un poids est fourni — pas de crash, pas de transmission
    incorrecte à un estimateur qui ne l'accepte pas en `.fit()`."""
    captured: dict = {}

    def fake_cross_val_score(estimator, X, y, cv=None, groups=None, scoring=None, n_jobs=None, fit_params=None):
        captured["fit_params"] = fit_params
        return np.array([0.5, 0.5, 0.5])

    monkeypatch.setattr(ml_training_module, "cross_val_score", fake_cross_val_score)

    df = _make_multiclass_df()
    split = split_dataset(df, "cible", ["f1", "f2"], "classification", None, 0.2, 42)
    n_train = len(split.X_train)
    rng = np.random.default_rng(0)
    y_train = rng.integers(0, 3, n_train)
    sample_weight = rng.uniform(0.5, 1.5, n_train)
    preprocessor_template = build_preprocessor(split.X_train)
    cv = _make_cv("classification", 3, None)
    config = TrainingConfig(optuna_trials=1, cv_folds=3, class_rebalancing=True)

    spec = MODEL_REGISTRY["knn"]
    _optimize_one_model(
        spec, split.X_train, y_train, "classification", cv, split.groups_train,
        preprocessor_template, config, lambda s, p: None, 0, 10,
        sample_weight=sample_weight,
    )

    assert captured["fit_params"] is None


def test_class_rebalancing_does_not_alter_train_test_split():
    """Anti-fuite (Lot A) : `class_rebalancing` n'est jamais transmis à
    `split_dataset` — le split train/test/CV reste rigoureusement identique
    avec ou sans rééquilibrage, seule la pondération vue par `.fit()` change."""
    df = _make_multiclass_df()
    split = split_dataset(df, "cible", ["f1", "f2"], "classification", None, 0.2, 42)

    cfg_off = TrainingConfig(optuna_trials=3, cv_folds=3, class_rebalancing=False)
    cfg_on = TrainingConfig(optuna_trials=3, cv_folds=3, class_rebalancing=True)
    result_off = train_and_evaluate(split, "classification", cfg_off, lambda s, p: None)
    result_on = train_and_evaluate(split, "classification", cfg_on, lambda s, p: None)

    assert result_off.model_card["n_train"] == result_on.model_card["n_train"] == len(split.X_train)
    assert result_off.model_card["n_test"] == result_on.model_card["n_test"] == len(split.X_test)


def test_class_rebalancing_improves_minority_recall_on_imbalanced_dataset():
    """Preuve que le flag agit réellement (pas seulement câblé) : sur un
    dataset déséquilibré (~92/8), activer le rééquilibrage améliore
    nettement le rappel de la classe minoritaire — au prix attendu d'un
    rappel global plus faible, l'arbitrage même que le message affiché à
    l'utilisateur décrit."""
    df = _make_imbalanced_classification_df()
    split = split_dataset(df, "cible", ["x1", "x2"], "classification", None, 0.2, 42)

    cfg_off = TrainingConfig(optuna_trials=3, cv_folds=3, class_rebalancing=False)
    cfg_on = TrainingConfig(optuna_trials=3, cv_folds=3, class_rebalancing=True)
    result_off = train_and_evaluate(split, "classification", cfg_off, lambda s, p: None)
    result_on = train_and_evaluate(split, "classification", cfg_on, lambda s, p: None)

    assert result_off.model_card["class_rebalancing_applied"] is False
    assert result_on.model_card["class_rebalancing_applied"] is True
    assert _minority_class_recall(result_on.evaluation) > _minority_class_recall(result_off.evaluation)
