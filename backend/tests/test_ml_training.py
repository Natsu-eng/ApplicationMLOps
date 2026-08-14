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

    # Beeswarm SHAP (Lot Explicabilité globale) : même bug potentiel que le
    # barres (forme 3D vs liste selon la version SHAP) — une série par
    # classe, jamais une seule "global" trompeuse en multiclasse.
    assert set(result.shap_beeswarm.keys()) == {"0", "1", "2"}
    for points in result.shap_beeswarm.values():
        assert points
        assert {"feature", "feature_value", "shap_value"} <= points[0].keys()


def test_group_anti_leak_split_reflected_in_model_card():
    rng = np.random.default_rng(3)
    n = 150
    df = pd.DataFrame(
        {"x": rng.normal(size=n), "groupe": rng.integers(0, 30, n), "cible": rng.normal(size=n)}
    )
    split = split_dataset(df, "cible", ["x"], "regression", "groupe", 0.2, 42)
    result = train_and_evaluate(split, "regression", _FAST_CONFIG, lambda s, p: None)
    assert result.model_card["anti_leak_grouping"] is True


def test_group_split_unseen_class_raises_actionable_error():
    """H8 (AUDIT_ROADMAP.md) — GroupShuffleSplit (split anti-fuite par
    groupe) ne stratifie pas : une classe rare concentrée dans un seul
    groupe peut atterrir entièrement en test, absente du train. Avant ce
    correctif, `LabelEncoder.transform` levait une `ValueError` sklearn
    brute ("y contains previously unseen labels") au lieu d'un message
    diagnosticable. Seed=0 choisi empiriquement pour reproduire le cas."""
    n_a, n_b = 100, 5
    df = pd.DataFrame(
        {
            "x": np.arange(n_a + n_b, dtype=float),
            "groupe": ["A"] * n_a + ["B"] * n_b,
            "cible": ["frequent"] * n_a + ["rare"] * n_b,
        }
    )
    split = split_dataset(df, "cible", ["x"], "classification", "groupe", 0.2, 0)

    with pytest.raises(RuntimeError, match="rare"):
        train_and_evaluate(split, "classification", _FAST_CONFIG, lambda s, p: None)


# ── H6 (AUDIT_ROADMAP.md) — le seed utilisateur doit varier les folds ──


def test_make_cv_folds_vary_with_seed():
    """Avant correctif, `_make_cv` hardcodait `random_state=42` : changer le
    seed de l'utilisateur ne faisait jamais varier les folds de CV. Preuve
    directe sur un vrai `KFold`/`StratifiedKFold` (pas un mock) : deux seeds
    différents doivent produire des découpages différents, un même seed doit
    rester reproductible."""
    X = np.arange(30).reshape(-1, 1)
    y = np.array([0, 1] * 15)

    cv_a1 = list(_make_cv("classification", 3, None, seed=1).split(X, y))
    cv_a2 = list(_make_cv("classification", 3, None, seed=1).split(X, y))
    cv_b = list(_make_cv("classification", 3, None, seed=2).split(X, y))

    # Même seed → mêmes folds (reproductibilité).
    for (train_a1, test_a1), (train_a2, test_a2) in zip(cv_a1, cv_a2):
        assert np.array_equal(train_a1, train_a2)
        assert np.array_equal(test_a1, test_a2)

    # Seed différent → au moins un fold différent.
    assert any(
        not np.array_equal(test_a1, test_b) for (_, test_a1), (_, test_b) in zip(cv_a1, cv_b)
    )


def test_make_cv_groupkfold_ignores_seed_deterministically():
    """GroupKFold n'accepte pas `random_state` (pas de `shuffle`) — doit
    rester construit sans erreur quel que soit le seed, et déterministe."""
    groups = np.array([0, 0, 1, 1, 2, 2])
    cv = _make_cv("regression", 3, groups, seed=7)
    assert cv.__class__.__name__ == "GroupKFold"


# ── Lot A — non-fuite préprocesseur/CV et calibration CQR groupée ──


def test_cv_estimator_is_pipeline_with_preprocessor_first_step(monkeypatch):
    """Preuve structurelle : l'estimateur passé à cross_validate est un
    Pipeline dont la 1re étape est le préprocesseur — garantit qu'il est
    cloné et refit à l'intérieur de chaque fold, jamais fit en amont sur
    tout le train (Lot A, fuite #1)."""
    captured: dict = {}

    def fake_cross_validate(estimator, X, y, cv=None, groups=None, scoring=None, n_jobs=None, fit_params=None):
        captured["estimator"] = estimator
        return {"test_score": np.array([0.5, 0.5, 0.5])}

    monkeypatch.setattr(ml_training_module, "cross_validate", fake_cross_validate)

    df = _make_regression_df()
    split = split_dataset(df, "cible", ["x1", "x2"], "regression", None, 0.2, 42)
    y_train = split.y_train.to_numpy(dtype=float)
    preprocessor_template = build_preprocessor(split.X_train)
    cv = _make_cv("regression", 3, None, 42)
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
    """Preuve que le Pipeline passé à cross_validate (donc cloné/refit par
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
    cv = _make_cv("regression", 3, None, 42)
    config = TrainingConfig(optuna_trials=1, cv_folds=3)

    captured: dict = {}

    def fake_cross_validate(estimator, X, y, cv=None, groups=None, scoring=None, n_jobs=None, fit_params=None):
        captured["estimator"] = estimator
        return {"test_score": np.array([0.5, 0.5, 0.5])}

    monkeypatch.setattr(ml_training_module, "cross_validate", fake_cross_validate)

    spec = MODEL_REGISTRY["lightgbm"]
    _optimize_one_model(
        spec, split.X_train, split.y_train.to_numpy(dtype=float),
        "regression", cv, None, preprocessor_template, config, lambda s, p: None, 0, 10,
    )

    estimator = captured["estimator"]
    # `.transformers` (pas `.transformers_`, réservé au ColumnTransformer fitté) :
    # le Pipeline capturé n'a pas encore été fit, cross_validate étant mocké.
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

    summary, beeswarm = _compute_shap_summary(model, "kernel", X, X[:10], ["f1", "f2", "f3"], None)
    assert len(summary) == 3
    assert all(isinstance(entry["importance"], float) for entry in summary)
    # Beeswarm (Lot Explicabilité globale) — classe positive seule en binaire
    # ("global"), un point par observation expliquée pour chaque variable.
    assert set(beeswarm.keys()) == {"global"}
    assert len(beeswarm["global"]) == 10 * 3


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

    shap_summary, shap_beeswarm, status = _compute_explainability(
        _DummyModel(), "kernel",
        np.zeros((10, n_features)), np.zeros((10, n_features)),
        feature_names, None, TrainingConfig(),
    )
    assert shap_summary == []
    assert shap_beeswarm == {}
    assert status["status"] == "degraded"
    assert status["message"]  # message non vide, en langage clair (pas de jargon brut)


def test_explainability_degrades_on_unexpected_shap_failure(monkeypatch):
    """Filet de sécurité générique : si le calcul SHAP échoue pour une raison
    imprévue, l'entraînement ne doit pas planter — dégradation avec message,
    pas d'exception qui remonte à l'appelant."""

    def _boom(*args, **kwargs):
        raise RuntimeError("échec simulé")

    monkeypatch.setattr(ml_training_module, "_compute_shap_summary", _boom)

    shap_summary, shap_beeswarm, status = ml_training_module._compute_explainability(
        object(), "tree", np.zeros((5, 2)), np.zeros((5, 2)), ["f1", "f2"], None, TrainingConfig(),
    )
    assert shap_summary == []
    assert shap_beeswarm == {}
    assert status["status"] == "degraded"
    assert status["message"]


def test_model_card_carries_ok_explainability_status_for_tree_models():
    """Non-régression : les boosters (routage 'tree') restent au statut 'ok'
    — le chantier SHAP par famille ne dégrade pas ce qui marchait déjà."""
    df = _make_regression_df()
    split = split_dataset(df, "cible", ["x1", "x2"], "regression", None, 0.2, 42)
    result = train_and_evaluate(split, "regression", _FAST_CONFIG, lambda s, p: None)
    assert result.model_card["explainability"] == {"status": "ok", "message": None}


# ── Lot Explicabilité globale — beeswarm/permutation/calibration/learning curve ──


def test_diagnostic_fields_populated_end_to_end_for_regression():
    """Régression : permutation + courbe d'apprentissage 'ok', calibration
    explicitement None (non applicable — pas un statut 'degraded', ce n'est
    juste pas pertinent pour une tâche sans probabilités)."""
    df = _make_regression_df()
    split = split_dataset(df, "cible", ["x1", "x2"], "regression", None, 0.2, 42)
    result = train_and_evaluate(split, "regression", _FAST_CONFIG, lambda s, p: None)

    assert result.model_card["permutation_importance_status"] == {"status": "ok", "message": None}
    assert result.model_card["learning_curve_status"] == {"status": "ok", "message": None}
    assert result.model_card["calibration_status"] is None
    assert result.calibration == {}
    assert result.permutation_importance
    assert all(isinstance(f["importance_mean"], float) for f in result.permutation_importance)
    assert result.learning_curve["train_sizes"]
    assert len(result.learning_curve["train_sizes"]) == len(result.learning_curve["val_scores_mean"])
    # Catalogue par défaut = boosters/RandomForest → routage "tree" → jamais vide.
    assert result.shap_beeswarm


def test_diagnostic_fields_populated_end_to_end_for_classification():
    """Classification : les 4 diagnostics sont 'ok', calibration non vide
    cette fois (contrairement à la régression)."""
    df = _make_binary_df()
    split = split_dataset(df, "cible", ["x1", "x2"], "classification", None, 0.2, 42)
    result = train_and_evaluate(split, "classification", _FAST_CONFIG, lambda s, p: None)

    assert result.model_card["permutation_importance_status"] == {"status": "ok", "message": None}
    assert result.model_card["learning_curve_status"] == {"status": "ok", "message": None}
    assert result.model_card["calibration_status"] == {"status": "ok", "message": None}
    assert result.calibration
    for curve in result.calibration.values():
        assert curve["mean_predicted"]
        assert len(curve["mean_predicted"]) == len(curve["fraction_positive"])
    assert result.permutation_importance
    assert result.learning_curve["train_sizes"]
    assert result.shap_beeswarm


def test_compute_permutation_importance_degrades_on_failure():
    """Même filet que `_compute_explainability` (Lot 5) : un estimateur
    incompatible ne fait jamais planter l'entraînement, juste dégrader avec
    un message clair."""
    from services.ml_training import _compute_permutation_importance

    summary, status = _compute_permutation_importance(
        object(), np.zeros((10, 2)), np.zeros(10), ["f1", "f2"], seed=0
    )
    assert summary == []
    assert status["status"] == "degraded"
    assert status["message"]


def test_compute_permutation_importance_survives_catboost_readonly_side_effect():
    """Bug réel trouvé en test end-to-end (worker, CatBoost gagnant sur un
    dataset à une seule variable numérique) : `CatBoostRegressor.predict()`
    passe le tableau numpy à son `Pool` interne SANS copie et le marque
    lui-même en lecture seule comme effet de bord (comportement propre à
    CatBoost, reproduit ici indépendamment du pipeline complet) —
    `permutation_importance` réutilise le MÊME tableau sur `n_repeats`
    répétitions : la 1re réussit, la 2e lève `ValueError: assignment
    destination is read-only`, quel que soit l'état d'origine du tableau
    passé (recopier en amont ne suffisait donc pas). Corrigé en passant un
    DataFrame à `permutation_importance` (réaffectation de colonne côté
    pandas, jamais d'écriture in-place dans le buffer verrouillé par
    CatBoost) — voir le commentaire dans `_compute_permutation_importance`."""
    from catboost import CatBoostRegressor

    from services.ml_training import _compute_permutation_importance

    rng = np.random.default_rng(4)
    n = 100
    X = rng.normal(size=(n, 1))
    y = X[:, 0] * 2 + rng.normal(0, 0.1, n)
    model = CatBoostRegressor(verbose=False, random_state=42, iterations=50).fit(X, y)

    # Sanity check de la prémisse du bug — CatBoost verrouille bien le
    # tableau qu'on lui passe en predict() (sinon ce test ne prouve rien).
    probe = X.copy()
    model.predict(probe)
    assert not probe.flags.writeable

    summary, status = _compute_permutation_importance(model, X, y, ["x"], seed=0)
    assert status == {"status": "ok", "message": None}
    assert len(summary) == 1


def test_compute_calibration_binary_produces_one_curve():
    """Binaire : une seule courbe (classe positive), pas une redondante par
    classe — même convention que le beeswarm SHAP (Lot Explicabilité
    globale) pour la même raison (symétrie des deux sorties)."""
    from services.ml_training import _compute_calibration

    rng = np.random.default_rng(0)
    n = 300
    y_test = rng.integers(0, 2, n)
    # Probabilités corrélées à y_test — évite un jeu dégénéré pour calibration_curve.
    p_pos = np.clip(y_test * 0.6 + rng.uniform(0, 0.35, n), 0.01, 0.99)
    proba_test = np.column_stack([1 - p_pos, p_pos])

    curves, status = _compute_calibration(y_test, proba_test, ["neg", "pos"])
    assert status["status"] == "ok"
    assert set(curves.keys()) == {"pos"}
    assert len(curves["pos"]["mean_predicted"]) == len(curves["pos"]["fraction_positive"])
    assert len(curves["pos"]["mean_predicted"]) > 0


def test_compute_calibration_multiclass_produces_one_curve_per_present_class():
    """Multiclasse : une courbe par classe (un-contre-tous), même motif que
    les courbes ROC/PR multiclasses déjà affichées (Lot E1-ter)."""
    from services.ml_training import _compute_calibration

    rng = np.random.default_rng(1)
    n = 300
    y_test = rng.integers(0, 3, n)
    class_names = ["a", "b", "c"]
    # Probabilités corrélées à la vraie classe (pic sur la bonne colonne + bruit).
    proba_test = np.zeros((n, 3))
    for i, cls in enumerate(y_test):
        proba_test[i] = rng.uniform(0, 0.2, 3)
        proba_test[i, cls] += 0.7
    proba_test = proba_test / proba_test.sum(axis=1, keepdims=True)

    curves, status = _compute_calibration(y_test, proba_test, class_names)
    assert status["status"] == "ok"
    assert set(curves.keys()) == {"a", "b", "c"}


def test_compute_calibration_degrades_on_failure(monkeypatch):
    """Filet générique — un échec inattendu du binning ne doit jamais faire
    planter l'entraînement (même motif que SHAP/permutation)."""

    def _boom(*args, **kwargs):
        raise RuntimeError("échec simulé")

    monkeypatch.setattr(ml_training_module, "calibration_curve", _boom)

    curves, status = ml_training_module._compute_calibration(
        np.array([0, 1, 0, 1]), np.array([[0.6, 0.4], [0.3, 0.7], [0.8, 0.2], [0.2, 0.8]]), ["neg", "pos"],
    )
    assert curves == {}
    assert status["status"] == "degraded"
    assert status["message"]


def test_learning_curve_pipeline_is_never_prefit(monkeypatch):
    """Preuve structurelle anti-fuite (même motif que
    `test_cv_estimator_is_pipeline_with_preprocessor_first_step`, Lot A) : la
    courbe d'apprentissage passe par un Pipeline (préprocesseur cloné, jamais
    déjà fit) et des données BRUTES (pas encore préprocessées) à
    `learning_curve` — sklearn refit ce pipeline dans chaque fold/taille,
    jamais sur le train complet déjà vu par le modèle final."""
    from sklearn.linear_model import Ridge

    from services.ml_training import _LEARNING_CURVE_TRAIN_SIZES, _compute_learning_curve

    captured: dict = {}

    def fake_learning_curve(estimator, X, y, cv=None, groups=None, train_sizes=None, scoring=None, n_jobs=None, random_state=None):
        captured["estimator"] = estimator
        captured["X"] = X
        captured["train_sizes"] = train_sizes
        n = len(train_sizes)
        return (
            np.array([10] * n),
            np.full((n, 3), 0.8),
            np.full((n, 3), 0.75),
        )

    monkeypatch.setattr(ml_training_module, "learning_curve", fake_learning_curve)

    df = _make_regression_df()
    split = split_dataset(df, "cible", ["x1", "x2"], "regression", None, 0.2, 42)
    y_train = split.y_train.to_numpy(dtype=float)
    preprocessor_template = build_preprocessor(split.X_train)
    cv = _make_cv("regression", 3, None, 42)
    config = TrainingConfig(optuna_trials=1, cv_folds=3)

    result, status = _compute_learning_curve(
        preprocessor_template, Ridge(alpha=1.0), split.X_train, y_train, cv, None, "regression", config,
    )

    assert status["status"] == "ok"
    estimator = captured["estimator"]
    assert isinstance(estimator, Pipeline)
    assert estimator.steps[0][0] == "preprocess"
    assert isinstance(estimator.named_steps["preprocess"], ColumnTransformer)
    assert isinstance(captured["X"], pd.DataFrame)  # jamais déjà préprocessé sur tout le train
    np.testing.assert_array_equal(captured["train_sizes"], _LEARNING_CURVE_TRAIN_SIZES)
    assert result["train_sizes"] == [10] * len(_LEARNING_CURVE_TRAIN_SIZES)


def test_learning_curve_degrades_on_failure(monkeypatch):
    """Même filet que les autres diagnostics du lot : un échec du fit ne
    doit jamais remonter à l'appelant."""
    from services.ml_training import _compute_learning_curve

    def _boom(*args, **kwargs):
        raise RuntimeError("échec simulé")

    monkeypatch.setattr(ml_training_module, "learning_curve", _boom)

    df = _make_regression_df()
    split = split_dataset(df, "cible", ["x1", "x2"], "regression", None, 0.2, 42)
    y_train = split.y_train.to_numpy(dtype=float)
    preprocessor_template = build_preprocessor(split.X_train)
    cv = _make_cv("regression", 3, None, 42)
    config = TrainingConfig(optuna_trials=1, cv_folds=3)

    from sklearn.linear_model import Ridge

    result, status = _compute_learning_curve(
        preprocessor_template, Ridge(alpha=1.0), split.X_train, y_train, cv, None, "regression", config,
    )
    assert result == {}
    assert status["status"] == "degraded"
    assert status["message"]


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
    `cross_validate` n'est pas spécifique aux arbres — le scaler de ces
    modèles est fit DANS le fold, jamais en amont (Lot 5, non-régression de
    l'anti-fuite Lot A pour les nouveaux modèles)."""
    captured: dict = {}

    def fake_cross_validate(estimator, X, y, cv=None, groups=None, scoring=None, n_jobs=None, fit_params=None):
        captured["estimator"] = estimator
        return {"test_score": np.array([0.7, 0.7, 0.7])}

    monkeypatch.setattr(ml_training_module, "cross_validate", fake_cross_validate)

    df = _make_regression_df()
    split = split_dataset(df, "cible", ["x1", "x2"], "regression", None, 0.2, 42)
    y_train = split.y_train.to_numpy(dtype=float)
    preprocessor_template = build_preprocessor(split.X_train)
    cv = _make_cv("regression", 3, None, 42)
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


def test_model_ids_restricts_catalog_to_explicit_selection(monkeypatch):
    """Mode expert (Lot E2) : `TrainingConfig.model_ids` remplace le
    sous-ensemble par défaut par une sélection explicite du catalogue complet
    (`subset="all"`) — ici un seul modèle, hors du sous-ensemble par défaut,
    pour vérifier que ce n'est pas juste un filtre no-op sur "default"."""
    called_ids: list[str] = []
    original = ml_training_module._optimize_one_model

    def _tracking(spec, *args, **kwargs):
        called_ids.append(spec.id)
        return original(spec, *args, **kwargs)

    monkeypatch.setattr(ml_training_module, "_optimize_one_model", _tracking)

    df = _make_regression_df()
    split = split_dataset(df, "cible", ["x1", "x2"], "regression", None, 0.2, 42)
    config = TrainingConfig(optuna_trials=3, cv_folds=3, model_ids=["extra_trees"])
    train_and_evaluate(split, "regression", config, lambda s, p: None)

    assert called_ids == ["extra_trees"]


def test_model_ids_empty_after_filtering_falls_back_to_default_subset(monkeypatch):
    """Garde-fou défensif : si `model_ids` ne désigne aucun modèle compatible
    avec la tâche (ne devrait jamais arriver, l'API filtre déjà — voir
    `routers/training.py`), le moteur retombe sur le sous-ensemble par défaut
    plutôt que de comparer un catalogue vide."""
    from services.ml_registry import models_for_task

    called_ids: list[str] = []
    original = ml_training_module._optimize_one_model

    def _tracking(spec, *args, **kwargs):
        called_ids.append(spec.id)
        return original(spec, *args, **kwargs)

    monkeypatch.setattr(ml_training_module, "_optimize_one_model", _tracking)

    df = _make_regression_df()
    split = split_dataset(df, "cible", ["x1", "x2"], "regression", None, 0.2, 42)
    config = TrainingConfig(optuna_trials=3, cv_folds=3, model_ids=["id_inexistant"])
    train_and_evaluate(split, "regression", config, lambda s, p: None)

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
    `model__sample_weight` de `cross_validate` pour un modèle qui le
    supporte (LightGBM)."""
    captured: dict = {}

    def fake_cross_validate(estimator, X, y, cv=None, groups=None, scoring=None, n_jobs=None, fit_params=None):
        captured["fit_params"] = fit_params
        return {"test_score": np.array([0.5, 0.5, 0.5])}

    monkeypatch.setattr(ml_training_module, "cross_validate", fake_cross_validate)

    df = _make_multiclass_df()
    split = split_dataset(df, "cible", ["f1", "f2"], "classification", None, 0.2, 42)
    n_train = len(split.X_train)
    rng = np.random.default_rng(0)
    y_train = rng.integers(0, 3, n_train)
    sample_weight = rng.uniform(0.5, 1.5, n_train)
    preprocessor_template = build_preprocessor(split.X_train)
    cv = _make_cv("classification", 3, None, 42)
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

    def fake_cross_validate(estimator, X, y, cv=None, groups=None, scoring=None, n_jobs=None, fit_params=None):
        captured["fit_params"] = fit_params
        return {"test_score": np.array([0.5, 0.5, 0.5])}

    monkeypatch.setattr(ml_training_module, "cross_validate", fake_cross_validate)

    df = _make_multiclass_df()
    split = split_dataset(df, "cible", ["f1", "f2"], "classification", None, 0.2, 42)
    n_train = len(split.X_train)
    rng = np.random.default_rng(0)
    y_train = rng.integers(0, 3, n_train)
    sample_weight = rng.uniform(0.5, 1.5, n_train)
    preprocessor_template = build_preprocessor(split.X_train)
    cv = _make_cv("classification", 3, None, 42)
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


# ── Lot D — leaderboard : tous les candidats, score de sélection ───────────


def test_optimize_one_model_captures_fold_scores_without_retraining():
    """Option A du cadrage Lot D : la variance inter-folds est récupérée
    pendant la MÊME recherche Optuna déjà en cours (`cross_validate` calcule
    un score par fold de toute façon ; on le garde via `trial.set_user_attr`
    au lieu de ne conserver que la moyenne) — aucun ré-entraînement
    supplémentaire. `fold_scores` porte un score par fold pour l'essai
    gagnant, cohérent avec `cv_score` (leur moyenne)."""
    df = _make_regression_df()
    split = split_dataset(df, "cible", ["x1", "x2"], "regression", None, 0.2, 42)
    y_train = split.y_train.to_numpy(dtype=float)
    preprocessor_template = build_preprocessor(split.X_train)
    cv = _make_cv("regression", 3, None, 42)
    config = TrainingConfig(optuna_trials=3, cv_folds=3)

    spec = MODEL_REGISTRY["lightgbm"]
    opt = _optimize_one_model(
        spec, split.X_train, y_train, "regression", cv, split.groups_train,
        preprocessor_template, config, lambda s, p: None, 0, 10,
    )
    assert opt.fold_scores is not None
    assert len(opt.fold_scores) == 3
    assert np.mean(opt.fold_scores) == pytest.approx(opt.cv_score, abs=1e-6)


def test_optimize_one_model_captures_regression_error_metric():
    """Lot D précision 1 : le R² seul n'est pas lisible pour un BE — une
    erreur en unité réelle (RMSE, validation croisée) est capturée par le
    même `cross_validate` (second scorer évalué sur les prédictions déjà
    produites par le fit de chaque fold), sans fit supplémentaire."""
    df = _make_regression_df()
    split = split_dataset(df, "cible", ["x1", "x2"], "regression", None, 0.2, 42)
    y_train = split.y_train.to_numpy(dtype=float)
    preprocessor_template = build_preprocessor(split.X_train)
    cv = _make_cv("regression", 3, None, 42)
    config = TrainingConfig(optuna_trials=3, cv_folds=3)

    spec = MODEL_REGISTRY["lightgbm"]
    opt = _optimize_one_model(
        spec, split.X_train, y_train, "regression", cv, split.groups_train,
        preprocessor_template, config, lambda s, p: None, 0, 10,
    )
    assert opt.fold_error is not None
    assert opt.fold_error > 0  # RMSE positif, dans l'unité de la cible


def test_optimize_one_model_classification_has_no_secondary_metric():
    """Le second scorer (erreur en unité réelle) est spécifique à la
    régression (cadrage Lot D) — jamais calculé en classification, où le
    score de sélection (AUC) n'a pas d'équivalent "erreur physique"."""
    df = _make_multiclass_df()
    split = split_dataset(df, "cible", ["f1", "f2"], "classification", None, 0.2, 42)
    y_train = split.y_train.to_numpy()
    preprocessor_template = build_preprocessor(split.X_train)
    cv = _make_cv("classification", 3, None, 42)
    config = TrainingConfig(optuna_trials=3, cv_folds=3)

    spec = MODEL_REGISTRY["lightgbm"]
    opt = _optimize_one_model(
        spec, split.X_train, y_train, "classification", cv, split.groups_train,
        preprocessor_template, config, lambda s, p: None, 0, 10,
    )
    assert opt.fold_error is None
    assert opt.fold_scores is not None and len(opt.fold_scores) == 3


def test_all_candidates_length_matches_default_catalog_and_exactly_one_winner():
    """TOUS les candidats du sous-ensemble par défaut sont exposés (pas
    seulement le gagnant) — condition de base du leaderboard (Lot D)."""
    from services.ml_registry import models_for_task

    df = _make_regression_df()
    split = split_dataset(df, "cible", ["x1", "x2"], "regression", None, 0.2, 42)
    result = train_and_evaluate(split, "regression", _FAST_CONFIG, lambda s, p: None)

    expected_labels = {spec.label("regression") for spec in models_for_task("regression", "default")}
    assert {c["algorithm"] for c in result.all_candidates} == expected_labels
    assert sum(c["is_winner"] for c in result.all_candidates) == 1
    winner_row = [c for c in result.all_candidates if c["is_winner"]][0]
    assert winner_row["algorithm"] == result.algorithm
    assert winner_row["selection_score"] == pytest.approx(result.metrics["cv_score"])


def test_all_candidates_ranked_by_selection_score_not_accuracy(monkeypatch):
    """Preuve que le classement du leaderboard (Lot D) suit le score de
    sélection (cv_score, AUC-based en classification) et non l'accuracy —
    scénario réaliste sur dataset déséquilibré : un modèle qui prédirait
    surtout la classe majoritaire afficherait une bonne accuracy mais un
    score de sélection médiocre (AUC proche du hasard). On force cet écart
    (cv_score fictif) pour prouver que c'est bien lui qui pilote le rang, et
    non une autre métrique qu'on aurait pu être tenté d'utiliser."""
    df = _make_imbalanced_classification_df()
    split = split_dataset(df, "cible", ["x1", "x2"], "classification", None, 0.2, 42)

    fake_catalog = [MODEL_REGISTRY["lightgbm"], MODEL_REGISTRY["random_forest"]]
    monkeypatch.setattr(ml_training_module, "models_for_task", lambda task_type, subset="all": fake_catalog)

    # lightgbm "perd" sur le score de sélection alors qu'un classement par
    # accuracy brute l'aurait placé devant (cas réaliste sur un dataset
    # déséquilibré : accuracy haute grâce à la classe majoritaire, AUC basse
    # car il ne distingue pas la classe rare) — random_forest gagne sur le
    # score de sélection, la seule métrique qui doit compter ici.
    fake_scores = {"lightgbm": 0.55, "random_forest": 0.91}
    original = ml_training_module._optimize_one_model

    def fake_optimize(spec, *args, **kwargs):
        opt = original(spec, *args, **kwargs)
        opt.cv_score = fake_scores[spec.id]
        return opt

    monkeypatch.setattr(ml_training_module, "_optimize_one_model", fake_optimize)

    result = train_and_evaluate(split, "classification", _FAST_CONFIG, lambda s, p: None)

    assert result.algorithm == "Forêt aléatoire (Random Forest)"
    ranks = {c["algorithm"]: c["rank"] for c in result.all_candidates}
    assert ranks["Forêt aléatoire (Random Forest)"] == 1
    assert ranks["LightGBM"] == 2
    winners = [c for c in result.all_candidates if c["is_winner"]]
    assert len(winners) == 1
    assert winners[0]["algorithm"] == "Forêt aléatoire (Random Forest)"


# ── Fix "bad allocation" — sparse préservé jusqu'au fit/SHAP ────────────────


def _make_high_cardinality_df(n=300, seed=17):
    """Analogue synthétique, sûr à faire tourner en test, du dataset réel
    ayant déclenché le diagnostic : une colonne quasi-identifiant (cardinalité
    ≈ nombre de lignes) qui, densifiée par un one-hot, exploserait en
    centaines de colonnes — ici bornée à une taille qui reste rapide à tester
    tout en restant représentative (n colonnes one-hot ≈ n lignes)."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    identifiant = np.array([f"ID{i:05d}" for i in range(n)])  # cardinalité = n (quasi-unique)
    y = (x1 + x2 > 0).astype(int)
    return pd.DataFrame({"identifiant": identifiant, "x1": x1, "x2": x2, "cible": y})


def test_high_cardinality_categorical_trains_without_densifying(monkeypatch):
    """Reproduit, en sûr et rapide, le scénario du diagnostic "bad
    allocation" : une colonne quasi-identifiant one-hotée ne doit JAMAIS
    déclencher de densification (`.todense()`) nulle part dans le pipeline
    pour le catalogue par défaut (100% famille "tree") — c'est la
    densification, pas la cardinalité en elle-même, qui faisait exploser la
    mémoire. Généralisé à N'IMPORTE QUEL dataset avec une colonne à
    cardinalité élevée, pas seulement celui du diagnostic."""
    import scipy.sparse as sp

    calls = {"count": 0}
    original_todense = sp.csr_matrix.todense

    def _tracking_todense(self, *args, **kwargs):
        calls["count"] += 1
        return original_todense(self, *args, **kwargs)

    monkeypatch.setattr(sp.csr_matrix, "todense", _tracking_todense)

    df = _make_high_cardinality_df()
    split = split_dataset(df, "cible", ["identifiant", "x1", "x2"], "classification", None, 0.2, 42)
    result = train_and_evaluate(split, "classification", _FAST_CONFIG, lambda s, p: None)

    assert calls["count"] == 0, "todense() appelé — une matrice a été densifiée inutilement (régression du fix)"
    assert result.algorithm  # l'entraînement a bien produit un résultat exploitable
    assert len(result.shap_summary) > 0  # SHAP calculé malgré l'entrée sparse (TreeExplainer)


def test_high_cardinality_regression_with_cqr_trains_without_densifying(monkeypatch):
    """Même preuve que ci-dessus, en régression — inclut le CQR (régresseurs
    de quantile LightGBM dédiés), deuxième point de densification supprimé
    par le fix."""
    import scipy.sparse as sp

    calls = {"count": 0}
    original_todense = sp.csr_matrix.todense

    def _tracking_todense(self, *args, **kwargs):
        calls["count"] += 1
        return original_todense(self, *args, **kwargs)

    monkeypatch.setattr(sp.csr_matrix, "todense", _tracking_todense)

    rng = np.random.default_rng(23)
    n = 300
    x1 = rng.normal(size=n)
    identifiant = np.array([f"ID{i:05d}" for i in range(n)])
    y = 3 * x1 + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"identifiant": identifiant, "x1": x1, "cible": y})

    split = split_dataset(df, "cible", ["identifiant", "x1"], "regression", None, 0.2, 42)
    result = train_and_evaluate(split, "regression", _FAST_CONFIG, lambda s, p: None)

    assert calls["count"] == 0, "todense() appelé — une matrice a été densifiée inutilement (régression du fix)"
    assert result.cqr is not None
    assert 0 <= result.cqr["empirical_coverage"] <= 1


def test_linear_explainer_still_works_with_sparse_upstream_input():
    """LinearExplainer exige un fond dense (SHAP) — vérifie que le fix (sparse
    conservé en amont, densifié seulement ici, borné) ne casse pas
    l'explicabilité pour la famille linéaire (mode expert futur, Lot E)."""
    from services.ml_training import _compute_explainability
    from services.ml_registry import MODEL_REGISTRY
    from sklearn.linear_model import Ridge

    df = _make_high_cardinality_df()
    df["cible"] = df["x1"] * 2 - df["x2"] + np.random.default_rng(1).normal(0, 0.1, len(df))
    split = split_dataset(df, "cible", ["identifiant", "x1", "x2"], "regression", None, 0.2, 42)

    preprocessor = build_preprocessor(split.X_train)
    X_train_proc = preprocessor.fit_transform(split.X_train)
    X_test_proc = preprocessor.transform(split.X_test)
    assert hasattr(X_train_proc, "todense")  # bien sparse en amont, condition du test

    model = Ridge(alpha=0.1, random_state=42).fit(X_train_proc, split.y_train.to_numpy(dtype=float))
    feature_names = list(preprocessor.get_feature_names_out())
    config = TrainingConfig(shap_sample_size=20)

    summary, beeswarm, status = _compute_explainability(
        model, "linear", X_train_proc, X_test_proc, feature_names, None, config
    )
    assert status["status"] == "ok"
    assert len(summary) == len(feature_names)
    assert set(beeswarm.keys()) == {"global"}
