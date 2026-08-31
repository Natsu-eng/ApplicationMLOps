"""Tests du script de déploiement autonome (retour utilisateur direct :
"tous les modèles ML — supervisé, non supervisé, vision — doivent pouvoir
être déployés dans d'autres plateformes par la suite") — vérifie que le
script généré est non seulement syntaxiquement valide, mais RÉELLEMENT
exécutable en dehors de ce projet (sous-processus séparé, aucun import
`domains.*`) et reproduit fidèlement `predict_one`/`predict_batch`."""
from __future__ import annotations

import ast
import json
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

from domains.shared.ml_preprocessing import split_dataset
from domains.training.services.deployment_export import detect_model_library, generate_deployment_script
from domains.training.services.engine import TrainingConfig, train_and_evaluate
from domains.training.services.inference import predict_one

_FAST_CONFIG = TrainingConfig(optuna_trials=3, cv_folds=3, model_ids=["lightgbm"])


def _regression_bundle():
    rng = np.random.default_rng(42)
    n = 150
    df = pd.DataFrame({"x1": rng.normal(50, 10, n), "x2": rng.normal(20, 5, n)})
    df["cible"] = 2.5 * df["x1"] - 1.2 * df["x2"] + rng.normal(0, 3, n)
    split = split_dataset(df, "cible", ["x1", "x2"], "regression", None, 0.2, 42)
    result = train_and_evaluate(split, "regression", _FAST_CONFIG, lambda s, p: None)
    return result.algorithm, result.pipeline_bundle


def _classification_bundle():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"x1": rng.normal(0, 1, n), "x2": rng.normal(0, 1, n)})
    df["cible"] = np.where(df["x1"] + df["x2"] > 0, "a", "b")
    split = split_dataset(df, "cible", ["x1", "x2"], "classification", None, 0.2, 42)
    result = train_and_evaluate(split, "classification", _FAST_CONFIG, lambda s, p: None)
    return result.algorithm, result.pipeline_bundle


def test_generated_script_is_valid_python_for_every_task_type():
    for algorithm, bundle in (_regression_bundle(), _classification_bundle()):
        script = generate_deployment_script(
            bundle=bundle, feature_columns=["x1", "x2"], algorithm=algorithm, task_type=bundle["task_type"],
            target_column="cible", artifact_filename="m.joblib", script_filename="m_deploiement.py",
        )
        ast.parse(script)  # lève SyntaxError si invalide


def test_detect_model_library_matches_the_real_lightgbm_module():
    _, bundle = _regression_bundle()
    name, pip_name = detect_model_library(bundle["model"])
    assert name == "LightGBM"
    assert pip_name == "lightgbm"


@pytest.mark.parametrize("bundle_factory", [_regression_bundle, _classification_bundle])
def test_standalone_script_reproduces_predict_one_in_a_real_subprocess(bundle_factory, tmp_path):
    """Le test le plus important de ce fichier — exécute le script généré
    dans un VRAI sous-processus Python séparé (pas un import direct, qui
    tricherait en réutilisant l'environnement/les modules déjà chargés de ce
    projet) et compare son résultat à `predict_one`, sur la MÊME observation
    et le MÊME bundle. Preuve directe de déployabilité hors DataLab Pro."""
    algorithm, bundle = bundle_factory()
    task_type = bundle["task_type"]

    artifact_path = tmp_path / "m.joblib"
    joblib.dump(bundle, artifact_path)
    script = generate_deployment_script(
        bundle=bundle, feature_columns=["x1", "x2"], algorithm=algorithm, task_type=task_type,
        target_column="cible", artifact_filename="m.joblib", script_filename="m_deploiement.py",
    )
    script_path = tmp_path / "m_deploiement.py"
    script_path.write_text(script, encoding="utf-8")

    row = {"x1": 50.0, "x2": 20.0}
    proc = subprocess.run(
        [sys.executable, str(script_path), "--predict", json.dumps(row)],
        capture_output=True, text=True, cwd=tmp_path, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    standalone_result = json.loads(proc.stdout)[0]

    reference = predict_one(bundle, ["x1", "x2"], row)

    if task_type == "classification":
        assert standalone_result["prediction"] == reference["prediction"]
    else:
        assert standalone_result["prediction"] == pytest.approx(reference["prediction"], abs=1e-6)
        if "interval" in reference and reference["interval"]:
            assert standalone_result["intervalle_bas"] == pytest.approx(reference["interval"]["low"], abs=1e-6)
            assert standalone_result["intervalle_haut"] == pytest.approx(reference["interval"]["high"], abs=1e-6)


def test_standalone_script_batch_mode_matches_row_count(tmp_path):
    algorithm, bundle = _regression_bundle()
    artifact_path = tmp_path / "m.joblib"
    joblib.dump(bundle, artifact_path)
    script = generate_deployment_script(
        bundle=bundle, feature_columns=["x1", "x2"], algorithm=algorithm, task_type="regression",
        target_column="cible", artifact_filename="m.joblib", script_filename="m_deploiement.py",
    )
    (tmp_path / "m_deploiement.py").write_text(script, encoding="utf-8")
    input_csv = tmp_path / "entree.csv"
    input_csv.write_text("x1,x2\n45,18\n55,22\n50,20\n", encoding="utf-8")
    output_csv = tmp_path / "sortie.csv"

    proc = subprocess.run(
        [sys.executable, str(tmp_path / "m_deploiement.py"), "--batch", str(input_csv), str(output_csv)],
        capture_output=True, text=True, cwd=tmp_path, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    result_df = pd.read_csv(output_csv)
    assert len(result_df) == 3
    assert "prediction" in result_df.columns


def test_standalone_script_rejects_missing_column_clearly(tmp_path):
    algorithm, bundle = _regression_bundle()
    joblib.dump(bundle, tmp_path / "m.joblib")
    script = generate_deployment_script(
        bundle=bundle, feature_columns=["x1", "x2"], algorithm=algorithm, task_type="regression",
        target_column="cible", artifact_filename="m.joblib", script_filename="m_deploiement.py",
    )
    (tmp_path / "m_deploiement.py").write_text(script, encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(tmp_path / "m_deploiement.py"), "--predict", json.dumps({"x1": 1.0})],
        capture_output=True, text=True, cwd=tmp_path, timeout=60,
    )
    assert proc.returncode != 0
    assert "x2" in proc.stderr
