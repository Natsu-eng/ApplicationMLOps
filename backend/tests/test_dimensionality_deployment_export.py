"""Tests de domains/dimensionality/services/deployment_export.py — même
rigueur que tests/test_clustering_deployment_export.py : le script généré
est vérifié en le faisant tourner dans un VRAI sous-processus Python séparé
(jamais un import direct, qui triche en réutilisant l'environnement du
process de test)."""
from __future__ import annotations

import ast
import json
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

from domains.dimensionality.services.deployment_export import generate_dimensionality_deployment_script
from domains.dimensionality.services.engine import DimensionalityConfig, train_and_evaluate_dimensionality
from domains.dimensionality.services.inference import project_point

_NOOP = lambda step, pct: None  # noqa: E731


def _make_two_blobs_df(n_per_group: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    group = np.repeat([0, 1], n_per_group)
    signal = np.where(group == 0, 0.0, 20.0) + rng.normal(0, 0.5, len(group))
    noise = rng.normal(0, 0.5, len(group))
    return pd.DataFrame({"signal": signal, "noise": noise})


def _train(algorithm_id: str):
    df = _make_two_blobs_df()
    return train_and_evaluate_dimensionality(df, DimensionalityConfig(algorithm_id=algorithm_id, seed=42), _NOOP)


@pytest.mark.parametrize("algorithm_id", ["pca", "umap", "tsne"])
def test_generated_script_is_valid_python_for_every_algorithm(algorithm_id):
    result = _train(algorithm_id)
    script = generate_dimensionality_deployment_script(
        feature_columns=result.feature_columns,
        algorithm=result.algorithm_label,
        algorithm_id=result.algorithm_id,
        artifact_filename="modele.joblib",
        script_filename="script.py",
    )
    ast.parse(script)


@pytest.mark.parametrize("algorithm_id", ["pca", "umap"])
def test_standalone_script_matches_project_point_in_a_real_subprocess(algorithm_id, tmp_path):
    result = _train(algorithm_id)
    bundle = result.pipeline_bundle
    probe_point = {"signal": 0.0, "noise": 0.0}
    reference = project_point(bundle, result.feature_columns, probe_point)
    assert reference["projection_method"] == "exact"

    artifact_path = tmp_path / "modele.joblib"
    joblib.dump(bundle, artifact_path)
    script = generate_dimensionality_deployment_script(
        feature_columns=result.feature_columns,
        algorithm=result.algorithm_label,
        algorithm_id=result.algorithm_id,
        artifact_filename=artifact_path.name,
        script_filename="script.py",
    )
    script_path = tmp_path / "script.py"
    script_path.write_text(script, encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(script_path), "--predict", json.dumps(probe_point)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        # UMAP compile ses noyaux numba au premier appel dans un process
        # neuf (cache JIT froid) — mesuré ~65s pour `fit_transform` seul en
        # isolation sur ce poste, largement au-delà des 60s habituels des
        # autres scripts de déploiement (PCA/scikit-learn purs, aucune
        # compilation JIT). Généreux uniquement pour ce cas, jamais pour
        # masquer un vrai plantage (le `assert returncode == 0, proc.stderr`
        # ci-dessous reste strict).
        timeout=180 if algorithm_id == "umap" else 60,
    )
    assert proc.returncode == 0, proc.stderr
    output = json.loads(proc.stdout)[0]
    assert output["x"] == pytest.approx(reference["x"], abs=1e-6)
    assert output["y"] == pytest.approx(reference["y"], abs=1e-6)


def test_standalone_script_rejects_new_points_for_tsne_clearly(tmp_path):
    """t-SNE est transductif — le script généré doit échouer PROPREMENT sur
    de nouvelles observations, jamais silencieusement faux."""
    result = _train("tsne")
    bundle = result.pipeline_bundle
    artifact_path = tmp_path / "modele.joblib"
    joblib.dump(bundle, artifact_path)
    script = generate_dimensionality_deployment_script(
        feature_columns=result.feature_columns,
        algorithm=result.algorithm_label,
        algorithm_id=result.algorithm_id,
        artifact_filename=artifact_path.name,
        script_filename="script.py",
    )
    script_path = tmp_path / "script.py"
    script_path.write_text(script, encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(script_path), "--predict", json.dumps({"signal": 0.0, "noise": 0.0})],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=60,
    )
    assert proc.returncode != 0
    assert "transductif" in proc.stderr


def test_standalone_script_batch_mode_matches_row_count(tmp_path):
    result = _train("pca")
    bundle = result.pipeline_bundle
    artifact_path = tmp_path / "modele.joblib"
    joblib.dump(bundle, artifact_path)
    script = generate_dimensionality_deployment_script(
        feature_columns=result.feature_columns,
        algorithm=result.algorithm_label,
        algorithm_id=result.algorithm_id,
        artifact_filename=artifact_path.name,
        script_filename="script.py",
    )
    script_path = tmp_path / "script.py"
    script_path.write_text(script, encoding="utf-8")

    input_df = pd.DataFrame({"signal": [0.0, 20.0], "noise": [0.0, 0.0], "id_ligne": ["a", "b"]})
    input_path = tmp_path / "entree.csv"
    output_path = tmp_path / "sortie.csv"
    input_df.to_csv(input_path, index=False)

    proc = subprocess.run(
        [sys.executable, str(script_path), "--batch", str(input_path), str(output_path)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    output_df = pd.read_csv(output_path)
    assert len(output_df) == 2
    assert "id_ligne" in output_df.columns
