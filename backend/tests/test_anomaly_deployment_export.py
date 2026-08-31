"""Tests de domains/anomalies/services/deployment_export.py — même rigueur
que tests/test_clustering_deployment_export.py : le script généré est
vérifié en le faisant tourner dans un VRAI sous-processus Python séparé
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

from domains.anomalies.services.deployment_export import generate_anomaly_deployment_script
from domains.anomalies.services.engine import AnomalyConfig, train_and_evaluate_anomalies
from domains.anomalies.services.inference import score_anomaly

_NOOP = lambda step, pct: None  # noqa: E731


def _make_dataset_with_injected_outliers(n_normal: int = 95, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    normal = rng.normal(0, 1, (n_normal, 3))
    outliers = np.array(
        [[15.0, 15.0, 15.0], [14.0, -14.0, 0.0], [-13.0, 13.0, -13.0], [16.0, 0.0, -16.0], [0.0, 16.0, 16.0]]
    )
    return pd.DataFrame(np.vstack([normal, outliers]), columns=["a", "b", "c"])


def _train():
    df = _make_dataset_with_injected_outliers()
    return train_and_evaluate_anomalies(df[["a", "b", "c"]], AnomalyConfig(top_n=10, seed=42), _NOOP)


def test_generated_script_is_valid_python():
    result = _train()
    script = generate_anomaly_deployment_script(
        feature_columns=result.feature_columns,
        artifact_filename="modele.joblib",
        script_filename="script.py",
    )
    ast.parse(script)


@pytest.mark.parametrize(
    "probe_point",
    [
        {"a": 0.0, "b": 0.0, "c": 0.0},  # au centre du groupe normal
        {"a": 20.0, "b": 20.0, "c": 20.0},  # très éloigné, doit ressortir anomalie
    ],
)
def test_standalone_script_matches_score_anomaly_in_a_real_subprocess(probe_point, tmp_path):
    result = _train()
    bundle = result.pipeline_bundle
    reference = score_anomaly(bundle, result.feature_columns, probe_point)

    artifact_path = tmp_path / "modele.joblib"
    joblib.dump(bundle, artifact_path)
    script = generate_anomaly_deployment_script(
        feature_columns=result.feature_columns,
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
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    output = json.loads(proc.stdout)[0]
    assert output["consensus_score"] == pytest.approx(reference["consensus_score"], abs=1e-9)
    assert output["is_anomaly_isolation_forest"] == reference["is_anomaly_isolation_forest"]
    assert output["is_anomaly_lof"] == reference["is_anomaly_lof"]
    assert output["is_anomaly_consensus"] == reference["is_anomaly_consensus"]
    assert output["agreement"] == reference["agreement"]


def test_standalone_script_batch_mode_matches_row_count(tmp_path):
    result = _train()
    bundle = result.pipeline_bundle
    artifact_path = tmp_path / "modele.joblib"
    joblib.dump(bundle, artifact_path)
    script = generate_anomaly_deployment_script(
        feature_columns=result.feature_columns,
        artifact_filename=artifact_path.name,
        script_filename="script.py",
    )
    script_path = tmp_path / "script.py"
    script_path.write_text(script, encoding="utf-8")

    input_df = pd.DataFrame({"a": [0.0, 20.0], "b": [0.0, 20.0], "c": [0.0, 20.0], "id_ligne": ["normal", "extreme"]})
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
    assert bool(output_df.loc[1, "is_anomaly_consensus"]) is True


def test_standalone_script_rejects_missing_column_clearly(tmp_path):
    result = _train()
    bundle = result.pipeline_bundle
    artifact_path = tmp_path / "modele.joblib"
    joblib.dump(bundle, artifact_path)
    script = generate_anomaly_deployment_script(
        feature_columns=result.feature_columns,
        artifact_filename=artifact_path.name,
        script_filename="script.py",
    )
    script_path = tmp_path / "script.py"
    script_path.write_text(script, encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(script_path), "--predict", json.dumps({"a": 0.0, "b": 0.0})],  # c manquant
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=60,
    )
    assert proc.returncode != 0
    assert "c" in proc.stderr
