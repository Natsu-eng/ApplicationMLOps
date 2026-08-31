"""Tests de domains/clustering/services/deployment_export.py — même rigueur
que tests/test_deployment_export.py (pilier supervisé) : le script généré
est vérifié en le faisant tourner dans un VRAI sous-processus Python séparé
(jamais un import direct, qui triche en réutilisant l'environnement du
process de test) pour les 3 familles d'algorithme (exact/approximate_centroid/
approximate_nearest_core)."""
from __future__ import annotations

import ast
import json
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

from domains.clustering.services.deployment_export import generate_clustering_deployment_script
from domains.clustering.services.engine import ClusteringConfig, train_and_evaluate_clustering
from domains.clustering.services.inference import assign_cluster

_NOOP = lambda step, pct: None  # noqa: E731


def _make_three_blobs_df(n_per_group: int = 50, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    group = np.repeat([0, 1, 2], n_per_group)
    centers = {0: (0, 0), 1: (12, 12), 2: (0, 12)}
    x1 = np.array([centers[g][0] for g in group]) + rng.normal(0, 0.5, len(group))
    x2 = np.array([centers[g][1] for g in group]) + rng.normal(0, 0.5, len(group))
    return pd.DataFrame({"x1": x1, "x2": x2})


def _train(algorithm_id: str):
    df = _make_three_blobs_df()
    return train_and_evaluate_clustering(df, ClusteringConfig(seed=42, algorithm_ids=[algorithm_id]), _NOOP)


def test_generated_script_is_valid_python_for_every_family():
    for algorithm_id in ("kmeans", "hierarchical", "dbscan"):
        result = _train(algorithm_id)
        script = generate_clustering_deployment_script(
            bundle=result.pipeline_bundle,
            feature_columns=result.feature_columns,
            algorithm=result.winning_label,
            family=result.model_card["family"],
            artifact_filename="modele.joblib",
            script_filename="script.py",
        )
        ast.parse(script)  # ne lève rien -> syntaxiquement valide


@pytest.mark.parametrize(
    "algorithm_id,probe_point,expected_method",
    [
        ("kmeans", {"x1": 0.0, "x2": 0.0}, "exact"),
        ("hierarchical", {"x1": 12.0, "x2": 12.0}, "approximate_centroid"),
        ("dbscan", {"x1": 0.1, "x2": 0.1}, "approximate_nearest_core"),
    ],
)
def test_standalone_script_matches_assign_cluster_in_a_real_subprocess(
    algorithm_id, probe_point, expected_method, tmp_path
):
    result = _train(algorithm_id)
    bundle = result.pipeline_bundle

    reference = assign_cluster(bundle, result.feature_columns, probe_point)
    assert reference["assignment_method"] == expected_method

    artifact_path = tmp_path / "modele.joblib"
    joblib.dump(bundle, artifact_path)
    script = generate_clustering_deployment_script(
        bundle=bundle,
        feature_columns=result.feature_columns,
        algorithm=result.winning_label,
        family=result.model_card["family"],
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
    assert output["assignment_method"] == expected_method
    assert output["is_noise"] == reference["is_noise"]
    if reference["cluster_id"] is None:
        assert output["cluster_id"] is None
    else:
        assert output["cluster_id"] == reference["cluster_id"]


def test_standalone_script_batch_mode_matches_row_count(tmp_path):
    result = _train("kmeans")
    bundle = result.pipeline_bundle
    artifact_path = tmp_path / "modele.joblib"
    joblib.dump(bundle, artifact_path)
    script = generate_clustering_deployment_script(
        bundle=bundle,
        feature_columns=result.feature_columns,
        algorithm=result.winning_label,
        family=result.model_card["family"],
        artifact_filename=artifact_path.name,
        script_filename="script.py",
    )
    script_path = tmp_path / "script.py"
    script_path.write_text(script, encoding="utf-8")

    input_df = pd.DataFrame({"x1": [0.0, 12.0, 0.0], "x2": [0.0, 12.0, 12.0], "id_ligne": ["a", "b", "c"]})
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
    assert len(output_df) == 3
    assert "id_ligne" in output_df.columns  # colonnes d'origine conservées
    assert set(output_df["assignment_method"]) == {"exact"}


def test_standalone_script_rejects_missing_column_clearly(tmp_path):
    result = _train("kmeans")
    bundle = result.pipeline_bundle
    artifact_path = tmp_path / "modele.joblib"
    joblib.dump(bundle, artifact_path)
    script = generate_clustering_deployment_script(
        bundle=bundle,
        feature_columns=result.feature_columns,
        algorithm=result.winning_label,
        family=result.model_card["family"],
        artifact_filename=artifact_path.name,
        script_filename="script.py",
    )
    script_path = tmp_path / "script.py"
    script_path.write_text(script, encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(script_path), "--predict", json.dumps({"x1": 0.0})],  # x2 manquant
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=60,
    )
    assert proc.returncode != 0
    assert "x2" in proc.stderr
