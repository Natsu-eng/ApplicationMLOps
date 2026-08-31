"""Tests de domains/vision/anomalies/services/deployment_export.py — même
rigueur que tests/test_vision_classification_deployment_export.py : le
script généré est vérifié en le faisant tourner dans un VRAI sous-processus
Python séparé, sur un modèle RÉELLEMENT entraîné (pas mocké)."""
from __future__ import annotations

import ast
import csv
import json
import subprocess
import sys

import numpy as np
import pytest
import torch
from PIL import Image

from domains.vision.anomalies.services.deployment_export import generate_vision_anomaly_deployment_script
from domains.vision.anomalies.services.engine import AnomalyVisionConfig, train_and_evaluate_anomaly_vision
from domains.vision.anomalies.services.inference import score_vision_anomaly
from domains.vision.anomalies.services.registry import get_anomaly_model_spec


def _make_normal(rng, size=(64, 64)):
    arr = np.full((size[1], size[0], 3), (120, 120, 120), dtype=np.uint8)
    noise = rng.integers(-10, 10, (size[1], size[0], 3))
    return np.clip(arr.astype(int) + noise, 0, 255).astype(np.uint8)


def _make_defect(rng, size=(64, 64)):
    arr = _make_normal(rng, size)
    arr[20:40, 20:40] = [220, 20, 20]
    return arr


def _write_mvtec_dataset(root, n_train_good=14, n_test_good=4, n_test_defect=4, seed=0):
    rng = np.random.default_rng(seed)
    (root / "train" / "good").mkdir(parents=True, exist_ok=True)
    for i in range(n_train_good):
        Image.fromarray(_make_normal(rng)).save(root / "train" / "good" / f"{i}.png")
    (root / "test" / "good").mkdir(parents=True, exist_ok=True)
    for i in range(n_test_good):
        Image.fromarray(_make_normal(rng)).save(root / "test" / "good" / f"{i}.png")
    (root / "test" / "scratch").mkdir(parents=True, exist_ok=True)
    for i in range(n_test_defect):
        Image.fromarray(_make_defect(rng)).save(root / "test" / "scratch" / f"{i}.png")
    return root


def _noop_progress(step: str, percent: int) -> None:
    pass


def _train(tmp_path, model_id="conv_autoencoder"):
    _write_mvtec_dataset(tmp_path / "dataset")
    config = AnomalyVisionConfig(model_id=model_id, num_epochs=2, batch_size=4)
    return train_and_evaluate_anomaly_vision(tmp_path / "dataset", config, _noop_progress)


@pytest.mark.parametrize("model_id", ["conv_autoencoder", "denoising_autoencoder", "conv_vae"])
def test_generated_script_is_valid_python_for_every_architecture(tmp_path, model_id):
    result = _train(tmp_path, model_id)
    spec = get_anomaly_model_spec(result.model_artifact["model_id"])
    script = generate_vision_anomaly_deployment_script(
        model_id=result.model_artifact["model_id"],
        model_label=spec.label,
        image_size=result.model_artifact["image_size"],
        threshold=result.threshold,
        artifact_filename="modele.pt",
        script_filename="script.py",
    )
    ast.parse(script)


@pytest.mark.parametrize("model_id", ["conv_autoencoder", "denoising_autoencoder", "conv_vae"])
def test_standalone_script_reproduces_score_in_a_real_subprocess(tmp_path, model_id):
    result = _train(tmp_path, model_id)
    artifact = result.model_artifact

    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    probe_path = probe_dir / "probe.png"
    rng = np.random.default_rng(123)
    Image.fromarray(_make_defect(rng)).save(probe_path)

    with Image.open(probe_path) as probe_image:
        reference = score_vision_anomaly(artifact, probe_image)

    artifact_path = tmp_path / "modele.pt"
    torch.save(artifact, artifact_path)
    spec = get_anomaly_model_spec(artifact["model_id"])
    script = generate_vision_anomaly_deployment_script(
        model_id=artifact["model_id"],
        model_label=spec.label,
        image_size=artifact["image_size"],
        threshold=result.threshold,
        artifact_filename=artifact_path.name,
        script_filename="script.py",
    )
    script_path = tmp_path / "script.py"
    script_path.write_text(script, encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(script_path), "--predict", str(probe_path)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    output = json.loads(proc.stdout)
    assert output["score_anomalie"] == pytest.approx(reference["anomaly_score"], abs=1e-5)
    assert output["seuil"] == pytest.approx(reference["threshold"], abs=1e-9)
    assert output["est_anomalie"] == reference["is_anomaly"]


def test_standalone_script_batch_mode_covers_every_image(tmp_path):
    result = _train(tmp_path)
    artifact = result.model_artifact
    artifact_path = tmp_path / "modele.pt"
    torch.save(artifact, artifact_path)
    spec = get_anomaly_model_spec(artifact["model_id"])
    script = generate_vision_anomaly_deployment_script(
        model_id=artifact["model_id"],
        model_label=spec.label,
        image_size=artifact["image_size"],
        threshold=result.threshold,
        artifact_filename=artifact_path.name,
        script_filename="script.py",
    )
    script_path = tmp_path / "script.py"
    script_path.write_text(script, encoding="utf-8")

    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    rng = np.random.default_rng(7)
    for i in range(3):
        Image.fromarray(_make_normal(rng)).save(batch_dir / f"img_{i}.png")
    output_csv = tmp_path / "sortie.csv"

    proc = subprocess.run(
        [sys.executable, str(script_path), "--batch", str(batch_dir), str(output_csv)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    with open(output_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert all(row["est_anomalie"] in {"True", "False"} for row in rows)
