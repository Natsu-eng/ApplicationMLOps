"""Tests de domains/vision/classification/services/deployment_export.py —
même rigueur que tests/test_clustering_deployment_export.py : le script
généré est vérifié en le faisant tourner dans un VRAI sous-processus Python
séparé (jamais un import direct, qui triche en réutilisant l'environnement
du process de test), sur un modèle RÉELLEMENT entraîné (pas mocké, même
approche que test_vision_classification_training.py)."""
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

from domains.vision.classification.services.deployment_export import generate_vision_classification_deployment_script
from domains.vision.classification.services.engine import (
    ClassificationConfig,
    build_eval_transform,
    train_and_evaluate_classification,
)
from domains.vision.classification.services.registry import get_backbone_spec


def _noop_progress(step: str, percent: int) -> None:
    pass


def _write_classification_dataset(root, class_counts: dict[str, int], size=(48, 48)):
    rng = np.random.default_rng(0)
    colors = {"classe_a": (220, 20, 20), "classe_b": (20, 20, 220)}
    for class_name, count in class_counts.items():
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        base_color = colors[class_name]
        for i in range(count):
            noise = rng.integers(-20, 20, (size[1], size[0], 3))
            arr = np.clip(np.array(base_color) + noise, 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(class_dir / f"{i}.png")
    return root


def _train(tmp_path):
    _write_classification_dataset(tmp_path / "dataset", {"classe_a": 8, "classe_b": 8})
    config = ClassificationConfig(backbone_id="mobilenet_v3_small", num_epochs=1, batch_size=4, image_size=48)
    return train_and_evaluate_classification(tmp_path / "dataset", config, _noop_progress)


def test_generated_script_is_valid_python(tmp_path):
    result = _train(tmp_path)
    spec = get_backbone_spec(result.model_artifact["backbone_id"])
    script = generate_vision_classification_deployment_script(
        backbone_id=result.model_artifact["backbone_id"],
        backbone_label=spec.label,
        class_names=result.class_names,
        image_size=result.model_artifact["image_size"],
        artifact_filename="modele.pt",
        script_filename="script.py",
    )
    ast.parse(script)


def test_standalone_script_reproduces_probabilities_in_a_real_subprocess(tmp_path):
    result = _train(tmp_path)
    artifact = result.model_artifact
    spec = get_backbone_spec(artifact["backbone_id"])

    # Référence — reconstruction directe (même logique que
    # services/gradcam.py::_rebuild_model, sans les hooks Grad-CAM inutiles
    # ici) sur UNE image de test réelle (pas depuis le dataset d'entraînement,
    # une image neuve comme un vrai déploiement en verrait).
    ref_model = spec.build_model(len(artifact["class_names"]), artifact["dropout_rate"])
    ref_model.load_state_dict(artifact["state_dict"])
    ref_model.eval()

    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    probe_path = probe_dir / "probe.png"
    rng = np.random.default_rng(123)
    arr = np.clip(np.array((220, 20, 20)) + rng.integers(-20, 20, (48, 48, 3)), 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(probe_path)

    transform = build_eval_transform(artifact["image_size"])
    image = Image.open(probe_path).convert("RGB")
    with torch.no_grad():
        ref_logits = ref_model(transform(image).unsqueeze(0))
        ref_probs = torch.softmax(ref_logits, dim=1)[0].tolist()

    artifact_path = tmp_path / "modele.pt"
    torch.save(artifact, artifact_path)
    script = generate_vision_classification_deployment_script(
        backbone_id=artifact["backbone_id"],
        backbone_label=spec.label,
        class_names=result.class_names,
        image_size=artifact["image_size"],
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
    for i, class_name in enumerate(result.class_names):
        assert output["probabilites"][class_name] == pytest.approx(ref_probs[i], abs=1e-5)


def test_standalone_script_batch_mode_covers_every_image(tmp_path):
    result = _train(tmp_path)
    artifact = result.model_artifact
    spec = get_backbone_spec(artifact["backbone_id"])
    artifact_path = tmp_path / "modele.pt"
    torch.save(artifact, artifact_path)
    script = generate_vision_classification_deployment_script(
        backbone_id=artifact["backbone_id"],
        backbone_label=spec.label,
        class_names=result.class_names,
        image_size=artifact["image_size"],
        artifact_filename=artifact_path.name,
        script_filename="script.py",
    )
    script_path = tmp_path / "script.py"
    script_path.write_text(script, encoding="utf-8")

    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    rng = np.random.default_rng(7)
    for i in range(3):
        arr = rng.integers(0, 255, (48, 48, 3)).astype("uint8")
        Image.fromarray(arr).save(batch_dir / f"img_{i}.png")
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
    assert all(row["prediction"] in result.class_names for row in rows)
