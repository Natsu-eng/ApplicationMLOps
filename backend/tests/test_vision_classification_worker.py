"""Tests de workers/vision_classification_worker.py (pilier Vision, Lot 15
sous-lot B) — bout en bout réel (pas mocké), même approche que
`test_anomaly_worker.py`. Un seul epoch, dataset minuscule : la durée réelle
CPU de ces tests reste de l'ordre de la dizaine de secondes par test."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from api.core.models import (
    Organization,
    VisionClassificationJob,
    VisionClassificationModel,
    VisionDataset,
)
from domains.vision.classification.worker import run_vision_classification_job


def _write_classification_dataset(root, n_per_class=8):
    rng = np.random.default_rng(1)
    for class_name, color in [("classe_a", (220, 20, 20)), ("classe_b", (20, 20, 220))]:
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            noise = rng.integers(-20, 20, (48, 48, 3))
            arr = np.clip(np.array(color) + noise, 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(class_dir / f"{i}.png")


def _make_job(db, tmp_path, structure_type="classification", dataset_status="ready", **config_overrides) -> VisionClassificationJob:
    org = Organization(name="Bureau test")
    db.add(org)
    db.flush()

    if structure_type == "classification":
        _write_classification_dataset(tmp_path)

    dataset = VisionDataset(
        organization_id=org.id,
        name="dataset.zip",
        structure_type=structure_type,
        storage_dir=str(tmp_path),
        n_images=16,
        n_classes=2,
        status=dataset_status,
    )
    db.add(dataset)
    db.flush()

    config = {
        "backbone_id": "mobilenet_v3_small",
        "num_epochs": 1,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "dropout_rate": 0.3,
        "freeze_backbone": True,
        "unfreeze_after_epoch": None,
        "seed": 42,
    }
    config.update(config_overrides)

    job = VisionClassificationJob(
        organization_id=org.id,
        vision_dataset_id=dataset.id,
        config_json=json.dumps(config),
        status="queued",
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def test_worker_persists_result_on_success(db_session, tmp_path):
    job = _make_job(db_session, tmp_path)
    run_vision_classification_job(job.id)

    db_session.expire_all()
    refreshed = db_session.query(VisionClassificationJob).filter(VisionClassificationJob.id == job.id).first()
    assert refreshed.status == "completed"
    assert refreshed.progress_percent == 100

    result = (
        db_session.query(VisionClassificationModel)
        .filter(VisionClassificationModel.vision_classification_job_id == job.id)
        .first()
    )
    assert result is not None
    assert result.backbone_id == "mobilenet_v3_small"
    assert json.loads(result.class_names_json) == ["classe_a", "classe_b"]
    assert result.n_train + result.n_val + result.n_test == 16
    assert Path(result.file_path).exists()


def test_worker_marks_job_failed_on_missing_dataset(db_session, tmp_path):
    job = _make_job(db_session, tmp_path)
    dataset = db_session.query(VisionDataset).filter(VisionDataset.id == job.vision_dataset_id).first()
    db_session.delete(dataset)
    db_session.commit()

    run_vision_classification_job(job.id)

    db_session.expire_all()
    refreshed = db_session.query(VisionClassificationJob).filter(VisionClassificationJob.id == job.id).first()
    assert refreshed.status == "failed"
    assert refreshed.error_message == "Dataset d'images introuvable ou non prêt"


def test_worker_rejects_mvtec_dataset(db_session, tmp_path):
    """Un dataset MVTec AD n'a pas de dossiers de classes — le worker doit
    refuser explicitement plutôt que de laisser ImageFolder produire un
    résultat incohérent (0 ou 2 "classes" train/test sans rapport avec une
    vraie tâche de classification)."""
    job = _make_job(db_session, tmp_path, structure_type="mvtec_ad")

    run_vision_classification_job(job.id)

    db_session.expire_all()
    refreshed = db_session.query(VisionClassificationJob).filter(VisionClassificationJob.id == job.id).first()
    assert refreshed.status == "failed"
    assert "normal/défaut" in refreshed.error_message


def test_worker_never_leaks_raw_traceback_on_failure(db_session, tmp_path, monkeypatch):
    import domains.vision.classification.worker as worker_module

    raw_exc = RuntimeError(
        'File "E:\\mlops\\app-analyse\\backend\\.venv\\Lib\\site-packages\\torch\\nn\\modules.py", line 42'
    )

    def _raise(*args, **kwargs):
        raise raw_exc

    monkeypatch.setattr(worker_module, "train_and_evaluate_classification", _raise)

    job = _make_job(db_session, tmp_path)
    run_vision_classification_job(job.id)

    db_session.expire_all()
    refreshed = db_session.query(VisionClassificationJob).filter(VisionClassificationJob.id == job.id).first()
    assert refreshed.status == "failed"
    assert "torch" not in refreshed.error_message
    assert "E:\\" not in refreshed.error_message
    assert "Traceback" not in refreshed.error_message
