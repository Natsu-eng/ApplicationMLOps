"""Tests de workers/vision_anomaly_worker.py (pilier Vision, Lot 15
sous-lot C) — bout en bout réel (pas mocké), même approche que
`test_vision_classification_worker.py`."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from api.core.models import (
    Organization,
    VisionAnomalyExampleRecord,
    VisionAnomalyJob,
    VisionAnomalyModel,
    VisionDataset,
)
from domains.vision.anomalies.worker import run_vision_anomaly_job


def _write_mvtec_dataset(root, n_train_good=14, n_test_good=4, n_test_defect=4):
    rng = np.random.default_rng(2)

    def normal():
        arr = np.full((64, 64, 3), (120, 120, 120), dtype=np.uint8)
        noise = rng.integers(-10, 10, (64, 64, 3))
        return np.clip(arr.astype(int) + noise, 0, 255).astype(np.uint8)

    def defect():
        arr = normal()
        arr[20:40, 20:40] = [220, 20, 20]
        return arr

    (root / "train" / "good").mkdir(parents=True, exist_ok=True)
    for i in range(n_train_good):
        Image.fromarray(normal()).save(root / "train" / "good" / f"{i}.png")
    (root / "test" / "good").mkdir(parents=True, exist_ok=True)
    for i in range(n_test_good):
        Image.fromarray(normal()).save(root / "test" / "good" / f"{i}.png")
    (root / "test" / "scratch").mkdir(parents=True, exist_ok=True)
    for i in range(n_test_defect):
        Image.fromarray(defect()).save(root / "test" / "scratch" / f"{i}.png")


def _make_job(db, tmp_path, structure_type="mvtec_ad", dataset_status="ready", **config_overrides) -> VisionAnomalyJob:
    org = Organization(name="Bureau test")
    db.add(org)
    db.flush()

    if structure_type == "mvtec_ad":
        _write_mvtec_dataset(tmp_path)

    dataset = VisionDataset(
        organization_id=org.id,
        name="dataset.zip",
        structure_type=structure_type,
        storage_dir=str(tmp_path),
        n_images=22,
        status=dataset_status,
    )
    db.add(dataset)
    db.flush()

    config = {
        "model_id": "conv_autoencoder",
        "num_epochs": 2,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "seed": 42,
        "mask_percentile": 0.95,
    }
    config.update(config_overrides)

    job = VisionAnomalyJob(
        organization_id=org.id,
        vision_dataset_id=dataset.id,
        config_json=json.dumps(config),
        status="queued",
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def test_worker_persists_result_and_examples_on_success(db_session, tmp_path):
    job = _make_job(db_session, tmp_path)
    run_vision_anomaly_job(job.id)

    db_session.expire_all()
    refreshed = db_session.query(VisionAnomalyJob).filter(VisionAnomalyJob.id == job.id).first()
    assert refreshed.status == "completed"
    assert refreshed.progress_percent == 100

    result = (
        db_session.query(VisionAnomalyModel)
        .filter(VisionAnomalyModel.vision_anomaly_job_id == job.id)
        .first()
    )
    assert result is not None
    assert result.model_id == "conv_autoencoder"
    assert Path(result.file_path).exists()

    examples = (
        db_session.query(VisionAnomalyExampleRecord)
        .filter(VisionAnomalyExampleRecord.vision_anomaly_model_id == result.id)
        .all()
    )
    assert len(examples) > 0
    assert all(e.heatmap_png.startswith("data:image/png;base64,") for e in examples)


def test_worker_trains_with_denoising_model(db_session, tmp_path):
    job = _make_job(db_session, tmp_path, model_id="denoising_autoencoder")
    run_vision_anomaly_job(job.id)

    db_session.expire_all()
    refreshed = db_session.query(VisionAnomalyJob).filter(VisionAnomalyJob.id == job.id).first()
    assert refreshed.status == "completed"
    result = db_session.query(VisionAnomalyModel).filter(VisionAnomalyModel.vision_anomaly_job_id == job.id).first()
    assert result.model_id == "denoising_autoencoder"


def test_worker_trains_with_vae_model(db_session, tmp_path):
    # Bout en bout sur le modèle dont la loss est branchée différemment
    # (compute_loss avec terme KL, voir vision_anomaly_training.py) —
    # vérifie que le branchement loss_kind=="vae" ne casse rien de bout en
    # bout, pas seulement au niveau du modèle isolé (test_vision_anomaly_registry.py).
    job = _make_job(db_session, tmp_path, model_id="conv_vae")
    run_vision_anomaly_job(job.id)

    db_session.expire_all()
    refreshed = db_session.query(VisionAnomalyJob).filter(VisionAnomalyJob.id == job.id).first()
    assert refreshed.status == "completed"
    result = db_session.query(VisionAnomalyModel).filter(VisionAnomalyModel.vision_anomaly_job_id == job.id).first()
    assert result.model_id == "conv_vae"


def test_worker_applies_custom_augmentation_and_val_ratio(db_session, tmp_path):
    # Répartition custom (val_ratio=0.3 au lieu du défaut 0.15) + preset
    # d'augmentation non trivial — vérifie que les deux nouveaux réglages
    # (Lot 6A, parité avec la classification) sont bien pris en compte de
    # bout en bout, pas seulement acceptés silencieusement.
    job = _make_job(db_session, tmp_path, augmentation_preset="standard", val_ratio=0.3)
    run_vision_anomaly_job(job.id)

    db_session.expire_all()
    refreshed = db_session.query(VisionAnomalyJob).filter(VisionAnomalyJob.id == job.id).first()
    assert refreshed.status == "completed"
    result = db_session.query(VisionAnomalyModel).filter(VisionAnomalyModel.vision_anomaly_job_id == job.id).first()
    # train/good/ compte 14 images (voir _write_mvtec_dataset) : sklearn
    # arrondit test_size au ceil (14*0.3=4.2 -> 5 en validation, 9 en train).
    assert result.n_val == 5
    assert result.n_train == 9


def test_worker_rejects_unknown_augmentation_preset(db_session, tmp_path):
    job = _make_job(db_session, tmp_path, augmentation_preset="extreme")
    run_vision_anomaly_job(job.id)

    db_session.expire_all()
    refreshed = db_session.query(VisionAnomalyJob).filter(VisionAnomalyJob.id == job.id).first()
    assert refreshed.status == "failed"
    assert "Preset d'augmentation inconnu" in refreshed.error_message


def test_worker_marks_job_failed_on_missing_dataset(db_session, tmp_path):
    job = _make_job(db_session, tmp_path)
    dataset = db_session.query(VisionDataset).filter(VisionDataset.id == job.vision_dataset_id).first()
    db_session.delete(dataset)
    db_session.commit()

    run_vision_anomaly_job(job.id)

    db_session.expire_all()
    refreshed = db_session.query(VisionAnomalyJob).filter(VisionAnomalyJob.id == job.id).first()
    assert refreshed.status == "failed"
    assert refreshed.error_message == "Dataset d'images introuvable ou non prêt"


def test_worker_rejects_classification_dataset(db_session, tmp_path):
    (tmp_path / "classe_a").mkdir()
    (tmp_path / "classe_b").mkdir()
    job = _make_job(db_session, tmp_path, structure_type="classification")

    run_vision_anomaly_job(job.id)

    db_session.expire_all()
    refreshed = db_session.query(VisionAnomalyJob).filter(VisionAnomalyJob.id == job.id).first()
    assert refreshed.status == "failed"
    assert "structure normal/défaut" in refreshed.error_message


def test_worker_never_leaks_raw_traceback_on_failure(db_session, tmp_path, monkeypatch):
    import domains.vision.anomalies.worker as worker_module

    raw_exc = RuntimeError(
        'File "E:\\mlops\\app-analyse\\backend\\.venv\\Lib\\site-packages\\torch\\nn\\modules.py", line 42'
    )

    def _raise(*args, **kwargs):
        raise raw_exc

    monkeypatch.setattr(worker_module, "train_and_evaluate_anomaly_vision", _raise)

    job = _make_job(db_session, tmp_path)
    run_vision_anomaly_job(job.id)

    db_session.expire_all()
    refreshed = db_session.query(VisionAnomalyJob).filter(VisionAnomalyJob.id == job.id).first()
    assert refreshed.status == "failed"
    assert "torch" not in refreshed.error_message
    assert "E:\\" not in refreshed.error_message
    assert "Traceback" not in refreshed.error_message
