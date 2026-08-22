"""Tests de domains/dimensionality/worker.py (Lot 13 — ML non supervisé) —
bout en bout réel (pas mocké), même approche que test_clustering_worker.py."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from api.core.models import Dataset, DimensionalityJob, DimensionalityModel, DimensionalityPoint, Organization
from domains.dimensionality.worker import run_dimensionality_job


def _write_temp_csv(df: pd.DataFrame) -> str:
    path = Path(tempfile.gettempdir()) / f"datalab_dimensionality_test_{np.random.default_rng().integers(1_000_000)}.csv"
    df.to_csv(path, index=False)
    return str(path)


def _make_two_blobs_df(n_per_group: int = 40, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    group = np.repeat([0, 1], n_per_group)
    signal = np.where(group == 0, 0.0, 20.0) + rng.normal(0, 0.5, len(group))
    noise = rng.normal(0, 0.5, len(group))
    return pd.DataFrame({"signal": signal, "noise": noise})


def _make_dimensionality_job(db, feature_columns: list[str], algorithm_id: str = "pca") -> DimensionalityJob:
    org = Organization(name="Bureau test")
    db.add(org)
    db.flush()

    df = _make_two_blobs_df()
    csv_path = _write_temp_csv(df)

    dataset = Dataset(
        organization_id=org.id,
        name="d.csv",
        file_path=csv_path,
        file_size_bytes=1,
        status="ready",
        columns_json=json.dumps([{"name": c, "dtype": str(df[c].dtype)} for c in df.columns]),
    )
    db.add(dataset)
    db.flush()

    job = DimensionalityJob(
        organization_id=org.id,
        dataset_id=dataset.id,
        feature_columns_json=json.dumps(feature_columns),
        config_json=json.dumps({"algorithm_id": algorithm_id, "seed": 42}),
        status="queued",
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def test_worker_persists_result_and_points(db_session):
    job = _make_dimensionality_job(db_session, feature_columns=["signal", "noise"])
    run_dimensionality_job(job.id)

    db_session.expire_all()
    refreshed = db_session.query(DimensionalityJob).filter(DimensionalityJob.id == job.id).first()
    assert refreshed.status == "completed"

    result = db_session.query(DimensionalityModel).filter(DimensionalityModel.dimensionality_job_id == job.id).first()
    assert result is not None
    assert result.algorithm_id == "pca"
    # La qualité de la PCA (variance concentrée sur PC1) est déjà couverte en
    # détail par test_dimensionality_training.py — ce test-ci vérifie la
    # PERSISTANCE, pas la précision du calcul. `signal`/`noise` sont deux
    # colonnes indépendantes : une fois standardisées (variance unitaire
    # chacune, étape du préprocesseur), aucune raison qu'une domine l'autre.
    variance_ratio = json.loads(result.variance_explained_json)
    assert 0.0 <= variance_ratio[0] <= 1.0

    points = (
        db_session.query(DimensionalityPoint).filter(DimensionalityPoint.dimensionality_job_id == job.id).all()
    )
    assert len(points) == 80  # 2 * n_per_group


def test_worker_marks_job_failed_on_missing_dataset(db_session):
    job = _make_dimensionality_job(db_session, feature_columns=["signal", "noise"])
    dataset = db_session.query(Dataset).filter(Dataset.id == job.dataset_id).first()
    db_session.delete(dataset)
    db_session.commit()

    run_dimensionality_job(job.id)

    db_session.expire_all()
    refreshed = db_session.query(DimensionalityJob).filter(DimensionalityJob.id == job.id).first()
    assert refreshed.status == "failed"
    assert refreshed.error_message == "Dataset introuvable ou non prêt"


def test_worker_never_leaks_raw_traceback_on_failure(db_session, monkeypatch):
    import domains.dimensionality.worker as dimensionality_worker_module

    raw_exc = RuntimeError(
        'File "E:\\mlops\\app-analyse\\backend\\.venv\\Lib\\site-packages\\sklearn\\decomposition\\_pca.py", line 42'
    )

    def _raise(*args, **kwargs):
        raise raw_exc

    monkeypatch.setattr(dimensionality_worker_module, "train_and_evaluate_dimensionality", _raise)

    job = _make_dimensionality_job(db_session, feature_columns=["signal", "noise"])
    run_dimensionality_job(job.id)

    db_session.expire_all()
    refreshed = db_session.query(DimensionalityJob).filter(DimensionalityJob.id == job.id).first()
    assert refreshed.status == "failed"
    assert "sklearn" not in refreshed.error_message
    assert "E:\\" not in refreshed.error_message
    assert "Traceback" not in refreshed.error_message


def test_worker_respects_algorithm_id_from_config(db_session):
    job = _make_dimensionality_job(db_session, feature_columns=["signal", "noise"], algorithm_id="tsne")
    run_dimensionality_job(job.id)

    db_session.expire_all()
    result = db_session.query(DimensionalityModel).filter(DimensionalityModel.dimensionality_job_id == job.id).first()
    assert result.algorithm_id == "tsne"
