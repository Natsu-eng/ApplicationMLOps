"""Tests de domains/anomalies/worker.py (Lot 14 — ML non supervisé) — bout en
bout réel (pas mocké), même approche que test_clustering_worker.py."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from api.core.models import AnomalyJob, AnomalyModel, AnomalyObservationRecord, Dataset, Organization
from domains.anomalies.worker import run_anomaly_job


def _write_temp_csv(df: pd.DataFrame) -> str:
    path = Path(tempfile.gettempdir()) / f"datalab_anomaly_test_{np.random.default_rng().integers(1_000_000)}.csv"
    df.to_csv(path, index=False)
    return str(path)


def _make_dataset_with_injected_outliers(n_normal: int = 45, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    normal = rng.normal(0, 1, (n_normal, 3))
    outliers = np.array([[15.0, 15.0, 15.0], [14.0, -14.0, 0.0], [-13.0, 13.0, -13.0]])
    return pd.DataFrame(np.vstack([normal, outliers]), columns=["a", "b", "c"])


def _make_anomaly_job(db, feature_columns: list[str], top_n: int = 10) -> AnomalyJob:
    org = Organization(name="Bureau test")
    db.add(org)
    db.flush()

    df = _make_dataset_with_injected_outliers()
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

    job = AnomalyJob(
        organization_id=org.id,
        dataset_id=dataset.id,
        feature_columns_json=json.dumps(feature_columns),
        config_json=json.dumps({"top_n": top_n, "seed": 42}),
        status="queued",
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def test_worker_persists_result_and_observations(db_session):
    job = _make_anomaly_job(db_session, feature_columns=["a", "b", "c"], top_n=10)
    run_anomaly_job(job.id)

    db_session.expire_all()
    refreshed = db_session.query(AnomalyJob).filter(AnomalyJob.id == job.id).first()
    assert refreshed.status == "completed"

    result = db_session.query(AnomalyModel).filter(AnomalyModel.anomaly_job_id == job.id).first()
    assert result is not None
    assert result.n_anomalies_consensus <= min(result.n_anomalies_isolation_forest, result.n_anomalies_lof)

    observations = (
        db_session.query(AnomalyObservationRecord).filter(AnomalyObservationRecord.anomaly_job_id == job.id).all()
    )
    assert len(observations) == 10
    ranks = sorted(o.rank for o in observations)
    assert ranks == list(range(1, 11))


def test_worker_marks_job_failed_on_missing_dataset(db_session):
    job = _make_anomaly_job(db_session, feature_columns=["a", "b", "c"])
    dataset = db_session.query(Dataset).filter(Dataset.id == job.dataset_id).first()
    db_session.delete(dataset)
    db_session.commit()

    run_anomaly_job(job.id)

    db_session.expire_all()
    refreshed = db_session.query(AnomalyJob).filter(AnomalyJob.id == job.id).first()
    assert refreshed.status == "failed"
    assert refreshed.error_message == "Dataset introuvable ou non prêt"


def test_worker_never_leaks_raw_traceback_on_failure(db_session, monkeypatch):
    import domains.anomalies.worker as anomaly_worker_module

    raw_exc = RuntimeError(
        'File "E:\\mlops\\app-analyse\\backend\\.venv\\Lib\\site-packages\\sklearn\\ensemble\\_iforest.py", line 42'
    )

    def _raise(*args, **kwargs):
        raise raw_exc

    monkeypatch.setattr(anomaly_worker_module, "train_and_evaluate_anomalies", _raise)

    job = _make_anomaly_job(db_session, feature_columns=["a", "b", "c"])
    run_anomaly_job(job.id)

    db_session.expire_all()
    refreshed = db_session.query(AnomalyJob).filter(AnomalyJob.id == job.id).first()
    assert refreshed.status == "failed"
    assert "sklearn" not in refreshed.error_message
    assert "E:\\" not in refreshed.error_message
    assert "Traceback" not in refreshed.error_message
