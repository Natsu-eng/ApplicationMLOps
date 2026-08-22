"""Tests de domains/clustering/worker.py (Lot 11+) — bout en bout réel
(pas mocké), même approche que tests/test_training_worker.py : le worker
ouvre sa propre session `SessionLocal`, qui pointe vers le même fichier
SQLite de test que `db_session` (voir conftest.py)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from api.core.models import ClusterCandidateRecord, ClusterModel, ClusteringJob, Dataset, Organization
from domains.clustering.worker import run_clustering_job


def _write_temp_csv(df: pd.DataFrame) -> str:
    path = Path(tempfile.gettempdir()) / f"datalab_clustering_test_{np.random.default_rng().integers(1_000_000)}.csv"
    df.to_csv(path, index=False)
    return str(path)


def _make_three_blobs_df(n_per_group: int = 40, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    group = np.repeat([0, 1, 2], n_per_group)
    centers = {0: (0, 0), 1: (12, 12), 2: (0, 12)}
    x1 = np.array([centers[g][0] for g in group]) + rng.normal(0, 0.7, len(group))
    x2 = np.array([centers[g][1] for g in group]) + rng.normal(0, 0.7, len(group))
    return pd.DataFrame({"x1": x1, "x2": x2})


def _make_clustering_job(db, feature_columns: list[str], algorithm_ids: list[str] | None = None) -> ClusteringJob:
    org = Organization(name="Bureau test")
    db.add(org)
    db.flush()

    df = _make_three_blobs_df()
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

    job = ClusteringJob(
        organization_id=org.id,
        dataset_id=dataset.id,
        feature_columns_json=json.dumps(feature_columns),
        config_json=json.dumps({"algorithm_ids": algorithm_ids, "seed": 42}),
        status="queued",
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def test_worker_persists_result_and_candidates(db_session):
    job = _make_clustering_job(db_session, feature_columns=["x1", "x2"])
    run_clustering_job(job.id)

    db_session.expire_all()
    refreshed = db_session.query(ClusteringJob).filter(ClusteringJob.id == job.id).first()
    assert refreshed.status == "completed"

    result = db_session.query(ClusterModel).filter(ClusterModel.clustering_job_id == job.id).first()
    assert result is not None
    assert result.n_clusters == 3
    metrics = json.loads(result.metrics_json)
    assert metrics["silhouette"] > 0.7

    profiles = json.loads(result.profiles_json)
    assert len(profiles) == 3
    assert sum(p["size"] for p in profiles) <= 120  # <= n_total (peut être < si bruit DBSCAN)

    model_card = json.loads(result.model_card_json)
    assert model_card["sampled"] is False
    assert model_card["n_samples_total"] == model_card["n_samples_used"] == 120

    candidates = (
        db_session.query(ClusterCandidateRecord).filter(ClusterCandidateRecord.clustering_job_id == job.id).all()
    )
    assert len(candidates) > 0
    winners = [c for c in candidates if c.is_winner]
    assert len(winners) == 1
    assert winners[0].algorithm == result.algorithm


def test_worker_winner_consistent_between_result_and_candidates(db_session):
    """Même garantie que le Lot D côté supervisé (test_worker_winner_consistent) :
    le résultat persisté et la ligne `is_winner=True` du leaderboard doivent
    toujours désigner le même algorithme, par construction (même transaction,
    mêmes variables source)."""
    job = _make_clustering_job(db_session, feature_columns=["x1", "x2"])
    run_clustering_job(job.id)

    db_session.expire_all()
    result = db_session.query(ClusterModel).filter(ClusterModel.clustering_job_id == job.id).first()
    winner_candidate = (
        db_session.query(ClusterCandidateRecord)
        .filter(ClusterCandidateRecord.clustering_job_id == job.id, ClusterCandidateRecord.is_winner.is_(True))
        .first()
    )
    assert winner_candidate.algorithm == result.algorithm
    assert winner_candidate.n_clusters == result.n_clusters


def test_worker_respects_algorithm_ids_from_config(db_session):
    job = _make_clustering_job(db_session, feature_columns=["x1", "x2"], algorithm_ids=["kmeans"])
    run_clustering_job(job.id)

    db_session.expire_all()
    candidates = (
        db_session.query(ClusterCandidateRecord).filter(ClusterCandidateRecord.clustering_job_id == job.id).all()
    )
    assert all("K-Means" in c.algorithm and "rapide" not in c.algorithm for c in candidates)


def test_worker_marks_job_failed_on_missing_dataset(db_session):
    job = _make_clustering_job(db_session, feature_columns=["x1", "x2"])
    dataset = db_session.query(Dataset).filter(Dataset.id == job.dataset_id).first()
    db_session.delete(dataset)
    db_session.commit()

    run_clustering_job(job.id)

    db_session.expire_all()
    refreshed = db_session.query(ClusteringJob).filter(ClusteringJob.id == job.id).first()
    assert refreshed.status == "failed"
    assert refreshed.error_message == "Dataset introuvable ou non prêt"


def test_worker_never_leaks_raw_traceback_on_clustering_failure(db_session, monkeypatch):
    import domains.clustering.worker as clustering_worker_module

    raw_exc = RuntimeError(
        'File "E:\\mlops\\app-analyse\\backend\\.venv\\Lib\\site-packages\\sklearn\\cluster\\_kmeans.py", line 42'
    )

    def _raise(*args, **kwargs):
        raise raw_exc

    monkeypatch.setattr(clustering_worker_module, "train_and_evaluate_clustering", _raise)

    job = _make_clustering_job(db_session, feature_columns=["x1", "x2"])
    run_clustering_job(job.id)

    db_session.expire_all()
    refreshed = db_session.query(ClusteringJob).filter(ClusteringJob.id == job.id).first()
    assert refreshed.status == "failed"
    assert "sklearn" not in refreshed.error_message
    assert "E:\\" not in refreshed.error_message
    assert "Traceback" not in refreshed.error_message
