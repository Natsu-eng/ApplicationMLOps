"""Tests de workers/training_worker.py (Lot 4c) — orchestration bout en bout
de la feature engineering : transformations amont appliquées avant le split,
config pipeline transmise à train_and_evaluate, spec persistée dans le bundle
et sur MLModel, distinction colonnes saisies (feature_columns_json) / colonnes
dérivées (jamais exposées comme telles).

`run_training_job` ouvre sa propre session DB (`SessionLocal`, voir
`api/core/database.py`) — comme documenté dans `conftest.py`, elle pointe
vers le même fichier SQLite de test que la fixture `db_session` tant que les
écritures sont commit avant l'appel."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from api.core.models import Dataset, MLModel, Organization, TrainingJob
from workers.training_worker import run_training_job


def _write_temp_csv(df: pd.DataFrame) -> str:
    path = Path(tempfile.gettempdir()) / f"datalab_worker_test_{np.random.default_rng().integers(1_000_000)}.csv"
    df.to_csv(path, index=False)
    return str(path)


def _make_job(db, feature_engineering_spec: dict | None, feature_columns: list[str]) -> TrainingJob:
    org = Organization(name="Bureau test")
    db.add(org)
    db.flush()

    rng = np.random.default_rng(21)
    n = 200
    base = pd.Timestamp("2022-01-01")
    df = pd.DataFrame({
        "date": [(base + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d") for d in rng.integers(0, 500, n)],
        "x": rng.normal(50, 10, n),
    })
    df["cible"] = df["x"] * 2 + rng.normal(0, 3, n)
    csv_path = _write_temp_csv(df)

    dataset = Dataset(
        organization_id=org.id, name="d.csv", file_path=csv_path, file_size_bytes=1,
        status="ready", columns_json=json.dumps([{"name": c, "dtype": str(df[c].dtype)} for c in df.columns]),
    )
    db.add(dataset)
    db.flush()

    job = TrainingJob(
        organization_id=org.id,
        dataset_id=dataset.id,
        task_type="regression",
        target_column="cible",
        feature_columns_json=json.dumps(feature_columns),
        config_json=json.dumps({"optuna_trials": 3, "cv_folds": 3}),
        feature_engineering_json=json.dumps(feature_engineering_spec) if feature_engineering_spec else None,
        status="queued",
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def test_worker_applies_upstream_feature_engineering_and_persists_spec(db_session):
    spec = {"version": 1, "upstream": [{"type": "datetime_decompose", "source_column": "date"}], "pipeline": {}}
    job = _make_job(db_session, spec, feature_columns=["date", "x"])

    run_training_job(job.id)

    db_session.expire_all()
    refreshed = db_session.query(TrainingJob).filter(TrainingJob.id == job.id).first()
    assert refreshed.status == "completed"

    model = db_session.query(MLModel).filter(MLModel.training_job_id == job.id).first()
    assert model is not None

    # Colonnes SAISIES (feature_columns_json) restent les colonnes brutes —
    # jamais les dérivées (précision 3 du cadrage).
    assert json.loads(model.feature_columns_json) == ["date", "x"]

    # La spec est bien persistée côté MLModel (transparence) et dans le bundle
    # joblib (c'est celle-ci qui compte pour l'inférence).
    assert json.loads(model.feature_engineering_json) == spec
    bundle = joblib.load(model.file_path)
    assert bundle["feature_engineering_spec"] == spec

    # Le préprocesseur a bien vu les colonnes dérivées, pas "date" brute.
    feature_names = bundle["feature_names"]
    assert any("date_annee" in name for name in feature_names)
    assert not any(name.startswith("date__") or name == "date" for name in feature_names)


def test_worker_without_feature_engineering_spec_behaves_as_before(db_session):
    job = _make_job(db_session, None, feature_columns=["x"])

    run_training_job(job.id)

    db_session.expire_all()
    refreshed = db_session.query(TrainingJob).filter(TrainingJob.id == job.id).first()
    assert refreshed.status == "completed"

    model = db_session.query(MLModel).filter(MLModel.training_job_id == job.id).first()
    assert model.feature_engineering_json is None
    bundle = joblib.load(model.file_path)
    assert "feature_engineering_spec" not in bundle
