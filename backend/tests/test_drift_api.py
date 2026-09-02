"""Tests de `GET /training/jobs/{id}/model/drift` (dérive des données) —
entraînement réel (pas mocké), même helper que test_predictions.py (copié,
pas importé : convention déjà suivie partout dans ce projet), mais avec un
VRAI fichier dataset sur disque cette fois — `read_dataset_dataframe` doit
pouvoir le relire (contrairement à `test_predictions.py`, qui n'a jamais
besoin de relire le dataset brut et utilise `file_path="unused"`)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from api.core.models import Dataset, MLModel, Prediction, TrainingJob
from domains.shared.drift import MIN_CURRENT_ROWS_FOR_DRIFT
from domains.shared.ml_preprocessing import split_dataset
from domains.training.services.engine import TrainingConfig, train_and_evaluate
from domains.training.services.versioning import next_version

_FAST_CONFIG = TrainingConfig(optuna_trials=3, cv_folds=3)
_RNG = np.random.default_rng(42)


def _register(client, email="owner@bureau.fr", org="Bureau"):
    resp = client.post(
        "/api/auth/register",
        json={"email": email, "nom": "Owner", "password": "motdepasse123", "organization_name": org},
    ).json()
    return {"Authorization": f"Bearer {resp['access_token']}"}


def _train_and_persist_model(db, organization_id: int, tmp_path: Path) -> TrainingJob:
    """Comme `test_predictions.py::_train_and_persist_model`, mais écrit un
    VRAI CSV sur disque (`dataset.file_path` pointe dessus) — nécessaire ici
    car `get_model_drift` relit le dataset d'entraînement comme référence,
    contrairement à `/predict` qui ne touche jamais le fichier brut."""
    n = 300
    df = pd.DataFrame({
        "x1": _RNG.normal(50, 10, n),
        "x2": _RNG.normal(20, 5, n),
        "cible": None,
    })
    df["cible"] = 2.5 * df["x1"] - 1.2 * df["x2"] + _RNG.normal(0, 3, n)
    dataset_path = tmp_path / f"dataset_{organization_id}.csv"
    df.to_csv(dataset_path, index=False)

    split = split_dataset(df, "cible", ["x1", "x2"], "regression", None, 0.2, 42)
    result = train_and_evaluate(split, "regression", _FAST_CONFIG, lambda s, p: None)

    artifact_path = Path(tempfile.gettempdir()) / f"datalab_test_model_drift_{organization_id}.joblib"
    joblib.dump(result.pipeline_bundle, artifact_path)

    dataset = Dataset(
        organization_id=organization_id,
        name="dataset_test.csv",
        file_path=str(dataset_path),
        file_size_bytes=dataset_path.stat().st_size,
        status="ready",
    )
    db.add(dataset)
    db.flush()

    job = TrainingJob(
        organization_id=organization_id,
        dataset_id=dataset.id,
        task_type="regression",
        target_column="cible",
        feature_columns_json=json.dumps(["x1", "x2"]),
        config_json=json.dumps({}),
        status="completed",
    )
    db.add(job)
    db.flush()

    model = MLModel(
        organization_id=organization_id,
        training_job_id=job.id,
        dataset_id=dataset.id,
        version=next_version(db, organization_id, dataset.id, job.target_column),
        algorithm=result.algorithm,
        task_type="regression",
        target_column="cible",
        feature_columns_json=json.dumps(["x1", "x2"]),
        feature_schema_json=json.dumps([{"name": "x1", "dtype": "float64"}, {"name": "x2", "dtype": "float64"}]),
        file_path=str(artifact_path),
        metrics_json=json.dumps(result.metrics),
        shap_summary_json=json.dumps(result.shap_summary),
        cqr_json=json.dumps(result.cqr),
        model_card_json=json.dumps(result.model_card),
        evaluation_json=json.dumps(result.evaluation),
    )
    db.add(model)
    db.commit()
    db.refresh(job)
    return job


def _insert_predictions(db, model_id: int, organization_id: int, x1_values: list[float]) -> None:
    for x1 in x1_values:
        db.add(Prediction(
            organization_id=organization_id,
            ml_model_id=model_id,
            input_json=json.dumps({"x1": x1, "x2": 20.0}),
            output_json=json.dumps({"prediction": 0.0}),
        ))
    db.commit()


def test_drift_404_before_completion(client):
    headers = _register(client)
    resp = client.get("/api/training/jobs/999999/model/drift", headers=headers)
    assert resp.status_code == 404


def test_drift_409_when_no_model_yet(client, db_session):
    headers = _register(client)
    job = TrainingJob(
        organization_id=1,
        dataset_id=1,
        task_type="regression",
        target_column="cible",
        feature_columns_json=json.dumps(["x1"]),
        config_json=json.dumps({}),
        status="running",
    )
    db_session.add(Dataset(id=1, organization_id=1, name="d.csv", file_path="unused", file_size_bytes=1, status="ready"))
    db_session.add(job)
    db_session.commit()

    resp = client.get(f"/api/training/jobs/{job.id}/model/drift", headers=headers)
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "MODELE_NON_DISPONIBLE"


def test_drift_reports_insufficient_data_below_threshold(client, db_session, tmp_path):
    headers = _register(client)
    job = _train_and_persist_model(db_session, organization_id=1, tmp_path=tmp_path)
    _insert_predictions(db_session, job.model.id, 1, [50.0] * (MIN_CURRENT_ROWS_FOR_DRIFT - 1))

    resp = client.get(f"/api/training/jobs/{job.id}/model/drift", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["insufficient_data"] is True
    assert body["features"] == []
    assert body["min_predictions_required"] == MIN_CURRENT_ROWS_FOR_DRIFT


def test_drift_detects_a_shifted_feature_once_enough_predictions_logged(client, db_session, tmp_path):
    """Le dataset d'entraînement a x1 ~ N(50, 10) — des prédictions envoyées
    massivement autour de x1=150 (10 écarts-types plus loin) doivent
    ressortir en dérive significative."""
    headers = _register(client)
    job = _train_and_persist_model(db_session, organization_id=1, tmp_path=tmp_path)
    shifted = list(_RNG.normal(150, 5, 100))
    _insert_predictions(db_session, job.model.id, 1, shifted)

    resp = client.get(f"/api/training/jobs/{job.id}/model/drift", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["insufficient_data"] is False
    assert body["n_predictions_analyzed"] == 100

    x1_entry = next(f for f in body["features"] if f["feature"] == "x1")
    assert x1_entry["severity"] == "significatif"
    assert body["n_significant"] >= 1


def test_drift_reports_stable_when_predictions_match_training_distribution(client, db_session, tmp_path):
    headers = _register(client)
    job = _train_and_persist_model(db_session, organization_id=1, tmp_path=tmp_path)
    matching = list(_RNG.normal(50, 10, 100))
    _insert_predictions(db_session, job.model.id, 1, matching)

    resp = client.get(f"/api/training/jobs/{job.id}/model/drift", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    x1_entry = next(f for f in body["features"] if f["feature"] == "x1")
    assert x1_entry["severity"] == "stable"


def test_drift_404_for_other_organization(client, db_session, tmp_path):
    _register(client, email="a@bureau.fr", org="Bureau A")
    job = _train_and_persist_model(db_session, organization_id=1, tmp_path=tmp_path)
    _insert_predictions(db_session, job.model.id, 1, list(_RNG.normal(50, 10, 100)))

    other_headers = _register(client, email="b@autre.fr", org="Bureau B")
    resp = client.get(f"/api/training/jobs/{job.id}/model/drift", headers=other_headers)
    assert resp.status_code == 404
