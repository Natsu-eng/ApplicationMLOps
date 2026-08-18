"""Tests de la persistance des prédictions (Lot 5, correctif I2,
AUDIT_DATALAB_2026-08-16.md §I2) — `POST /training/jobs/{id}/predict`
persiste désormais chaque prédiction réussie, `GET .../predictions`
l'expose, `services/prediction_retention.py` la purge après
`prediction_retention_days`.

Entraînement réel (pas mocké), exécuté directement en process de test —
même helper que test_inference.py (copié, pas importé : chaque fichier de
test reste autonome, convention déjà suivie partout dans ce projet)."""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from api.core.models import Dataset, MLModel, Prediction, TrainingJob
from services.ml_preprocessing import split_dataset
from services.ml_training import TrainingConfig, train_and_evaluate
from services.model_versioning import next_version

_FAST_CONFIG = TrainingConfig(optuna_trials=3, cv_folds=3)


def _register(client, email="owner@bureau.fr", org="Bureau"):
    resp = client.post(
        "/auth/register",
        json={"email": email, "nom": "Owner", "password": "motdepasse123", "organization_name": org},
    ).json()
    return {"Authorization": f"Bearer {resp['access_token']}"}


def _train_and_persist_model(db, organization_id: int) -> TrainingJob:
    """Entraîne un vrai modèle de régression et l'insère en base, comme le
    ferait workers/training_worker.py — sans passer par Redis."""
    rng = np.random.default_rng(42)
    n = 150
    df = pd.DataFrame({
        "x1": rng.normal(50, 10, n),
        "x2": rng.normal(20, 5, n),
        "cible": None,
    })
    df["cible"] = 2.5 * df["x1"] - 1.2 * df["x2"] + rng.normal(0, 3, n)

    split = split_dataset(df, "cible", ["x1", "x2"], "regression", None, 0.2, 42)
    result = train_and_evaluate(split, "regression", _FAST_CONFIG, lambda s, p: None)

    artifact_path = Path(tempfile.gettempdir()) / f"datalab_test_model_predictions_{organization_id}.joblib"
    joblib.dump(result.pipeline_bundle, artifact_path)

    dataset = Dataset(
        organization_id=organization_id,
        name="dataset_test.csv",
        file_path="unused",
        file_size_bytes=1,
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


def test_successful_prediction_is_persisted_and_listed(client, db_session):
    headers = _register(client)
    job = _train_and_persist_model(db_session, organization_id=1)

    resp = client.post(
        f"/training/jobs/{job.id}/predict", headers=headers, json={"data": {"x1": 50, "x2": 20}}
    )
    assert resp.status_code == 200

    history = client.get(f"/training/jobs/{job.id}/predictions", headers=headers)
    assert history.status_code == 200
    entries = history.json()["entries"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["input"] == {"x1": 50, "x2": 20}
    assert entry["prediction"] == resp.json()["prediction"]
    assert entry["interval"] == resp.json()["interval"]
    assert entry["requested_by"] == "Owner"


def test_failed_prediction_is_not_persisted(client, db_session):
    headers = _register(client)
    job = _train_and_persist_model(db_session, organization_id=1)

    resp = client.post(f"/training/jobs/{job.id}/predict", headers=headers, json={"data": {"x1": 50}})
    assert resp.status_code == 400

    history = client.get(f"/training/jobs/{job.id}/predictions", headers=headers)
    assert history.json()["entries"] == []


def test_persisted_output_never_includes_the_local_explanation(client, db_session):
    """`explanation` (SHAP local) est recalculable à la demande — jamais
    persistée (voir api/core/models.py::Prediction, docstring)."""
    headers = _register(client)
    job = _train_and_persist_model(db_session, organization_id=1)

    resp = client.post(
        f"/training/jobs/{job.id}/predict", headers=headers, json={"data": {"x1": 50, "x2": 20}}
    )
    assert resp.json()["explanation"] is not None  # bien renvoyée dans la réponse HTTP...

    row = db_session.query(Prediction).filter(Prediction.ml_model_id == job.model.id).one()
    assert "explanation" not in json.loads(row.output_json)  # ...mais jamais persistée


def test_predictions_are_isolated_by_organization(client, db_session):
    headers_a = _register(client, email="a@bureau-a.fr", org="Bureau A")
    job_a = _train_and_persist_model(db_session, organization_id=1)
    client.post(f"/training/jobs/{job_a.id}/predict", headers=headers_a, json={"data": {"x1": 50, "x2": 20}})

    headers_b = _register(client, email="b@bureau-b.fr", org="Bureau B")

    # Org B ne peut ni lister ni avoir accès au job d'org A.
    resp = client.get(f"/training/jobs/{job_a.id}/predictions", headers=headers_b)
    assert resp.status_code == 404


def test_old_predictions_are_purged_on_the_next_prediction(client, db_session):
    headers = _register(client)
    job = _train_and_persist_model(db_session, organization_id=1)

    old_date = datetime.now(timezone.utc) - timedelta(days=100)  # au-delà des 90 jours par défaut
    db_session.add(Prediction(
        organization_id=1,
        ml_model_id=job.model.id,
        requested_by_id=None,
        input_json=json.dumps({"x1": 1, "x2": 1}),
        output_json=json.dumps({"prediction": 1.0}),
        created_at=old_date,
    ))
    db_session.commit()

    resp = client.post(
        f"/training/jobs/{job.id}/predict", headers=headers, json={"data": {"x1": 50, "x2": 20}}
    )
    assert resp.status_code == 200

    remaining = db_session.query(Prediction).filter(Prediction.organization_id == 1).all()
    assert len(remaining) == 1  # l'ancienne a été purgée, seule la nouvelle reste
    # SQLite (dev/test) relit un datetime naïf malgré DateTime(timezone=True)
    # — même écart documenté dans services/job_watchdog.py::_as_aware_utc.
    remaining_created_at = remaining[0].created_at.replace(tzinfo=timezone.utc)
    assert remaining_created_at > old_date
