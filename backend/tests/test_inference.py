"""Tests de services/ml_inference.py et de l'endpoint POST /training/jobs/{id}/predict
(Lot 4a) — referme la boucle « entraîner sans pouvoir utiliser » du Lot 3.

L'entraînement est réel (pas mocké) mais exécuté directement en process de
test — pas de Redis/worker nécessaire ; le job et le modèle sont insérés en
base comme le ferait `workers/training_worker.py`, pour tester l'endpoint
de prédiction dans des conditions représentatives.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from api.core.models import Dataset, MLModel, TrainingJob
from services.feature_engineering import apply_upstream_feature_engineering
from services.ml_inference import InferenceError, predict_one
from services.ml_preprocessing import split_dataset
from services.ml_training import TrainingConfig, train_and_evaluate

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
    df = pd.DataFrame(
        {
            "x1": rng.normal(50, 10, n),
            "x2": rng.normal(20, 5, n),
            "cible": None,
        }
    )
    df["cible"] = 2.5 * df["x1"] - 1.2 * df["x2"] + rng.normal(0, 3, n)

    split = split_dataset(df, "cible", ["x1", "x2"], "regression", None, 0.2, 42)
    result = train_and_evaluate(split, "regression", _FAST_CONFIG, lambda s, p: None)

    artifact_path = Path(tempfile.gettempdir()) / "datalab_test_model.joblib"
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


def test_predict_returns_plausible_value_with_interval(client, db_session):
    headers = _register(client)
    # Base fraîche par test (voir conftest._fresh_database) : l'organisation
    # créée par _register est systématiquement la première, id=1.
    job = _train_and_persist_model(db_session, organization_id=1)

    resp = client.post(
        f"/training/jobs/{job.id}/predict",
        headers=headers,
        json={"data": {"x1": 50, "x2": 20}},
    )
    assert resp.status_code == 200
    body = resp.json()
    # x1=50, x2=20 est proche du centre de la distribution d'entraînement
    # (moyennes 50/20) : la prédiction doit être proche de 2.5*50 - 1.2*20 = 101
    assert 70 <= body["prediction"] <= 130
    assert body["interval"] is not None
    assert body["interval"]["low"] <= body["prediction"] <= body["interval"]["high"]


def test_predict_rejects_missing_feature(client, db_session):
    headers = _register(client)
    job = _train_and_persist_model(db_session, organization_id=1)

    resp = client.post(f"/training/jobs/{job.id}/predict", headers=headers, json={"data": {"x1": 50}})
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "PREDICTION_IMPOSSIBLE"


# ── Lot 4c — rejeu de la feature engineering amont à l'inférence ──────────


def _train_bundle_with_datetime_spec():
    rng = np.random.default_rng(3)
    n = 150
    base = pd.Timestamp("2022-01-01")
    df = pd.DataFrame({
        "date": [(base + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d") for d in rng.integers(0, 500, n)],
        "x": rng.normal(50, 10, n),
    })
    df["cible"] = df["x"] * 2 + rng.normal(0, 3, n)

    spec = {"version": 1, "upstream": [{"type": "datetime_decompose", "source_column": "date"}], "pipeline": {}}
    raw_feature_columns = ["date", "x"]
    df_transformed, effective_columns = apply_upstream_feature_engineering(df, raw_feature_columns, spec)

    split = split_dataset(df_transformed, "cible", effective_columns, "regression", None, 0.2, 42)
    result = train_and_evaluate(split, "regression", _FAST_CONFIG, lambda s, p: None)
    result.pipeline_bundle["feature_engineering_spec"] = spec
    return result.pipeline_bundle, raw_feature_columns, df


def test_predict_replays_datetime_feature_engineering_identically_to_training():
    """Preuve de cohérence train↔inférence : la prédiction obtenue via
    `predict_one` (rejeu de la spec amont) doit être identique à celle
    obtenue en appliquant manuellement la même transformation puis le même
    préprocesseur/modèle — aucune divergence d'implémentation possible
    puisque c'est la même fonction partagée dans les deux cas."""
    bundle, raw_feature_columns, df = _train_bundle_with_datetime_spec()

    row = {"date": df.loc[0, "date"], "x": float(df.loc[0, "x"])}
    result = predict_one(bundle, raw_feature_columns, row)

    # Reproduction manuelle, indépendante de predict_one, pour comparaison.
    manual_df, _ = apply_upstream_feature_engineering(
        pd.DataFrame([row]), raw_feature_columns, bundle["feature_engineering_spec"]
    )
    X_proc = bundle["preprocessor"].transform(manual_df)
    X_proc = np.asarray(X_proc.todense()) if hasattr(X_proc, "todense") else np.asarray(X_proc)
    expected_prediction = float(bundle["model"].predict(X_proc)[0])

    assert result["prediction"] == pytest.approx(expected_prediction, rel=1e-9)


def test_predict_rejects_incompatible_feature_engineering_spec_version():
    bundle, raw_feature_columns, df = _train_bundle_with_datetime_spec()
    bundle["feature_engineering_spec"] = {**bundle["feature_engineering_spec"], "version": 999}

    row = {"date": df.loc[0, "date"], "x": float(df.loc[0, "x"])}
    with pytest.raises(InferenceError):
        predict_one(bundle, raw_feature_columns, row)
