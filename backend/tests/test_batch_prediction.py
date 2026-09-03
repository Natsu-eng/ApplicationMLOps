"""Tests de la prédiction en lot (retour utilisateur : "batch prediction —
upload d'un fichier, prédictions pour toutes les lignes") — même approche
que test_inference.py (entraînement réel, pas mocké) pour la partie moteur,
et que test_training_api.py (file RQ mockée) pour la partie router."""
from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from api.core.config import get_settings
from api.core.models import Dataset, MLModel, TrainingJob
from domains.training.batch_prediction_worker import run_batch_prediction_job
from domains.training.services.engine import TrainingConfig, train_and_evaluate
from domains.shared.ml_preprocessing import split_dataset
from domains.training.services.versioning import next_version

_FAST_CONFIG = TrainingConfig(optuna_trials=3, cv_folds=3)


def _register(client, email="owner@bureau.fr", org="Bureau"):
    resp = client.post(
        "/api/auth/register",
        json={"email": email, "nom": "Owner", "password": "motdepasse123", "organization_name": org},
    ).json()
    return {"Authorization": f"Bearer {resp['access_token']}"}


def _train_and_persist_model(db, organization_id: int) -> TrainingJob:
    """Même construction que test_inference.py::_train_and_persist_model —
    entraînement réel, inséré en base comme le ferait
    workers/training_worker.py, sans passer par Redis."""
    rng = np.random.default_rng(42)
    n = 150
    df = pd.DataFrame({"x1": rng.normal(50, 10, n), "x2": rng.normal(20, 5, n)})
    df["cible"] = 2.5 * df["x1"] - 1.2 * df["x2"] + rng.normal(0, 3, n)

    split = split_dataset(df, "cible", ["x1", "x2"], "regression", None, 0.2, 42)
    result = train_and_evaluate(split, "regression", _FAST_CONFIG, lambda s, p: None)

    artifact_path = Path(tempfile.gettempdir()) / f"datalab_test_batch_model_{organization_id}.joblib"
    joblib.dump(result.pipeline_bundle, artifact_path)

    dataset = Dataset(
        organization_id=organization_id, name="dataset_test.csv", file_path="unused", file_size_bytes=1, status="ready"
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


def _upload_predict_batch(client, headers, job_id, content=b"x1,x2\n45,18\n55,22\n", filename="a_predire.csv"):
    with patch("domains.training.batch_prediction_router.analysis_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        return client.post(
            f"/api/training/jobs/{job_id}/predict-batch",
            headers=headers,
            files={"file": (filename, io.BytesIO(content), "text/csv")},
        )


def test_create_batch_prediction_job_enqueues_and_returns_queued(client, db_session):
    headers = _register(client)
    job = _train_and_persist_model(db_session, organization_id=1)

    resp = _upload_predict_batch(client, headers, job.id)

    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "queued"
    assert body["training_job_id"] == job.id
    assert body["input_filename"] == "a_predire.csv"


def test_create_batch_prediction_job_rejects_model_not_ready(client, db_session):
    headers = _register(client)
    dataset = Dataset(organization_id=1, name="d.csv", file_path="unused", file_size_bytes=1, status="ready")
    db_session.add(dataset)
    db_session.flush()
    pending_job = TrainingJob(
        organization_id=1,
        dataset_id=dataset.id,
        task_type="regression",
        target_column="cible",
        feature_columns_json=json.dumps(["x1"]),
        config_json="{}",
        status="running",
    )
    db_session.add(pending_job)
    db_session.commit()

    resp = _upload_predict_batch(client, headers, pending_job.id)
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "MODELE_NON_DISPONIBLE"


def test_create_batch_prediction_job_rejects_empty_file(client, db_session):
    headers = _register(client)
    job = _train_and_persist_model(db_session, organization_id=1)

    resp = _upload_predict_batch(client, headers, job.id, content=b"")
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "DATASET_FICHIER_VIDE"


def test_create_batch_prediction_job_rejects_unsupported_format(client, db_session):
    headers = _register(client)
    job = _train_and_persist_model(db_session, organization_id=1)

    resp = _upload_predict_batch(client, headers, job.id, content=b"not a real file", filename="a.exe")
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "DATASET_FORMAT_NON_SUPPORTE"


def test_create_batch_prediction_job_404_for_other_organization(client, db_session):
    _register(client, "a@bureau-a.fr", "Bureau A")
    job_a = _train_and_persist_model(db_session, organization_id=1)

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    resp = _upload_predict_batch(client, headers_b, job_a.id)
    assert resp.status_code == 404


def test_quota_shared_with_other_job_types(client, db_session):
    """Même discipline que les autres types de job (voir
    test_vision_classification_api.py::test_quota_shared_with_other_job_types)
    — une prédiction en lot occupe un slot de worker comme n'importe quel
    autre job, jamais un type "gratuit" qui contournerait le quota."""
    headers = _register(client)
    job = _train_and_persist_model(db_session, organization_id=1)
    settings = get_settings()
    for _ in range(settings.max_concurrent_jobs_per_org):
        db_session.add(TrainingJob(
            organization_id=1, dataset_id=job.dataset_id, task_type="regression", target_column="cible",
            feature_columns_json="[]", config_json="{}", status="running",
        ))
    db_session.commit()

    resp = _upload_predict_batch(client, headers, job.id)
    assert resp.status_code == 429
    assert resp.json()["detail"]["code"] == "QUOTA_ENTRAINEMENTS_ATTEINT"


def test_batch_prediction_end_to_end_produces_a_downloadable_csv(client, db_session):
    """Entraînement réel + fichier réel + worker réel (pas de mock au-delà de
    la file RQ à la création) — vérifie toute la chaîne : upload -> worker
    -> résultat téléchargeable, colonnes d'origine préservées."""
    headers = _register(client)
    job = _train_and_persist_model(db_session, organization_id=1)

    content = b"id_ligne,x1,x2\nr1,45,18\nr2,55,22\n"
    create_resp = _upload_predict_batch(client, headers, job.id, content=content)
    batch_id = create_resp.json()["id"]

    run_batch_prediction_job(batch_id)
    db_session.expire_all()

    status_resp = client.get(f"/api/training/batch-predictions/{batch_id}", headers=headers)
    assert status_resp.json()["status"] == "completed"
    assert status_resp.json()["n_rows"] == 2

    download_resp = client.get(f"/api/training/batch-predictions/{batch_id}/download", headers=headers)
    assert download_resp.status_code == 200
    result_df = pd.read_csv(io.BytesIO(download_resp.content))
    assert list(result_df["id_ligne"]) == ["r1", "r2"]
    assert "prediction" in result_df.columns
    assert "intervalle_bas" in result_df.columns
    assert "intervalle_haut" in result_df.columns


def test_download_rejects_before_completion(client, db_session):
    headers = _register(client)
    job = _train_and_persist_model(db_session, organization_id=1)
    create_resp = _upload_predict_batch(client, headers, job.id)
    batch_id = create_resp.json()["id"]

    resp = client.get(f"/api/training/batch-predictions/{batch_id}/download", headers=headers)
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "RESULTAT_INDISPONIBLE"


def test_batch_prediction_end_to_end_produces_a_downloadable_excel(client, db_session):
    """Retour utilisateur direct : "on doit télécharger aussi les
    prédictions en format excel pour voir directement" — même résultat que
    le CSV, généré à la volée, jamais un second fichier persisté."""
    headers = _register(client)
    job = _train_and_persist_model(db_session, organization_id=1)

    content = b"id_ligne,x1,x2\nr1,45,18\nr2,55,22\n"
    create_resp = _upload_predict_batch(client, headers, job.id, content=content)
    batch_id = create_resp.json()["id"]

    run_batch_prediction_job(batch_id)
    db_session.expire_all()

    download_resp = client.get(f"/api/training/batch-predictions/{batch_id}/download-excel", headers=headers)
    assert download_resp.status_code == 200
    assert download_resp.headers["content-type"] == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    result_df = pd.read_excel(io.BytesIO(download_resp.content))
    assert list(result_df["id_ligne"]) == ["r1", "r2"]
    assert "prediction" in result_df.columns
    assert "intervalle_bas" in result_df.columns
    assert "intervalle_haut" in result_df.columns


def test_download_excel_rejects_before_completion(client, db_session):
    headers = _register(client)
    job = _train_and_persist_model(db_session, organization_id=1)
    create_resp = _upload_predict_batch(client, headers, job.id)
    batch_id = create_resp.json()["id"]

    resp = client.get(f"/api/training/batch-predictions/{batch_id}/download-excel", headers=headers)
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "RESULTAT_INDISPONIBLE"


def test_worker_marks_job_failed_on_missing_column(client, db_session):
    headers = _register(client)
    job = _train_and_persist_model(db_session, organization_id=1)
    # x2 manquante dans le fichier uploadé.
    create_resp = _upload_predict_batch(client, headers, job.id, content=b"x1\n45\n55\n")
    batch_id = create_resp.json()["id"]

    run_batch_prediction_job(batch_id)
    db_session.expire_all()

    status_resp = client.get(f"/api/training/batch-predictions/{batch_id}", headers=headers)
    assert status_resp.json()["status"] == "failed"
    assert "x2" in status_resp.json()["error_message"]


def test_list_batch_predictions_isolated_between_organizations(client, db_session):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    job_a = _train_and_persist_model(db_session, organization_id=1)
    _upload_predict_batch(client, headers_a, job_a.id)

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    resp = client.get("/api/training/batch-predictions", headers=headers_b)
    assert resp.json() == []


def test_get_batch_prediction_404_for_other_organization(client, db_session):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    job_a = _train_and_persist_model(db_session, organization_id=1)
    batch_id = _upload_predict_batch(client, headers_a, job_a.id).json()["id"]

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    resp = client.get(f"/api/training/batch-predictions/{batch_id}", headers=headers_b)
    assert resp.status_code == 404


def test_cancel_queued_batch_prediction(client, db_session):
    headers = _register(client)
    job = _train_and_persist_model(db_session, organization_id=1)
    batch_id = _upload_predict_batch(client, headers, job.id).json()["id"]

    resp = client.post(f"/api/training/batch-predictions/{batch_id}/cancel", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"

    again = client.post(f"/api/training/batch-predictions/{batch_id}/cancel", headers=headers)
    assert again.status_code == 409


def test_delete_batch_prediction_removes_it_from_history(client, db_session):
    headers = _register(client)
    job = _train_and_persist_model(db_session, organization_id=1)
    batch_id = _upload_predict_batch(client, headers, job.id).json()["id"]

    resp = client.delete(f"/api/training/batch-predictions/{batch_id}", headers=headers)
    assert resp.status_code == 204
    assert client.get(f"/api/training/batch-predictions/{batch_id}", headers=headers).status_code == 404


def test_events_stream_closes_immediately_on_terminal_job(client, db_session):
    headers = _register(client)
    job = _train_and_persist_model(db_session, organization_id=1)
    batch_id = _upload_predict_batch(client, headers, job.id).json()["id"]
    run_batch_prediction_job(batch_id)
    db_session.expire_all()

    resp = client.get(f"/api/training/batch-predictions/{batch_id}/events", headers=headers)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert '"status": "completed"' in resp.text
