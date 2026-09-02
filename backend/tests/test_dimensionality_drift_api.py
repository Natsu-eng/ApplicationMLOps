"""Tests de `GET /dimensionality/jobs/{id}/drift` — voir
test_clustering_drift_api.py pour le raisonnement complet, identique ici."""
from __future__ import annotations

import io
import json
from unittest.mock import patch

import numpy as np

from api.core.models import DimensionalityProjectionLog
from domains.dimensionality.worker import run_dimensionality_job
from domains.shared.drift import MIN_CURRENT_ROWS_FOR_DRIFT

_RNG = np.random.default_rng(42)


def _register(client, email="owner@bureau.fr", org="Bureau"):
    resp = client.post(
        "/api/auth/register",
        json={"email": email, "nom": "Owner", "password": "motdepasse123", "organization_name": org},
    ).json()
    return {"Authorization": f"Bearer {resp['access_token']}"}


def _upload_dataset(client, headers, name="d.csv", n=200):
    x1 = _RNG.normal(50, 10, n)
    x2 = _RNG.normal(20, 5, n)
    rows = "\n".join(f"{x1[i]},{x2[i]},cat{i % 3}" for i in range(n))
    content = f"x1,x2,categorie\n{rows}\n".encode()
    resp = client.post("/api/datasets", headers=headers, files={"file": (name, io.BytesIO(content), "text/csv")})
    return resp.json()


def _create_and_complete_job(client, headers, dataset_id):
    with patch("domains.dimensionality.router.analysis_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        job = client.post(
            "/api/dimensionality/jobs", headers=headers, json={"dataset_id": dataset_id, "feature_columns": ["x1", "x2"]}
        ).json()
    run_dimensionality_job(job["id"])
    return job


def _insert_logs(db, dimensionality_job_id: int, organization_id: int, x1_values: list[float]) -> None:
    for x1 in x1_values:
        db.add(DimensionalityProjectionLog(
            organization_id=organization_id,
            dimensionality_job_id=dimensionality_job_id,
            input_json=json.dumps({"x1": x1, "x2": 20.0}),
        ))
    db.commit()


def test_drift_409_when_no_result_yet(client):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    with patch("domains.dimensionality.router.analysis_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        job = client.post(
            "/api/dimensionality/jobs", headers=headers, json={"dataset_id": dataset["id"], "feature_columns": ["x1", "x2"]}
        ).json()
    resp = client.get(f"/api/dimensionality/jobs/{job['id']}/drift", headers=headers)
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "RESULTAT_INDISPONIBLE"


def test_project_endpoint_persists_a_drift_log_entry(client, db_session):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job = _create_and_complete_job(client, headers, dataset["id"])
    db_session.expire_all()

    resp = client.post(f"/api/dimensionality/jobs/{job['id']}/project", headers=headers, json={"data": {"x1": 50, "x2": 20}})
    assert resp.status_code == 200

    count = (
        db_session.query(DimensionalityProjectionLog)
        .filter(DimensionalityProjectionLog.dimensionality_job_id == job["id"])
        .count()
    )
    assert count == 1


def test_drift_reports_insufficient_data_below_threshold(client, db_session):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job = _create_and_complete_job(client, headers, dataset["id"])
    db_session.expire_all()
    _insert_logs(db_session, job["id"], 1, [50.0] * (MIN_CURRENT_ROWS_FOR_DRIFT - 1))

    resp = client.get(f"/api/dimensionality/jobs/{job['id']}/drift", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["insufficient_data"] is True
    assert body["features"] == []


def test_drift_detects_a_shifted_feature(client, db_session):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job = _create_and_complete_job(client, headers, dataset["id"])
    db_session.expire_all()
    shifted = list(_RNG.normal(150, 5, 100))
    _insert_logs(db_session, job["id"], 1, shifted)

    resp = client.get(f"/api/dimensionality/jobs/{job['id']}/drift", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["insufficient_data"] is False
    x1_entry = next(f for f in body["features"] if f["feature"] == "x1")
    assert x1_entry["severity"] == "significatif"


def test_drift_404_for_other_organization(client, db_session):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    dataset_a = _upload_dataset(client, headers_a)
    job = _create_and_complete_job(client, headers_a, dataset_a["id"])
    db_session.expire_all()
    _insert_logs(db_session, job["id"], 1, list(_RNG.normal(50, 10, 100)))

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    resp = client.get(f"/api/dimensionality/jobs/{job['id']}/drift", headers=headers_b)
    assert resp.status_code == 404
