"""Tests API du router anomalies (Lot 14 — ML non supervisé)."""
from __future__ import annotations

import io
from unittest.mock import patch

from api.core.config import get_settings
from workers.anomaly_worker import run_anomaly_job


def _register(client, email="owner@bureau.fr", org="Bureau"):
    resp = client.post(
        "/auth/register",
        json={"email": email, "nom": "Owner", "password": "motdepasse123", "organization_name": org},
    ).json()
    return {"Authorization": f"Bearer {resp['access_token']}"}


def _upload_dataset(client, headers, name="d.csv", n=60):
    rows = "\n".join(f"{i},{i * 2},cat{i % 3}" for i in range(n))
    content = f"x1,x2,categorie\n{rows}\n".encode()
    resp = client.post("/datasets", headers=headers, files={"file": (name, io.BytesIO(content), "text/csv")})
    return resp.json()


def _create_job(client, headers, dataset_id, **overrides):
    body = {"dataset_id": dataset_id, "feature_columns": ["x1", "x2"]}
    body.update(overrides)
    with patch("api.routers.anomalies.analysis_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        return client.post("/anomalies/jobs", headers=headers, json=body)


def test_create_job_enqueues_and_returns_queued(client):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    resp = _create_job(client, headers, dataset["id"])
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "queued"
    assert body["feature_columns"] == ["x1", "x2"]


def test_create_job_rejects_top_n_out_of_range(client):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    resp = _create_job(client, headers, dataset["id"], top_n=99999)
    assert resp.status_code == 422  # validation Pydantic (Field ge/le)


def test_create_job_rejects_unknown_feature_columns(client):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    resp = _create_job(client, headers, dataset["id"], feature_columns=["colonne_inexistante"])
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "COLONNES_INCONNUES"


def test_create_job_rejects_missing_dataset(client):
    headers = _register(client)
    resp = _create_job(client, headers, dataset_id=999999)
    assert resp.status_code == 404


def test_list_jobs_isolated_between_organizations(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    dataset_a = _upload_dataset(client, headers_a, "a.csv")
    _create_job(client, headers_a, dataset_a["id"])

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    resp = client.get("/anomalies/jobs", headers=headers_b)
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_job_404_for_other_organization(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    dataset_a = _upload_dataset(client, headers_a, "a.csv")
    job = _create_job(client, headers_a, dataset_a["id"]).json()

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    resp = client.get(f"/anomalies/jobs/{job['id']}", headers=headers_b)
    assert resp.status_code == 404


def test_result_endpoint_404_before_completion(client):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"]).json()
    resp = client.get(f"/anomalies/jobs/{job['id']}/result", headers=headers)
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "RESULTAT_INDISPONIBLE"


def test_delete_job_removes_it_from_history(client):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"]).json()

    resp = client.delete(f"/anomalies/jobs/{job['id']}", headers=headers)
    assert resp.status_code == 204
    assert client.get(f"/anomalies/jobs/{job['id']}", headers=headers).status_code == 404


def test_quota_shared_across_all_four_job_types(client):
    """Mélange délibéré des 4 types de job pour vérifier que le quota est
    bien compté ensemble, tous types confondus — pas seulement 2 à la fois
    comme les tests dédiés de chaque module."""
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    limit = get_settings().max_concurrent_jobs_per_org

    with patch("api.routers.training.training_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        client.post("/training/jobs", headers=headers, json={"dataset_id": dataset["id"], "target_column": "x1"})

    with patch("api.routers.clustering.analysis_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        client.post(
            "/clustering/jobs", headers=headers, json={"dataset_id": dataset["id"], "feature_columns": ["x1", "x2"]}
        )

    with patch("api.routers.dimensionality.analysis_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        client.post(
            "/dimensionality/jobs",
            headers=headers,
            json={"dataset_id": dataset["id"], "feature_columns": ["x1", "x2"]},
        )

    remaining_slots = limit - 3
    for _ in range(remaining_slots):
        resp = _create_job(client, headers, dataset["id"])
        assert resp.status_code == 201

    over_limit = _create_job(client, headers, dataset["id"])
    assert over_limit.status_code == 429
    assert over_limit.json()["detail"]["code"] == "QUOTA_ENTRAINEMENTS_ATTEINT"


def test_result_and_observations_after_completion(client, db_session):
    headers = _register(client)
    dataset = _upload_dataset(client, headers, n=60)
    job = _create_job(client, headers, dataset["id"], top_n=15).json()

    run_anomaly_job(job["id"])
    db_session.expire_all()

    result_resp = client.get(f"/anomalies/jobs/{job['id']}/result", headers=headers)
    assert result_resp.status_code == 200
    result_body = result_resp.json()
    assert result_body["n_samples_used"] == 60
    assert sum(result_body["score_histogram"]["counts"]) == 60

    obs_resp = client.get(f"/anomalies/jobs/{job['id']}/observations", headers=headers)
    assert obs_resp.status_code == 200
    observations = obs_resp.json()
    assert len(observations) == 15
    assert observations == sorted(observations, key=lambda o: o["rank"])
    assert all(o["agreement"] in {"both", "isolation_forest_only", "lof_only", "none"} for o in observations)


def test_observations_isolated_between_organizations(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    dataset_a = _upload_dataset(client, headers_a, "a.csv")
    job = _create_job(client, headers_a, dataset_a["id"]).json()

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    resp = client.get(f"/anomalies/jobs/{job['id']}/observations", headers=headers_b)
    assert resp.status_code == 404
