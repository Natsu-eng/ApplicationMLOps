"""Tests API du router anomalies (Lot 14 — ML non supervisé)."""
from __future__ import annotations

import io
import json
from unittest.mock import patch

from api.core.config import get_settings
from api.core.models import AnomalyJob
from domains.anomalies.worker import run_anomaly_job


def _register(client, email="owner@bureau.fr", org="Bureau"):
    resp = client.post(
        "/api/auth/register",
        json={"email": email, "nom": "Owner", "password": "motdepasse123", "organization_name": org},
    ).json()
    return {"Authorization": f"Bearer {resp['access_token']}"}


def _upload_dataset(client, headers, name="d.csv", n=60):
    rows = "\n".join(f"{i},{i * 2},cat{i % 3}" for i in range(n))
    content = f"x1,x2,categorie\n{rows}\n".encode()
    resp = client.post("/api/datasets", headers=headers, files={"file": (name, io.BytesIO(content), "text/csv")})
    return resp.json()


def _create_job(client, headers, dataset_id, **overrides):
    body = {"dataset_id": dataset_id, "feature_columns": ["x1", "x2"]}
    body.update(overrides)
    with patch("domains.anomalies.router.analysis_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        return client.post("/api/anomalies/jobs", headers=headers, json=body)


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


def test_create_job_rejects_contamination_out_of_range(client):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    resp = _create_job(client, headers, dataset["id"], contamination=0.6)
    assert resp.status_code == 422  # validation Pydantic (Field gt=0, le=0.5)


def test_create_job_rejects_zero_contamination(client):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    resp = _create_job(client, headers, dataset["id"], contamination=0.0)
    assert resp.status_code == 422


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
    resp = client.get("/api/anomalies/jobs", headers=headers_b)
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_job_404_for_other_organization(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    dataset_a = _upload_dataset(client, headers_a, "a.csv")
    job = _create_job(client, headers_a, dataset_a["id"]).json()

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    resp = client.get(f"/api/anomalies/jobs/{job['id']}", headers=headers_b)
    assert resp.status_code == 404


def test_result_endpoint_404_before_completion(client):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"]).json()
    resp = client.get(f"/api/anomalies/jobs/{job['id']}/result", headers=headers)
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "RESULTAT_INDISPONIBLE"


def test_delete_job_removes_it_from_history(client):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"]).json()

    resp = client.delete(f"/api/anomalies/jobs/{job['id']}", headers=headers)
    assert resp.status_code == 204
    assert client.get(f"/api/anomalies/jobs/{job['id']}", headers=headers).status_code == 404


# ── Lot 7, §J.2 — annulation (garde une trace, contrairement à la suppression) ─


def test_cancel_queued_job_marks_it_cancelled_and_keeps_history(client):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"]).json()

    resp = client.post(f"/api/anomalies/jobs/{job['id']}/cancel", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"
    assert client.get(f"/api/anomalies/jobs/{job['id']}", headers=headers).json()["status"] == "cancelled"


# ── Lot 7, §J.2 — notifications SSE ──────────────────────────────────────────


def test_events_stream_closes_immediately_on_terminal_job(client):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"]).json()
    client.post(f"/api/anomalies/jobs/{job['id']}/cancel", headers=headers)

    resp = client.get(f"/api/anomalies/jobs/{job['id']}/events", headers=headers)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert '"status": "cancelled"' in resp.text


def test_events_stream_404_for_other_organization(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    dataset_a = _upload_dataset(client, headers_a, "a.csv")
    job = _create_job(client, headers_a, dataset_a["id"]).json()

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    resp = client.get(f"/api/anomalies/jobs/{job['id']}/events", headers=headers_b)
    assert resp.status_code == 404


def test_cancel_rejects_already_completed_job(client, db_session):
    headers = _register(client)
    dataset = _upload_dataset(client, headers, n=60)
    job = _create_job(client, headers, dataset["id"]).json()
    run_anomaly_job(job["id"])
    db_session.expire_all()

    resp = client.post(f"/api/anomalies/jobs/{job['id']}/cancel", headers=headers)
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "JOB_NON_ANNULABLE"


def test_cancel_404_for_other_organization(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    dataset_a = _upload_dataset(client, headers_a, "a.csv")
    job = _create_job(client, headers_a, dataset_a["id"]).json()

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    resp = client.post(f"/api/anomalies/jobs/{job['id']}/cancel", headers=headers_b)
    assert resp.status_code == 404


# ── Lot 7, §J.2 — relance depuis une configuration existante ────────────────


def test_rerun_creates_a_new_job_with_the_same_configuration(client):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    original = _create_job(client, headers, dataset["id"], top_n=15, contamination=0.1).json()

    with patch("domains.anomalies.router.analysis_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        resp = client.post(f"/api/anomalies/jobs/{original['id']}/rerun", headers=headers)
    assert resp.status_code == 201
    body = resp.json()
    assert body["id"] != original["id"]
    assert body["dataset_id"] == original["dataset_id"]
    assert body["feature_columns"] == original["feature_columns"]
    assert body["status"] == "queued"


def test_rerun_preserves_auto_contamination(client, db_session):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    original = _create_job(client, headers, dataset["id"]).json()

    with patch("domains.anomalies.router.analysis_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        resp = client.post(f"/api/anomalies/jobs/{original['id']}/rerun", headers=headers)
    assert resp.status_code == 201
    new_job = db_session.query(AnomalyJob).filter(AnomalyJob.id == resp.json()["id"]).first()
    assert json.loads(new_job.config_json)["contamination"] == "auto"


def test_rerun_404_for_other_organization(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    dataset_a = _upload_dataset(client, headers_a, "a.csv")
    job = _create_job(client, headers_a, dataset_a["id"]).json()

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    resp = client.post(f"/api/anomalies/jobs/{job['id']}/rerun", headers=headers_b)
    assert resp.status_code == 404


def test_quota_shared_across_all_four_job_types(client):
    """Mélange délibéré des 4 types de job pour vérifier que le quota est
    bien compté ensemble, tous types confondus — pas seulement 2 à la fois
    comme les tests dédiés de chaque module."""
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    limit = get_settings().max_concurrent_jobs_per_org

    with patch("domains.training.router.training_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        client.post("/api/training/jobs", headers=headers, json={"dataset_id": dataset["id"], "target_column": "x1"})

    with patch("domains.clustering.router.analysis_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        client.post(
            "/api/clustering/jobs", headers=headers, json={"dataset_id": dataset["id"], "feature_columns": ["x1", "x2"]}
        )

    with patch("domains.dimensionality.router.analysis_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        client.post(
            "/api/dimensionality/jobs",
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

    result_resp = client.get(f"/api/anomalies/jobs/{job['id']}/result", headers=headers)
    assert result_resp.status_code == 200
    result_body = result_resp.json()
    assert result_body["n_samples_used"] == 60
    assert sum(result_body["score_histogram"]["counts"]) == 60

    obs_resp = client.get(f"/api/anomalies/jobs/{job['id']}/observations", headers=headers)
    assert obs_resp.status_code == 200
    observations = obs_resp.json()
    assert len(observations) == 15
    assert observations == sorted(observations, key=lambda o: o["rank"])
    assert all(o["agreement"] in {"both", "isolation_forest_only", "lof_only", "none"} for o in observations)


def test_explicit_contamination_flows_through_to_model_card(client, db_session):
    headers = _register(client)
    dataset = _upload_dataset(client, headers, n=60)
    job = _create_job(client, headers, dataset["id"], contamination=0.1).json()

    run_anomaly_job(job["id"])
    db_session.expire_all()

    result_resp = client.get(f"/api/anomalies/jobs/{job['id']}/result", headers=headers)
    assert result_resp.status_code == 200
    assert result_resp.json()["model_card"]["contamination"] == 0.1


def test_default_contamination_is_auto_in_model_card(client, db_session):
    headers = _register(client)
    dataset = _upload_dataset(client, headers, n=60)
    job = _create_job(client, headers, dataset["id"]).json()

    run_anomaly_job(job["id"])
    db_session.expire_all()

    result_resp = client.get(f"/api/anomalies/jobs/{job['id']}/result", headers=headers)
    assert result_resp.json()["model_card"]["contamination"] == "auto"


def test_observations_isolated_between_organizations(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    dataset_a = _upload_dataset(client, headers_a, "a.csv")
    job = _create_job(client, headers_a, dataset_a["id"]).json()

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    resp = client.get(f"/api/anomalies/jobs/{job['id']}/observations", headers=headers_b)
    assert resp.status_code == 404
