"""Tests API du router clustering (Lot 11+ — ML non supervisé)."""
from __future__ import annotations

import io
from unittest.mock import patch

from api.core.config import get_settings
from workers.clustering_worker import run_clustering_job


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
    with patch("api.routers.clustering.analysis_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        return client.post("/api/clustering/jobs", headers=headers, json=body)


def test_algorithms_catalog_lists_registry(client):
    headers = _register(client)
    resp = client.get("/api/clustering/algorithms-catalog", headers=headers)
    assert resp.status_code == 200
    algos = resp.json()["algorithms"]
    ids = {a["id"] for a in algos}
    assert {"kmeans", "dbscan", "hierarchical", "minibatch_kmeans"} <= ids
    defaults = {a["id"] for a in algos if a["is_default"]}
    assert defaults == {"kmeans", "dbscan", "hierarchical"}


def test_create_job_enqueues_and_returns_queued(client):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    resp = _create_job(client, headers, dataset["id"])
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "queued"
    assert body["feature_columns"] == ["x1", "x2"]


def test_create_job_rejects_unknown_feature_columns(client):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    resp = _create_job(client, headers, dataset["id"], feature_columns=["colonne_inexistante"])
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "COLONNES_INCONNUES"


def test_create_job_rejects_empty_feature_columns(client):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    resp = _create_job(client, headers, dataset["id"], feature_columns=[])
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "COLONNES_MANQUANTES"


def test_create_job_rejects_unknown_algorithm_ids(client):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    resp = _create_job(client, headers, dataset["id"], algorithm_ids=["algorithme_magique"])
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "ALGORITHMES_INCONNUS"


def test_create_job_rejects_missing_dataset(client):
    headers = _register(client)
    resp = _create_job(client, headers, dataset_id=999999)
    assert resp.status_code == 404


def test_list_jobs_isolated_between_organizations(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    dataset_a = _upload_dataset(client, headers_a, "a.csv")
    _create_job(client, headers_a, dataset_a["id"])

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    resp = client.get("/api/clustering/jobs", headers=headers_b)
    assert resp.status_code == 200
    assert resp.json() == []

    resp_a = client.get("/api/clustering/jobs", headers=headers_a)
    assert len(resp_a.json()) == 1


def test_get_job_404_for_other_organization(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    dataset_a = _upload_dataset(client, headers_a, "a.csv")
    job = _create_job(client, headers_a, dataset_a["id"]).json()

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    resp = client.get(f"/api/clustering/jobs/{job['id']}", headers=headers_b)
    assert resp.status_code == 404


def test_result_endpoint_404_before_completion(client):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"]).json()
    resp = client.get(f"/api/clustering/jobs/{job['id']}/result", headers=headers)
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "RESULTAT_INDISPONIBLE"


def test_predict_endpoint_assigns_new_observation_after_completion(client, db_session):
    headers = _register(client)
    dataset = _upload_dataset(client, headers, n=60)
    job = _create_job(client, headers, dataset["id"], algorithm_ids=["kmeans"]).json()

    run_clustering_job(job["id"])
    db_session.expire_all()

    resp = client.post(
        f"/api/clustering/jobs/{job['id']}/predict", headers=headers, json={"data": {"x1": 0, "x2": 0}}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["assignment_method"] == "exact"
    assert body["cluster_id"] is not None
    assert body["is_noise"] is False


def test_predict_endpoint_rejects_missing_feature(client, db_session):
    headers = _register(client)
    dataset = _upload_dataset(client, headers, n=60)
    job = _create_job(client, headers, dataset["id"], algorithm_ids=["kmeans"]).json()

    run_clustering_job(job["id"])
    db_session.expire_all()

    resp = client.post(f"/api/clustering/jobs/{job['id']}/predict", headers=headers, json={"data": {"x1": 0}})
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "ASSIGNATION_IMPOSSIBLE"


def test_predict_endpoint_409_before_completion(client):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"]).json()
    resp = client.post(f"/api/clustering/jobs/{job['id']}/predict", headers=headers, json={"data": {"x1": 0, "x2": 0}})
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "RESULTAT_INDISPONIBLE"


def test_delete_job_removes_it_from_history(client):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"]).json()

    resp = client.delete(f"/api/clustering/jobs/{job['id']}", headers=headers)
    assert resp.status_code == 204
    assert client.get(f"/api/clustering/jobs/{job['id']}", headers=headers).status_code == 404


# ── Lot 7, §J.2 — annulation (garde une trace, contrairement à la suppression) ─


def test_cancel_queued_job_marks_it_cancelled_and_keeps_history(client):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"]).json()

    resp = client.post(f"/api/clustering/jobs/{job['id']}/cancel", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"

    # Contrairement à DELETE, le job reste consultable dans l'historique.
    still_there = client.get(f"/api/clustering/jobs/{job['id']}", headers=headers)
    assert still_there.status_code == 200
    assert still_there.json()["status"] == "cancelled"


def test_cancel_rejects_already_completed_job(client, db_session):
    headers = _register(client)
    dataset = _upload_dataset(client, headers, n=60)
    job = _create_job(client, headers, dataset["id"], algorithm_ids=["kmeans"]).json()
    run_clustering_job(job["id"])
    db_session.expire_all()

    resp = client.post(f"/api/clustering/jobs/{job['id']}/cancel", headers=headers)
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "JOB_NON_ANNULABLE"


def test_cancel_404_for_other_organization(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    dataset_a = _upload_dataset(client, headers_a, "a.csv")
    job = _create_job(client, headers_a, dataset_a["id"]).json()

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    resp = client.post(f"/api/clustering/jobs/{job['id']}/cancel", headers=headers_b)
    assert resp.status_code == 404


# ── Lot 7, §J.2 — relance depuis une configuration existante ────────────────


def test_rerun_creates_a_new_job_with_the_same_configuration(client):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    original = _create_job(client, headers, dataset["id"], algorithm_ids=["kmeans"], seed=7).json()

    with patch("api.routers.clustering.analysis_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        resp = client.post(f"/api/clustering/jobs/{original['id']}/rerun", headers=headers)
    assert resp.status_code == 201
    body = resp.json()
    assert body["id"] != original["id"]
    assert body["dataset_id"] == original["dataset_id"]
    assert body["feature_columns"] == original["feature_columns"]
    assert body["status"] == "queued"


def test_rerun_404_for_other_organization(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    dataset_a = _upload_dataset(client, headers_a, "a.csv")
    job = _create_job(client, headers_a, dataset_a["id"]).json()

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    resp = client.post(f"/api/clustering/jobs/{job['id']}/rerun", headers=headers_b)
    assert resp.status_code == 404


def test_quota_is_shared_between_supervised_and_clustering_jobs(client):
    """Un seul worker physique traite les deux types de job — la quota doit
    compter les deux ensemble, pas des limites séparées qui laisseraient une
    organisation saturer le worker en cumulant supervisé + clustering."""
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    limit = get_settings().max_concurrent_jobs_per_org

    # Un job supervisé (mocké) consomme déjà un slot.
    with patch("api.routers.training.training_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        client.post("/api/training/jobs", headers=headers, json={"dataset_id": dataset["id"], "target_column": "x1"})

    # Puis (limit - 1) jobs de clustering pour atteindre la limite.
    for _ in range(limit - 1):
        resp = _create_job(client, headers, dataset["id"])
        assert resp.status_code == 201

    over_limit = _create_job(client, headers, dataset["id"])
    assert over_limit.status_code == 429
    assert over_limit.json()["detail"]["code"] == "QUOTA_ENTRAINEMENTS_ATTEINT"
