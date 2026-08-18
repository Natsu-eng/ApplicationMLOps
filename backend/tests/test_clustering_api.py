"""Tests API du router clustering (Lot 11+ — ML non supervisé)."""
from __future__ import annotations

import io
from unittest.mock import patch

from api.core.config import get_settings


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


def test_delete_job_removes_it_from_history(client):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"]).json()

    resp = client.delete(f"/api/clustering/jobs/{job['id']}", headers=headers)
    assert resp.status_code == 204
    assert client.get(f"/api/clustering/jobs/{job['id']}", headers=headers).status_code == 404


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
