"""GET /dashboard/summary (Lot 4, correctif I3, AUDIT_DATALAB_2026-08-16.md
§C.2.4) — remplace les 8 appels de liste complets faits par `Dashboard.tsx`
au montage par un seul aller-retour agrégé."""
from __future__ import annotations

import io
from unittest.mock import patch


def _register(client, email="owner@bureau.fr", org="Bureau"):
    resp = client.post(
        "/api/auth/register",
        json={"email": email, "nom": "Owner", "password": "motdepasse123", "organization_name": org},
    ).json()
    return {"Authorization": f"Bearer {resp['access_token']}"}


def _upload_dataset(client, headers, name="d.csv"):
    content = b"x1,x2,cible\n" + b"\n".join(f"{i},{i * 2},{i % 2}".encode() for i in range(20))
    resp = client.post("/api/datasets", headers=headers, files={"file": (name, io.BytesIO(content), "text/csv")})
    return resp.json()


def test_summary_reflects_empty_organization(client):
    headers = _register(client)
    resp = client.get("/api/dashboard/summary", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["members_count"] == 1  # le owner lui-même
    assert body["datasets_count"] == 0
    assert body["recent_datasets"] == []
    assert body["supervised_count"] == 0
    assert body["unsupervised_count"] == 0
    assert body["vision_count"] == 0
    assert body["active_count"] == 0
    assert body["recent_supervised"] == []


def test_summary_counts_datasets_and_recent_list(client):
    headers = _register(client)
    for i in range(3):
        _upload_dataset(client, headers, name=f"d{i}.csv")

    body = client.get("/api/dashboard/summary", headers=headers).json()
    assert body["datasets_count"] == 3
    assert len(body["recent_datasets"]) == 3
    # Le plus récent en tête (created_at desc), même ordre que GET /datasets.
    assert body["recent_datasets"][0]["name"] == "d2.csv"


def test_summary_counts_jobs_per_pillar(client):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)

    with patch("api.routers.training.training_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        client.post("/api/training/jobs", headers=headers, json={"dataset_id": dataset["id"], "target_column": "cible"})

    with patch("api.routers.clustering.analysis_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        client.post(
            "/api/clustering/jobs", headers=headers, json={"dataset_id": dataset["id"], "feature_columns": ["x1", "x2"]}
        )
        client.post(
            "/api/clustering/jobs", headers=headers, json={"dataset_id": dataset["id"], "feature_columns": ["x1", "x2"]}
        )

    body = client.get("/api/dashboard/summary", headers=headers).json()
    assert body["supervised_count"] == 1
    assert body["unsupervised_count"] == 2  # clustering seul ici, mais compté dans le total non supervisé
    assert body["vision_count"] == 0
    assert body["active_count"] == 3  # les 3 jobs sont "queued" (worker mocké, jamais exécuté)
    assert len(body["recent_supervised"]) == 1
    assert len(body["recent_clustering"]) == 2


def test_summary_isolated_between_organizations(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    _upload_dataset(client, headers_a, "a.csv")

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")

    body_b = client.get("/api/dashboard/summary", headers=headers_b).json()
    assert body_b["datasets_count"] == 0
    assert body_b["recent_datasets"] == []
    assert body_b["members_count"] == 1  # uniquement le owner de B, jamais celui de A


def test_summary_recent_supervised_matches_list_training_jobs_shape(client):
    """Même schéma que GET /training/jobs (dataset_name, headline_metric,
    etc.) — réutilise `to_summary`, jamais une forme dupliquée."""
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    with patch("api.routers.training.training_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        client.post("/api/training/jobs", headers=headers, json={"dataset_id": dataset["id"], "target_column": "cible"})

    from_list = client.get("/api/training/jobs", headers=headers).json()[0]
    from_summary = client.get("/api/dashboard/summary", headers=headers).json()["recent_supervised"][0]
    assert from_list == from_summary
