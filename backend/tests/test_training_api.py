"""Tests du router training (Lot 3) — validation et isolation.

La file RQ est mockée : ces tests valident la logique de l'endpoint (pas
l'exécution réelle du worker, qui exige Redis — couverte séparément et sans
dépendance externe par test_ml_training.py, qui appelle directement
`train_and_evaluate`)."""
from __future__ import annotations

import io
from unittest.mock import patch


def _register(client, email="owner@bureau.fr", org="Bureau"):
    resp = client.post(
        "/auth/register",
        json={"email": email, "nom": "Owner", "password": "motdepasse123", "organization_name": org},
    ).json()
    return {"Authorization": f"Bearer {resp['access_token']}"}


def _upload_dataset(client, headers):
    # 50 lignes, cible à valeurs toutes distinctes : au-delà du seuil de
    # cardinalité de detect_task_type (20), pour être détecté sans ambiguïté
    # comme régression (voir services/ml_task.py).
    rows = "\n".join(f"{i},{i * 2},{i * 3}" for i in range(50))
    content = f"x1,x2,cible\n{rows}\n".encode()
    resp = client.post("/datasets", headers=headers, files={"file": ("d.csv", io.BytesIO(content), "text/csv")})
    return resp.json()


@patch("api.routers.training.training_queue")
def test_create_job_enqueues_and_returns_summary(mock_queue, client):
    mock_queue.enqueue.return_value.id = "fake-rq-id"
    headers = _register(client)
    dataset = _upload_dataset(client, headers)

    resp = client.post(
        "/training/jobs", headers=headers, json={"dataset_id": dataset["id"], "target_column": "cible"}
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "queued"
    assert body["task_type"] == "regression"
    mock_queue.enqueue.assert_called_once()


@patch("api.routers.training.training_queue")
def test_create_job_rejects_unknown_target_column(mock_queue, client):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)

    resp = client.post(
        "/training/jobs",
        headers=headers,
        json={"dataset_id": dataset["id"], "target_column": "colonne_inexistante"},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "COLONNE_CIBLE_INTROUVABLE"


@patch("api.routers.training.training_queue")
def test_create_job_rejects_unready_dataset(mock_queue, client):
    headers = _register(client)
    resp = client.post(
        "/training/jobs", headers=headers, json={"dataset_id": 999999, "target_column": "cible"}
    )
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "DATASET_INTROUVABLE"


@patch("api.routers.training.training_queue")
def test_training_job_isolation_between_organizations(mock_queue, client):
    mock_queue.enqueue.return_value.id = "fake-rq-id"
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    dataset = _upload_dataset(client, headers_a)

    job = client.post(
        "/training/jobs", headers=headers_a, json={"dataset_id": dataset["id"], "target_column": "cible"}
    ).json()

    assert client.get("/training/jobs", headers=headers_b).json() == []
    assert client.get(f"/training/jobs/{job['id']}", headers=headers_b).status_code == 404


@patch("api.routers.training.training_queue")
def test_delete_removes_job_from_history(mock_queue, client):
    mock_queue.enqueue.return_value.id = "fake-rq-id"
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job = client.post(
        "/training/jobs", headers=headers, json={"dataset_id": dataset["id"], "target_column": "cible"}
    ).json()

    assert client.delete(f"/training/jobs/{job['id']}", headers=headers).status_code == 204
    assert client.get(f"/training/jobs/{job['id']}", headers=headers).status_code == 404
    assert client.get("/training/jobs", headers=headers).json() == []


@patch("api.routers.training.training_queue")
def test_delete_rejects_cross_organization(mock_queue, client):
    mock_queue.enqueue.return_value.id = "fake-rq-id"
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    dataset = _upload_dataset(client, headers_a)
    job = client.post(
        "/training/jobs", headers=headers_a, json={"dataset_id": dataset["id"], "target_column": "cible"}
    ).json()

    resp = client.delete(f"/training/jobs/{job['id']}", headers=headers_b)
    assert resp.status_code == 404
    # toujours là côté organisation A — la tentative de B n'a rien supprimé
    assert client.get(f"/training/jobs/{job['id']}", headers=headers_a).status_code == 200
