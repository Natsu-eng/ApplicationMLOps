"""Tests API du router dimensionality (Lot 13 — ML non supervisé)."""
from __future__ import annotations

import io
import json
from unittest.mock import patch

from api.core.config import get_settings
from api.core.models import DimensionalityJob
from workers.dimensionality_worker import run_dimensionality_job


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
    with patch("api.routers.dimensionality.analysis_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        return client.post("/dimensionality/jobs", headers=headers, json=body)


def test_algorithms_catalog_lists_registry(client):
    headers = _register(client)
    resp = client.get("/dimensionality/algorithms-catalog", headers=headers)
    assert resp.status_code == 200
    algos = resp.json()["algorithms"]
    ids = {a["id"] for a in algos}
    assert {"pca", "tsne"} <= ids
    defaults = {a["id"] for a in algos if a["is_default"]}
    assert defaults == {"pca"}


def test_create_job_enqueues_and_returns_queued(client):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    resp = _create_job(client, headers, dataset["id"])
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "queued"
    assert body["algorithm_id"] == "pca"


def test_create_job_rejects_unknown_algorithm_id(client):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    resp = _create_job(client, headers, dataset["id"], algorithm_id="methode_magique")
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "ALGORITHME_INCONNU"


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
    resp = client.get("/dimensionality/jobs", headers=headers_b)
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_job_404_for_other_organization(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    dataset_a = _upload_dataset(client, headers_a, "a.csv")
    job = _create_job(client, headers_a, dataset_a["id"]).json()

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    resp = client.get(f"/dimensionality/jobs/{job['id']}", headers=headers_b)
    assert resp.status_code == 404


def test_result_endpoint_404_before_completion(client):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"]).json()
    resp = client.get(f"/dimensionality/jobs/{job['id']}/result", headers=headers)
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "RESULTAT_INDISPONIBLE"


def test_delete_job_removes_it_from_history(client):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"]).json()

    resp = client.delete(f"/dimensionality/jobs/{job['id']}", headers=headers)
    assert resp.status_code == 204
    assert client.get(f"/dimensionality/jobs/{job['id']}", headers=headers).status_code == 404


def test_quota_shared_across_supervised_clustering_and_dimensionality(client):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    limit = get_settings().max_concurrent_jobs_per_org

    with patch("api.routers.training.training_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        client.post("/training/jobs", headers=headers, json={"dataset_id": dataset["id"], "target_column": "x1"})

    for _ in range(limit - 1):
        resp = _create_job(client, headers, dataset["id"])
        assert resp.status_code == 201

    over_limit = _create_job(client, headers, dataset["id"])
    assert over_limit.status_code == 429
    assert over_limit.json()["detail"]["code"] == "QUOTA_ENTRAINEMENTS_ATTEINT"


def test_result_points_and_color_by_after_completion(client, db_session):
    """Enfile via l'API (respecte les validations réelles), puis exécute le
    worker directement dans le test (comme les autres suites du projet ne
    tournent pas de vrai process RQ) pour vérifier les 3 endpoints de
    lecture d'un résultat terminé, y compris la coloration à la demande."""
    headers = _register(client)
    dataset = _upload_dataset(client, headers, n=60)
    job = _create_job(client, headers, dataset["id"], algorithm_id="pca").json()

    run_dimensionality_job(job["id"])
    db_session.expire_all()

    result_resp = client.get(f"/dimensionality/jobs/{job['id']}/result", headers=headers)
    assert result_resp.status_code == 200
    result_body = result_resp.json()
    assert result_body["algorithm_id"] == "pca"
    assert len(result_body["loadings"]) > 0

    points_resp = client.get(f"/dimensionality/jobs/{job['id']}/points", headers=headers)
    assert points_resp.status_code == 200
    points = points_resp.json()
    assert len(points) == 60
    assert points == sorted(points, key=lambda p: p["row_index"])

    color_resp = client.get(f"/dimensionality/jobs/{job['id']}/color-by?column=categorie", headers=headers)
    assert color_resp.status_code == 200
    color_body = color_resp.json()
    assert color_body["kind"] == "categorical"
    assert len(color_body["values"]) == 60

    unknown_column_resp = client.get(f"/dimensionality/jobs/{job['id']}/color-by?column=n_existe_pas", headers=headers)
    assert unknown_column_resp.status_code == 400
    assert unknown_column_resp.json()["detail"]["code"] == "COLONNE_INCONNUE"


def test_color_by_isolated_between_organizations(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    dataset_a = _upload_dataset(client, headers_a, "a.csv")
    job = _create_job(client, headers_a, dataset_a["id"]).json()

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    resp = client.get(f"/dimensionality/jobs/{job['id']}/color-by?column=x1", headers=headers_b)
    assert resp.status_code == 404
