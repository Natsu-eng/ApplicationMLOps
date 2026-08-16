"""Tests API du router vision/anomalies (pilier Vision, Lot 15 sous-lot C)."""
from __future__ import annotations

import io
import zipfile
from unittest.mock import patch

import numpy as np
from PIL import Image

from api.core.config import get_settings
from workers.vision_anomaly_worker import run_vision_anomaly_job


def _register(client, email="owner@bureau.fr", org="Bureau"):
    resp = client.post(
        "/auth/register",
        json={"email": email, "nom": "Owner", "password": "motdepasse123", "organization_name": org},
    ).json()
    return {"Authorization": f"Bearer {resp['access_token']}"}


def _png_bytes(color=(120, 120, 120), size=(48, 48)) -> bytes:
    rng = np.random.default_rng(0)
    arr = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    noise = rng.integers(-10, 10, (size[1], size[0], 3))
    arr = np.clip(arr.astype(int) + noise, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _mvtec_zip_bytes(n_train_good=12, n_test_good=3, n_test_defect=3) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for i in range(n_train_good):
            zf.writestr(f"train/good/{i}.png", _png_bytes((120, 120, 120)))
        for i in range(n_test_good):
            zf.writestr(f"test/good/{i}.png", _png_bytes((120, 120, 120)))
        for i in range(n_test_defect):
            zf.writestr(f"test/scratch/{i}.png", _png_bytes((220, 20, 20)))
    return buf.getvalue()


def _classification_zip_bytes(n_per_class=4, n_classes=2) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for c in range(n_classes):
            for i in range(n_per_class):
                zf.writestr(f"classe_{c}/img_{i}.png", _png_bytes((10 * c, 10 * c, 10 * c)))
    return buf.getvalue()


def _upload_vision_dataset(client, headers, content, name="dataset.zip"):
    return client.post(
        "/vision/datasets", headers=headers, files={"file": (name, io.BytesIO(content), "application/zip")}
    ).json()


def _create_job(client, headers, vision_dataset_id, **overrides):
    body = {"vision_dataset_id": vision_dataset_id, "num_epochs": 2, "batch_size": 4}
    body.update(overrides)
    with patch("api.routers.vision_anomalies.training_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        return client.post("/vision/anomalies/jobs", headers=headers, json=body)


def test_list_models(client):
    headers = _register(client)
    resp = client.get("/vision/anomalies/models", headers=headers)
    assert resp.status_code == 200
    ids = {m["id"] for m in resp.json()}
    assert "conv_autoencoder" in ids


def test_create_job_enqueues_and_returns_queued(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers, _mvtec_zip_bytes())
    resp = _create_job(client, headers, dataset["id"])
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "queued"
    assert body["model_id"] == "conv_autoencoder"


def test_create_job_rejects_unknown_model(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers, _mvtec_zip_bytes())
    resp = _create_job(client, headers, dataset["id"], model_id="patchcore")
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "MODELE_INCONNU"


def test_create_job_rejects_classification_dataset(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers, _classification_zip_bytes())
    assert dataset["structure_type"] == "classification"

    resp = _create_job(client, headers, dataset["id"])
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "VISION_DATASET_STRUCTURE_INVALIDE"


def test_create_job_rejects_missing_dataset(client):
    headers = _register(client)
    resp = _create_job(client, headers, vision_dataset_id=999999)
    assert resp.status_code == 404


def test_list_jobs_isolated_between_organizations(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    dataset_a = _upload_vision_dataset(client, headers_a, _mvtec_zip_bytes())
    _create_job(client, headers_a, dataset_a["id"])

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    resp = client.get("/vision/anomalies/jobs", headers=headers_b)
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_job_404_for_other_organization(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    dataset_a = _upload_vision_dataset(client, headers_a, _mvtec_zip_bytes())
    job = _create_job(client, headers_a, dataset_a["id"]).json()

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    resp = client.get(f"/vision/anomalies/jobs/{job['id']}", headers=headers_b)
    assert resp.status_code == 404


def test_result_endpoint_404_before_completion(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers, _mvtec_zip_bytes())
    job = _create_job(client, headers, dataset["id"]).json()
    resp = client.get(f"/vision/anomalies/jobs/{job['id']}/result", headers=headers)
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "RESULTAT_INDISPONIBLE"


def test_delete_job_removes_it_from_history(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers, _mvtec_zip_bytes())
    job = _create_job(client, headers, dataset["id"]).json()

    resp = client.delete(f"/vision/anomalies/jobs/{job['id']}", headers=headers)
    assert resp.status_code == 204
    assert client.get(f"/vision/anomalies/jobs/{job['id']}", headers=headers).status_code == 404


def test_quota_shared_with_other_job_types(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers, _mvtec_zip_bytes())
    limit = get_settings().max_concurrent_jobs_per_org

    classification_dataset = _upload_vision_dataset(client, headers, _classification_zip_bytes())
    with patch("api.routers.vision_classification.training_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        client.post(
            "/vision/classification/jobs", headers=headers,
            json={"vision_dataset_id": classification_dataset["id"], "num_epochs": 1},
        )

    remaining_slots = limit - 1
    for _ in range(remaining_slots):
        resp = _create_job(client, headers, dataset["id"])
        assert resp.status_code == 201

    over_limit = _create_job(client, headers, dataset["id"])
    assert over_limit.status_code == 429
    assert over_limit.json()["detail"]["code"] == "QUOTA_ENTRAINEMENTS_ATTEINT"


def test_result_and_examples_after_completion(client, db_session):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers, _mvtec_zip_bytes())
    job = _create_job(client, headers, dataset["id"]).json()

    run_vision_anomaly_job(job["id"])
    db_session.expire_all()

    result_resp = client.get(f"/vision/anomalies/jobs/{job['id']}/result", headers=headers)
    assert result_resp.status_code == 200
    body = result_resp.json()
    assert body["n_test"] == 6
    assert len(body["history"]) == 2
    assert len(body["confusion_matrix"]) == 2

    examples_resp = client.get(f"/vision/anomalies/jobs/{job['id']}/examples", headers=headers)
    assert examples_resp.status_code == 200
    examples = examples_resp.json()
    assert len(examples) > 0
    scores = [e["anomaly_score"] for e in examples]
    assert scores == sorted(scores, reverse=True)

    job_resp = client.get(f"/vision/anomalies/jobs/{job['id']}", headers=headers)
    assert job_resp.json()["status"] == "completed"
    assert job_resp.json()["roc_auc"] is not None


def test_examples_isolated_between_organizations(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    dataset_a = _upload_vision_dataset(client, headers_a, _mvtec_zip_bytes())
    job = _create_job(client, headers_a, dataset_a["id"]).json()

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    resp = client.get(f"/vision/anomalies/jobs/{job['id']}/examples", headers=headers_b)
    assert resp.status_code == 404
