"""Tests API du router vision/anomalies (pilier Vision, Lot 15 sous-lot C)."""
from __future__ import annotations

import io
import zipfile
from unittest.mock import patch

import numpy as np
from PIL import Image

from api.core.config import get_settings
from api.core.models import VisionAnomalyJob
from workers.vision_anomaly_worker import run_vision_anomaly_job


def _register(client, email="owner@bureau.fr", org="Bureau"):
    resp = client.post(
        "/api/auth/register",
        json={"email": email, "nom": "Owner", "password": "motdepasse123", "organization_name": org},
    ).json()
    return {"Authorization": f"Bearer {resp['access_token']}"}


def _png_bytes(color=(120, 120, 120), size=(48, 48), variant: int = 0) -> bytes:
    """`variant` rend deux images de même couleur/bruit bit-à-bit distinctes
    — nécessaire depuis la déduplication à l'ingestion (Lot 0.1, correctif
    C1) : la graine fixe (`default_rng(0)`) produisait avant ce correctif le
    même bruit à chaque appel, donc des images "différentes" en réalité
    identiques, y compris entre train/good et test/good (exactement la
    fuite corrigée par C1 — sans `variant`, cette fixture la déclenchait
    involontairement)."""
    rng = np.random.default_rng(variant)
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
            zf.writestr(f"train/good/{i}.png", _png_bytes((120, 120, 120), variant=i + 1))
        for i in range(n_test_good):
            zf.writestr(f"test/good/{i}.png", _png_bytes((120, 120, 120), variant=1000 + i))
        for i in range(n_test_defect):
            zf.writestr(f"test/scratch/{i}.png", _png_bytes((220, 20, 20), variant=2000 + i))
    return buf.getvalue()


def _classification_zip_bytes(n_per_class=4, n_classes=2) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for c in range(n_classes):
            for i in range(n_per_class):
                zf.writestr(f"classe_{c}/img_{i}.png", _png_bytes((10 * c, 10 * c, 10 * c), variant=i + 1))
    return buf.getvalue()


def _upload_vision_dataset(client, headers, content, name="dataset.zip"):
    return client.post(
        "/api/vision/datasets", headers=headers, files={"files": (name, io.BytesIO(content), "application/zip")}
    ).json()


def _create_job(client, headers, vision_dataset_id, **overrides):
    body = {"vision_dataset_id": vision_dataset_id, "num_epochs": 2, "batch_size": 4}
    body.update(overrides)
    with patch("api.routers.vision_anomalies.vision_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        return client.post("/api/vision/anomalies/jobs", headers=headers, json=body)


def test_list_models(client):
    headers = _register(client)
    resp = client.get("/api/vision/anomalies/models", headers=headers)
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
    resp = client.get("/api/vision/anomalies/jobs", headers=headers_b)
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_job_404_for_other_organization(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    dataset_a = _upload_vision_dataset(client, headers_a, _mvtec_zip_bytes())
    job = _create_job(client, headers_a, dataset_a["id"]).json()

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    resp = client.get(f"/api/vision/anomalies/jobs/{job['id']}", headers=headers_b)
    assert resp.status_code == 404


def test_result_endpoint_404_before_completion(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers, _mvtec_zip_bytes())
    job = _create_job(client, headers, dataset["id"]).json()
    resp = client.get(f"/api/vision/anomalies/jobs/{job['id']}/result", headers=headers)
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "RESULTAT_INDISPONIBLE"


def test_delete_job_removes_it_from_history(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers, _mvtec_zip_bytes())
    job = _create_job(client, headers, dataset["id"]).json()

    resp = client.delete(f"/api/vision/anomalies/jobs/{job['id']}", headers=headers)
    assert resp.status_code == 204
    assert client.get(f"/api/vision/anomalies/jobs/{job['id']}", headers=headers).status_code == 404


# ── Lot 7, §J.2 — annulation (garde une trace, contrairement à la suppression) ─


def test_cancel_queued_job_marks_it_cancelled_and_keeps_history(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers, _mvtec_zip_bytes())
    job = _create_job(client, headers, dataset["id"]).json()

    resp = client.post(f"/api/vision/anomalies/jobs/{job['id']}/cancel", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"
    assert client.get(f"/api/vision/anomalies/jobs/{job['id']}", headers=headers).json()["status"] == "cancelled"


# ── Lot 7, §J.2 — notifications SSE ──────────────────────────────────────────


def test_events_stream_closes_immediately_on_terminal_job(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers, _mvtec_zip_bytes())
    job = _create_job(client, headers, dataset["id"]).json()
    client.post(f"/api/vision/anomalies/jobs/{job['id']}/cancel", headers=headers)

    resp = client.get(f"/api/vision/anomalies/jobs/{job['id']}/events", headers=headers)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert '"status": "cancelled"' in resp.text


def test_events_stream_404_for_other_organization(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    dataset_a = _upload_vision_dataset(client, headers_a, _mvtec_zip_bytes())
    job = _create_job(client, headers_a, dataset_a["id"]).json()

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    resp = client.get(f"/api/vision/anomalies/jobs/{job['id']}/events", headers=headers_b)
    assert resp.status_code == 404


def test_cancel_rejects_already_completed_job(client, db_session):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers, _mvtec_zip_bytes())
    job_id = _create_job(client, headers, dataset["id"]).json()["id"]

    job = db_session.query(VisionAnomalyJob).filter(VisionAnomalyJob.id == job_id).first()
    job.status = "completed"
    db_session.commit()

    resp = client.post(f"/api/vision/anomalies/jobs/{job_id}/cancel", headers=headers)
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "JOB_NON_ANNULABLE"


def test_cancel_404_for_other_organization(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    dataset_a = _upload_vision_dataset(client, headers_a, _mvtec_zip_bytes())
    job = _create_job(client, headers_a, dataset_a["id"]).json()

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    resp = client.post(f"/api/vision/anomalies/jobs/{job['id']}/cancel", headers=headers_b)
    assert resp.status_code == 404


# ── Lot 7, §J.2 — relance depuis une configuration existante ────────────────


def test_rerun_creates_a_new_job_with_the_same_configuration(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers, _mvtec_zip_bytes())
    original = _create_job(client, headers, dataset["id"], num_epochs=3).json()

    with patch("api.routers.vision_anomalies.vision_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        resp = client.post(f"/api/vision/anomalies/jobs/{original['id']}/rerun", headers=headers)
    assert resp.status_code == 201
    body = resp.json()
    assert body["id"] != original["id"]
    assert body["vision_dataset_id"] == original["vision_dataset_id"]
    assert body["model_id"] == original["model_id"]
    assert body["status"] == "queued"


def test_rerun_404_for_other_organization(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    dataset_a = _upload_vision_dataset(client, headers_a, _mvtec_zip_bytes())
    job = _create_job(client, headers_a, dataset_a["id"]).json()

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    resp = client.post(f"/api/vision/anomalies/jobs/{job['id']}/rerun", headers=headers_b)
    assert resp.status_code == 404


def test_quota_shared_with_other_job_types(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers, _mvtec_zip_bytes())
    limit = get_settings().max_concurrent_jobs_per_org

    classification_dataset = _upload_vision_dataset(client, headers, _classification_zip_bytes())
    with patch("api.routers.vision_classification.vision_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        client.post(
            "/api/vision/classification/jobs", headers=headers,
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

    result_resp = client.get(f"/api/vision/anomalies/jobs/{job['id']}/result", headers=headers)
    assert result_resp.status_code == 200
    body = result_resp.json()
    assert body["n_test"] == 6
    assert len(body["history"]) == 2
    assert len(body["confusion_matrix"]) == 2

    examples_resp = client.get(f"/api/vision/anomalies/jobs/{job['id']}/examples", headers=headers)
    assert examples_resp.status_code == 200
    examples = examples_resp.json()
    assert len(examples) > 0
    scores = [e["anomaly_score"] for e in examples]
    assert scores == sorted(scores, reverse=True)

    job_resp = client.get(f"/api/vision/anomalies/jobs/{job['id']}", headers=headers)
    assert job_resp.json()["status"] == "completed"
    assert job_resp.json()["roc_auc"] is not None


def test_examples_isolated_between_organizations(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    dataset_a = _upload_vision_dataset(client, headers_a, _mvtec_zip_bytes())
    job = _create_job(client, headers_a, dataset_a["id"]).json()

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    resp = client.get(f"/api/vision/anomalies/jobs/{job['id']}/examples", headers=headers_b)
    assert resp.status_code == 404
