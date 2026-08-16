"""Tests API du router vision/classification (pilier Vision, Lot 15
sous-lot B)."""
from __future__ import annotations

import io
import zipfile
from unittest.mock import patch

from PIL import Image

from api.core.config import get_settings
from workers.vision_classification_worker import run_vision_classification_job


def _register(client, email="owner@bureau.fr", org="Bureau"):
    resp = client.post(
        "/auth/register",
        json={"email": email, "nom": "Owner", "password": "motdepasse123", "organization_name": org},
    ).json()
    return {"Authorization": f"Bearer {resp['access_token']}"}


def _png_bytes(color=(255, 0, 0), size=(48, 48), variant: int = 0) -> bytes:
    """`variant` rend deux images de même couleur bit-à-bit distinctes —
    nécessaire depuis la déduplication à l'ingestion (Lot 0.1, correctif
    C1) : sans lui, les images "différentes" de ces fixtures étaient en
    réalité des doublons exacts, désormais réduits à une seule copie."""
    img = Image.new("RGB", size, color)
    if variant:
        img.putpixel((0, 0), (variant % 256, (variant * 7) % 256, (variant * 13) % 256))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _classification_zip_bytes(n_per_class=8, n_classes=2) -> bytes:
    buf = io.BytesIO()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    with zipfile.ZipFile(buf, "w") as zf:
        for c in range(n_classes):
            for i in range(n_per_class):
                zf.writestr(f"classe_{c}/img_{i}.png", _png_bytes(colors[c % len(colors)], variant=i + 1))
    return buf.getvalue()


def _upload_vision_dataset(client, headers, content=None, name="dataset.zip"):
    content = content if content is not None else _classification_zip_bytes()
    return client.post(
        "/vision/datasets", headers=headers, files={"file": (name, io.BytesIO(content), "application/zip")}
    ).json()


def _create_job(client, headers, vision_dataset_id, **overrides):
    body = {"vision_dataset_id": vision_dataset_id, "num_epochs": 1, "batch_size": 4}
    body.update(overrides)
    with patch("api.routers.vision_classification.training_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        return client.post("/vision/classification/jobs", headers=headers, json=body)


def test_list_backbones(client):
    headers = _register(client)
    resp = client.get("/vision/classification/backbones", headers=headers)
    assert resp.status_code == 200
    ids = {b["id"] for b in resp.json()}
    assert "mobilenet_v3_small" in ids
    assert "resnet18" in ids


def test_create_job_enqueues_and_returns_queued(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers)
    resp = _create_job(client, headers, dataset["id"])
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "queued"
    assert body["backbone_id"] == "mobilenet_v3_small"


def test_create_job_rejects_unknown_backbone(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers)
    resp = _create_job(client, headers, dataset["id"], backbone_id="resnet152")
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "BACKBONE_INCONNU"


def test_create_job_rejects_epochs_out_of_range(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers)
    resp = _create_job(client, headers, dataset["id"], num_epochs=999)
    assert resp.status_code == 422


def test_create_job_rejects_missing_dataset(client):
    headers = _register(client)
    resp = _create_job(client, headers, vision_dataset_id=999999)
    assert resp.status_code == 404


def test_create_job_rejects_mvtec_dataset(client):
    headers = _register(client)
    files = {}
    for i in range(6):
        files[f"train/good/{i}.png"] = _png_bytes((10, 10, 10), variant=i + 1)
    for i in range(3):
        files[f"test/good/{i}.png"] = _png_bytes((20, 20, 20), variant=i + 1)
    for i in range(3):
        files[f"test/scratch/{i}.png"] = _png_bytes((200, 0, 0), variant=i + 1)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    dataset = _upload_vision_dataset(client, headers, content=buf.getvalue())
    assert dataset["structure_type"] == "mvtec_ad"

    resp = _create_job(client, headers, dataset["id"])
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "VISION_DATASET_STRUCTURE_INVALIDE"


def test_list_jobs_isolated_between_organizations(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    dataset_a = _upload_vision_dataset(client, headers_a)
    _create_job(client, headers_a, dataset_a["id"])

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    resp = client.get("/vision/classification/jobs", headers=headers_b)
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_job_404_for_other_organization(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    dataset_a = _upload_vision_dataset(client, headers_a)
    job = _create_job(client, headers_a, dataset_a["id"]).json()

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    resp = client.get(f"/vision/classification/jobs/{job['id']}", headers=headers_b)
    assert resp.status_code == 404


def test_result_endpoint_404_before_completion(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"]).json()
    resp = client.get(f"/vision/classification/jobs/{job['id']}/result", headers=headers)
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "RESULTAT_INDISPONIBLE"


def test_delete_job_removes_it_from_history(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"]).json()

    resp = client.delete(f"/vision/classification/jobs/{job['id']}", headers=headers)
    assert resp.status_code == 204
    assert client.get(f"/vision/classification/jobs/{job['id']}", headers=headers).status_code == 404


def test_quota_shared_with_other_job_types(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers)
    limit = get_settings().max_concurrent_jobs_per_org

    with patch("api.routers.anomalies.training_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        tabular = client.post(
            "/datasets", headers=headers,
            files={"file": ("d.csv", io.BytesIO(b"x1,x2\n1,2\n3,4\n5,6\n7,8\n"), "text/csv")},
        ).json()
        client.post(
            "/anomalies/jobs", headers=headers,
            json={"dataset_id": tabular["id"], "feature_columns": ["x1", "x2"]},
        )

    remaining_slots = limit - 1
    for _ in range(remaining_slots):
        resp = _create_job(client, headers, dataset["id"])
        assert resp.status_code == 201

    over_limit = _create_job(client, headers, dataset["id"])
    assert over_limit.status_code == 429
    assert over_limit.json()["detail"]["code"] == "QUOTA_ENTRAINEMENTS_ATTEINT"


def test_result_after_completion(client, db_session):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"]).json()

    run_vision_classification_job(job["id"])
    db_session.expire_all()

    result_resp = client.get(f"/vision/classification/jobs/{job['id']}/result", headers=headers)
    assert result_resp.status_code == 200
    body = result_resp.json()
    assert body["class_names"] == ["classe_0", "classe_1"]
    assert body["n_train"] + body["n_val"] + body["n_test"] == 16
    assert len(body["history"]) == 1
    assert len(body["confusion_matrix"]) == 2

    job_resp = client.get(f"/vision/classification/jobs/{job['id']}", headers=headers)
    assert job_resp.json()["status"] == "completed"
    assert job_resp.json()["test_accuracy"] is not None


def test_explain_returns_gradcam_heatmap(client, db_session):
    """Sous-lot D (Grad-CAM) — endpoint synchrone, mêmes conventions que
    `POST /training/jobs/{id}/predict`."""
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"]).json()
    run_vision_classification_job(job["id"])
    db_session.expire_all()

    resp = client.post(
        f"/vision/classification/jobs/{job['id']}/explain",
        headers=headers,
        files={"file": ("test.png", io.BytesIO(_png_bytes()), "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_label"] in {"classe_0", "classe_1"}
    assert body["target_label"] == body["predicted_label"]
    assert set(body["probabilities"]) == {"classe_0", "classe_1"}
    assert body["heatmap_png"].startswith("data:image/png;base64,")


def test_explain_with_explicit_target_label(client, db_session):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"]).json()
    run_vision_classification_job(job["id"])
    db_session.expire_all()

    resp = client.post(
        f"/vision/classification/jobs/{job['id']}/explain",
        headers=headers,
        files={"file": ("test.png", io.BytesIO(_png_bytes()), "image/png")},
        data={"target_label": "classe_1"},
    )
    assert resp.status_code == 200
    assert resp.json()["target_label"] == "classe_1"


def test_explain_rejects_unknown_target_label(client, db_session):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"]).json()
    run_vision_classification_job(job["id"])
    db_session.expire_all()

    resp = client.post(
        f"/vision/classification/jobs/{job['id']}/explain",
        headers=headers,
        files={"file": ("test.png", io.BytesIO(_png_bytes()), "image/png")},
        data={"target_label": "classe_inexistante"},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "CLASSE_INCONNUE"


def test_explain_rejects_invalid_image(client, db_session):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"]).json()
    run_vision_classification_job(job["id"])
    db_session.expire_all()

    resp = client.post(
        f"/vision/classification/jobs/{job['id']}/explain",
        headers=headers,
        files={"file": ("test.png", io.BytesIO(b"not an image"), "image/png")},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "IMAGE_INVALIDE"


def test_explain_404_before_completion(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"]).json()

    resp = client.post(
        f"/vision/classification/jobs/{job['id']}/explain",
        headers=headers,
        files={"file": ("test.png", io.BytesIO(_png_bytes()), "image/png")},
    )
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "RESULTAT_INDISPONIBLE"
