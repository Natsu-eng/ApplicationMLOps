"""Tests API du router vision/classification (pilier Vision, Lot 15
sous-lot B)."""
from __future__ import annotations

import io
import zipfile
from unittest.mock import patch

from PIL import Image

from api.core.config import get_settings
from api.core.models import VisionClassificationJob
from domains.vision.classification.worker import run_vision_classification_job


def _register(client, email="owner@bureau.fr", org="Bureau"):
    resp = client.post(
        "/api/auth/register",
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
        "/api/vision/datasets", headers=headers, files={"files": (name, io.BytesIO(content), "application/zip")}
    ).json()


def _create_job(client, headers, vision_dataset_id, **overrides):
    body = {"vision_dataset_id": vision_dataset_id, "num_epochs": 1, "batch_size": 4}
    body.update(overrides)
    with patch("domains.vision.classification.router.vision_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        return client.post("/api/vision/classification/jobs", headers=headers, json=body)


def test_list_backbones(client):
    headers = _register(client)
    resp = client.get("/api/vision/classification/backbones", headers=headers)
    assert resp.status_code == 200
    ids = {b["id"] for b in resp.json()}
    # Lot 6A — registre étendu de 2 à 7 backbones (voir DECISIONS.md).
    assert ids == {
        "mobilenet_v3_small", "mobilenet_v3_large", "resnet18", "resnet34",
        "efficientnet_b0", "shufflenet_v2", "densenet121",
    }


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


def test_create_job_rejects_unknown_augmentation_preset(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers)
    resp = _create_job(client, headers, dataset["id"], augmentation_preset="extreme")
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "AUGMENTATION_PRESET_INCONNU"


def test_create_job_defaults_to_standard_augmentation(client, db_session):
    import json as _json

    from api.core.models import VisionClassificationJob

    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers)
    with patch("domains.vision.classification.router.vision_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        job_id = client.post(
            "/api/vision/classification/jobs", headers=headers,
            json={"vision_dataset_id": dataset["id"], "num_epochs": 1, "batch_size": 4},
        ).json()["id"]

    job = db_session.query(VisionClassificationJob).filter(VisionClassificationJob.id == job_id).first()
    assert _json.loads(job.config_json)["augmentation_preset"] == "standard"


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
    resp = client.get("/api/vision/classification/jobs", headers=headers_b)
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_job_404_for_other_organization(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    dataset_a = _upload_vision_dataset(client, headers_a)
    job = _create_job(client, headers_a, dataset_a["id"]).json()

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    resp = client.get(f"/api/vision/classification/jobs/{job['id']}", headers=headers_b)
    assert resp.status_code == 404


def test_result_endpoint_404_before_completion(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"]).json()
    resp = client.get(f"/api/vision/classification/jobs/{job['id']}/result", headers=headers)
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "RESULTAT_INDISPONIBLE"


def test_delete_job_removes_it_from_history(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"]).json()

    resp = client.delete(f"/api/vision/classification/jobs/{job['id']}", headers=headers)
    assert resp.status_code == 204
    assert client.get(f"/api/vision/classification/jobs/{job['id']}", headers=headers).status_code == 404


# ── Lot 7, §J.2 — annulation (garde une trace, contrairement à la suppression) ─


def test_cancel_queued_job_marks_it_cancelled_and_keeps_history(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"]).json()

    resp = client.post(f"/api/vision/classification/jobs/{job['id']}/cancel", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"
    assert client.get(f"/api/vision/classification/jobs/{job['id']}", headers=headers).json()["status"] == "cancelled"


# ── Lot 7, §J.2 — notifications SSE ──────────────────────────────────────────


def test_events_stream_closes_immediately_on_terminal_job(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"]).json()
    client.post(f"/api/vision/classification/jobs/{job['id']}/cancel", headers=headers)

    resp = client.get(f"/api/vision/classification/jobs/{job['id']}/events", headers=headers)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert '"status": "cancelled"' in resp.text


def test_events_stream_404_for_other_organization(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    dataset_a = _upload_vision_dataset(client, headers_a)
    job = _create_job(client, headers_a, dataset_a["id"]).json()

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    resp = client.get(f"/api/vision/classification/jobs/{job['id']}/events", headers=headers_b)
    assert resp.status_code == 404


def test_cancel_rejects_already_completed_job(client, db_session):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers)
    job_id = _create_job(client, headers, dataset["id"]).json()["id"]

    job = db_session.query(VisionClassificationJob).filter(VisionClassificationJob.id == job_id).first()
    job.status = "completed"
    db_session.commit()

    resp = client.post(f"/api/vision/classification/jobs/{job_id}/cancel", headers=headers)
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "JOB_NON_ANNULABLE"


def test_cancel_404_for_other_organization(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    dataset_a = _upload_vision_dataset(client, headers_a)
    job = _create_job(client, headers_a, dataset_a["id"]).json()

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    resp = client.post(f"/api/vision/classification/jobs/{job['id']}/cancel", headers=headers_b)
    assert resp.status_code == 404


# ── Lot 7, §J.2 — relance depuis une configuration existante ────────────────


def test_rerun_creates_a_new_job_with_the_same_configuration(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers)
    original = _create_job(client, headers, dataset["id"], num_epochs=3).json()

    with patch("domains.vision.classification.router.vision_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        resp = client.post(f"/api/vision/classification/jobs/{original['id']}/rerun", headers=headers)
    assert resp.status_code == 201
    body = resp.json()
    assert body["id"] != original["id"]
    assert body["vision_dataset_id"] == original["vision_dataset_id"]
    assert body["backbone_id"] == original["backbone_id"]
    assert body["status"] == "queued"


def test_rerun_404_for_other_organization(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    dataset_a = _upload_vision_dataset(client, headers_a)
    job = _create_job(client, headers_a, dataset_a["id"]).json()

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    resp = client.post(f"/api/vision/classification/jobs/{job['id']}/rerun", headers=headers_b)
    assert resp.status_code == 404


def test_quota_shared_with_other_job_types(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers)
    limit = get_settings().max_concurrent_jobs_per_org

    with patch("domains.anomalies.router.analysis_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        tabular = client.post(
            "/api/datasets", headers=headers,
            files={"file": ("d.csv", io.BytesIO(b"x1,x2\n1,2\n3,4\n5,6\n7,8\n"), "text/csv")},
        ).json()
        client.post(
            "/api/anomalies/jobs", headers=headers,
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

    result_resp = client.get(f"/api/vision/classification/jobs/{job['id']}/result", headers=headers)
    assert result_resp.status_code == 200
    body = result_resp.json()
    assert body["class_names"] == ["classe_0", "classe_1"]
    assert body["n_train"] + body["n_val"] + body["n_test"] == 16
    assert len(body["history"]) == 1
    assert len(body["confusion_matrix"]) == 2
    # Lot 6A (correctif 16G) — binaire (2 classes) : une seule courbe ROC/PR,
    # persistée par le worker et exposée telle quelle par l'endpoint.
    assert set(body["roc_curves"].keys()) == {"classe_1"}
    assert body["test_roc_auc"] is not None
    # Onglet "Fiabilité" (retour utilisateur : "d'autres fonctionnalités
    # modernes que les autres plateformes n'offrent pas").
    assert set(body["calibration"].keys()) <= {"classe_1"}

    job_resp = client.get(f"/api/vision/classification/jobs/{job['id']}", headers=headers)
    assert job_resp.json()["status"] == "completed"
    assert job_resp.json()["test_accuracy"] is not None


def test_explain_blocked_after_too_many_attempts(client, db_session):
    """Lot 1.4 (§C.2.7/§D.4, AUDIT_DATALAB_2026-08-16.md) — /explain charge
    un modèle torch à chaque appel, le plus coûteux des endpoints étendus
    par ce correctif, et n'avait jusqu'ici aucune limite."""
    from api.core.config import get_settings

    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"]).json()
    run_vision_classification_job(job["id"])
    db_session.expire_all()

    limit = get_settings().explain_rate_limit_max_attempts
    responses = [
        client.post(
            f"/api/vision/classification/jobs/{job['id']}/explain",
            headers=headers,
            files={"file": ("test.png", io.BytesIO(_png_bytes()), "image/png")},
        )
        for _ in range(limit)
    ]
    assert all(r.status_code == 200 for r in responses)

    blocked = client.post(
        f"/api/vision/classification/jobs/{job['id']}/explain",
        headers=headers,
        files={"file": ("test.png", io.BytesIO(_png_bytes()), "image/png")},
    )
    assert blocked.status_code == 429
    assert blocked.json()["detail"]["code"] == "TROP_DE_REQUETES"


def test_explain_returns_gradcam_heatmap(client, db_session):
    """Sous-lot D (Grad-CAM) — endpoint synchrone, mêmes conventions que
    `POST /training/jobs/{id}/predict`."""
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"]).json()
    run_vision_classification_job(job["id"])
    db_session.expire_all()

    resp = client.post(
        f"/api/vision/classification/jobs/{job['id']}/explain",
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
        f"/api/vision/classification/jobs/{job['id']}/explain",
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
        f"/api/vision/classification/jobs/{job['id']}/explain",
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
        f"/api/vision/classification/jobs/{job['id']}/explain",
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
        f"/api/vision/classification/jobs/{job['id']}/explain",
        headers=headers,
        files={"file": ("test.png", io.BytesIO(_png_bytes()), "image/png")},
    )
    assert resp.status_code == 404


# ── explain-dataset-examples (retour utilisateur direct : "Grad-CAM devrait
# supporter le batch, pas une image à la fois") ─────────────────────────


def test_explain_batch_returns_a_result_per_image(client, db_session):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"]).json()
    run_vision_classification_job(job["id"])
    db_session.expire_all()

    resp = client.post(
        f"/api/vision/classification/jobs/{job['id']}/explain-dataset-examples",
        headers=headers,
        json={"relative_paths": ["classe_0/img_0.png", "classe_1/img_0.png"]},
    )
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert [r["relative_path"] for r in results] == ["classe_0/img_0.png", "classe_1/img_0.png"]
    for r in results:
        assert r["error"] is None
        assert r["predicted_label"] in {"classe_0", "classe_1"}
        assert r["heatmap_png"].startswith("data:image/png;base64,")


def test_explain_batch_reports_missing_image_without_failing_the_others(client, db_session):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"]).json()
    run_vision_classification_job(job["id"])
    db_session.expire_all()

    resp = client.post(
        f"/api/vision/classification/jobs/{job['id']}/explain-dataset-examples",
        headers=headers,
        json={"relative_paths": ["classe_0/img_0.png", "classe_0/introuvable.png"]},
    )
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert results[0]["error"] is None
    assert results[0]["predicted_label"] is not None
    assert results[1]["error"] is not None
    assert results[1]["predicted_label"] is None


def test_explain_batch_rejects_directory_traversal_as_a_per_item_error_not_a_crash(client, db_session):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"]).json()
    run_vision_classification_job(job["id"])
    db_session.expire_all()

    resp = client.post(
        f"/api/vision/classification/jobs/{job['id']}/explain-dataset-examples",
        headers=headers,
        json={"relative_paths": ["../../../etc/passwd"]},
    )
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert results[0]["error"] is not None
    assert results[0]["heatmap_png"] is None


def test_explain_batch_rejects_more_than_the_max_batch_size(client, db_session):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"]).json()
    run_vision_classification_job(job["id"])
    db_session.expire_all()

    resp = client.post(
        f"/api/vision/classification/jobs/{job['id']}/explain-dataset-examples",
        headers=headers,
        json={"relative_paths": [f"classe_0/img_{i % 8}.png" for i in range(13)]},
    )
    assert resp.status_code == 422


def test_explain_batch_rejects_empty_list(client, db_session):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"]).json()
    run_vision_classification_job(job["id"])
    db_session.expire_all()

    resp = client.post(
        f"/api/vision/classification/jobs/{job['id']}/explain-dataset-examples",
        headers=headers,
        json={"relative_paths": []},
    )
    assert resp.status_code == 422


def test_explain_batch_404_before_completion(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"]).json()

    resp = client.post(
        f"/api/vision/classification/jobs/{job['id']}/explain-dataset-examples",
        headers=headers,
        json={"relative_paths": ["classe_0/img_0.png"]},
    )
    assert resp.status_code == 404


def test_explain_batch_404_for_other_organization(client, db_session):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    dataset_a = _upload_vision_dataset(client, headers_a)
    job = _create_job(client, headers_a, dataset_a["id"]).json()
    run_vision_classification_job(job["id"])
    db_session.expire_all()

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    resp = client.post(
        f"/api/vision/classification/jobs/{job['id']}/explain-dataset-examples",
        headers=headers_b,
        json={"relative_paths": ["classe_0/img_0.png"]},
    )
    assert resp.status_code == 404


def test_explain_batch_shares_the_explain_rate_limit(client, db_session):
    """Un appel batch consomme le MÊME quota horaire que `/explain` — sinon
    le plafond anti-abus de l'endpoint le plus coûteux serait contournable
    en passant systématiquement par le batch."""
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"]).json()
    run_vision_classification_job(job["id"])
    db_session.expire_all()

    limit = get_settings().explain_rate_limit_max_attempts
    for _ in range(limit):
        resp = client.post(
            f"/api/vision/classification/jobs/{job['id']}/explain-dataset-examples",
            headers=headers,
            json={"relative_paths": ["classe_0/img_0.png"]},
        )
        assert resp.status_code == 200

    blocked = client.post(
        f"/api/vision/classification/jobs/{job['id']}/explain-dataset-examples",
        headers=headers,
        json={"relative_paths": ["classe_0/img_0.png"]},
    )
    assert blocked.status_code == 429


# ── Mode expert : comparatif de backbones (retour utilisateur direct, parité
# avec `model_ids` du ML tabulaire) ─────────────────────────────────────────


def test_create_job_accepts_backbone_ids_comparison(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers)
    resp = _create_job(client, headers, dataset["id"], backbone_ids=["mobilenet_v3_small", "shufflenet_v2"])
    assert resp.status_code == 201


def test_create_job_rejects_a_single_backbone_id_in_the_comparison_list(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers)
    resp = _create_job(client, headers, dataset["id"], backbone_ids=["mobilenet_v3_small"])
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "COMPARATIF_BACKBONES_INVALIDE"


def test_create_job_rejects_too_many_backbone_ids(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers)
    resp = _create_job(
        client, headers, dataset["id"],
        backbone_ids=["mobilenet_v3_small", "resnet18", "resnet34", "mobilenet_v3_large", "efficientnet_b0"],
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "COMPARATIF_BACKBONES_INVALIDE"


def test_create_job_rejects_duplicate_backbone_ids(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers)
    resp = _create_job(
        client, headers, dataset["id"], backbone_ids=["mobilenet_v3_small", "mobilenet_v3_small"]
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "COMPARATIF_BACKBONES_INVALIDE"


def test_create_job_rejects_unknown_backbone_in_the_comparison_list(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers)
    resp = _create_job(client, headers, dataset["id"], backbone_ids=["mobilenet_v3_small", "resnet152"])
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "BACKBONE_INCONNU"


def test_comparison_result_includes_a_leaderboard_and_the_winner_is_persisted(client, db_session):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers)
    job = _create_job(
        client, headers, dataset["id"], backbone_ids=["mobilenet_v3_small", "shufflenet_v2"]
    ).json()
    run_vision_classification_job(job["id"])
    db_session.expire_all()

    result = client.get(f"/api/vision/classification/jobs/{job['id']}/result", headers=headers).json()
    candidates = result["model_card"]["candidates"]
    assert len(candidates) == 2
    assert sum(1 for c in candidates if c["selected"]) == 1
    # Le backbone persisté comme modèle final EST celui marqué "selected".
    winner = next(c for c in candidates if c["selected"])
    assert result["backbone_id"] == winner["backbone_id"]


# ── Mode expert : résolution d'entrée (retour utilisateur direct — "vision
# n'offre pas de réduire/augmenter la taille des images") ──────────────────


def test_create_job_accepts_a_non_default_image_size(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers)
    resp = _create_job(client, headers, dataset["id"], image_size=64)
    assert resp.status_code == 201


def test_create_job_rejects_an_unknown_image_size(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers)
    resp = _create_job(client, headers, dataset["id"], image_size=100)
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "TAILLE_IMAGE_INCONNUE"


def test_result_reflects_the_image_size_actually_used(client, db_session):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"], image_size=64).json()
    run_vision_classification_job(job["id"])
    db_session.expire_all()

    result = client.get(f"/api/vision/classification/jobs/{job['id']}/result", headers=headers).json()
    assert result["model_card"]["image_size"] == 64


# ── /model/export-script (Lot 6B, §F.2 — script de déploiement autonome) ──


def test_export_deployment_script_returns_a_python_file_after_completion(client, db_session):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"]).json()
    run_vision_classification_job(job["id"])
    db_session.expire_all()

    resp = client.get(f"/api/vision/classification/jobs/{job['id']}/model/export-script", headers=headers)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/x-python")
    assert "def predict(" in resp.text
    assert "domains" not in resp.text  # jamais de dépendance à ce projet


def test_export_deployment_script_reflects_the_actual_backbone_and_image_size(client, db_session):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"], backbone_id="resnet18", image_size=64).json()
    run_vision_classification_job(job["id"])
    db_session.expire_all()

    resp = client.get(f"/api/vision/classification/jobs/{job['id']}/model/export-script", headers=headers)
    assert resp.status_code == 200
    assert '"resnet18"' in resp.text
    assert "64" in resp.text


def test_export_deployment_script_409_before_completion(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"]).json()
    resp = client.get(f"/api/vision/classification/jobs/{job['id']}/model/export-script", headers=headers)
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "MODELE_NON_DISPONIBLE"
