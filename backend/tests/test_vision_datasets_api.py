"""Tests API du router vision/datasets (pilier Vision, Lot 15 sous-lot A)."""
from __future__ import annotations

import io
import zipfile

from PIL import Image


def _register(client, email="owner@bureau.fr", org="Bureau"):
    resp = client.post(
        "/api/auth/register",
        json={"email": email, "nom": "Owner", "password": "motdepasse123", "organization_name": org},
    ).json()
    return {"Authorization": f"Bearer {resp['access_token']}"}


def _png_bytes(color=(255, 0, 0), size=(32, 32), variant: int = 0) -> bytes:
    """`variant` rend deux images de même couleur bit-à-bit distinctes —
    nécessaire depuis la déduplication (Lot 0.1, correctif C1) : voir
    test_vision_datasets_service.py::_png_bytes pour le raisonnement complet."""
    img = Image.new("RGB", size, color)
    if variant:
        img.putpixel((0, 0), (variant % 256, (variant * 7) % 256, (variant * 13) % 256))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _build_zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


def _classification_zip_bytes(n_per_class=4, n_classes=2) -> bytes:
    files = {}
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for c in range(n_classes):
        for i in range(n_per_class):
            files[f"classe_{c}/img_{i}.png"] = _png_bytes(colors[c % len(colors)], variant=i + 1)
    return _build_zip(files)


def _upload_vision_dataset(client, headers, content: bytes, name="dataset.zip"):
    return client.post(
        "/api/vision/datasets", headers=headers, files={"file": (name, io.BytesIO(content), "application/zip")}
    )


def test_upload_valid_classification_dataset(client):
    headers = _register(client)
    resp = _upload_vision_dataset(client, headers, _classification_zip_bytes())
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "ready"
    assert body["structure_type"] == "classification"
    assert body["n_classes"] == 2
    assert body["n_images"] == 8
    assert set(body["class_distribution"]) == {"classe_0", "classe_1"}


def test_upload_blocked_after_too_many_attempts(client):
    """Lot 1.4 (§C.2.7/§D.4, AUDIT_DATALAB_2026-08-16.md) — même correctif
    que le dataset tabulaire (test_datasets.py), compteur indépendant
    ("vision_dataset_upload" vs "dataset_upload")."""
    from api.core.config import get_settings

    headers = _register(client)
    limit = get_settings().upload_rate_limit_max_attempts
    responses = [_upload_vision_dataset(client, headers, _classification_zip_bytes()) for _ in range(limit)]
    assert all(r.status_code == 201 for r in responses)

    blocked = _upload_vision_dataset(client, headers, _classification_zip_bytes())
    assert blocked.status_code == 429
    assert blocked.json()["detail"]["code"] == "TROP_DE_REQUETES"


def test_upload_rejects_non_zip_extension(client):
    headers = _register(client)
    resp = client.post(
        "/api/vision/datasets", headers=headers, files={"file": ("dataset.png", io.BytesIO(b"x"), "image/png")}
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "VISION_DATASET_FORMAT_NON_SUPPORTE"


def test_upload_rejects_empty_archive(client):
    headers = _register(client)
    resp = client.post(
        "/api/vision/datasets", headers=headers, files={"file": ("dataset.zip", io.BytesIO(b""), "application/zip")}
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "VISION_DATASET_FICHIER_VIDE"


def test_upload_unrecognized_structure_stored_as_error(client):
    """Un ZIP invalide reçoit quand même une entrée en base (status "error",
    message diagnostiqué) plutôt qu'une simple 4xx — même convention que
    l'upload de dataset tabulaire (api/routers/datasets.py)."""
    headers = _register(client)
    content = _build_zip({"photo1.png": _png_bytes(), "photo2.png": _png_bytes()})
    resp = _upload_vision_dataset(client, headers, content)
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "error"
    assert body["error_message"]


def test_list_datasets_isolated_between_organizations(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    _upload_vision_dataset(client, headers_a, _classification_zip_bytes())

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    resp = client.get("/api/vision/datasets", headers=headers_b)
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_dataset_404_for_other_organization(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    dataset = _upload_vision_dataset(client, headers_a, _classification_zip_bytes()).json()

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    resp = client.get(f"/api/vision/datasets/{dataset['id']}", headers=headers_b)
    assert resp.status_code == 404


def test_delete_dataset_removes_it_and_its_files(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers, _classification_zip_bytes()).json()

    resp = client.delete(f"/api/vision/datasets/{dataset['id']}", headers=headers)
    assert resp.status_code == 204
    assert client.get(f"/api/vision/datasets/{dataset['id']}", headers=headers).status_code == 404


def test_get_dataset_detail_includes_validation_report(client):
    headers = _register(client)
    files = {f"classe_0/img_{i}.png": _png_bytes(variant=i + 1) for i in range(4)}
    files["classe_0/broken.png"] = b"not an image"
    files.update({f"classe_1/img_{i}.png": _png_bytes((0, 255, 0), variant=i + 1) for i in range(4)})
    content = _build_zip(files)
    dataset = _upload_vision_dataset(client, headers, content).json()

    resp = client.get(f"/api/vision/datasets/{dataset['id']}", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["validation_report"]["n_corrupted"] == 1


def test_get_image_returns_file(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers, _classification_zip_bytes()).json()

    resp = client.get(f"/api/vision/datasets/{dataset['id']}/image", headers=headers, params={"path": "classe_0/img_0.png"})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"


def test_get_image_404_for_unknown_path(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers, _classification_zip_bytes()).json()

    resp = client.get(f"/api/vision/datasets/{dataset['id']}/image", headers=headers, params={"path": "classe_0/nope.png"})
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "IMAGE_INTROUVABLE"


def test_get_image_rejects_path_traversal(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers, _classification_zip_bytes()).json()

    resp = client.get(
        f"/api/vision/datasets/{dataset['id']}/image", headers=headers, params={"path": "../../../../etc/passwd"}
    )
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "IMAGE_INTROUVABLE"


def test_get_image_isolated_between_organizations(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    dataset_a = _upload_vision_dataset(client, headers_a, _classification_zip_bytes()).json()

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    resp = client.get(
        f"/api/vision/datasets/{dataset_a['id']}/image", headers=headers_b, params={"path": "classe_0/img_0.png"}
    )
    assert resp.status_code == 404


def test_list_images_returns_paths_for_class(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers, _classification_zip_bytes(n_per_class=4)).json()

    resp = client.get(f"/api/vision/datasets/{dataset['id']}/images", headers=headers, params={"class_name": "classe_0"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["class_name"] == "classe_0"
    assert body["total"] == 4
    assert len(body["paths"]) == 4
    assert all(p.startswith("classe_0/") for p in body["paths"])


def test_list_images_caps_at_gallery_limit(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers, _classification_zip_bytes(n_per_class=80)).json()

    resp = client.get(f"/api/vision/datasets/{dataset['id']}/images", headers=headers, params={"class_name": "classe_0"})
    body = resp.json()
    assert body["total"] == 80
    assert len(body["paths"]) == 60  # MAX_GALLERY_IMAGES_PER_CLASS


def test_list_images_404_for_unknown_class(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers, _classification_zip_bytes()).json()

    resp = client.get(f"/api/vision/datasets/{dataset['id']}/images", headers=headers, params={"class_name": "classe_inexistante"})
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "CLASSE_INTROUVABLE"


def test_list_images_rejects_path_traversal(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers, _classification_zip_bytes()).json()

    resp = client.get(
        f"/api/vision/datasets/{dataset['id']}/images", headers=headers, params={"class_name": "../../../../etc"}
    )
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "CLASSE_INTROUVABLE"


def test_list_images_isolated_between_organizations(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    dataset_a = _upload_vision_dataset(client, headers_a, _classification_zip_bytes()).json()

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    resp = client.get(
        f"/api/vision/datasets/{dataset_a['id']}/images", headers=headers_b, params={"class_name": "classe_0"}
    )
    assert resp.status_code == 404
