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
        "/api/vision/datasets", headers=headers, files={"files": (name, io.BytesIO(content), "application/zip")}
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


def test_upload_rejects_unsupported_extension(client):
    """Lot 6A — un seul fichier dont l'extension n'est ni .zip/.tar/
    .tar.gz/.tgz est rejeté avant même la lecture du contenu (vérification
    rapide côté nom de fichier, voir validate_archive_extension). Plus
    d'un fichier bascule automatiquement sur le chemin "import de dossier"
    (VISION_DATASET_FORMAT_NON_SUPPORTE ne s'applique qu'à ce cas précis :
    un seul fichier, extension inconnue)."""
    headers = _register(client)
    resp = client.post(
        "/api/vision/datasets", headers=headers, files={"files": ("dataset.png", io.BytesIO(b"x"), "image/png")}
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "VISION_DATASET_FORMAT_NON_SUPPORTE"


def test_upload_of_a_tar_gz_archive_is_accepted(client):
    """Lot 6A — le téléchargement officiel MVTec AD est distribué en
    .tar.xz (jamais .zip) : le format doit être accepté, pas seulement
    toléré en façade (voir services/vision_datasets.py::_extract_tar_members)."""
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for c in range(2):
            for i in range(4):
                content = _png_bytes([(255, 0, 0), (0, 255, 0)][c], variant=i + 1)
                info = tarfile.TarInfo(name=f"classe_{c}/img_{i}.png")
                info.size = len(content)
                tf.addfile(info, io.BytesIO(content))
    resp = client.post(
        "/api/vision/datasets", headers=_register(client),
        files={"files": ("dataset.tar.gz", io.BytesIO(buf.getvalue()), "application/gzip")},
    )
    assert resp.status_code == 201
    assert resp.json()["status"] == "ready"


def test_upload_of_a_folder_multiple_files_is_accepted(client):
    """Lot 6A — plusieurs fichiers sous le même champ "files" (pas
    d'archive à ouvrir), chaque nom de fichier porte son chemin relatif
    complet (webkitRelativePath côté navigateur)."""
    files = [
        ("files", (f"mon_dataset/classe_{c}/img_{i}.png", io.BytesIO(_png_bytes([(255, 0, 0), (0, 255, 0)][c], variant=i + 1)), "image/png"))
        for c in range(2) for i in range(4)
    ]
    resp = client.post("/api/vision/datasets", headers=_register(client), files=files)
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "ready"
    assert body["name"] == "mon_dataset"


def test_upload_rejects_empty_archive(client):
    headers = _register(client)
    resp = client.post(
        "/api/vision/datasets", headers=headers, files={"files": ("dataset.zip", io.BytesIO(b""), "application/zip")}
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


def test_get_dataset_detail_includes_augmentation_recommendation(client):
    """Lot 6A (correctif I9) — fondée sur la classe la plus petite (8 < 20
    ici) : "forte" attendu (voir services/vision_classification_training.py
    ::recommend_augmentation_preset)."""
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers, _classification_zip_bytes(n_per_class=8)).json()

    resp = client.get(f"/api/vision/datasets/{dataset['id']}", headers=headers)
    assert resp.json()["recommended_augmentation_preset"] == "forte"


def test_augmentation_recommendation_absent_for_mvtec_dataset(client):
    """Sans objet pour un dataset "mvtec_ad" — l'augmentation d'images ne
    concerne que l'entraînement de classification."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for i in range(12):
            zf.writestr(f"train/good/{i}.png", _png_bytes((120, 120, 120), variant=i + 1))
        for i in range(3):
            zf.writestr(f"test/good/{i}.png", _png_bytes((120, 120, 120), variant=1000 + i))
        for i in range(3):
            zf.writestr(f"test/scratch/{i}.png", _png_bytes((220, 20, 20), variant=2000 + i))

    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers, buf.getvalue()).json()

    resp = client.get(f"/api/vision/datasets/{dataset['id']}", headers=headers)
    assert resp.json()["structure_type"] == "mvtec_ad"
    assert resp.json()["recommended_augmentation_preset"] is None


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


# ── Aperçu d'augmentation (Lot 6A) ──────────────────────────────────────────


def _mvtec_zip_bytes(n_train_good=6, n_test_good=3, n_test_defect=3) -> bytes:
    files = {}
    for i in range(n_train_good):
        files[f"train/good/{i}.png"] = _png_bytes((120, 120, 120), variant=i + 1)
    for i in range(n_test_good):
        files[f"test/good/{i}.png"] = _png_bytes((120, 120, 120), variant=i + 100)
    for i in range(n_test_defect):
        files[f"test/scratch/{i}.png"] = _png_bytes((200, 0, 0), variant=i + 200)
    return _build_zip(files)


def test_augmentation_preview_returns_pairs_for_classification_dataset(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers, _classification_zip_bytes(n_per_class=4)).json()

    resp = client.get(
        f"/api/vision/datasets/{dataset['id']}/augmentation-preview", headers=headers, params={"preset": "standard"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["preset"] == "standard"
    assert 1 <= len(body["pairs"]) <= 3
    for pair in body["pairs"]:
        assert pair["original_png"].startswith("data:image/png;base64,")
        assert pair["augmented_png"].startswith("data:image/png;base64,")


def test_augmentation_preview_samples_train_good_for_mvtec_dataset(client):
    headers = _register(client)
    dataset = client.post(
        "/api/vision/datasets", headers=headers, files={"files": ("d.zip", io.BytesIO(_mvtec_zip_bytes()), "application/zip")}
    ).json()

    resp = client.get(
        f"/api/vision/datasets/{dataset['id']}/augmentation-preview", headers=headers, params={"preset": "legere"}
    )
    assert resp.status_code == 200
    assert len(resp.json()["pairs"]) > 0


def test_augmentation_preview_none_preset_still_returns_pairs(client):
    """preset="aucune" ne transforme rien - l'apercu doit quand meme
    renvoyer les images (original == augmented pixel pour pixel n'est pas
    verifie ici, juste que l'endpoint ne casse pas sur ce cas limite)."""
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers, _classification_zip_bytes(n_per_class=4)).json()

    resp = client.get(
        f"/api/vision/datasets/{dataset['id']}/augmentation-preview", headers=headers, params={"preset": "aucune"}
    )
    assert resp.status_code == 200
    assert len(resp.json()["pairs"]) > 0


def test_augmentation_preview_rejects_unknown_preset(client):
    headers = _register(client)
    dataset = _upload_vision_dataset(client, headers, _classification_zip_bytes()).json()

    resp = client.get(
        f"/api/vision/datasets/{dataset['id']}/augmentation-preview", headers=headers, params={"preset": "extreme"}
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "AUGMENTATION_PRESET_INCONNU"


def test_augmentation_preview_404_for_other_organization(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    dataset_a = _upload_vision_dataset(client, headers_a, _classification_zip_bytes()).json()

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    resp = client.get(
        f"/api/vision/datasets/{dataset_a['id']}/augmentation-preview", headers=headers_b, params={"preset": "standard"}
    )
    assert resp.status_code == 404
