"""Tests de `MaxJsonBodySizeMiddleware` (api/main.py) — Lot 1.4 (§C.2.7,
AUDIT_DATALAB_2026-08-16.md). `POST /training/jobs/{id}/predict` acceptait
un dictionnaire JSON arbitraire sans aucune limite de taille avant ce
correctif ; testé ici via `/auth/register` (endpoint JSON simple, pas
besoin de dataset/job pour exercer la limite elle-même)."""
from __future__ import annotations

from api.core.config import get_settings


def test_oversized_json_body_rejected_with_413(client):
    limit_bytes = get_settings().max_json_body_size_mb * 1024 * 1024
    padding = "x" * (limit_bytes + 1024)
    resp = client.post(
        "/auth/register",
        json={"email": "trop@bureau.fr", "nom": padding, "password": "motdepasse123", "organization_name": "Bureau"},
    )
    assert resp.status_code == 413
    assert resp.json()["detail"]["code"] == "CORPS_TROP_VOLUMINEUX"


def test_normal_json_body_not_affected(client):
    resp = client.post(
        "/auth/register",
        json={"email": "normal@bureau.fr", "nom": "Normal", "password": "motdepasse123", "organization_name": "Bureau"},
    )
    assert resp.status_code == 201


def test_multipart_upload_not_affected_by_json_limit(client):
    """La limite ne doit JAMAIS s'appliquer aux uploads (multipart) — ils
    ont leurs propres limites, plus élevées, vérifiées ailleurs."""
    import io

    resp = client.post(
        "/auth/register",
        json={"email": "uploader@bureau.fr", "nom": "Uploader", "password": "motdepasse123", "organization_name": "Bureau"},
    )
    headers = {"Authorization": f"Bearer {resp.json()['access_token']}"}
    # Une ligne CSV minuscule, bien en dessous de max_json_body_size_mb ET
    # de max_upload_size_mb — prouve juste que le chemin multipart n'est
    # jamais intercepté par la vérification Content-Type: application/json.
    upload = client.post(
        "/datasets", headers=headers, files={"file": ("t.csv", io.BytesIO(b"a,b\n1,2\n"), "text/csv")}
    )
    assert upload.status_code == 201
