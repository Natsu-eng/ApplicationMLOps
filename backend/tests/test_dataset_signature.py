"""Signature réelle des fichiers uploadés + garde anti-bombe zip pour
`.xlsx` (Phase 1, AUDIT_BACKEND_2026-08-23.md §C.2) — avant ce correctif,
seule l'extension déclarée par le nom de fichier était vérifiée."""
from __future__ import annotations

import io
import zipfile

import pandas as pd
import pytest

from domains.shared import dataset_io


def _register_and_headers(client, email="sig@bureau.fr"):
    resp = client.post(
        "/api/auth/register",
        json={"email": email, "nom": "Sig", "password": "motdepasse123", "organization_name": "Bureau"},
    )
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_genuine_csv_uploads_normally(client):
    headers = _register_and_headers(client, "a@bureau.fr")
    content = b"a,b\n1,2\n3,4\n"
    resp = client.post(
        "/api/datasets", headers=headers, files={"file": ("data.csv", io.BytesIO(content), "text/csv")}
    )
    assert resp.status_code == 201
    assert resp.json()["status"] == "ready"


def test_xlsx_renamed_to_csv_is_rejected(client):
    """Le contenu (signature ZIP) ne correspond pas à l'extension déclarée
    (.csv, format texte attendu) — refusé même si l'extension seule aurait
    été acceptée."""
    headers = _register_and_headers(client, "b@bureau.fr")
    buffer = io.BytesIO()
    pd.DataFrame({"x": [1, 2, 3]}).to_excel(buffer, index=False)
    fake_csv_content = buffer.getvalue()  # contenu binaire .xlsx réel, nommé .csv

    resp = client.post(
        "/api/datasets", headers=headers, files={"file": ("fichier.csv", io.BytesIO(fake_csv_content), "text/csv")}
    )
    assert resp.status_code == 201  # jamais un crash — dégradation propre (voir router)
    body = resp.json()
    assert body["status"] == "error"
    assert "extension déclarée" in body["error_message"]


def test_garbage_renamed_to_xlsx_is_rejected(client):
    headers = _register_and_headers(client, "c@bureau.fr")
    resp = client.post(
        "/api/datasets",
        headers=headers,
        files={"file": ("classeur.xlsx", io.BytesIO(b"ceci n'est pas un fichier Excel"), "application/octet-stream")},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "error"
    assert "classeur Excel" in body["error_message"]


def test_genuine_small_xlsx_uploads_normally(client):
    headers = _register_and_headers(client, "d@bureau.fr")
    buffer = io.BytesIO()
    pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]}).to_excel(buffer, index=False)
    resp = client.post(
        "/api/datasets",
        headers=headers,
        files={"file": ("classeur.xlsx", io.BytesIO(buffer.getvalue()), "application/octet-stream")},
    )
    assert resp.status_code == 201
    assert resp.json()["status"] == "ready"


def test_xlsx_zip_bomb_rejected_by_uncompressed_size_cap(client, monkeypatch, tmp_path):
    """Reproduit une bombe zip sans réellement écrire des centaines de Mo :
    seuil abaissé pour le test, contenu réel juste au-dessus."""
    monkeypatch.setattr(dataset_io, "MAX_XLSX_UNCOMPRESSED_BYTES", 1000)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("huge.xml", b"0" * 5000)  # 5000 > seuil abaisse (1000)
    content = buffer.getvalue()
    assert content.startswith(b"PK\x03\x04")

    path = tmp_path / "bomb.xlsx"
    path.write_bytes(content)

    with pytest.raises(dataset_io.DatasetParsingError, match="bombe zip"):
        dataset_io.read_dataframe(path, ".xlsx")


def test_parquet_signature_checked():
    """Unitaire (pas besoin de passer par l'API) — vérifie directement la
    fonction de signature, format parquet reconnu par en-tête ET pied de
    fichier "PAR1"."""
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        f.write(b"not a real parquet file")
        path = Path(f.name)
    try:
        with pytest.raises(dataset_io.DatasetParsingError, match="Parquet"):
            dataset_io.read_dataframe(path, ".parquet")
    finally:
        path.unlink(missing_ok=True)
