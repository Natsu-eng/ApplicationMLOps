"""Tests du router datasets (Lot 2) — upload, schéma, isolation, suppression."""
from __future__ import annotations

import io


def _register(client, email="owner@bureau.fr", org="Bureau"):
    resp = client.post(
        "/auth/register",
        json={"email": email, "nom": "Owner", "password": "motdepasse123", "organization_name": org},
    ).json()
    return {"Authorization": f"Bearer {resp['access_token']}"}


def _csv_file(content: str = "a,b\n1,2\n3,4\n"):
    return {"file": ("test.csv", io.BytesIO(content.encode()), "text/csv")}


def test_upload_csv_computes_schema(client):
    headers = _register(client)
    resp = client.post("/datasets", headers=headers, files=_csv_file())
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "ready"
    assert body["row_count"] == 2
    assert body["column_count"] == 2
    assert {c["name"] for c in body["columns"]} == {"a", "b"}


def test_upload_rejects_unsupported_extension(client):
    headers = _register(client)
    resp = client.post(
        "/datasets", headers=headers, files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")}
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "DATASET_FORMAT_NON_SUPPORTE"


def test_upload_rejects_empty_file(client):
    headers = _register(client)
    resp = client.post("/datasets", headers=headers, files={"file": ("vide.csv", io.BytesIO(b""), "text/csv")})
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "DATASET_FICHIER_VIDE"


def test_dataset_isolation_between_organizations(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")

    created = client.post("/datasets", headers=headers_a, files=_csv_file()).json()

    assert client.get("/datasets", headers=headers_b).json() == []
    assert client.get(f"/datasets/{created['id']}", headers=headers_b).status_code == 404


def test_delete_removes_dataset(client):
    headers = _register(client)
    created = client.post("/datasets", headers=headers, files=_csv_file()).json()

    assert client.delete(f"/datasets/{created['id']}", headers=headers).status_code == 204
    assert client.get(f"/datasets/{created['id']}", headers=headers).status_code == 404


def test_preview_returns_sample_rows(client):
    headers = _register(client)
    created = client.post("/datasets", headers=headers, files=_csv_file()).json()

    resp = client.get(f"/datasets/{created['id']}/preview", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["columns"] == ["a", "b"]
    assert body["sample_size"] == 2


def test_eda_returns_stats_and_correlations(client):
    headers = _register(client)
    created = client.post("/datasets", headers=headers, files=_csv_file()).json()

    resp = client.get(f"/datasets/{created['id']}/eda", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["row_count"] == 2
    assert len(body["column_stats"]) == 2
    assert body["correlation_matrix"]["columns"] == ["a", "b"]


def test_histogram_returns_numeric_bins(client):
    headers = _register(client)
    created = client.post("/datasets", headers=headers, files=_csv_file()).json()

    resp = client.get(f"/datasets/{created['id']}/histogram", headers=headers, params={"column": "a"})
    assert resp.status_code == 200
    assert resp.json()["kind"] == "numeric"


def test_histogram_rejects_unknown_column(client):
    headers = _register(client)
    created = client.post("/datasets", headers=headers, files=_csv_file()).json()

    resp = client.get(
        f"/datasets/{created['id']}/histogram", headers=headers, params={"column": "inexistante"}
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "COLONNE_INTROUVABLE"


def test_quality_check_returns_warnings_list(client):
    headers = _register(client)
    created = client.post("/datasets", headers=headers, files=_csv_file()).json()

    resp = client.get(
        f"/datasets/{created['id']}/quality-check", headers=headers, params={"target_column": "b"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["warnings"], list)


def test_quality_check_rejects_unknown_target_column(client):
    headers = _register(client)
    created = client.post("/datasets", headers=headers, files=_csv_file()).json()

    resp = client.get(
        f"/datasets/{created['id']}/quality-check", headers=headers, params={"target_column": "inexistante"}
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "COLONNE_INTROUVABLE"


def test_quality_check_isolation_between_organizations(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")

    created = client.post("/datasets", headers=headers_a, files=_csv_file()).json()

    resp = client.get(
        f"/datasets/{created['id']}/quality-check", headers=headers_b, params={"target_column": "b"}
    )
    assert resp.status_code == 404
