"""Tests du router datasets (Lot 2) — upload, schéma, isolation, suppression."""
from __future__ import annotations

import io
from unittest.mock import patch


def _register(client, email="owner@bureau.fr", org="Bureau"):
    resp = client.post(
        "/api/auth/register",
        json={"email": email, "nom": "Owner", "password": "motdepasse123", "organization_name": org},
    ).json()
    return {"Authorization": f"Bearer {resp['access_token']}"}


def _csv_file(content: str = "a,b\n1,2\n3,4\n"):
    return {"file": ("test.csv", io.BytesIO(content.encode()), "text/csv")}


def test_upload_csv_computes_schema(client):
    headers = _register(client)
    resp = client.post("/api/datasets", headers=headers, files=_csv_file())
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "ready"
    assert body["row_count"] == 2
    assert body["column_count"] == 2
    assert {c["name"] for c in body["columns"]} == {"a", "b"}


# ── Lot 5, correctif P2 — hash de contenu + détection de doublon ────────────


def test_upload_computes_a_sha256_content_hash(client):
    headers = _register(client)
    body = client.post("/api/datasets", headers=headers, files=_csv_file()).json()
    assert body["content_hash"] is not None
    assert len(body["content_hash"]) == 64  # hex SHA-256
    assert body["duplicate_of_dataset_id"] is None


def test_uploading_the_same_content_twice_flags_the_duplicate(client):
    headers = _register(client)
    first = client.post("/api/datasets", headers=headers, files=_csv_file()).json()
    second = client.post(
        "/api/datasets", headers=headers, files=_csv_file()  # même contenu, autre appel
    ).json()

    assert second["content_hash"] == first["content_hash"]
    assert second["duplicate_of_dataset_id"] == first["id"]
    # Jamais bloquant : le second upload aboutit quand même.
    assert second["status"] == "ready"


def test_uploading_different_content_never_flags_a_duplicate(client):
    headers = _register(client)
    client.post("/api/datasets", headers=headers, files=_csv_file("a,b\n1,2\n")).json()
    second = client.post("/api/datasets", headers=headers, files=_csv_file("a,b\n9,9\n9,9\n9,9\n")).json()
    assert second["duplicate_of_dataset_id"] is None


def test_duplicate_detection_is_isolated_by_organization(client):
    """Le même fichier uploadé par deux organisations DIFFÉRENTES n'est
    jamais un doublon l'un de l'autre — l'isolation multi-tenant prime."""
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    client.post("/api/datasets", headers=headers_a, files=_csv_file())

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    second = client.post("/api/datasets", headers=headers_b, files=_csv_file()).json()
    assert second["duplicate_of_dataset_id"] is None


def test_upload_blocked_after_too_many_attempts(client):
    """Lot 1.4 (§C.2.7/§D.4, AUDIT_DATALAB_2026-08-16.md) — l'upload n'avait
    jusqu'ici aucune limite de débit (contrairement à /auth/login)."""
    from api.core.config import get_settings

    headers = _register(client)
    limit = get_settings().upload_rate_limit_max_attempts
    responses = [client.post("/api/datasets", headers=headers, files=_csv_file()) for _ in range(limit)]
    assert all(r.status_code == 201 for r in responses)

    blocked = client.post("/api/datasets", headers=headers, files=_csv_file())
    assert blocked.status_code == 429
    assert blocked.json()["detail"]["code"] == "TROP_DE_REQUETES"


def test_upload_rejects_unsupported_extension(client):
    headers = _register(client)
    resp = client.post(
        "/api/datasets", headers=headers, files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")}
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "DATASET_FORMAT_NON_SUPPORTE"


def test_upload_rejects_empty_file(client):
    headers = _register(client)
    resp = client.post("/api/datasets", headers=headers, files={"file": ("vide.csv", io.BytesIO(b""), "text/csv")})
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "DATASET_FICHIER_VIDE"


def test_dataset_isolation_between_organizations(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")

    created = client.post("/api/datasets", headers=headers_a, files=_csv_file()).json()

    assert client.get("/api/datasets", headers=headers_b).json() == []
    assert client.get(f"/api/datasets/{created['id']}", headers=headers_b).status_code == 404


def test_delete_removes_dataset(client):
    headers = _register(client)
    created = client.post("/api/datasets", headers=headers, files=_csv_file()).json()

    assert client.delete(f"/api/datasets/{created['id']}", headers=headers).status_code == 204
    assert client.get(f"/api/datasets/{created['id']}", headers=headers).status_code == 404


# ── Lot 7, §J.3 — avertissement de suppression en cascade (décompte d'usage) ─


def test_usage_is_zero_for_a_fresh_dataset(client):
    headers = _register(client)
    created = client.post("/api/datasets", headers=headers, files=_csv_file()).json()

    resp = client.get(f"/api/datasets/{created['id']}/usage", headers=headers)
    assert resp.status_code == 200
    assert resp.json() == {
        "training_jobs": 0,
        "clustering_jobs": 0,
        "dimensionality_jobs": 0,
        "anomaly_jobs": 0,
        "total": 0,
    }


@patch("api.routers.clustering.analysis_queue")
@patch("api.routers.training.training_queue")
def test_usage_counts_jobs_referencing_this_dataset(mock_training_queue, mock_clustering_queue, client):
    mock_training_queue.enqueue.return_value.id = "fake-rq-id"
    mock_clustering_queue.enqueue.return_value.id = "fake-rq-id"
    headers = _register(client)
    rows = "\n".join(f"{i},{i * 2}\n" for i in range(20))
    content = f"x1,x2\n{rows}".encode()
    dataset = client.post(
        "/api/datasets", headers=headers, files={"file": ("d.csv", io.BytesIO(content), "text/csv")}
    ).json()

    client.post(
        "/api/training/jobs", headers=headers, json={"dataset_id": dataset["id"], "target_column": "x2"}
    )
    client.post(
        "/api/clustering/jobs", headers=headers, json={"dataset_id": dataset["id"], "feature_columns": ["x1", "x2"]}
    )

    resp = client.get(f"/api/datasets/{dataset['id']}/usage", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["training_jobs"] == 1
    assert body["clustering_jobs"] == 1
    assert body["total"] == 2


def test_usage_404_for_other_organization(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    dataset_a = client.post("/api/datasets", headers=headers_a, files=_csv_file()).json()

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    resp = client.get(f"/api/datasets/{dataset_a['id']}/usage", headers=headers_b)
    assert resp.status_code == 404


def test_preview_returns_sample_rows(client):
    headers = _register(client)
    created = client.post("/api/datasets", headers=headers, files=_csv_file()).json()

    resp = client.get(f"/api/datasets/{created['id']}/preview", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["columns"] == ["a", "b"]
    assert body["sample_size"] == 2


def test_eda_returns_stats_and_correlations(client):
    headers = _register(client)
    created = client.post("/api/datasets", headers=headers, files=_csv_file()).json()

    resp = client.get(f"/api/datasets/{created['id']}/eda", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["row_count"] == 2
    assert len(body["column_stats"]) == 2
    assert body["correlation_matrix"]["columns"] == ["a", "b"]
    # Nouveaux champs Lot B toujours présents, même sans target_column
    assert "categorical_correlation_matrix" in body
    assert "outlier_summary" in body
    assert "top_correlated_pairs" in body
    assert body["target_distribution"] is None


def _richer_csv_file():
    rows = "\n".join(f"{i},{'a' if i % 2 == 0 else 'b'},{i * 2}" for i in range(30))
    content = f"valeur,categorie,cible\n{rows}\n"
    return {"file": ("richer.csv", io.BytesIO(content.encode()), "text/csv")}


def test_eda_with_target_column_includes_target_distribution(client):
    headers = _register(client)
    created = client.post("/api/datasets", headers=headers, files=_richer_csv_file()).json()

    resp = client.get(
        f"/api/datasets/{created['id']}/eda", headers=headers, params={"target_column": "cible"}
    )
    assert resp.status_code == 200
    assert resp.json()["target_distribution"] is not None


def test_eda_rejects_unknown_target_column(client):
    headers = _register(client)
    created = client.post("/api/datasets", headers=headers, files=_csv_file()).json()

    resp = client.get(
        f"/api/datasets/{created['id']}/eda", headers=headers, params={"target_column": "inexistante"}
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "COLONNE_INTROUVABLE"


def test_feature_by_target_returns_groups(client):
    headers = _register(client)
    created = client.post("/api/datasets", headers=headers, files=_richer_csv_file()).json()

    resp = client.get(
        f"/api/datasets/{created['id']}/feature-by-target",
        headers=headers,
        params={"feature": "valeur", "target": "categorie"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert {g["class_name"] for g in body["groups"]} == {"a", "b"}


def test_feature_by_target_rejects_non_numeric_feature(client):
    headers = _register(client)
    created = client.post("/api/datasets", headers=headers, files=_richer_csv_file()).json()

    resp = client.get(
        f"/api/datasets/{created['id']}/feature-by-target",
        headers=headers,
        params={"feature": "categorie", "target": "cible"},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "FEATURE_NON_NUMERIQUE"


def test_feature_by_target_isolation_between_organizations(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")

    created = client.post("/api/datasets", headers=headers_a, files=_richer_csv_file()).json()

    resp = client.get(
        f"/api/datasets/{created['id']}/feature-by-target",
        headers=headers_b,
        params={"feature": "valeur", "target": "categorie"},
    )
    assert resp.status_code == 404


def test_histogram_returns_numeric_bins(client):
    headers = _register(client)
    created = client.post("/api/datasets", headers=headers, files=_csv_file()).json()

    resp = client.get(f"/api/datasets/{created['id']}/histogram", headers=headers, params={"column": "a"})
    assert resp.status_code == 200
    assert resp.json()["kind"] == "numeric"


def test_histogram_rejects_unknown_column(client):
    headers = _register(client)
    created = client.post("/api/datasets", headers=headers, files=_csv_file()).json()

    resp = client.get(
        f"/api/datasets/{created['id']}/histogram", headers=headers, params={"column": "inexistante"}
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "COLONNE_INTROUVABLE"


def test_quality_check_returns_warnings_list(client):
    headers = _register(client)
    created = client.post("/api/datasets", headers=headers, files=_csv_file()).json()

    resp = client.get(
        f"/api/datasets/{created['id']}/quality-check", headers=headers, params={"target_column": "b"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["warnings"], list)


def test_quality_check_rejects_unknown_target_column(client):
    headers = _register(client)
    created = client.post("/api/datasets", headers=headers, files=_csv_file()).json()

    resp = client.get(
        f"/api/datasets/{created['id']}/quality-check", headers=headers, params={"target_column": "inexistante"}
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "COLONNE_INTROUVABLE"


def test_quality_check_isolation_between_organizations(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")

    created = client.post("/api/datasets", headers=headers_a, files=_csv_file()).json()

    resp = client.get(
        f"/api/datasets/{created['id']}/quality-check", headers=headers_b, params={"target_column": "b"}
    )
    assert resp.status_code == 404


# ── target_column optionnel (Lot Nettoyage guidé des variables) ─────────────


def test_quality_check_without_target_column_returns_structural_warnings(client):
    """Permet d'appeler ce endpoint dès l'exploration d'un dataset (page
    Données/EDA), avant même de choisir une cible pour un entraînement."""
    headers = _register(client)
    n = 100
    content = "id_client,toujours_pareil,x\n" + "\n".join(
        f"C{i},42,{i}" for i in range(n)
    ) + "\n"
    created = client.post(
        "/api/datasets", headers=headers, files={"file": ("d.csv", io.BytesIO(content.encode()), "text/csv")}
    ).json()

    resp = client.get(f"/api/datasets/{created['id']}/quality-check", headers=headers)
    assert resp.status_code == 200
    codes = {w["code"] for w in resp.json()["warnings"]}
    assert "colonne_constante" in codes
    assert "cardinalite_excessive" in codes
    # Aucune détection nécessitant une cible ne peut apparaître sans cible fournie.
    assert "fuite_cible" not in codes
    assert "desequilibre_classes" not in codes
