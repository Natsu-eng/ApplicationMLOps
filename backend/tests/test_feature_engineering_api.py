"""Tests API du Lot 4c : GET /datasets/{id}/feature-engineering-suggestions
et POST /training/jobs (champ feature_engineering)."""
from __future__ import annotations

import io
import json
from unittest.mock import patch

from api.core.models import TrainingJob


def _register(client, email="owner@bureau.fr", org="Bureau"):
    resp = client.post(
        "/auth/register",
        json={"email": email, "nom": "Owner", "password": "motdepasse123", "organization_name": org},
    ).json()
    return {"Authorization": f"Bearer {resp['access_token']}"}


def _upload_rich_dataset(client, headers):
    """Dataset avec une colonne date (texte), une colonne à cardinalité
    excessive, une paire corrélée et une colonne à forte proportion de
    valeurs manquantes — de quoi déclencher les 4 familles de suggestions."""
    rows = []
    for i in range(120):
        date = f"2022-{(i % 12) + 1:02d}-01"
        ville = f"ville_{i % 90}"  # forte cardinalité
        surface = 50 + i
        consommation = surface * 3  # quasi colinéaire
        revenu = "" if i % 3 == 0 else str(1000 + i)  # ~33% manquant
        cible = surface * 2 - consommation
        rows.append(f"{date},{ville},{surface},{consommation},{revenu},{cible}")
    content = ("date,ville,surface,consommation,revenu,cible\n" + "\n".join(rows) + "\n").encode()
    resp = client.post("/datasets", headers=headers, files={"file": ("d.csv", io.BytesIO(content), "text/csv")})
    return resp.json()


def test_feature_engineering_suggestions_endpoint_returns_all_families(client):
    headers = _register(client)
    dataset = _upload_rich_dataset(client, headers)

    resp = client.get(
        f"/datasets/{dataset['id']}/feature-engineering-suggestions",
        headers=headers,
        params={"target_column": "cible"},
    )
    assert resp.status_code == 200
    codes = {s["code"] for s in resp.json()["suggestions"]}
    assert codes == {
        "decomposition_date", "ratio_colonnes_correlees",
        "regroupement_frequence", "imputation_configurable",
        "exclusion_variable",  # "ville" (cardinalité excessive) — Lot Nettoyage guidé des variables
    }


def test_feature_engineering_suggestions_endpoint_rejects_unknown_target(client):
    headers = _register(client)
    dataset = _upload_rich_dataset(client, headers)

    resp = client.get(
        f"/datasets/{dataset['id']}/feature-engineering-suggestions",
        headers=headers,
        params={"target_column": "inexistante"},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "COLONNE_INTROUVABLE"


def _upload_simple_dataset(client, headers):
    rows = "\n".join(f"2022-01-{(i % 28) + 1:02d},{i}" for i in range(60))
    content = f"date,cible\n{rows}\n".encode()
    resp = client.post("/datasets", headers=headers, files={"file": ("d2.csv", io.BytesIO(content), "text/csv")})
    return resp.json()


@patch("api.routers.training.training_queue")
def test_create_job_persists_versioned_feature_engineering_spec(mock_queue, db_session, client):
    mock_queue.enqueue.return_value.id = "fake-rq-id"
    headers = _register(client)
    dataset = _upload_simple_dataset(client, headers)

    resp = client.post(
        "/training/jobs",
        headers=headers,
        json={
            "dataset_id": dataset["id"],
            "target_column": "cible",
            "feature_engineering": {
                "upstream": [{"type": "datetime_decompose", "source_column": "date"}],
                "pipeline": {},
            },
        },
    )
    assert resp.status_code == 201
    job_id = resp.json()["id"]

    db_session.expire_all()
    job = db_session.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    stored = json.loads(job.feature_engineering_json)
    assert stored == {
        "version": 1,
        "upstream": [{"type": "datetime_decompose", "source_column": "date"}],
        "pipeline": {},
    }


@patch("api.routers.training.training_queue")
def test_create_job_rejects_unknown_transformation_type(mock_queue, client):
    headers = _register(client)
    dataset = _upload_simple_dataset(client, headers)

    resp = client.post(
        "/training/jobs",
        headers=headers,
        json={
            "dataset_id": dataset["id"],
            "target_column": "cible",
            "feature_engineering": {"upstream": [{"type": "target_encoding", "column": "date"}], "pipeline": {}},
        },
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "TRANSFORMATION_INCONNUE"


@patch("api.routers.training.training_queue")
def test_create_job_rejects_unknown_column_in_feature_engineering(mock_queue, client):
    headers = _register(client)
    dataset = _upload_simple_dataset(client, headers)

    resp = client.post(
        "/training/jobs",
        headers=headers,
        json={
            "dataset_id": dataset["id"],
            "target_column": "cible",
            "feature_engineering": {
                "upstream": [{"type": "datetime_decompose", "source_column": "colonne_inexistante"}],
                "pipeline": {},
            },
        },
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "COLONNES_INCONNUES"


@patch("api.routers.training.training_queue")
def test_create_job_without_feature_engineering_stores_null(mock_queue, db_session, client):
    mock_queue.enqueue.return_value.id = "fake-rq-id"
    headers = _register(client)
    dataset = _upload_simple_dataset(client, headers)

    resp = client.post(
        "/training/jobs", headers=headers, json={"dataset_id": dataset["id"], "target_column": "cible"}
    )
    assert resp.status_code == 201
    job_id = resp.json()["id"]

    db_session.expire_all()
    job = db_session.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    assert job.feature_engineering_json is None
