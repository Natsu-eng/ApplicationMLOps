"""Tests du Lot 9 — registre de modèles versionné (promotion + export +
liste du registre). L'artefact (bundle joblib) existait depuis le Lot 3 ;
ce lot ajoute la promotion (staging/production) et l'export."""
from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import joblib

from api.core.models import MLModel, TrainingJob
from services.model_versioning import next_version


def _register(client, email="owner@bureau.fr", org="Bureau"):
    resp = client.post(
        "/api/auth/register",
        json={"email": email, "nom": "Owner", "password": "motdepasse123", "organization_name": org},
    ).json()
    return {"Authorization": f"Bearer {resp['access_token']}"}


def _upload_dataset(client, headers, name="d.csv"):
    rows = "\n".join(f"{i},{i * 2},{i * 3}" for i in range(50))
    content = f"x1,x2,cible\n{rows}\n".encode()
    resp = client.post("/api/datasets", headers=headers, files={"file": (name, io.BytesIO(content), "text/csv")})
    return resp.json()


def _create_job(client, headers, dataset_id, target_column="cible"):
    with patch("api.routers.training.training_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        return client.post(
            "/api/training/jobs", headers=headers, json={"dataset_id": dataset_id, "target_column": target_column}
        ).json()


def _complete_job(db_session, job_id, org_id, algorithm="LightGBM", artifact_path=None):
    job = db_session.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    job.status = "completed"
    if artifact_path is None:
        artifact_path = Path(tempfile.gettempdir()) / f"datalab_registry_test_{job_id}.joblib"
        joblib.dump({"dummy": "bundle"}, artifact_path)
    db_session.add(
        MLModel(
            organization_id=org_id,
            training_job_id=job.id,
            dataset_id=job.dataset_id,
            version=next_version(db_session, org_id, job.dataset_id, job.target_column),
            algorithm=algorithm,
            task_type="regression",
            target_column=job.target_column,
            feature_columns_json=json.dumps(["x1", "x2"]),
            file_path=str(artifact_path),
            metrics_json=json.dumps({"r2_test": 0.9, "cv_score": 0.88}),
        )
    )
    db_session.commit()
    db_session.refresh(job)
    return job


# ── Promotion ────────────────────────────────────────────────────────────


def test_promote_to_staging_sets_stage_and_timestamp(client, db_session):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"])
    org_id = db_session.query(TrainingJob).filter(TrainingJob.id == job["id"]).first().organization_id
    _complete_job(db_session, job["id"], org_id)

    resp = client.post(f"/api/training/jobs/{job['id']}/model/promote", headers=headers, json={"stage": "staging"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["stage"] == "staging"
    assert body["promoted_at"] is not None


def test_promote_to_production_demotes_previous_production_for_same_dataset_and_target(client, db_session):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job1 = _create_job(client, headers, dataset["id"])
    job2 = _create_job(client, headers, dataset["id"])
    org_id = db_session.query(TrainingJob).filter(TrainingJob.id == job1["id"]).first().organization_id
    _complete_job(db_session, job1["id"], org_id, algorithm="LightGBM")
    _complete_job(db_session, job2["id"], org_id, algorithm="CatBoost")

    r1 = client.post(f"/api/training/jobs/{job1['id']}/model/promote", headers=headers, json={"stage": "production"})
    assert r1.json()["stage"] == "production"

    r2 = client.post(f"/api/training/jobs/{job2['id']}/model/promote", headers=headers, json={"stage": "production"})
    assert r2.json()["stage"] == "production"

    # job1 doit avoir été démis en "staging" — jamais "none", le modèle
    # reste un candidat connu, juste plus la référence en production.
    model1 = client.get(f"/api/training/jobs/{job1['id']}/model", headers=headers).json()
    assert model1["stage"] == "staging"


def test_promote_does_not_demote_production_of_a_different_target(client, db_session):
    """Deux jobs sur le MÊME dataset mais des cibles DIFFÉRENTES ne sont pas
    en concurrence pour "production" — ce sont deux problèmes distincts."""
    headers = _register(client)
    rows = "\n".join(f"{i},{i * 2},{i * 3}" for i in range(50))
    content = f"x1,x2,cible\n{rows}\n".encode()
    dataset = client.post(
        "/api/datasets", headers=headers, files={"file": ("d.csv", io.BytesIO(content), "text/csv")}
    ).json()

    job1 = _create_job(client, headers, dataset["id"], target_column="cible")
    job2 = _create_job(client, headers, dataset["id"], target_column="x2")
    org_id = db_session.query(TrainingJob).filter(TrainingJob.id == job1["id"]).first().organization_id
    _complete_job(db_session, job1["id"], org_id)
    _complete_job(db_session, job2["id"], org_id)

    client.post(f"/api/training/jobs/{job1['id']}/model/promote", headers=headers, json={"stage": "production"})
    client.post(f"/api/training/jobs/{job2['id']}/model/promote", headers=headers, json={"stage": "production"})

    model1 = client.get(f"/api/training/jobs/{job1['id']}/model", headers=headers).json()
    assert model1["stage"] == "production"  # jamais démis, cible différente


def test_promote_to_none_clears_stage_and_timestamp(client, db_session):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"])
    org_id = db_session.query(TrainingJob).filter(TrainingJob.id == job["id"]).first().organization_id
    _complete_job(db_session, job["id"], org_id)
    client.post(f"/api/training/jobs/{job['id']}/model/promote", headers=headers, json={"stage": "production"})

    resp = client.post(f"/api/training/jobs/{job['id']}/model/promote", headers=headers, json={"stage": "none"})
    body = resp.json()
    assert body["stage"] is None
    assert body["promoted_at"] is None


def test_promote_rejects_invalid_stage(client, db_session):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"])
    org_id = db_session.query(TrainingJob).filter(TrainingJob.id == job["id"]).first().organization_id
    _complete_job(db_session, job["id"], org_id)

    resp = client.post(f"/api/training/jobs/{job['id']}/model/promote", headers=headers, json={"stage": "deleted"})
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "STAGE_INVALIDE"


def test_promote_requires_a_completed_job(client, db_session):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"])  # jamais complété, pas de modèle

    resp = client.post(f"/api/training/jobs/{job['id']}/model/promote", headers=headers, json={"stage": "staging"})
    assert resp.status_code == 409


# ── Export ───────────────────────────────────────────────────────────────


def test_export_returns_the_artifact_file(client, db_session):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"])
    org_id = db_session.query(TrainingJob).filter(TrainingJob.id == job["id"]).first().organization_id
    _complete_job(db_session, job["id"], org_id)

    resp = client.get(f"/api/training/jobs/{job['id']}/model/export", headers=headers)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/octet-stream"
    assert "attachment" in resp.headers.get("content-disposition", "")
    # Le contenu renvoyé doit être un bundle joblib valide et rechargeable.
    loaded = joblib.load(io.BytesIO(resp.content))
    assert loaded == {"dummy": "bundle"}


def test_export_rejects_missing_artifact_on_disk(client, db_session):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"])
    org_id = db_session.query(TrainingJob).filter(TrainingJob.id == job["id"]).first().organization_id
    _complete_job(db_session, job["id"], org_id, artifact_path=Path(tempfile.gettempdir()) / "n_existe_pas.joblib")

    resp = client.get(f"/api/training/jobs/{job['id']}/model/export", headers=headers)
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "ARTEFACT_INTROUVABLE"


def test_export_isolated_between_organizations(client, db_session):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    dataset_a = _upload_dataset(client, headers_a, "a.csv")
    job_a = _create_job(client, headers_a, dataset_a["id"])
    org_a_id = db_session.query(TrainingJob).filter(TrainingJob.id == job_a["id"]).first().organization_id
    _complete_job(db_session, job_a["id"], org_a_id)

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    resp = client.get(f"/api/training/jobs/{job_a['id']}/model/export", headers=headers_b)
    assert resp.status_code == 404


# ── Registre ─────────────────────────────────────────────────────────────


def test_registry_lists_only_promoted_models(client, db_session):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job1 = _create_job(client, headers, dataset["id"])
    job2 = _create_job(client, headers, dataset["id"])
    org_id = db_session.query(TrainingJob).filter(TrainingJob.id == job1["id"]).first().organization_id
    _complete_job(db_session, job1["id"], org_id, algorithm="LightGBM")
    _complete_job(db_session, job2["id"], org_id, algorithm="CatBoost")  # jamais promu

    client.post(f"/api/training/jobs/{job1['id']}/model/promote", headers=headers, json={"stage": "production"})

    resp = client.get("/api/training/models/registry", headers=headers)
    assert resp.status_code == 200
    entries = resp.json()["entries"]
    assert len(entries) == 1
    assert entries[0]["algorithm"] == "LightGBM"
    assert entries[0]["stage"] == "production"


def test_registry_isolated_between_organizations(client, db_session):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    dataset_a = _upload_dataset(client, headers_a, "a.csv")
    job_a = _create_job(client, headers_a, dataset_a["id"])
    org_a_id = db_session.query(TrainingJob).filter(TrainingJob.id == job_a["id"]).first().organization_id
    _complete_job(db_session, job_a["id"], org_a_id)
    client.post(f"/api/training/jobs/{job_a['id']}/model/promote", headers=headers_a, json={"stage": "production"})

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    resp = client.get("/api/training/models/registry", headers=headers_b)
    assert resp.json()["entries"] == []


# ── Lot 5, correctif P1 — versions réelles ──────────────────────────────────


def test_second_model_on_the_same_problem_gets_version_2(client, db_session):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job1 = _create_job(client, headers, dataset["id"])
    job2 = _create_job(client, headers, dataset["id"])
    org_id = db_session.query(TrainingJob).filter(TrainingJob.id == job1["id"]).first().organization_id
    _complete_job(db_session, job1["id"], org_id, algorithm="LightGBM")
    _complete_job(db_session, job2["id"], org_id, algorithm="CatBoost")

    m1 = client.get(f"/api/training/jobs/{job1['id']}/model", headers=headers).json()
    m2 = client.get(f"/api/training/jobs/{job2['id']}/model", headers=headers).json()
    assert m1["version"] == 1
    assert m2["version"] == 2


def test_model_on_a_different_target_starts_its_own_version_sequence(client, db_session):
    headers = _register(client)
    rows = "\n".join(f"{i},{i * 2},{i * 3}" for i in range(50))
    content = f"x1,x2,cible\n{rows}\n".encode()
    dataset = client.post(
        "/api/datasets", headers=headers, files={"file": ("d.csv", io.BytesIO(content), "text/csv")}
    ).json()
    job1 = _create_job(client, headers, dataset["id"], target_column="cible")
    job2 = _create_job(client, headers, dataset["id"], target_column="x2")
    org_id = db_session.query(TrainingJob).filter(TrainingJob.id == job1["id"]).first().organization_id
    _complete_job(db_session, job1["id"], org_id)
    _complete_job(db_session, job2["id"], org_id)

    m1 = client.get(f"/api/training/jobs/{job1['id']}/model", headers=headers).json()
    m2 = client.get(f"/api/training/jobs/{job2['id']}/model", headers=headers).json()
    assert m1["version"] == 1
    assert m2["version"] == 1  # lignée séparée (cible différente), pas 2


def test_promote_to_archived_is_accepted_and_excluded_from_the_registry(client, db_session):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"])
    org_id = db_session.query(TrainingJob).filter(TrainingJob.id == job["id"]).first().organization_id
    _complete_job(db_session, job["id"], org_id)
    client.post(f"/api/training/jobs/{job['id']}/model/promote", headers=headers, json={"stage": "production"})

    resp = client.post(f"/api/training/jobs/{job['id']}/model/promote", headers=headers, json={"stage": "archived"})
    assert resp.status_code == 200
    assert resp.json()["stage"] == "archived"

    registry = client.get("/api/training/models/registry", headers=headers).json()["entries"]
    assert registry == []  # jamais dans le registre actif, même s'il fut "production"


def test_versions_endpoint_lists_the_whole_lineage_most_recent_first(client, db_session):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job1 = _create_job(client, headers, dataset["id"])
    job2 = _create_job(client, headers, dataset["id"])
    org_id = db_session.query(TrainingJob).filter(TrainingJob.id == job1["id"]).first().organization_id
    _complete_job(db_session, job1["id"], org_id, algorithm="LightGBM")
    _complete_job(db_session, job2["id"], org_id, algorithm="CatBoost")

    resp = client.get(f"/api/training/jobs/{job1['id']}/model/versions", headers=headers)
    assert resp.status_code == 200
    entries = resp.json()["entries"]
    assert [e["version"] for e in entries] == [2, 1]
    assert [e["algorithm"] for e in entries] == ["CatBoost", "LightGBM"]
    # Le job_id de la version 1 permet de la repromouvoir (rollback).
    assert entries[1]["job_id"] == job1["id"]


def test_rollback_via_repromoting_an_earlier_version(client, db_session):
    """Pas d'endpoint dédié : repromouvoir une version antérieure démet
    automatiquement la version courante — même mécanisme qu'une promotion
    normale (voir promote_model, docstring)."""
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job1 = _create_job(client, headers, dataset["id"])
    job2 = _create_job(client, headers, dataset["id"])
    org_id = db_session.query(TrainingJob).filter(TrainingJob.id == job1["id"]).first().organization_id
    _complete_job(db_session, job1["id"], org_id, algorithm="LightGBM")
    _complete_job(db_session, job2["id"], org_id, algorithm="CatBoost")

    client.post(f"/api/training/jobs/{job1['id']}/model/promote", headers=headers, json={"stage": "production"})
    client.post(f"/api/training/jobs/{job2['id']}/model/promote", headers=headers, json={"stage": "production"})
    # Rollback : version 1 reprend la production.
    resp = client.post(f"/api/training/jobs/{job1['id']}/model/promote", headers=headers, json={"stage": "production"})
    assert resp.json()["stage"] == "production"

    m2 = client.get(f"/api/training/jobs/{job2['id']}/model", headers=headers).json()
    assert m2["stage"] == "staging"  # démis par le rollback


def test_history_endpoint_reflects_promotions_across_the_lineage(client, db_session):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job1 = _create_job(client, headers, dataset["id"])
    job2 = _create_job(client, headers, dataset["id"])
    org_id = db_session.query(TrainingJob).filter(TrainingJob.id == job1["id"]).first().organization_id
    _complete_job(db_session, job1["id"], org_id, algorithm="LightGBM")
    _complete_job(db_session, job2["id"], org_id, algorithm="CatBoost")

    client.post(f"/api/training/jobs/{job1['id']}/model/promote", headers=headers, json={"stage": "production"})
    client.post(f"/api/training/jobs/{job2['id']}/model/promote", headers=headers, json={"stage": "production"})

    resp = client.get(f"/api/training/jobs/{job1['id']}/model/history", headers=headers)
    assert resp.status_code == 200
    entries = resp.json()["entries"]
    assert len(entries) == 2  # 2 promotions au total sur cette lignée
    assert entries[0]["stage"] == "production"  # la plus récente d'abord
    assert entries[0]["version"] == 2
    assert entries[0]["actor"] == "Owner"
    assert entries[1]["version"] == 1


def test_versions_and_history_isolated_between_organizations(client, db_session):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    dataset_a = _upload_dataset(client, headers_a, "a.csv")
    job_a = _create_job(client, headers_a, dataset_a["id"])
    org_a_id = db_session.query(TrainingJob).filter(TrainingJob.id == job_a["id"]).first().organization_id
    _complete_job(db_session, job_a["id"], org_a_id)
    client.post(f"/api/training/jobs/{job_a['id']}/model/promote", headers=headers_a, json={"stage": "production"})

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    assert client.get(f"/api/training/jobs/{job_a['id']}/model/versions", headers=headers_b).status_code == 404
    assert client.get(f"/api/training/jobs/{job_a['id']}/model/history", headers=headers_b).status_code == 404
