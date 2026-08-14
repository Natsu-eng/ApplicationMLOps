"""Tests du Lot 10 — durcissement SaaS : journal d'audit + quota
d'entraînements concurrents par organisation. Portée technique volontaire
(pas de plans tarifaires/facturation, hors périmètre de ce lot)."""
from __future__ import annotations

import io
from unittest.mock import patch

from api.core.config import get_settings


def _register(client, email="owner@bureau.fr", org="Bureau"):
    resp = client.post(
        "/auth/register",
        json={"email": email, "nom": "Owner", "password": "motdepasse123", "organization_name": org},
    ).json()
    return {"Authorization": f"Bearer {resp['access_token']}"}


def _upload_dataset(client, headers, name="d.csv"):
    rows = "\n".join(f"{i},{i * 2},{i * 3}" for i in range(50))
    content = f"x1,x2,cible\n{rows}\n".encode()
    resp = client.post("/datasets", headers=headers, files={"file": (name, io.BytesIO(content), "text/csv")})
    return resp.json()


def _create_job(client, headers, dataset_id):
    with patch("api.routers.training.training_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        return client.post(
            "/training/jobs", headers=headers, json={"dataset_id": dataset_id, "target_column": "cible"}
        )


# ── Journal d'audit ──────────────────────────────────────────────────────


def test_audit_log_records_member_added(client):
    headers = _register(client)
    client.post(
        "/auth/team/members",
        headers=headers,
        json={"email": "membre@bureau.fr", "nom": "Membre", "password": "motdepasse123"},
    )

    resp = client.get("/auth/team/audit-log", headers=headers)
    assert resp.status_code == 200
    entries = resp.json()
    assert any(e["action"] == "member.added" and e["details"]["email"] == "membre@bureau.fr" for e in entries)
    assert entries[0]["actor_name"] == "Owner"


def test_audit_log_records_dataset_deleted(client):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    client.delete(f"/datasets/{dataset['id']}", headers=headers)

    resp = client.get("/auth/team/audit-log", headers=headers)
    entries = resp.json()
    deleted = [e for e in entries if e["action"] == "dataset.deleted"]
    assert len(deleted) == 1
    assert deleted[0]["target_id"] == dataset["id"]
    assert deleted[0]["details"]["name"] == "d.csv"


def test_audit_log_records_training_job_deleted(client):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"]).json()
    client.delete(f"/training/jobs/{job['id']}", headers=headers)

    resp = client.get("/auth/team/audit-log", headers=headers)
    entries = resp.json()
    assert any(e["action"] == "training_job.deleted" and e["target_id"] == job["id"] for e in entries)


def test_audit_log_records_model_promotion(client, db_session):
    from api.core.models import MLModel, TrainingJob
    import json as _json

    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"]).json()
    org_id = db_session.query(TrainingJob).filter(TrainingJob.id == job["id"]).first().organization_id
    job_row = db_session.query(TrainingJob).filter(TrainingJob.id == job["id"]).first()
    job_row.status = "completed"
    db_session.add(MLModel(
        organization_id=org_id, training_job_id=job_row.id, algorithm="LightGBM", task_type="regression",
        target_column="cible", feature_columns_json=_json.dumps(["x1", "x2"]), file_path="unused.joblib",
        metrics_json=_json.dumps({"r2_test": 0.9}),
    ))
    db_session.commit()

    client.post(f"/training/jobs/{job['id']}/model/promote", headers=headers, json={"stage": "production"})

    entries = client.get("/auth/team/audit-log", headers=headers).json()
    promoted = [e for e in entries if e["action"] == "model.promoted"]
    assert len(promoted) == 1
    assert promoted[0]["details"]["stage"] == "production"


def test_audit_log_restricted_to_owner(client):
    headers = _register(client)
    client.post(
        "/auth/team/members",
        headers=headers,
        json={"email": "membre@bureau.fr", "nom": "Membre", "password": "motdepasse123"},
    )
    member_login = client.post(
        "/auth/login", data={"username": "membre@bureau.fr", "password": "motdepasse123"}
    ).json()
    member_headers = {"Authorization": f"Bearer {member_login['access_token']}"}

    resp = client.get("/auth/team/audit-log", headers=member_headers)
    assert resp.status_code == 403


def test_audit_log_isolated_between_organizations(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    dataset_a = _upload_dataset(client, headers_a, "a.csv")
    client.delete(f"/datasets/{dataset_a['id']}", headers=headers_a)

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    entries_b = client.get("/auth/team/audit-log", headers=headers_b).json()
    assert not any(e["action"] == "dataset.deleted" for e in entries_b)


# ── Quota d'entraînements concurrents ────────────────────────────────────


def test_quota_blocks_creation_beyond_the_limit(client):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    limit = get_settings().max_concurrent_jobs_per_org

    responses = [_create_job(client, headers, dataset["id"]) for _ in range(limit)]
    assert all(r.status_code == 201 for r in responses)

    over_limit = _create_job(client, headers, dataset["id"])
    assert over_limit.status_code == 429
    assert over_limit.json()["detail"]["code"] == "QUOTA_ENTRAINEMENTS_ATTEINT"


def test_quota_does_not_count_completed_or_failed_jobs(client, db_session):
    from api.core.models import TrainingJob

    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    limit = get_settings().max_concurrent_jobs_per_org

    jobs = [_create_job(client, headers, dataset["id"]).json() for _ in range(limit)]
    # Termine tous les jobs "actifs" — ils ne doivent plus compter dans le quota.
    for j in jobs:
        row = db_session.query(TrainingJob).filter(TrainingJob.id == j["id"]).first()
        row.status = "completed"
    db_session.commit()

    resp = _create_job(client, headers, dataset["id"])
    assert resp.status_code == 201


def test_quota_isolated_between_organizations(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    dataset_a = _upload_dataset(client, headers_a, "a.csv")
    limit = get_settings().max_concurrent_jobs_per_org
    for _ in range(limit):
        assert _create_job(client, headers_a, dataset_a["id"]).status_code == 201
    # Org A a atteint son quota — org B doit pouvoir lancer normalement.
    assert _create_job(client, headers_a, dataset_a["id"]).status_code == 429

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    dataset_b = _upload_dataset(client, headers_b, "b.csv")
    assert _create_job(client, headers_b, dataset_b["id"]).status_code == 201
