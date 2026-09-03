"""Tests du Lot 10 — durcissement SaaS : journal d'audit + quota
d'entraînements concurrents par organisation. Portée technique volontaire
(pas de plans tarifaires/facturation, hors périmètre de ce lot)."""
from __future__ import annotations

import io
import time
from unittest.mock import patch

from api.core.config import get_settings


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


def _create_job(client, headers, dataset_id):
    with patch("domains.training.router.training_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        return client.post(
            "/api/training/jobs", headers=headers, json={"dataset_id": dataset_id, "target_column": "cible"}
        )


# ── Journal d'audit ──────────────────────────────────────────────────────


def test_audit_log_records_member_added(client):
    headers = _register(client)
    client.post(
        "/api/auth/team/members",
        headers=headers,
        json={"email": "membre@bureau.fr", "nom": "Membre", "password": "motdepasse123"},
    )

    resp = client.get("/api/auth/team/audit-log", headers=headers)
    assert resp.status_code == 200
    entries = resp.json()
    assert any(e["action"] == "member.added" and e["details"]["email"] == "membre@bureau.fr" for e in entries)
    assert entries[0]["actor_name"] == "Owner"


def test_audit_log_records_dataset_deleted(client):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    client.delete(f"/api/datasets/{dataset['id']}", headers=headers)

    resp = client.get("/api/auth/team/audit-log", headers=headers)
    entries = resp.json()
    deleted = [e for e in entries if e["action"] == "dataset.deleted"]
    assert len(deleted) == 1
    assert deleted[0]["target_id"] == dataset["id"]
    assert deleted[0]["details"]["name"] == "d.csv"


def test_audit_log_records_training_job_deleted(client):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"]).json()
    client.delete(f"/api/training/jobs/{job['id']}", headers=headers)

    resp = client.get("/api/auth/team/audit-log", headers=headers)
    entries = resp.json()
    assert any(e["action"] == "training_job.deleted" and e["target_id"] == job["id"] for e in entries)


# ── Traçabilité du pilier non supervisé (AUDIT_PILIER2_ET_REFONTE_UX.md, P1)
# — jusqu'ici seul le pilier supervisé traçait ses suppressions, les 3
# modules non supervisés (clustering, réduction de dimension, anomalies)
# étaient invisibles du journal d'audit. Petits datasets synthétiques (60
# lignes), jobs jamais réellement exécutés (RQ mocké, statut "queued") — on
# vérifie la traçabilité de la suppression, pas le moteur ML.


def _upload_unsupervised_dataset(client, headers, name="d.csv", n=60):
    rows = "\n".join(f"{i},{i * 2},cat{i % 3}" for i in range(n))
    content = f"x1,x2,categorie\n{rows}\n".encode()
    resp = client.post("/api/datasets", headers=headers, files={"file": (name, io.BytesIO(content), "text/csv")})
    return resp.json()


def test_audit_log_records_clustering_job_deleted(client):
    headers = _register(client)
    dataset = _upload_unsupervised_dataset(client, headers)
    with patch("domains.clustering.router.analysis_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        job = client.post(
            "/api/clustering/jobs", headers=headers, json={"dataset_id": dataset["id"], "feature_columns": ["x1", "x2"]}
        ).json()
    client.delete(f"/api/clustering/jobs/{job['id']}", headers=headers)

    entries = client.get("/api/auth/team/audit-log", headers=headers).json()
    assert any(e["action"] == "clustering_job.deleted" and e["target_id"] == job["id"] for e in entries)


def test_audit_log_records_dimensionality_job_deleted(client):
    headers = _register(client)
    dataset = _upload_unsupervised_dataset(client, headers)
    with patch("domains.dimensionality.router.analysis_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        job = client.post(
            "/api/dimensionality/jobs", headers=headers, json={"dataset_id": dataset["id"], "feature_columns": ["x1", "x2"]}
        ).json()
    client.delete(f"/api/dimensionality/jobs/{job['id']}", headers=headers)

    entries = client.get("/api/auth/team/audit-log", headers=headers).json()
    assert any(e["action"] == "dimensionality_job.deleted" and e["target_id"] == job["id"] for e in entries)


def test_audit_log_records_anomaly_job_deleted(client):
    headers = _register(client)
    dataset = _upload_unsupervised_dataset(client, headers)
    with patch("domains.anomalies.router.analysis_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        job = client.post(
            "/api/anomalies/jobs", headers=headers, json={"dataset_id": dataset["id"], "feature_columns": ["x1", "x2"]}
        ).json()
    client.delete(f"/api/anomalies/jobs/{job['id']}", headers=headers)

    entries = client.get("/api/auth/team/audit-log", headers=headers).json()
    assert any(e["action"] == "anomaly_job.deleted" and e["target_id"] == job["id"] for e in entries)


def test_audit_log_records_model_promotion(client, db_session):
    import json as _json

    from api.core.models import MLModel, TrainingJob
    from domains.training.services.versioning import next_version

    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job = _create_job(client, headers, dataset["id"]).json()
    org_id = db_session.query(TrainingJob).filter(TrainingJob.id == job["id"]).first().organization_id
    job_row = db_session.query(TrainingJob).filter(TrainingJob.id == job["id"]).first()
    job_row.status = "completed"
    db_session.add(MLModel(
        organization_id=org_id, training_job_id=job_row.id, dataset_id=job_row.dataset_id,
        version=next_version(db_session, org_id, job_row.dataset_id, job_row.target_column),
        algorithm="LightGBM", task_type="regression",
        target_column="cible", feature_columns_json=_json.dumps(["x1", "x2"]), file_path="unused.joblib",
        metrics_json=_json.dumps({"r2_test": 0.9}),
    ))
    db_session.commit()

    client.post(f"/api/training/jobs/{job['id']}/model/promote", headers=headers, json={"stage": "production"})

    entries = client.get("/api/auth/team/audit-log", headers=headers).json()
    promoted = [e for e in entries if e["action"] == "model.promoted"]
    assert len(promoted) == 1
    assert promoted[0]["details"]["stage"] == "production"


def test_audit_log_restricted_to_owner(client):
    headers = _register(client)
    client.post(
        "/api/auth/team/members",
        headers=headers,
        json={"email": "membre@bureau.fr", "nom": "Membre", "password": "motdepasse123"},
    )
    # Le membre remplace d'abord le mot de passe provisoire fixé par le
    # propriétaire : sans cela il serait refusé par la garde
    # AUTH_MDP_PROVISOIRE, et ce test vérifierait le mauvais mécanisme —
    # il passerait au vert sans jamais éprouver le contrôle de rôle.
    first = client.post(
        "/api/auth/login", data={"username": "membre@bureau.fr", "password": "motdepasse123"}
    ).json()
    client.patch(
        "/api/auth/me/password",
        headers={"Authorization": f"Bearer {first['access_token']}"},
        json={
            "current_password": "motdepasse123",
            "new_password": "sonpropremdp456",
            "new_password_confirm": "sonpropremdp456",
        },
    )
    # Un jeton émis dans la même seconde que la révocation est rejeté
    # (granularité de `iat`, voir `get_current_user`).
    time.sleep(1.05)
    member_login = client.post(
        "/api/auth/login", data={"username": "membre@bureau.fr", "password": "sonpropremdp456"}
    ).json()
    member_headers = {"Authorization": f"Bearer {member_login['access_token']}"}

    resp = client.get("/api/auth/team/audit-log", headers=member_headers)
    assert resp.status_code == 403
    assert resp.json()["detail"]["code"] == "AUTH_OWNER_REQUIS"


def test_audit_log_isolated_between_organizations(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    dataset_a = _upload_dataset(client, headers_a, "a.csv")
    client.delete(f"/api/datasets/{dataset_a['id']}", headers=headers_a)

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    entries_b = client.get("/api/auth/team/audit-log", headers=headers_b).json()
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


# ── Watchdog de jobs orphelins (H2, AUDIT_ROADMAP.md) ────────────────────


def test_stale_running_job_is_reconciled_and_frees_quota(client, db_session):
    """Un job 'running' sans signal de vie depuis plus de
    stale_job_timeout_minutes (worker mort) ne doit jamais bloquer le quota
    indéfiniment — reclassé 'failed' à la prochaine tentative de création."""
    from datetime import datetime, timedelta, timezone

    from api.core.models import TrainingJob

    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    limit = get_settings().max_concurrent_jobs_per_org
    timeout = get_settings().stale_job_timeout_minutes

    jobs = [_create_job(client, headers, dataset["id"]).json() for _ in range(limit)]
    stale_time = datetime.now(timezone.utc) - timedelta(minutes=timeout + 5)
    for j in jobs:
        row = db_session.query(TrainingJob).filter(TrainingJob.id == j["id"]).first()
        row.status = "running"
        row.started_at = stale_time
        row.progress_updated_at = stale_time
    db_session.commit()

    resp = _create_job(client, headers, dataset["id"])
    assert resp.status_code == 201

    for j in jobs:
        row = db_session.query(TrainingJob).filter(TrainingJob.id == j["id"]).first()
        assert row.status == "failed"
        assert row.error_message  # message actionnable, jamais vide


def test_recent_running_job_is_not_reconciled(client, db_session):
    """Un job 'running' avec un signal de vie récent est un entraînement
    réellement en cours — ne doit jamais être reclassé par erreur."""
    from datetime import datetime, timezone

    from api.core.models import TrainingJob

    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    limit = get_settings().max_concurrent_jobs_per_org

    jobs = [_create_job(client, headers, dataset["id"]).json() for _ in range(limit)]
    now = datetime.now(timezone.utc)
    for j in jobs:
        row = db_session.query(TrainingJob).filter(TrainingJob.id == j["id"]).first()
        row.status = "running"
        row.started_at = now
        row.progress_updated_at = now
    db_session.commit()

    resp = _create_job(client, headers, dataset["id"])
    assert resp.status_code == 429

    for j in jobs:
        row = db_session.query(TrainingJob).filter(TrainingJob.id == j["id"]).first()
        assert row.status == "running"
