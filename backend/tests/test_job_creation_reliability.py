"""Idempotence de création de job + échec propre à l'enfilage (Phase 2,
AUDIT_BACKEND_2026-08-23.md, Axe F.4/F.5) — `domains/shared/job_creation.py`,
partagé par les 6 domaines à job. Testé sur `training` comme domaine de
référence (même pattern que `test_idor_regression.py` pour l'isolation
multi-tenant) : chaque domaine délègue aux deux mêmes fonctions partagées,
un test par domaine n'apporterait pas de garantie supplémentaire."""

from __future__ import annotations

import io
from unittest.mock import patch

from api.core.models import TrainingJob


def _register(client, email="reliability@bureau.fr"):
    resp = client.post(
        "/api/auth/register",
        json={"email": email, "nom": "Fiab", "password": "motdepasse123", "organization_name": "Bureau"},
    )
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _upload_dataset(client, headers, name="d.csv"):
    rows = "\n".join(f"{i},{i * 2},{i * 3}" for i in range(50))
    content = f"x1,x2,cible\n{rows}\n".encode()
    resp = client.post("/api/datasets", headers=headers, files={"file": (name, io.BytesIO(content), "text/csv")})
    return resp.json()


def _create_job(client, headers, dataset_id, idempotency_key: str | None = None):
    request_headers = dict(headers)
    if idempotency_key is not None:
        request_headers["Idempotency-Key"] = idempotency_key
    with patch("domains.training.router.training_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        return client.post(
            "/api/training/jobs",
            headers=request_headers,
            json={"dataset_id": dataset_id, "target_column": "cible"},
        )


def test_same_idempotency_key_returns_the_same_job(client, db_session):
    headers = _register(client, "idem-a@bureau.fr")
    dataset = _upload_dataset(client, headers)

    first = _create_job(client, headers, dataset["id"], idempotency_key="clic-du-bouton-1")
    second = _create_job(client, headers, dataset["id"], idempotency_key="clic-du-bouton-1")

    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["id"] == second.json()["id"]
    assert db_session.query(TrainingJob).count() == 1  # un seul job créé en base, pas deux


def test_different_idempotency_keys_create_distinct_jobs(client, db_session):
    headers = _register(client, "idem-b@bureau.fr")
    dataset = _upload_dataset(client, headers)

    first = _create_job(client, headers, dataset["id"], idempotency_key="tentative-1")
    second = _create_job(client, headers, dataset["id"], idempotency_key="tentative-2")

    assert first.json()["id"] != second.json()["id"]
    assert db_session.query(TrainingJob).count() == 2


def test_no_idempotency_key_never_deduplicates(client, db_session):
    """Comportement historique préservé : sans en-tête, chaque requête crée
    un job distinct — l'idempotence est une protection OPT-IN, jamais une
    déduplication automatique par contenu."""
    headers = _register(client, "idem-c@bureau.fr")
    dataset = _upload_dataset(client, headers)

    first = _create_job(client, headers, dataset["id"])
    second = _create_job(client, headers, dataset["id"])

    assert first.json()["id"] != second.json()["id"]
    assert db_session.query(TrainingJob).count() == 2


def test_idempotency_key_scoped_by_organization(client, db_session):
    """La même clé, utilisée par deux organisations différentes, ne doit
    JAMAIS faire fuiter le job de l'une vers l'autre (voir
    `resolve_idempotent_job_id`, scope explicite par organization_id)."""
    headers_a = _register(client, "org-a@bureau-a.fr")
    headers_b = _register(client, "org-b@bureau-b.fr")
    dataset_a = _upload_dataset(client, headers_a)
    dataset_b = _upload_dataset(client, headers_b)

    resp_a = _create_job(client, headers_a, dataset_a["id"], idempotency_key="meme-cle")
    resp_b = _create_job(client, headers_b, dataset_b["id"], idempotency_key="meme-cle")

    assert resp_a.json()["id"] != resp_b.json()["id"]
    assert db_session.query(TrainingJob).count() == 2


def test_enqueue_failure_marks_job_failed_instead_of_orphaned_queued(client, db_session):
    """LE test central du correctif F5 — avant lui, un job dont l'enfilage
    échouait restait "queued" avec rq_job_id=NULL pour toujours, invisible à
    job_watchdog.py (qui ne couvre que "running"). Vérifie que ce cas ne
    peut plus se produire : le job est marqué "failed" DANS LA MÊME
    requête, et le client reçoit un 503 explicite plutôt qu'un 201
    mensonger."""
    headers = _register(client, "enqueue-fail@bureau.fr")
    dataset = _upload_dataset(client, headers)

    with patch("domains.training.router.training_queue") as mock_queue:
        mock_queue.enqueue.side_effect = ConnectionError("Redis indisponible (simulation)")
        resp = client.post(
            "/api/training/jobs", headers=headers, json={"dataset_id": dataset["id"], "target_column": "cible"}
        )

    assert resp.status_code == 503
    assert resp.json()["detail"]["code"] == "FILE_INDISPONIBLE"

    job = db_session.query(TrainingJob).one()
    assert job.status == "failed"  # jamais "queued" orphelin
    assert job.rq_job_id is None
    assert job.error_message  # message actionnable persisté, pas vide
