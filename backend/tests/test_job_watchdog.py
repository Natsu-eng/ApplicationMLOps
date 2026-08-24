"""Réconciliation des jobs orphelins — `"running"` (H2, AUDIT_ROADMAP.md,
comportement historique) ET, depuis la Phase 2 (AUDIT_BACKEND_2026-08-23.md
§F3), `"queued"` dont le job RQ sous-jacent a disparu de Redis."""

from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from api.core.job_queue import redis_conn
from api.core.models import TrainingJob
from domains.shared.job_watchdog import reconcile_stale_jobs


def _register(client, email="watchdog@bureau.fr"):
    resp = client.post(
        "/api/auth/register",
        json={"email": email, "nom": "Watchdog", "password": "motdepasse123", "organization_name": "Bureau"},
    )
    return resp.json()


def _create_queued_job(client, headers, dataset_id) -> int:
    with patch("domains.training.router.training_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "rq-id-jamais-persiste-en-redis"
        resp = client.post(
            "/api/training/jobs", headers=headers, json={"dataset_id": dataset_id, "target_column": "cible"}
        )
    return resp.json()["id"]


def test_reconcile_marks_stale_running_job_as_failed(client, db_session):
    tokens = _register(client, "running-stale@bureau.fr")
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    rows = "\n".join(f"{i},{i * 2},{i * 3}" for i in range(50))
    dataset = client.post(
        "/api/datasets",
        headers=headers,
        files={"file": ("d.csv", io.BytesIO(f"x1,x2,cible\n{rows}\n".encode()), "text/csv")},
    ).json()
    job_id = _create_queued_job(client, headers, dataset["id"])

    job = db_session.query(TrainingJob).filter(TrainingJob.id == job_id).one()
    job.status = "running"
    job.started_at = datetime.now(timezone.utc) - timedelta(minutes=90)
    job.progress_updated_at = datetime.now(timezone.utc) - timedelta(minutes=90)
    db_session.commit()

    reconciled = reconcile_stale_jobs(db_session, job.organization_id, stale_after_minutes=40)

    assert reconciled == 1
    db_session.refresh(job)
    assert job.status == "failed"
    assert job.finished_at is not None


def test_reconcile_leaves_fresh_running_job_alone(client, db_session):
    tokens = _register(client, "running-fresh@bureau.fr")
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    rows = "\n".join(f"{i},{i * 2},{i * 3}" for i in range(50))
    dataset = client.post(
        "/api/datasets",
        headers=headers,
        files={"file": ("d.csv", io.BytesIO(f"x1,x2,cible\n{rows}\n".encode()), "text/csv")},
    ).json()
    job_id = _create_queued_job(client, headers, dataset["id"])

    job = db_session.query(TrainingJob).filter(TrainingJob.id == job_id).one()
    job.status = "running"
    job.started_at = datetime.now(timezone.utc)
    job.progress_updated_at = datetime.now(timezone.utc)
    db_session.commit()

    reconciled = reconcile_stale_jobs(db_session, job.organization_id, stale_after_minutes=40)

    assert reconciled == 0
    db_session.refresh(job)
    assert job.status == "running"


def test_reconcile_marks_queued_job_as_failed_when_rq_job_is_gone(client, db_session):
    """Le coeur du correctif Phase 2 (§F3) — un job `"queued"` dont le
    `rq_job_id` ne correspond à aucun job réellement présent dans Redis
    (jamais persisté, ici — simule une perte Redis) est désormais détecté,
    alors qu'avant ce correctif il serait resté `"queued"` pour toujours."""
    tokens = _register(client, "queued-lost@bureau.fr")
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    rows = "\n".join(f"{i},{i * 2},{i * 3}" for i in range(50))
    dataset = client.post(
        "/api/datasets",
        headers=headers,
        files={"file": ("d.csv", io.BytesIO(f"x1,x2,cible\n{rows}\n".encode()), "text/csv")},
    ).json()
    job_id = _create_queued_job(client, headers, dataset["id"])

    job = db_session.query(TrainingJob).filter(TrainingJob.id == job_id).one()
    assert job.status == "queued"
    assert job.rq_job_id == "rq-id-jamais-persiste-en-redis"  # jamais réellement enfilé (queue mockée)
    # Délai de grâce (voir job_watchdog.py) : un job tout juste créé n'est
    # jamais vérifié dans Redis, quel que soit son `rq_job_id` — simule ici
    # un job réellement resté "queued" sans jamais avoir été pris.
    job.created_at = datetime.now(timezone.utc) - timedelta(minutes=90)
    db_session.commit()

    reconciled = reconcile_stale_jobs(db_session, job.organization_id, stale_after_minutes=40)

    assert reconciled == 1
    db_session.refresh(job)
    assert job.status == "failed"
    assert "perdu" in job.error_message.lower()


def test_reconcile_marks_queued_job_with_null_rq_job_id_as_failed(client, db_session):
    """Résidu possible d'avant le correctif F5 (`job_creation.py`), ou
    incohérence de données — un `"queued"` sans `rq_job_id` du tout n'est
    jamais un état transitoire légitime."""
    tokens = _register(client, "queued-null@bureau.fr")
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    rows = "\n".join(f"{i},{i * 2},{i * 3}" for i in range(50))
    dataset = client.post(
        "/api/datasets",
        headers=headers,
        files={"file": ("d.csv", io.BytesIO(f"x1,x2,cible\n{rows}\n".encode()), "text/csv")},
    ).json()
    job_id = _create_queued_job(client, headers, dataset["id"])

    job = db_session.query(TrainingJob).filter(TrainingJob.id == job_id).one()
    job.rq_job_id = None
    job.created_at = datetime.now(timezone.utc) - timedelta(minutes=90)
    db_session.commit()

    reconciled = reconcile_stale_jobs(db_session, job.organization_id, stale_after_minutes=40)

    assert reconciled == 1
    db_session.refresh(job)
    assert job.status == "failed"


def test_reconcile_leaves_freshly_queued_job_alone_even_if_rq_job_is_gone(client, db_session):
    """Délai de grâce (voir job_watchdog.py) — un job `"queued"` tout juste
    créé n'est jamais vérifié dans Redis, même si son `rq_job_id` ne
    correspond à rien (queue mockée en test, ou brève fenêtre de cohérence
    éventuelle en production juste après l'enfilage). Bug réel trouvé en
    exécutant la suite COMPLÈTE (jamais visible fichier par fichier) :
    sans ce délai, `test_saas_hardening.py::test_quota_blocks_creation_beyond_the_limit`
    et les tests de quota similaires échouaient — chaque job nouvellement
    créé (file RQ mockée dans ces tests) était immédiatement réconcilié
    `"failed"` à la création du suivant, le quota ne se déclenchait jamais."""
    tokens = _register(client, "queued-fresh-gone@bureau.fr")
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    rows = "\n".join(f"{i},{i * 2},{i * 3}" for i in range(50))
    dataset = client.post(
        "/api/datasets",
        headers=headers,
        files={"file": ("d.csv", io.BytesIO(f"x1,x2,cible\n{rows}\n".encode()), "text/csv")},
    ).json()
    job_id = _create_queued_job(client, headers, dataset["id"])

    job = db_session.query(TrainingJob).filter(TrainingJob.id == job_id).one()
    assert job.status == "queued"
    assert job.rq_job_id == "rq-id-jamais-persiste-en-redis"  # jamais réellement enfilé (queue mockée)

    reconciled = reconcile_stale_jobs(db_session, job.organization_id, stale_after_minutes=40)

    assert reconciled == 0
    db_session.refresh(job)
    assert job.status == "queued"


def test_reconcile_leaves_queued_job_alone_when_rq_job_really_exists(client, db_session):
    """Un job réellement en attente dans Redis (juste pas encore pris par
    un worker — charge normale) ne doit jamais être déclaré perdu."""
    tokens = _register(client, "queued-real@bureau.fr")
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    rows = "\n".join(f"{i},{i * 2},{i * 3}" for i in range(50))
    dataset = client.post(
        "/api/datasets",
        headers=headers,
        files={"file": ("d.csv", io.BytesIO(f"x1,x2,cible\n{rows}\n".encode()), "text/csv")},
    ).json()

    with patch("domains.training.router.training_queue") as mock_queue_router:
        # Enfile pour de vrai sur la file "training" réelle (Redis local de
        # test), jamais consommée par un worker dans ce test — simule
        # fidèlement "en attente, pas encore pris".
        from api.core.job_queue import training_queue

        def _real_enqueue(*args, **kwargs):
            return training_queue.enqueue(*args, **kwargs)

        mock_queue_router.enqueue.side_effect = _real_enqueue
        resp = client.post(
            "/api/training/jobs", headers=headers, json={"dataset_id": dataset["id"], "target_column": "cible"}
        )
    job_id = resp.json()["id"]
    job = db_session.query(TrainingJob).filter(TrainingJob.id == job_id).one()
    assert job.rq_job_id  # réellement enfilé

    try:
        reconciled = reconcile_stale_jobs(db_session, job.organization_id, stale_after_minutes=40)
        assert reconciled == 0
        db_session.refresh(job)
        assert job.status == "queued"
    finally:
        # Nettoyage — sans quoi ce job réel resterait dans Redis pour le
        # process de test suivant.
        redis_conn.delete(f"rq:job:{job.rq_job_id}")
