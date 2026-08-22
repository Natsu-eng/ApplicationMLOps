"""Pagination par curseur (Lot 4, correctif I3, AUDIT_DATALAB_2026-08-16.md
§C.2.4) — teste le comportement à travers UN endpoint représentatif
(`GET /training/jobs`), le motif étant identique sur les 5 autres
(clustering/dimensionality/anomalies/vision classification/vision
anomalies) : `api/core/pagination.py::paginate_by_id`, partagé par les 6.
Les tests de non-régression des 5 autres endpoints (déjà couverts par leurs
suites respectives) confirment que le motif appliqué est identique."""
from __future__ import annotations

from unittest.mock import patch

from api.core.models import TrainingJob


def _register(client, email="owner@bureau.fr"):
    resp = client.post(
        "/api/auth/register",
        json={"email": email, "nom": "Owner", "password": "motdepasse123", "organization_name": "Bureau"},
    ).json()
    return {"Authorization": f"Bearer {resp['access_token']}"}


def _upload_dataset(client, headers, name="d.csv"):
    import io
    content = b"x1,x2,cible\n" + b"\n".join(f"{i},{i*2},{i%2}".encode() for i in range(20))
    resp = client.post("/api/datasets", headers=headers, files={"file": (name, io.BytesIO(content), "text/csv")})
    return resp.json()


def _create_jobs(client, db_session, headers, dataset_id, n):
    """Marque chaque job "completed" immédiatement après création (via
    `db_session`, pas l'API) — sans ça, `max_concurrent_jobs_per_org`
    (3 par défaut, `services/job_quota.py`) bloque toute création au-delà
    de 3 jobs "queued" simultanés, ce qui n'a rien à voir avec ce que ces
    tests de pagination vérifient (le nombre TOTAL de jobs, actifs ou non)."""
    with patch("domains.training.router.training_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        for _ in range(n):
            resp = client.post("/api/training/jobs", headers=headers, json={"dataset_id": dataset_id, "target_column": "cible"})
            job = db_session.query(TrainingJob).filter(TrainingJob.id == resp.json()["id"]).first()
            job.status = "completed"
            db_session.commit()


def test_no_pagination_params_returns_everything_unlimited(client, db_session):
    """Comportement d'avant ce lot, inchangé — aucun appelant existant
    (Dashboard, pages Historique) n'est cassé par ce correctif."""
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    _create_jobs(client, db_session, headers, dataset["id"], 7)

    resp = client.get("/api/training/jobs", headers=headers)
    assert resp.status_code == 200
    assert len(resp.json()) == 7
    assert "X-Next-Cursor" not in resp.headers


def test_limit_returns_only_the_requested_page_size(client, db_session):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    _create_jobs(client, db_session, headers, dataset["id"], 7)

    resp = client.get("/api/training/jobs", headers=headers, params={"limit": 3})
    assert resp.status_code == 200
    assert len(resp.json()) == 3


def test_next_cursor_header_present_when_more_pages_exist(client, db_session):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    _create_jobs(client, db_session, headers, dataset["id"], 7)

    resp = client.get("/api/training/jobs", headers=headers, params={"limit": 3})
    assert "X-Next-Cursor" in resp.headers


def test_next_cursor_header_absent_on_last_page(client, db_session):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    _create_jobs(client, db_session, headers, dataset["id"], 3)

    resp = client.get("/api/training/jobs", headers=headers, params={"limit": 3})
    assert len(resp.json()) == 3
    assert "X-Next-Cursor" not in resp.headers  # exactement 3, pas de page suivante


def test_cursor_advances_to_the_next_page_without_overlap_or_gap(client, db_session):
    """Le curseur pris de la 1ère page redonne exactement le reste, sans
    doublon ni omission — le vrai test de correction d'une pagination."""
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    _create_jobs(client, db_session, headers, dataset["id"], 7)

    all_jobs = client.get("/api/training/jobs", headers=headers).json()
    all_ids_by_creation_order = [j["id"] for j in all_jobs]  # déjà trié created_at desc

    page1 = client.get("/api/training/jobs", headers=headers, params={"limit": 3})
    page1_ids = [j["id"] for j in page1.json()]
    cursor = page1.headers["X-Next-Cursor"]

    page2 = client.get("/api/training/jobs", headers=headers, params={"limit": 3, "cursor": cursor})
    page2_ids = [j["id"] for j in page2.json()]
    cursor2 = page2.headers["X-Next-Cursor"]

    page3 = client.get("/api/training/jobs", headers=headers, params={"limit": 3, "cursor": cursor2})
    page3_ids = [j["id"] for j in page3.json()]
    assert "X-Next-Cursor" not in page3.headers  # dernière page (7 = 3+3+1)

    assert page1_ids + page2_ids + page3_ids == all_ids_by_creation_order
    assert len(set(page1_ids + page2_ids + page3_ids)) == 7  # aucun doublon


def test_pagination_isolated_between_organizations(client, db_session):
    """Le curseur d'une organisation ne doit jamais laisser fuir les jobs
    d'une autre — même filtre organization_id qu'avant ce lot."""
    headers_a = _register(client, "a@bureau-a.fr")
    dataset_a = _upload_dataset(client, headers_a, "a.csv")
    _create_jobs(client, db_session, headers_a, dataset_a["id"], 5)

    headers_b = _register(client, "b@bureau-b.fr")
    dataset_b = _upload_dataset(client, headers_b, "b.csv")
    _create_jobs(client, db_session, headers_b, dataset_b["id"], 2)

    resp = client.get("/api/training/jobs", headers=headers_b, params={"limit": 10})
    assert len(resp.json()) == 2
    assert "X-Next-Cursor" not in resp.headers
