"""Corrélation request_id -- API vers job/worker et journal d'audit (Phase 3,
AUDIT_BACKEND_2026-08-23.md, Axe I) -- avant ce correctif, un job traite en
tache de fond (RQ) et une entree d'audit n'avaient aucun lien retrouvable
avec la requete HTTP qui les avait produits."""

from __future__ import annotations

from unittest.mock import patch

from api.core.models import AuditLog, TrainingJob

_FIXED_REQUEST_ID = "d290f1ee-6c54-4b01-90e6-d701748f0851"


def _register(client, email="rid@bureau.fr"):
    resp = client.post(
        "/api/auth/register",
        json={"email": email, "nom": "Owner", "password": "motdepasse123", "organization_name": "Bureau"},
        headers={"X-Request-ID": _FIXED_REQUEST_ID},
    )
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _upload_dataset(client, headers):
    import io

    rows = "\n".join(f"{i},{i * 2},{i * 3}" for i in range(50))
    return client.post(
        "/api/datasets",
        headers=headers,
        files={"file": ("d.csv", io.BytesIO(f"x1,x2,cible\n{rows}\n".encode()), "text/csv")},
    ).json()


def test_job_creation_stores_the_request_id_of_its_creating_request(client, db_session):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)

    with patch("domains.training.router.training_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        resp = client.post(
            "/api/training/jobs",
            headers={**headers, "X-Request-ID": _FIXED_REQUEST_ID},
            json={"dataset_id": dataset["id"], "target_column": "cible"},
        )
    assert resp.status_code == 201
    job = db_session.query(TrainingJob).filter(TrainingJob.id == resp.json()["id"]).one()
    assert job.request_id == _FIXED_REQUEST_ID


def test_a_different_request_produces_a_different_job_request_id(client, db_session):
    """Non-régression du cas trivial : deux jobs créés par deux requêtes
    distinctes n'héritent jamais du même request_id par accident (variable
    de contexte mal réinitialisée entre deux requêtes du même process)."""
    headers = _register(client)
    dataset = _upload_dataset(client, headers)

    with patch("domains.training.router.training_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        first = client.post(
            "/api/training/jobs", headers=headers, json={"dataset_id": dataset["id"], "target_column": "cible"}
        )
        second = client.post(
            "/api/training/jobs", headers=headers, json={"dataset_id": dataset["id"], "target_column": "cible"}
        )
    job_a = db_session.query(TrainingJob).filter(TrainingJob.id == first.json()["id"]).one()
    job_b = db_session.query(TrainingJob).filter(TrainingJob.id == second.json()["id"]).one()
    assert job_a.request_id is not None
    assert job_b.request_id is not None
    assert job_a.request_id != job_b.request_id


def test_audit_log_entry_carries_the_request_id_of_its_producing_request(client, db_session):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)

    del_resp = client.delete(f"/api/datasets/{dataset['id']}", headers={**headers, "X-Request-ID": _FIXED_REQUEST_ID})
    assert del_resp.status_code == 204

    entry = (
        db_session.query(AuditLog)
        .filter(AuditLog.action == "dataset.deleted", AuditLog.target_id == dataset["id"])
        .one()
    )
    assert entry.request_id == _FIXED_REQUEST_ID

    # Exposé via l'API elle-même, pas seulement en base.
    listed = client.get("/api/auth/team/audit-log", headers=headers).json()
    assert listed[0]["request_id"] == _FIXED_REQUEST_ID


def test_audit_log_request_id_is_none_outside_an_http_request():
    """`log_action` peut en théorie être appelée hors d'une requête HTTP
    (script, tâche de fond future) — `request_id_var` vaut alors `"-"`
    (défaut du ContextVar, voir observability.py) : ne doit jamais être
    stocké tel quel, une chaîne littérale `"-"` en base serait moins
    honnête qu'un vrai NULL."""
    from domains.shared.audit import log_action

    class _FakeSession:
        added = None

        def add(self, obj):
            self.added = obj

    db = _FakeSession()
    log_action(db, organization_id=1, actor_id=1, action="test.action")
    assert db.added.request_id is None


def test_openapi_schema_exposes_the_error_code_catalog(client):
    """Phase 3, §5 — catalogue d'erreurs stable et découvrable via
    `/openapi.json`, plutôt que uniquement lisible dans le code source."""
    schema = client.get("/openapi.json").json()
    assert "x-error-codes" in schema
    assert "ERREUR_INTERNE" in schema["x-error-codes"]
    assert "QUOTA_ENTRAINEMENTS_ATTEINT" in schema["x-error-codes"]
    assert "DATASET_INTROUVABLE" in schema["x-error-codes"]
