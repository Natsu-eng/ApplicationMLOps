"""Tests de GET /training/jobs/compare (Lot D-bis) — comparaison inter-jobs.

Le Lot D comparait déjà les modèles D'UN MÊME job (leaderboard intra-job,
voir test_training_api.py::test_candidates_endpoint_*) ; ce lot compare
PLUSIEURS jobs entre eux (config, métriques)."""
from __future__ import annotations

import io
import json
from unittest.mock import patch

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


def _complete_job(db_session, job_id, org_id, algorithm="LightGBM", r2_test=0.9, cv_score=0.88):
    job = db_session.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    job.status = "completed"
    db_session.add(
        MLModel(
            organization_id=org_id,
            training_job_id=job.id,
            dataset_id=job.dataset_id,
            version=next_version(db_session, org_id, job.dataset_id, job.target_column),
            algorithm=algorithm,
            task_type="regression",
            target_column="cible",
            feature_columns_json=json.dumps(["x1", "x2"]),
            file_path="unused.joblib",
            metrics_json=json.dumps({"r2_test": r2_test, "cv_score": cv_score}),
        )
    )
    db_session.commit()
    return job


def _create_job(client, headers, dataset_id, **overrides):
    # Context manager plutôt que décorateur @patch : appelée depuis un
    # SITE D'APPEL avec ses propres arguments positionnels (client, headers,
    # dataset_id), pas depuis pytest comme une fonction de test — l'ordre
    # d'injection du mock par @patch (ajouté en DERNIER argument positionnel,
    # convention `def test_x(self, mock)`) ne correspond pas à un appel
    # explicite comme celui-ci.
    with patch("api.routers.training.training_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        payload = {"dataset_id": dataset_id, "target_column": "cible", **overrides}
        return client.post("/api/training/jobs", headers=headers, json=payload).json()


def test_compare_returns_entries_in_requested_order(client, db_session):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job1 = _create_job(client, headers, dataset["id"])
    job2 = _create_job(client, headers, dataset["id"])
    org_id = db_session.query(TrainingJob).filter(TrainingJob.id == job1["id"]).first().organization_id
    _complete_job(db_session, job1["id"], org_id, algorithm="LightGBM", r2_test=0.90)
    _complete_job(db_session, job2["id"], org_id, algorithm="CatBoost", r2_test=0.85)

    resp = client.get(
        "/api/training/jobs/compare", headers=headers, params=[("job_ids", job2["id"]), ("job_ids", job1["id"])]
    )
    assert resp.status_code == 200
    body = resp.json()
    assert [e["job_id"] for e in body["entries"]] == [job2["id"], job1["id"]]
    assert body["entries"][0]["algorithm"] == "CatBoost"
    assert body["entries"][1]["algorithm"] == "LightGBM"
    assert body["entries"][0]["metrics"]["r2_test"] == 0.85


@patch("api.routers.training.training_queue")
def test_compare_flags_differing_config_and_ignores_equal_fields(mock_queue, client, db_session):
    mock_queue.enqueue.return_value.id = "fake-rq-id"
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job1 = client.post(
        "/api/training/jobs",
        headers=headers,
        json={"dataset_id": dataset["id"], "target_column": "cible", "cv_folds": 4, "seed": 42},
    ).json()
    job2 = client.post(
        "/api/training/jobs",
        headers=headers,
        json={"dataset_id": dataset["id"], "target_column": "cible", "cv_folds": 6, "seed": 42},
    ).json()
    org_id = db_session.query(TrainingJob).filter(TrainingJob.id == job1["id"]).first().organization_id
    _complete_job(db_session, job1["id"], org_id)
    _complete_job(db_session, job2["id"], org_id)

    resp = client.get(
        "/api/training/jobs/compare", headers=headers, params=[("job_ids", job1["id"]), ("job_ids", job2["id"])]
    )
    body = resp.json()
    assert "cv_folds" in body["differing_config_fields"]
    assert "seed" not in body["differing_config_fields"]


@patch("api.routers.training.training_queue")
def test_compare_model_ids_diff_by_set_not_by_order(mock_queue, client, db_session):
    """`model_ids` doit être comparé par ENSEMBLE — le même sous-ensemble de
    modèles choisi dans un ordre différent n'est pas une vraie différence de
    configuration."""
    mock_queue.enqueue.return_value.id = "fake-rq-id"
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job1 = client.post(
        "/api/training/jobs",
        headers=headers,
        json={"dataset_id": dataset["id"], "target_column": "cible", "model_ids": ["lightgbm", "catboost"]},
    ).json()
    job2 = client.post(
        "/api/training/jobs",
        headers=headers,
        json={"dataset_id": dataset["id"], "target_column": "cible", "model_ids": ["catboost", "lightgbm"]},
    ).json()
    org_id = db_session.query(TrainingJob).filter(TrainingJob.id == job1["id"]).first().organization_id
    _complete_job(db_session, job1["id"], org_id)
    _complete_job(db_session, job2["id"], org_id)

    resp = client.get(
        "/api/training/jobs/compare", headers=headers, params=[("job_ids", job1["id"]), ("job_ids", job2["id"])]
    )
    assert "model_ids" not in resp.json()["differing_config_fields"]


def test_compare_requires_at_least_two_jobs(client, db_session):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job1 = _create_job(client, headers, dataset["id"])

    resp = client.get("/api/training/jobs/compare", headers=headers, params=[("job_ids", job1["id"])])
    assert resp.status_code in (400, 422)  # 422 si Query(min_length=2) rejette avant le handler


def test_compare_rejects_unknown_job_id(client, db_session):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job1 = _create_job(client, headers, dataset["id"])

    resp = client.get(
        "/api/training/jobs/compare", headers=headers, params=[("job_ids", job1["id"]), ("job_ids", 999999)]
    )
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "TRAINING_JOB_INTROUVABLE"


def test_compare_isolation_between_organizations(client, db_session):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    dataset_a = _upload_dataset(client, headers_a, "a.csv")
    job_a1 = _create_job(client, headers_a, dataset_a["id"])
    job_a2 = _create_job(client, headers_a, dataset_a["id"])

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    dataset_b = _upload_dataset(client, headers_b, "b.csv")
    job_b1 = _create_job(client, headers_b, dataset_b["id"])

    # L'organisation B tente de comparer un job qui lui appartient avec un
    # job de l'organisation A — traité comme absent (jamais un indice
    # d'existence croisée), pas une fuite d'isolation.
    resp = client.get(
        "/api/training/jobs/compare", headers=headers_b, params=[("job_ids", job_b1["id"]), ("job_ids", job_a1["id"])]
    )
    assert resp.status_code == 404

    # Contrôle : org A peut bien comparer ses deux propres jobs.
    resp_ok = client.get(
        "/api/training/jobs/compare", headers=headers_a, params=[("job_ids", job_a1["id"]), ("job_ids", job_a2["id"])]
    )
    assert resp_ok.status_code == 200
