"""GET /dashboard/summary (Lot 4, correctif I3, AUDIT_DATALAB_2026-08-16.md
§C.2.4) — remplace les 8 appels de liste complets faits par `Dashboard.tsx`
au montage par un seul aller-retour agrégé."""
from __future__ import annotations

import io
from unittest.mock import patch


def _register(client, email="owner@bureau.fr", org="Bureau"):
    resp = client.post(
        "/api/auth/register",
        json={"email": email, "nom": "Owner", "password": "motdepasse123", "organization_name": org},
    ).json()
    return {"Authorization": f"Bearer {resp['access_token']}"}


def _upload_dataset(client, headers, name="d.csv"):
    content = b"x1,x2,cible\n" + b"\n".join(f"{i},{i * 2},{i % 2}".encode() for i in range(20))
    resp = client.post("/api/datasets", headers=headers, files={"file": (name, io.BytesIO(content), "text/csv")})
    return resp.json()


def test_summary_reflects_empty_organization(client):
    headers = _register(client)
    resp = client.get("/api/dashboard/summary", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["members_count"] == 1  # le owner lui-même
    assert body["datasets_count"] == 0
    assert body["recent_datasets"] == []
    assert body["supervised_count"] == 0
    assert body["unsupervised_count"] == 0
    assert body["vision_count"] == 0
    assert body["active_count"] == 0
    assert body["recent_supervised"] == []


def test_summary_counts_datasets_and_recent_list(client):
    headers = _register(client)
    for i in range(3):
        _upload_dataset(client, headers, name=f"d{i}.csv")

    body = client.get("/api/dashboard/summary", headers=headers).json()
    assert body["datasets_count"] == 3
    assert len(body["recent_datasets"]) == 3
    # Le plus récent en tête (created_at desc), même ordre que GET /datasets.
    assert body["recent_datasets"][0]["name"] == "d2.csv"


def test_summary_counts_jobs_per_pillar(client):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)

    with patch("domains.training.router.training_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        client.post("/api/training/jobs", headers=headers, json={"dataset_id": dataset["id"], "target_column": "cible"})

    with patch("domains.clustering.router.analysis_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        client.post(
            "/api/clustering/jobs", headers=headers, json={"dataset_id": dataset["id"], "feature_columns": ["x1", "x2"]}
        )
        client.post(
            "/api/clustering/jobs", headers=headers, json={"dataset_id": dataset["id"], "feature_columns": ["x1", "x2"]}
        )

    body = client.get("/api/dashboard/summary", headers=headers).json()
    assert body["supervised_count"] == 1
    assert body["unsupervised_count"] == 2  # clustering seul ici, mais compté dans le total non supervisé
    assert body["vision_count"] == 0
    assert body["active_count"] == 3  # les 3 jobs sont "queued" (worker mocké, jamais exécuté)
    assert len(body["recent_supervised"]) == 1
    assert len(body["recent_clustering"]) == 2


def test_summary_isolated_between_organizations(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    _upload_dataset(client, headers_a, "a.csv")

    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")

    body_b = client.get("/api/dashboard/summary", headers=headers_b).json()
    assert body_b["datasets_count"] == 0
    assert body_b["recent_datasets"] == []
    assert body_b["members_count"] == 1  # uniquement le owner de B, jamais celui de A


def test_summary_recent_supervised_matches_list_training_jobs_shape(client):
    """Même schéma que GET /training/jobs (dataset_name, headline_metric,
    etc.) — réutilise `to_summary`, jamais une forme dupliquée."""
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    with patch("domains.training.router.training_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        client.post("/api/training/jobs", headers=headers, json={"dataset_id": dataset["id"], "target_column": "cible"})

    from_list = client.get("/api/training/jobs", headers=headers).json()[0]
    from_summary = client.get("/api/dashboard/summary", headers=headers).json()["recent_supervised"][0]
    assert from_list == from_summary


# ── Fiabilité des modèles actifs (retour utilisateur : maquette de refonte
# — "Fiabilité moyenne : part des modèles dont le verdict est utilisable",
# calculée sur les modèles ML tabulaire en staging/production) ────────────


def _insert_model(db_session, organization_id: int, stage, delta_r2: float = 0.02):
    """Modèle inséré directement en base — pas d'entraînement réel
    nécessaire, seul le verdict calculé depuis metrics_json compte ici
    (même principe que test_model_verdict.py, mais via to_model_detail())."""
    import json

    from api.core.models import Dataset, MLModel, TrainingJob
    from domains.training.services.versioning import next_version

    dataset = Dataset(
        organization_id=organization_id, name="d.csv", file_path="unused", file_size_bytes=1, status="ready"
    )
    db_session.add(dataset)
    db_session.flush()
    job = TrainingJob(
        organization_id=organization_id,
        dataset_id=dataset.id,
        task_type="regression",
        target_column="cible",
        feature_columns_json=json.dumps(["x1", "x2"]),
        config_json=json.dumps({}),
        status="completed",
    )
    db_session.add(job)
    db_session.flush()
    model = MLModel(
        organization_id=organization_id,
        training_job_id=job.id,
        dataset_id=dataset.id,
        version=next_version(db_session, organization_id, dataset.id, job.target_column),
        algorithm="LightGBM",
        task_type="regression",
        target_column="cible",
        feature_columns_json=json.dumps(["x1", "x2"]),
        feature_schema_json=json.dumps([{"name": "x1", "dtype": "float64"}, {"name": "x2", "dtype": "float64"}]),
        file_path="unused",
        metrics_json=json.dumps({"r2_train": 0.90 + delta_r2, "r2_test": 0.90, "delta_r2": delta_r2}),
        evaluation_json=json.dumps({}),
        stage=stage,
    )
    db_session.add(model)
    db_session.commit()
    return model


def test_reliability_is_none_without_any_staged_or_production_model(client):
    headers = _register(client)
    resp = client.get("/api/dashboard/summary", headers=headers)
    assert resp.json()["active_models_reliability_pct"] is None


def test_reliability_is_full_when_every_active_model_has_no_critical_claim(client, db_session):
    headers = _register(client)
    org_id = client.get("/api/auth/me", headers=headers).json()["organization_id"]
    _insert_model(db_session, org_id, "production", delta_r2=0.01)  # pas de surapprentissage marqué
    _insert_model(db_session, org_id, "staging", delta_r2=0.02)

    resp = client.get("/api/dashboard/summary", headers=headers)
    assert resp.json()["active_models_reliability_pct"] == 1.0


def test_reliability_drops_when_an_active_model_has_a_critical_claim(client, db_session):
    headers = _register(client)
    org_id = client.get("/api/auth/me", headers=headers).json()["organization_id"]
    _insert_model(db_session, org_id, "production", delta_r2=0.01)  # sain
    _insert_model(db_session, org_id, "staging", delta_r2=0.30)  # surapprentissage marqué -> critique

    resp = client.get("/api/dashboard/summary", headers=headers)
    assert resp.json()["active_models_reliability_pct"] == 0.5


def test_reliability_ignores_models_without_a_stage(client, db_session):
    """Un modèle jamais promu (`stage` absent/"none") ne compte pas dans le
    dénominateur — sinon le score refléterait tout l'historique
    d'entraînement, pas ce qui est réellement actif."""
    headers = _register(client)
    org_id = client.get("/api/auth/me", headers=headers).json()["organization_id"]
    _insert_model(db_session, org_id, "production", delta_r2=0.01)
    _insert_model(db_session, org_id, None, delta_r2=0.30)  # jamais promu, ignoré malgré le surapprentissage

    resp = client.get("/api/dashboard/summary", headers=headers)
    assert resp.json()["active_models_reliability_pct"] == 1.0
