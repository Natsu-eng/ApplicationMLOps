"""Tests du router training (Lot 3) — validation et isolation.

La file RQ est mockée : ces tests valident la logique de l'endpoint (pas
l'exécution réelle du worker, qui exige Redis — couverte séparément et sans
dépendance externe par test_ml_training.py, qui appelle directement
`train_and_evaluate`)."""
from __future__ import annotations

import io
import json
from unittest.mock import patch

from api.core.models import MLModel, ModelCandidate, TrainingJob
from api.routers.training import _headline_metric


def _register(client, email="owner@bureau.fr", org="Bureau"):
    resp = client.post(
        "/auth/register",
        json={"email": email, "nom": "Owner", "password": "motdepasse123", "organization_name": org},
    ).json()
    return {"Authorization": f"Bearer {resp['access_token']}"}


def _upload_dataset(client, headers):
    # 50 lignes, cible à valeurs toutes distinctes : au-delà du seuil de
    # cardinalité de detect_task_type (20), pour être détecté sans ambiguïté
    # comme régression (voir services/ml_task.py).
    rows = "\n".join(f"{i},{i * 2},{i * 3}" for i in range(50))
    content = f"x1,x2,cible\n{rows}\n".encode()
    resp = client.post("/datasets", headers=headers, files={"file": ("d.csv", io.BytesIO(content), "text/csv")})
    return resp.json()


@patch("api.routers.training.training_queue")
def test_create_job_enqueues_and_returns_summary(mock_queue, client):
    mock_queue.enqueue.return_value.id = "fake-rq-id"
    headers = _register(client)
    dataset = _upload_dataset(client, headers)

    resp = client.post(
        "/training/jobs", headers=headers, json={"dataset_id": dataset["id"], "target_column": "cible"}
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "queued"
    assert body["task_type"] == "regression"
    mock_queue.enqueue.assert_called_once()


@patch("api.routers.training.training_queue")
def test_create_job_rejects_unknown_target_column(mock_queue, client):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)

    resp = client.post(
        "/training/jobs",
        headers=headers,
        json={"dataset_id": dataset["id"], "target_column": "colonne_inexistante"},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "COLONNE_CIBLE_INTROUVABLE"


@patch("api.routers.training.training_queue")
def test_create_job_rejects_unready_dataset(mock_queue, client):
    headers = _register(client)
    resp = client.post(
        "/training/jobs", headers=headers, json={"dataset_id": 999999, "target_column": "cible"}
    )
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "DATASET_INTROUVABLE"


@patch("api.routers.training.training_queue")
def test_training_job_isolation_between_organizations(mock_queue, client):
    mock_queue.enqueue.return_value.id = "fake-rq-id"
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    dataset = _upload_dataset(client, headers_a)

    job = client.post(
        "/training/jobs", headers=headers_a, json={"dataset_id": dataset["id"], "target_column": "cible"}
    ).json()

    assert client.get("/training/jobs", headers=headers_b).json() == []
    assert client.get(f"/training/jobs/{job['id']}", headers=headers_b).status_code == 404


@patch("api.routers.training.training_queue")
def test_delete_removes_job_from_history(mock_queue, client):
    mock_queue.enqueue.return_value.id = "fake-rq-id"
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job = client.post(
        "/training/jobs", headers=headers, json={"dataset_id": dataset["id"], "target_column": "cible"}
    ).json()

    assert client.delete(f"/training/jobs/{job['id']}", headers=headers).status_code == 204
    assert client.get(f"/training/jobs/{job['id']}", headers=headers).status_code == 404
    assert client.get("/training/jobs", headers=headers).json() == []


@patch("api.routers.training.training_queue")
def test_delete_completed_job_with_model(mock_queue, client, db_session):
    """Reproduit un bug réel trouvé en usage (Postgres) : la relation
    `TrainingJob.model` sans `passive_deletes=True` tentait de mettre à NULL
    `ml_models.training_job_id` (colonne NOT NULL) avant la suppression du
    job, au lieu de laisser le `ON DELETE CASCADE` de la contrainte FK s'en
    charger côté base — supprimer n'importe quel entraînement déjà terminé
    (donc avec un modèle associé) échouait systématiquement en 500. Le test
    précédent (`test_delete_removes_job_from_history`) ne le couvrait pas
    car il supprime un job juste après création, avant qu'un modèle lui
    soit associé. Voir `api/core/models.py::TrainingJob.model`."""
    mock_queue.enqueue.return_value.id = "fake-rq-id"
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job_id = client.post(
        "/training/jobs", headers=headers, json={"dataset_id": dataset["id"], "target_column": "cible"}
    ).json()["id"]

    job = db_session.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    job.status = "completed"
    db_session.add(
        MLModel(
            organization_id=job.organization_id,
            training_job_id=job.id,
            algorithm="LightGBM",
            task_type="regression",
            target_column="cible",
            feature_columns_json=json.dumps(["x1", "x2"]),
            file_path="unused.joblib",
            metrics_json=json.dumps({}),
        )
    )
    db_session.commit()

    assert client.delete(f"/training/jobs/{job_id}", headers=headers).status_code == 204
    assert client.get(f"/training/jobs/{job_id}", headers=headers).status_code == 404


@patch("api.routers.training.training_queue")
def test_delete_rejects_cross_organization(mock_queue, client):
    mock_queue.enqueue.return_value.id = "fake-rq-id"
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    dataset = _upload_dataset(client, headers_a)
    job = client.post(
        "/training/jobs", headers=headers_a, json={"dataset_id": dataset["id"], "target_column": "cible"}
    ).json()

    resp = client.delete(f"/training/jobs/{job['id']}", headers=headers_b)
    assert resp.status_code == 404
    # toujours là côté organisation A — la tentative de B n'a rien supprimé
    assert client.get(f"/training/jobs/{job['id']}", headers=headers_a).status_code == 200


# ── Lot D — carte d'historique : score de sélection, pas l'accuracy brute ──


def test_headline_metric_uses_selection_score_not_accuracy_for_classification():
    """Bug trouvé lors de l'audit leaderboard (Lot D) : sur un dataset
    déséquilibré, l'accuracy peut afficher un score flatteur alors que le
    modèle rate systématiquement la classe rare. La carte d'historique doit
    afficher `cv_score` (le score qui a réellement départagé les candidats,
    voir `_classification_selection_score`), jamais `accuracy`."""
    metrics = {"accuracy": 0.95, "cv_score": 0.61, "roc_auc": 0.60}
    headline = _headline_metric("classification", metrics)
    assert headline == {"name": "cv_score", "value": 0.61}


def test_headline_metric_regression_unchanged():
    """Fix isolé à la classification (précision explicite du cadrage) — la
    régression garde r2_test, déjà la bonne métrique (pas de piège
    d'accuracy en régression)."""
    metrics = {"r2_test": 0.87, "cv_score": 0.85}
    headline = _headline_metric("regression", metrics)
    assert headline == {"name": "r2_test", "value": 0.87}


# ── Lot D — leaderboard : endpoint GET /jobs/{id}/candidates ───────────────


def _complete_job_with_model_and_candidates(db_session, job_id, org_id, add_candidates=True):
    job = db_session.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    job.status = "completed"
    db_session.add(
        MLModel(
            organization_id=org_id,
            training_job_id=job.id,
            algorithm="LightGBM",
            task_type="regression",
            target_column="cible",
            feature_columns_json=json.dumps(["x1", "x2"]),
            file_path="unused.joblib",
            metrics_json=json.dumps({"r2_test": 0.9, "cv_score": 0.88}),
        )
    )
    if add_candidates:
        db_session.add(ModelCandidate(
            organization_id=org_id, training_job_id=job.id, algorithm="LightGBM", family="arbre_ensemble",
            selection_score=0.88, is_winner=True, rank=1,
            fold_scores_json=json.dumps([0.85, 0.9, 0.89]),
            secondary_metric=1.2, secondary_metric_label="RMSE (validation croisée)",
        ))
        db_session.add(ModelCandidate(
            organization_id=org_id, training_job_id=job.id, algorithm="CatBoost", family="arbre_ensemble",
            selection_score=0.80, is_winner=False, rank=2,
            fold_scores_json=json.dumps([0.78, 0.81, 0.81]),
            secondary_metric=1.5, secondary_metric_label="RMSE (validation croisée)",
        ))
    db_session.commit()
    return job


@patch("api.routers.training.training_queue")
def test_candidates_endpoint_returns_leaderboard_sorted_by_rank(mock_queue, client, db_session):
    mock_queue.enqueue.return_value.id = "fake-rq-id"
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job = client.post(
        "/training/jobs", headers=headers, json={"dataset_id": dataset["id"], "target_column": "cible"}
    ).json()

    job_row = db_session.query(TrainingJob).filter(TrainingJob.id == job["id"]).first()
    _complete_job_with_model_and_candidates(db_session, job["id"], job_row.organization_id)

    resp = client.get(f"/training/jobs/{job['id']}/candidates", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["selection_metric_label"] == "R² (validation croisée)"
    assert [c["algorithm"] for c in body["candidates"]] == ["LightGBM", "CatBoost"]
    assert body["candidates"][0]["is_winner"] is True
    assert body["candidates"][1]["is_winner"] is False
    assert body["candidates"][0]["fold_scores"] == [0.85, 0.9, 0.89]
    assert body["candidates"][0]["secondary_metric_label"] == "RMSE (validation croisée)"


@patch("api.routers.training.training_queue")
def test_candidates_endpoint_backward_compatible_for_pre_lot_jobs(mock_queue, client, db_session):
    """Rétrocompatibilité (Lot D) : un job terminé avant ce lot n'a aucune
    ligne `ModelCandidate` — l'endpoint doit renvoyer une liste vide, jamais
    une erreur, pour que le frontend affiche proprement le seul gagnant déjà
    disponible via `GET /jobs/{id}/model`."""
    mock_queue.enqueue.return_value.id = "fake-rq-id"
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job = client.post(
        "/training/jobs", headers=headers, json={"dataset_id": dataset["id"], "target_column": "cible"}
    ).json()

    job_row = db_session.query(TrainingJob).filter(TrainingJob.id == job["id"]).first()
    _complete_job_with_model_and_candidates(db_session, job["id"], job_row.organization_id, add_candidates=False)

    resp = client.get(f"/training/jobs/{job['id']}/candidates", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["candidates"] == []
    assert body["selection_metric_label"] == "R² (validation croisée)"


@patch("api.routers.training.training_queue")
def test_candidates_endpoint_isolated_between_organizations(mock_queue, client, db_session):
    mock_queue.enqueue.return_value.id = "fake-rq-id"
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    dataset = _upload_dataset(client, headers_a)
    job = client.post(
        "/training/jobs", headers=headers_a, json={"dataset_id": dataset["id"], "target_column": "cible"}
    ).json()

    job_row = db_session.query(TrainingJob).filter(TrainingJob.id == job["id"]).first()
    _complete_job_with_model_and_candidates(db_session, job["id"], job_row.organization_id)

    assert client.get(f"/training/jobs/{job['id']}/candidates", headers=headers_b).status_code == 404
    assert client.get(f"/training/jobs/{job['id']}/candidates", headers=headers_a).status_code == 200
