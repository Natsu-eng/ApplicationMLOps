"""Tests du router training (Lot 3) — validation et isolation.

La file RQ est mockée : ces tests valident la logique de l'endpoint (pas
l'exécution réelle du worker, qui exige Redis — couverte séparément et sans
dépendance externe par test_ml_training.py, qui appelle directement
`train_and_evaluate`)."""
from __future__ import annotations

import io
import json
from unittest.mock import patch

from api.core.config import get_settings
from api.core.models import MLModel, ModelCandidate, TrainingJob
from api.routers.training import _headline_metric
from services.model_versioning import next_version


def _register(client, email="owner@bureau.fr", org="Bureau"):
    resp = client.post(
        "/api/auth/register",
        json={"email": email, "nom": "Owner", "password": "motdepasse123", "organization_name": org},
    ).json()
    return {"Authorization": f"Bearer {resp['access_token']}"}


def _upload_dataset(client, headers):
    # 50 lignes, cible à valeurs toutes distinctes : au-delà du seuil de
    # cardinalité de detect_task_type (20), pour être détecté sans ambiguïté
    # comme régression (voir services/ml_task.py).
    rows = "\n".join(f"{i},{i * 2},{i * 3}" for i in range(50))
    content = f"x1,x2,cible\n{rows}\n".encode()
    resp = client.post("/api/datasets", headers=headers, files={"file": ("d.csv", io.BytesIO(content), "text/csv")})
    return resp.json()


@patch("api.routers.training.training_queue")
def test_create_job_enqueues_and_returns_summary(mock_queue, client):
    mock_queue.enqueue.return_value.id = "fake-rq-id"
    headers = _register(client)
    dataset = _upload_dataset(client, headers)

    resp = client.post(
        "/api/training/jobs", headers=headers, json={"dataset_id": dataset["id"], "target_column": "cible"}
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
        "/api/training/jobs",
        headers=headers,
        json={"dataset_id": dataset["id"], "target_column": "colonne_inexistante"},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "COLONNE_CIBLE_INTROUVABLE"


@patch("api.routers.training.training_queue")
def test_create_job_rejects_unready_dataset(mock_queue, client):
    headers = _register(client)
    resp = client.post(
        "/api/training/jobs", headers=headers, json={"dataset_id": 999999, "target_column": "cible"}
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
        "/api/training/jobs", headers=headers_a, json={"dataset_id": dataset["id"], "target_column": "cible"}
    ).json()

    assert client.get("/api/training/jobs", headers=headers_b).json() == []
    assert client.get(f"/api/training/jobs/{job['id']}", headers=headers_b).status_code == 404


@patch("api.routers.training.training_queue")
def test_delete_removes_job_from_history(mock_queue, client):
    mock_queue.enqueue.return_value.id = "fake-rq-id"
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job = client.post(
        "/api/training/jobs", headers=headers, json={"dataset_id": dataset["id"], "target_column": "cible"}
    ).json()

    assert client.delete(f"/api/training/jobs/{job['id']}", headers=headers).status_code == 204
    assert client.get(f"/api/training/jobs/{job['id']}", headers=headers).status_code == 404
    assert client.get("/api/training/jobs", headers=headers).json() == []


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
        "/api/training/jobs", headers=headers, json={"dataset_id": dataset["id"], "target_column": "cible"}
    ).json()["id"]

    job = db_session.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    job.status = "completed"
    db_session.add(
        MLModel(
            organization_id=job.organization_id,
            training_job_id=job.id,
            dataset_id=job.dataset_id,
            version=next_version(db_session, job.organization_id, job.dataset_id, job.target_column),
            algorithm="LightGBM",
            task_type="regression",
            target_column="cible",
            feature_columns_json=json.dumps(["x1", "x2"]),
            file_path="unused.joblib",
            metrics_json=json.dumps({}),
        )
    )
    db_session.commit()

    assert client.delete(f"/api/training/jobs/{job_id}", headers=headers).status_code == 204
    assert client.get(f"/api/training/jobs/{job_id}", headers=headers).status_code == 404


@patch("api.routers.training.training_queue")
def test_delete_rejects_cross_organization(mock_queue, client):
    mock_queue.enqueue.return_value.id = "fake-rq-id"
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    dataset = _upload_dataset(client, headers_a)
    job = client.post(
        "/api/training/jobs", headers=headers_a, json={"dataset_id": dataset["id"], "target_column": "cible"}
    ).json()

    resp = client.delete(f"/api/training/jobs/{job['id']}", headers=headers_b)
    assert resp.status_code == 404


# ── Lot 7, §J.2 — annulation (garde une trace, contrairement à la suppression) ─


@patch("api.routers.training.training_queue")
def test_cancel_queued_job_marks_it_cancelled_and_keeps_history(mock_queue, client):
    mock_queue.enqueue.return_value.id = "fake-rq-id"
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job = client.post(
        "/api/training/jobs", headers=headers, json={"dataset_id": dataset["id"], "target_column": "cible"}
    ).json()

    resp = client.post(f"/api/training/jobs/{job['id']}/cancel", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"
    assert client.get(f"/api/training/jobs/{job['id']}", headers=headers).json()["status"] == "cancelled"


@patch("api.routers.training.training_queue")
def test_cancel_rejects_already_completed_job(mock_queue, client, db_session):
    mock_queue.enqueue.return_value.id = "fake-rq-id"
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job_id = client.post(
        "/api/training/jobs", headers=headers, json={"dataset_id": dataset["id"], "target_column": "cible"}
    ).json()["id"]

    job = db_session.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    job.status = "completed"
    db_session.commit()

    resp = client.post(f"/api/training/jobs/{job_id}/cancel", headers=headers)
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "JOB_NON_ANNULABLE"


@patch("api.routers.training.training_queue")
def test_cancel_404_for_other_organization(mock_queue, client):
    mock_queue.enqueue.return_value.id = "fake-rq-id"
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    dataset = _upload_dataset(client, headers_a)
    job = client.post(
        "/api/training/jobs", headers=headers_a, json={"dataset_id": dataset["id"], "target_column": "cible"}
    ).json()

    resp = client.post(f"/api/training/jobs/{job['id']}/cancel", headers=headers_b)
    assert resp.status_code == 404
    # toujours là côté organisation A — la tentative de B n'a rien supprimé
    assert client.get(f"/api/training/jobs/{job['id']}", headers=headers_a).status_code == 200


# ── Lot E2 — mode guidé/expert : catalogue de modèles + manettes ───────────


@patch("api.routers.training.training_queue")
def test_model_endpoint_exposes_global_explainability_fields(mock_queue, client, db_session):
    mock_queue.enqueue.return_value.id = "fake-rq-id"
    """GET /training/jobs/{id}/model (Lot Explicabilité globale) : les 4
    nouveaux champs (beeswarm/permutation/calibration/learning_curve)
    traversent bien json.dumps (worker) → colonnes DB → json.loads (API) →
    réponse HTTP, round-trip jamais testé avant ce lot."""
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job_id = client.post(
        "/api/training/jobs", headers=headers, json={"dataset_id": dataset["id"], "target_column": "cible"}
    ).json()["id"]

    job = db_session.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    job.status = "completed"
    db_session.add(
        MLModel(
            organization_id=job.organization_id,
            training_job_id=job.id,
            dataset_id=job.dataset_id,
            version=next_version(db_session, job.organization_id, job.dataset_id, job.target_column),
            algorithm="LightGBM",
            task_type="regression",
            target_column="cible",
            feature_columns_json=json.dumps(["x1", "x2"]),
            file_path="unused.joblib",
            metrics_json=json.dumps({}),
            shap_beeswarm_json=json.dumps({"global": [{"feature": "x1", "feature_value": 1.0, "shap_value": 0.5}]}),
            permutation_importance_json=json.dumps([{"feature": "x1", "importance_mean": 0.1, "importance_std": 0.01}]),
            calibration_json=None,  # régression : non applicable
            learning_curve_json=json.dumps({"train_sizes": [10, 20], "metric_label": "R²"}),
        )
    )
    db_session.commit()

    resp = client.get(f"/api/training/jobs/{job_id}/model", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["shap_beeswarm"] == {"global": [{"feature": "x1", "feature_value": 1.0, "shap_value": 0.5}]}
    assert body["permutation_importance"] == [{"feature": "x1", "importance_mean": 0.1, "importance_std": 0.01}]
    assert body["calibration"] is None
    assert body["learning_curve"]["train_sizes"] == [10, 20]


@patch("api.routers.training.training_queue")
def test_model_endpoint_degrades_cleanly_for_pre_lot_jobs(mock_queue, client, db_session):
    mock_queue.enqueue.return_value.id = "fake-rq-id"
    """Rétrocompatibilité : un modèle entraîné avant ce lot n'a aucune de ces
    4 colonnes (NULL) — l'API doit répondre avec des valeurs par défaut
    ([]/{}/None), jamais une 500 ou un champ manquant côté frontend."""
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    job_id = client.post(
        "/api/training/jobs", headers=headers, json={"dataset_id": dataset["id"], "target_column": "cible"}
    ).json()["id"]

    job = db_session.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    job.status = "completed"
    db_session.add(
        MLModel(
            organization_id=job.organization_id,
            training_job_id=job.id,
            dataset_id=job.dataset_id,
            version=next_version(db_session, job.organization_id, job.dataset_id, job.target_column),
            algorithm="LightGBM",
            task_type="regression",
            target_column="cible",
            feature_columns_json=json.dumps(["x1", "x2"]),
            file_path="unused.joblib",
            metrics_json=json.dumps({}),
            # shap_beeswarm_json/permutation_importance_json/calibration_json/
            # learning_curve_json omis — NULL, comme un job pré-lot.
        )
    )
    db_session.commit()

    resp = client.get(f"/api/training/jobs/{job_id}/model", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["shap_beeswarm"] == {}
    assert body["permutation_importance"] == []
    assert body["calibration"] is None
    assert body["learning_curve"] is None


def test_models_catalog_lists_all_nine_registry_entries(client):
    headers = _register(client)
    resp = client.get("/api/training/models-catalog", headers=headers)
    assert resp.status_code == 200
    models = resp.json()["models"]
    assert len(models) == 9

    default_ids = {m["id"] for m in models if m["is_default"]}
    assert default_ids == {"lightgbm", "xgboost", "catboost", "random_forest"}

    slow_ids = {m["id"] for m in models if m["slow"]}
    assert slow_ids == {"svm", "knn"}

    naive_bayes = next(m for m in models if m["id"] == "naive_bayes")
    assert naive_bayes["supported_tasks"] == ["classification"]


@patch("api.routers.training.training_queue")
def test_create_job_without_expert_fields_uses_unchanged_server_defaults(mock_queue, client, db_session):
    """Non-régression (Lot E2) : mode expert OFF (aucun champ expert envoyé)
    doit produire exactement le `config_json` d'avant ce lot — mêmes défauts
    serveur, `model_ids` absent (sous-ensemble par défaut, inchangé)."""
    mock_queue.enqueue.return_value.id = "fake-rq-id"
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    settings = get_settings()

    job = client.post(
        "/api/training/jobs", headers=headers, json={"dataset_id": dataset["id"], "target_column": "cible"}
    ).json()

    job_row = db_session.query(TrainingJob).filter(TrainingJob.id == job["id"]).first()
    config = json.loads(job_row.config_json)
    assert config["seed"] == settings.model_seed
    assert config["cqr_alpha"] == settings.cqr_alpha
    assert config["cv_folds"] == settings.cv_folds_default
    assert config["optuna_trials"] == settings.optuna_trials_default
    assert config["model_ids"] is None


@patch("api.routers.training.training_queue")
def test_create_job_with_expert_fields_threads_them_into_config(mock_queue, client, db_session):
    mock_queue.enqueue.return_value.id = "fake-rq-id"
    headers = _register(client)
    dataset = _upload_dataset(client, headers)

    job = client.post(
        "/api/training/jobs",
        headers=headers,
        json={
            "dataset_id": dataset["id"],
            "target_column": "cible",
            "seed": 7,
            "cqr_alpha": 0.1,
            "cv_folds": 6,
            "model_ids": ["lightgbm", "extra_trees"],
        },
    ).json()

    job_row = db_session.query(TrainingJob).filter(TrainingJob.id == job["id"]).first()
    config = json.loads(job_row.config_json)
    assert config["seed"] == 7
    assert config["cqr_alpha"] == 0.1
    assert config["cv_folds"] == 6
    assert config["model_ids"] == ["lightgbm", "extra_trees"]


@patch("api.routers.training.training_queue")
def test_create_job_rejects_unknown_model_id(mock_queue, client):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)

    resp = client.post(
        "/api/training/jobs",
        headers=headers,
        json={"dataset_id": dataset["id"], "target_column": "cible", "model_ids": ["modele_inexistant"]},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "MODELES_INCONNUS"


@patch("api.routers.training.training_queue")
def test_create_job_rejects_model_ids_incompatible_with_detected_task(mock_queue, client):
    """Le dataset de `_upload_dataset` est détecté en régression — Naive
    Bayes (classification uniquement, voir `ml_registry.MODEL_REGISTRY`)
    n'est donc compatible avec aucune tâche possible ici."""
    headers = _register(client)
    dataset = _upload_dataset(client, headers)

    resp = client.post(
        "/api/training/jobs",
        headers=headers,
        json={"dataset_id": dataset["id"], "target_column": "cible", "model_ids": ["naive_bayes"]},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "AUCUN_MODELE_COMPATIBLE"


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
            dataset_id=job.dataset_id,
            version=next_version(db_session, org_id, job.dataset_id, job.target_column),
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
        "/api/training/jobs", headers=headers, json={"dataset_id": dataset["id"], "target_column": "cible"}
    ).json()

    job_row = db_session.query(TrainingJob).filter(TrainingJob.id == job["id"]).first()
    _complete_job_with_model_and_candidates(db_session, job["id"], job_row.organization_id)

    resp = client.get(f"/api/training/jobs/{job['id']}/candidates", headers=headers)
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
        "/api/training/jobs", headers=headers, json={"dataset_id": dataset["id"], "target_column": "cible"}
    ).json()

    job_row = db_session.query(TrainingJob).filter(TrainingJob.id == job["id"]).first()
    _complete_job_with_model_and_candidates(db_session, job["id"], job_row.organization_id, add_candidates=False)

    resp = client.get(f"/api/training/jobs/{job['id']}/candidates", headers=headers)
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
        "/api/training/jobs", headers=headers_a, json={"dataset_id": dataset["id"], "target_column": "cible"}
    ).json()

    job_row = db_session.query(TrainingJob).filter(TrainingJob.id == job["id"]).first()
    _complete_job_with_model_and_candidates(db_session, job["id"], job_row.organization_id)

    assert client.get(f"/api/training/jobs/{job['id']}/candidates", headers=headers_b).status_code == 404
    assert client.get(f"/api/training/jobs/{job['id']}/candidates", headers=headers_a).status_code == 200


# ── Lot 7, §J.1 — estimation de durée avant lancement ───────────────────────


def test_estimate_duration_degrades_honestly_without_history(client):
    headers = _register(client)
    dataset = _upload_dataset(client, headers)
    resp = client.get(f"/api/training/estimate-duration?dataset_id={dataset['id']}", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["estimated_seconds"] is None


def test_estimate_duration_404_for_missing_dataset(client):
    headers = _register(client)
    resp = client.get("/api/training/estimate-duration?dataset_id=999999", headers=headers)
    assert resp.status_code == 404


def test_estimate_duration_isolated_between_organizations(client):
    headers_a = _register(client, "a@bureau-a.fr", "Bureau A")
    dataset_a = _upload_dataset(client, headers_a)
    headers_b = _register(client, "b@bureau-b.fr", "Bureau B")
    resp = client.get(f"/api/training/estimate-duration?dataset_id={dataset_a['id']}", headers=headers_b)
    assert resp.status_code == 404
