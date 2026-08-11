"""Tests de workers/training_worker.py (Lot 4c) — orchestration bout en bout
de la feature engineering : transformations amont appliquées avant le split,
config pipeline transmise à train_and_evaluate, spec persistée dans le bundle
et sur MLModel, distinction colonnes saisies (feature_columns_json) / colonnes
dérivées (jamais exposées comme telles).

`run_training_job` ouvre sa propre session DB (`SessionLocal`, voir
`api/core/database.py`) — comme documenté dans `conftest.py`, elle pointe
vers le même fichier SQLite de test que la fixture `db_session` tant que les
écritures sont commit avant l'appel."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from api.core.models import Dataset, MLModel, ModelCandidate, Organization, TrainingJob
import workers.training_worker as training_worker_module
from workers.training_worker import _user_safe_error_message, run_training_job


def _write_temp_csv(df: pd.DataFrame) -> str:
    path = Path(tempfile.gettempdir()) / f"datalab_worker_test_{np.random.default_rng().integers(1_000_000)}.csv"
    df.to_csv(path, index=False)
    return str(path)


def _make_job(db, feature_engineering_spec: dict | None, feature_columns: list[str]) -> TrainingJob:
    org = Organization(name="Bureau test")
    db.add(org)
    db.flush()

    rng = np.random.default_rng(21)
    n = 200
    base = pd.Timestamp("2022-01-01")
    df = pd.DataFrame({
        "date": [(base + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d") for d in rng.integers(0, 500, n)],
        "x": rng.normal(50, 10, n),
    })
    df["cible"] = df["x"] * 2 + rng.normal(0, 3, n)
    csv_path = _write_temp_csv(df)

    dataset = Dataset(
        organization_id=org.id, name="d.csv", file_path=csv_path, file_size_bytes=1,
        status="ready", columns_json=json.dumps([{"name": c, "dtype": str(df[c].dtype)} for c in df.columns]),
    )
    db.add(dataset)
    db.flush()

    job = TrainingJob(
        organization_id=org.id,
        dataset_id=dataset.id,
        task_type="regression",
        target_column="cible",
        feature_columns_json=json.dumps(feature_columns),
        config_json=json.dumps({"optuna_trials": 3, "cv_folds": 3}),
        feature_engineering_json=json.dumps(feature_engineering_spec) if feature_engineering_spec else None,
        status="queued",
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def test_worker_applies_upstream_feature_engineering_and_persists_spec(db_session):
    spec = {"version": 1, "upstream": [{"type": "datetime_decompose", "source_column": "date"}], "pipeline": {}}
    job = _make_job(db_session, spec, feature_columns=["date", "x"])

    run_training_job(job.id)

    db_session.expire_all()
    refreshed = db_session.query(TrainingJob).filter(TrainingJob.id == job.id).first()
    assert refreshed.status == "completed"

    model = db_session.query(MLModel).filter(MLModel.training_job_id == job.id).first()
    assert model is not None

    # Colonnes SAISIES (feature_columns_json) restent les colonnes brutes —
    # jamais les dérivées (précision 3 du cadrage).
    assert json.loads(model.feature_columns_json) == ["date", "x"]

    # La spec est bien persistée côté MLModel (transparence) et dans le bundle
    # joblib (c'est celle-ci qui compte pour l'inférence).
    assert json.loads(model.feature_engineering_json) == spec
    bundle = joblib.load(model.file_path)
    assert bundle["feature_engineering_spec"] == spec

    # Le préprocesseur a bien vu les colonnes dérivées, pas "date" brute.
    feature_names = bundle["feature_names"]
    assert any("date_annee" in name for name in feature_names)
    assert not any(name.startswith("date__") or name == "date" for name in feature_names)


def test_worker_without_feature_engineering_spec_behaves_as_before(db_session):
    job = _make_job(db_session, None, feature_columns=["x"])

    run_training_job(job.id)

    db_session.expire_all()
    refreshed = db_session.query(TrainingJob).filter(TrainingJob.id == job.id).first()
    assert refreshed.status == "completed"

    model = db_session.query(MLModel).filter(MLModel.training_job_id == job.id).first()
    assert model.feature_engineering_json is None
    bundle = joblib.load(model.file_path)
    assert "feature_engineering_spec" not in bundle


# ── Lot D — leaderboard : persistance de TOUS les candidats ─────────────────


def test_worker_persists_all_default_catalog_candidates_not_only_winner(db_session):
    """Avant ce lot, seul le gagnant était persisté (`MLModel`) — le travail
    de comparaison réel du moteur restait invisible. Le worker doit
    maintenant persister une ligne `ModelCandidate` par modèle du
    sous-ensemble par défaut, pas seulement celui retenu."""
    from services.ml_registry import models_for_task

    job = _make_job(db_session, None, feature_columns=["x"])
    run_training_job(job.id)

    db_session.expire_all()
    rows = db_session.query(ModelCandidate).filter(ModelCandidate.training_job_id == job.id).all()
    expected_labels = {spec.label("regression") for spec in models_for_task("regression", "default")}
    assert {r.algorithm for r in rows} == expected_labels
    assert sum(r.is_winner for r in rows) == 1


def test_worker_respects_model_ids_from_config_json(db_session):
    """Mode expert (Lot E2), bout en bout via le worker : `config_json`
    porte `model_ids` comme n'importe quel autre paramètre existant
    (`optuna_trials`, `cv_folds`...) — vérifie que `TrainingConfig(**config)`
    le reçoit bien et que seul le sous-ensemble demandé est comparé."""
    org = Organization(name="Bureau test")
    db_session.add(org)
    db_session.flush()

    rng = np.random.default_rng(11)
    n = 150
    df = pd.DataFrame({"x": rng.normal(50, 10, n)})
    df["cible"] = df["x"] * 2 + rng.normal(0, 3, n)
    csv_path = _write_temp_csv(df)

    dataset = Dataset(
        organization_id=org.id, name="d.csv", file_path=csv_path, file_size_bytes=1,
        status="ready", columns_json=json.dumps([{"name": c, "dtype": str(df[c].dtype)} for c in df.columns]),
    )
    db_session.add(dataset)
    db_session.flush()

    job = TrainingJob(
        organization_id=org.id,
        dataset_id=dataset.id,
        task_type="regression",
        target_column="cible",
        feature_columns_json=json.dumps(["x"]),
        config_json=json.dumps({"optuna_trials": 3, "cv_folds": 3, "model_ids": ["extra_trees"]}),
        status="queued",
    )
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)

    run_training_job(job.id)

    db_session.expire_all()
    refreshed = db_session.query(TrainingJob).filter(TrainingJob.id == job.id).first()
    assert refreshed.status == "completed"

    rows = db_session.query(ModelCandidate).filter(ModelCandidate.training_job_id == job.id).all()
    from services.ml_registry import MODEL_REGISTRY

    assert {r.algorithm for r in rows} == {MODEL_REGISTRY["extra_trees"].label("regression")}


def test_worker_winner_consistent_between_ml_model_and_candidate_row(db_session):
    """Exigence de cohérence du cadrage Lot D : le gagnant dans `MLModel` et
    la ligne `ModelCandidate` avec `is_winner=True` désignent TOUJOURS le
    même modèle (même algorithme, même score de sélection) — les deux sont
    écrits dans la même transaction, à partir des mêmes valeurs (voir
    `services/ml_training.py::train_and_evaluate`), jamais recalculés
    séparément."""
    job = _make_job(db_session, None, feature_columns=["x"])
    run_training_job(job.id)

    db_session.expire_all()
    model = db_session.query(MLModel).filter(MLModel.training_job_id == job.id).first()
    winners = (
        db_session.query(ModelCandidate)
        .filter(ModelCandidate.training_job_id == job.id, ModelCandidate.is_winner.is_(True))
        .all()
    )
    assert len(winners) == 1
    winner = winners[0]
    assert winner.algorithm == model.algorithm
    metrics = json.loads(model.metrics_json)
    assert winner.selection_score == pytest.approx(metrics["cv_score"])


def test_worker_candidates_carry_fold_scores_and_regression_secondary_metric(db_session):
    """Option A (variance inter-folds) + précision 1 (erreur en unité réelle
    pour la régression) du cadrage Lot D, bout en bout via le worker."""
    job = _make_job(db_session, None, feature_columns=["x"])
    run_training_job(job.id)

    db_session.expire_all()
    rows = db_session.query(ModelCandidate).filter(ModelCandidate.training_job_id == job.id).all()
    assert len(rows) > 0
    for row in rows:
        assert row.fold_scores_json is not None
        assert len(json.loads(row.fold_scores_json)) >= 1
        assert row.secondary_metric is not None  # RMSE — régression uniquement, toujours dispo ici
        assert row.secondary_metric_label == "RMSE (validation croisée)"


# ── Volet A — échouer proprement : jamais de trace brute affichée ──────────


def _assert_no_raw_technical_leak(message: str) -> None:
    """Un message d'échec utilisateur ne doit jamais contenir de trace Python
    ni de chemin de fichier interne — voir diagnostic "bad allocation"."""
    lowered = message.lower()
    assert "traceback" not in lowered
    assert ".py" not in lowered
    assert "e:\\" not in lowered and "e:/" not in lowered
    assert "file \"" not in lowered


@pytest.mark.parametrize(
    "exc,expected_substring",
    [
        (MemoryError("Unable to allocate 512. MiB for an array"), "mémoire disponible"),
        (RuntimeError("_catboost.CatBoostError: bad allocation"), "mémoire disponible"),
        (ValueError("Unable to allocate array"), "mémoire disponible"),
        (RuntimeError("colonne cible introuvable dans un contexte inattendu"), "raison technique"),
    ],
)
def test_user_safe_error_message_translates_known_causes(exc, expected_substring):
    message = _user_safe_error_message(exc)
    assert expected_substring in message
    _assert_no_raw_technical_leak(message)


def test_worker_never_leaks_raw_traceback_on_training_failure(db_session, monkeypatch):
    """Bout en bout : une exception technique levée pendant l'entraînement
    (ex. "bad allocation" CatBoost, chemin de fichier réel dans le message
    d'origine) ne doit jamais atteindre `job.error_message` telle quelle —
    seul un message traduit, sûr, doit être persisté. Le détail complet
    reste dans les logs serveur (non vérifié ici, hors périmètre du test)."""
    raw_exc = RuntimeError(
        'File "E:\\mlops\\app-analyse\\backend\\.venv\\Lib\\site-packages\\catboost\\core.py", line 5827, '
        "in fit\n_catboost.CatBoostError: bad allocation"
    )

    def _raise(*args, **kwargs):
        raise raw_exc

    monkeypatch.setattr(training_worker_module, "train_and_evaluate", _raise)

    job = _make_job(db_session, None, feature_columns=["x"])
    run_training_job(job.id)

    db_session.expire_all()
    refreshed = db_session.query(TrainingJob).filter(TrainingJob.id == job.id).first()
    assert refreshed.status == "failed"
    assert "mémoire disponible" in refreshed.error_message
    _assert_no_raw_technical_leak(refreshed.error_message)


def test_worker_data_leakage_message_unaffected_by_safe_translation(db_session, monkeypatch):
    """`DataLeakageError` porte déjà un message français auto-rédigé, sûr —
    ce chemin d'exception reste inchangé par la traduction générique du
    Volet A (pas de sur-correction d'un cas déjà propre)."""
    from services.ml_preprocessing import DataLeakageError

    def _raise(*args, **kwargs):
        raise DataLeakageError("Fuite détectée entre train et test sur 3 groupe(s) de la colonne 'groupe'")

    monkeypatch.setattr(training_worker_module, "train_and_evaluate", _raise)

    job = _make_job(db_session, None, feature_columns=["x"])
    run_training_job(job.id)

    db_session.expire_all()
    refreshed = db_session.query(TrainingJob).filter(TrainingJob.id == job.id).first()
    assert refreshed.status == "failed"
    assert refreshed.error_message == "Fuite détectée entre train et test sur 3 groupe(s) de la colonne 'groupe'"
