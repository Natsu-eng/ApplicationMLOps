"""Tests de services/duration_estimate.py (Lot 7, §J.1) — estimation de
durée dérivée de l'historique réel, jamais d'une constante inventée."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from api.core.models import Dataset, Organization, TrainingJob
from domains.training.services.duration_estimate import MIN_COMPLETED_JOBS_FOR_ESTIMATE, estimate_training_duration


def _make_org(db) -> int:
    org = Organization(name="Bureau test")
    db.add(org)
    db.flush()
    return org.id


def _make_completed_job(db, organization_id: int, n_rows: int, duration_seconds: float, **config_overrides):
    dataset = Dataset(
        organization_id=organization_id,
        name="d.csv",
        file_path="unused",
        file_size_bytes=1,
        status="ready",
        row_count=n_rows,
    )
    db.add(dataset)
    db.flush()

    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    config = {"optuna_trials": 20, "cv_folds": 4, **config_overrides}
    job = TrainingJob(
        organization_id=organization_id,
        dataset_id=dataset.id,
        task_type="regression",
        target_column="cible",
        feature_columns_json=json.dumps(["x1"]),
        config_json=json.dumps(config),
        status="completed",
        started_at=started,
        finished_at=started + timedelta(seconds=duration_seconds),
    )
    db.add(job)
    db.flush()
    return job


def test_degrades_honestly_with_no_history(db_session):
    org_id = _make_org(db_session)
    result = estimate_training_duration(db_session, org_id, n_rows=1000, n_models=4, n_trials=20, n_folds=4)
    assert result.status == "degraded"
    assert result.estimated_seconds is None
    assert result.based_on_n_jobs == 0


def test_degrades_honestly_below_minimum_history(db_session):
    org_id = _make_org(db_session)
    for _ in range(MIN_COMPLETED_JOBS_FOR_ESTIMATE - 1):
        _make_completed_job(db_session, org_id, n_rows=1000, duration_seconds=60, model_ids=["lightgbm"])
    result = estimate_training_duration(db_session, org_id, n_rows=1000, n_models=1, n_trials=20, n_folds=4)
    assert result.status == "degraded"


def test_estimate_scales_with_configured_parameters(db_session):
    """Base : 1000 lignes, 1 modèle, 20 essais, 4 folds -> 80s. Doubler le
    nombre de modèles doit environ doubler l'estimation, dérivée du même
    taux historique — preuve que le calcul est réellement proportionnel,
    pas une valeur fixe recopiée."""
    org_id = _make_org(db_session)
    for _ in range(MIN_COMPLETED_JOBS_FOR_ESTIMATE):
        _make_completed_job(db_session, org_id, n_rows=1000, duration_seconds=80, model_ids=["lightgbm"])

    single_model = estimate_training_duration(db_session, org_id, n_rows=1000, n_models=1, n_trials=20, n_folds=4)
    double_model = estimate_training_duration(db_session, org_id, n_rows=1000, n_models=2, n_trials=20, n_folds=4)

    assert single_model.status == "estimated"
    assert single_model.estimated_seconds is not None
    assert double_model.estimated_seconds == single_model.estimated_seconds * 2


def test_jobs_without_dataset_row_count_are_ignored(db_session):
    org_id = _make_org(db_session)
    # Dataset sans row_count (jamais renseigné, ex. échec de parsing) — ne
    # doit jamais produire une division par zéro ni fausser le taux.
    for _ in range(MIN_COMPLETED_JOBS_FOR_ESTIMATE):
        job = _make_completed_job(db_session, org_id, n_rows=1000, duration_seconds=50, model_ids=["lightgbm"])
    # Un job supplémentaire sans row_count, mélangé aux valides.
    dataset = Dataset(
        organization_id=org_id, name="d2.csv", file_path="unused", file_size_bytes=1, status="ready", row_count=None
    )
    db_session.add(dataset)
    db_session.flush()
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    db_session.add(
        TrainingJob(
            organization_id=org_id,
            dataset_id=dataset.id,
            task_type="regression",
            target_column="cible",
            feature_columns_json=json.dumps(["x1"]),
            config_json=json.dumps({"model_ids": ["lightgbm"], "optuna_trials": 20, "cv_folds": 4}),
            status="completed",
            started_at=started,
            finished_at=started + timedelta(seconds=99999),  # aberrant si jamais pris en compte
        )
    )
    db_session.flush()

    result = estimate_training_duration(db_session, org_id, n_rows=1000, n_models=1, n_trials=20, n_folds=4)
    assert result.status == "estimated"
    assert result.estimated_seconds == 50.0
