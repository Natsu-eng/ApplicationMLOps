"""Tests de services/model_versioning.py::next_version (Lot 5, correctif
P1, AUDIT_DATALAB_2026-08-16.md §P1) — logique pure, base de test réelle
mais sans passer par l'API (juste des lignes MLModel insérées directement)."""
from __future__ import annotations

import json

from api.core.models import Dataset, MLModel, Organization, TrainingJob
from services.model_versioning import next_version


def _make_org_and_dataset(db_session, org_name="Bureau"):
    org = Organization(name=org_name)
    db_session.add(org)
    db_session.flush()
    dataset = Dataset(organization_id=org.id, name="d.csv", file_path="x", file_size_bytes=1, status="ready")
    db_session.add(dataset)
    db_session.flush()
    return org, dataset


def _add_model(db_session, org_id, dataset_id, target_column, version, training_job_id):
    db_session.add(MLModel(
        organization_id=org_id, training_job_id=training_job_id, dataset_id=dataset_id, version=version,
        algorithm="LightGBM", task_type="regression", target_column=target_column,
        feature_columns_json=json.dumps([]), file_path="x", metrics_json=json.dumps({}),
    ))


def _add_job(db_session, org_id, dataset_id, target_column):
    job = TrainingJob(
        organization_id=org_id, dataset_id=dataset_id, task_type="regression", target_column=target_column,
        feature_columns_json=json.dumps([]), config_json=json.dumps({}), status="completed",
    )
    db_session.add(job)
    db_session.flush()
    return job


def test_first_model_on_a_problem_gets_version_1(db_session):
    org, dataset = _make_org_and_dataset(db_session)
    assert next_version(db_session, org.id, dataset.id, "cible") == 1


def test_next_version_increments_from_the_existing_maximum(db_session):
    org, dataset = _make_org_and_dataset(db_session)
    job = _add_job(db_session, org.id, dataset.id, "cible")
    _add_model(db_session, org.id, dataset.id, "cible", version=1, training_job_id=job.id)
    db_session.commit()
    assert next_version(db_session, org.id, dataset.id, "cible") == 2


def test_next_version_is_independent_per_target_column(db_session):
    org, dataset = _make_org_and_dataset(db_session)
    job = _add_job(db_session, org.id, dataset.id, "cible_a")
    _add_model(db_session, org.id, dataset.id, "cible_a", version=1, training_job_id=job.id)
    db_session.commit()
    assert next_version(db_session, org.id, dataset.id, "cible_b") == 1


def test_next_version_is_independent_per_dataset(db_session):
    org, dataset1 = _make_org_and_dataset(db_session)
    dataset2 = Dataset(organization_id=org.id, name="d2.csv", file_path="x", file_size_bytes=1, status="ready")
    db_session.add(dataset2)
    db_session.flush()
    job = _add_job(db_session, org.id, dataset1.id, "cible")
    _add_model(db_session, org.id, dataset1.id, "cible", version=1, training_job_id=job.id)
    db_session.commit()
    assert next_version(db_session, org.id, dataset2.id, "cible") == 1


def test_next_version_is_independent_per_organization(db_session):
    org1, dataset1 = _make_org_and_dataset(db_session, "Bureau A")
    org2, dataset2 = _make_org_and_dataset(db_session, "Bureau B")
    job = _add_job(db_session, org1.id, dataset1.id, "cible")
    _add_model(db_session, org1.id, dataset1.id, "cible", version=1, training_job_id=job.id)
    db_session.commit()
    assert next_version(db_session, org2.id, dataset2.id, "cible") == 1
