"""Fonction exécutée par le worker RQ — un process séparé de l'API (voir
`rq worker training --url $REDIS_URL`, lancé par un conteneur dédié en
Docker, voir docker-compose.yml).

Tourne dans un process à part : ouvre sa propre session DB (pas de
`Depends(get_db)`, qui est un mécanisme FastAPI), et persiste la progression
directement en base à chaque étape — le polling de l'API
(`GET /training/jobs/{id}`) lit cette même table, pas un état RQ interne.
"""
from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime, timezone

import joblib

from api.core.database import SessionLocal
from api.core.models import Dataset, MLModel, TrainingJob
from api.core.storage import model_file_path
from services.datasets import read_dataframe
from services.ml_preprocessing import DataLeakageError, split_dataset
from services.ml_training import TrainingConfig, train_and_evaluate

logger = logging.getLogger("datalab.training_worker")


def _make_progress_callback(db, job: TrainingJob):
    def callback(step: str, percent: int) -> None:
        job.progress_step = step
        job.progress_percent = percent
        db.commit()

    return callback


def run_training_job(job_id: int) -> None:
    """Point d'entrée RQ — enfilé par `POST /training/jobs`
    (voir api/routers/training.py)."""
    db = SessionLocal()
    try:
        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if job is None:
            logger.error("[Training] Job %s introuvable", job_id)
            return

        job.status = "running"
        job.started_at = datetime.now(timezone.utc)
        db.commit()

        try:
            dataset = db.query(Dataset).filter(Dataset.id == job.dataset_id).first()
            if dataset is None or dataset.status != "ready":
                raise RuntimeError("Dataset introuvable ou non prêt")

            from pathlib import Path

            df = read_dataframe(Path(dataset.file_path), Path(dataset.file_path).suffix)

            feature_columns = json.loads(job.feature_columns_json)
            config_dict = json.loads(job.config_json)
            config = TrainingConfig(**config_dict)

            progress_cb = _make_progress_callback(db, job)
            split = split_dataset(
                df=df,
                target=job.target_column,
                feature_columns=feature_columns,
                task_type=job.task_type,
                group_column=job.group_column,
                test_size=config.test_size,
                seed=config.seed,
            )

            result = train_and_evaluate(split, job.task_type, config, progress_cb)

            artifact_path = model_file_path(job.organization_id, job.id)
            joblib.dump(result.pipeline_bundle, artifact_path)

            ml_model = MLModel(
                organization_id=job.organization_id,
                training_job_id=job.id,
                algorithm=result.algorithm,
                task_type=job.task_type,
                target_column=job.target_column,
                feature_columns_json=json.dumps(feature_columns),
                file_path=str(artifact_path),
                metrics_json=json.dumps(result.metrics),
                shap_summary_json=json.dumps(result.shap_summary),
                cqr_json=json.dumps(result.cqr) if result.cqr else None,
                model_card_json=json.dumps(result.model_card),
            )
            db.add(ml_model)

            job.status = "completed"
            job.progress_step = "Terminé"
            job.progress_percent = 100
            job.finished_at = datetime.now(timezone.utc)
            db.commit()
            logger.info("[Training] Job %s terminé — %s retenu", job_id, result.algorithm)

        except DataLeakageError as exc:
            job.status = "failed"
            job.error_message = str(exc)
            job.finished_at = datetime.now(timezone.utc)
            db.commit()
            logger.warning("[Training] Job %s — fuite détectée : %s", job_id, exc)

        except Exception as exc:  # toute erreur d'entraînement ne doit jamais faire planter le worker
            job.status = "failed"
            job.error_message = str(exc)
            job.finished_at = datetime.now(timezone.utc)
            db.commit()
            logger.error("[Training] Job %s échoué : %s\n%s", job_id, exc, traceback.format_exc())

    finally:
        db.close()
