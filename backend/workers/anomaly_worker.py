"""Fonction exécutée par le worker RQ pour un job de détection d'anomalies —
même process worker que `training_worker.py`/`clustering_worker.py`/
`dimensionality_worker.py`, enfilée sur `analysis_queue` (Lot 4, correctif
I6 — voir `api/core/job_queue.py`). Mêmes conventions : session DB propre,
progression persistée à chaque étape, `_user_safe_error_message` copié
(pas importé)."""
from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path

import joblib

from api.core.database import SessionLocal
from api.core.models import AnomalyJob, AnomalyModel, AnomalyObservationRecord, Dataset
from api.core.storage import anomaly_model_file_path
from services.anomaly_training import AnomalyConfig, train_and_evaluate_anomalies
from services.datasets import read_dataset_dataframe
from services.ml_preprocessing import TrainingAbortedError

logger = logging.getLogger("datalab.anomaly_worker")


def _user_safe_error_message(exc: Exception) -> str:
    text = str(exc).lower()
    is_memory_error = (
        isinstance(exc, MemoryError)
        or "bad allocation" in text
        or "unable to allocate" in text
        or "out of memory" in text
    )
    if is_memory_error:
        return (
            "La détection d'anomalies a dépassé la mémoire disponible. Essayez de réduire le nombre de "
            "variables sélectionnées, ou la taille du jeu de données."
        )
    return "La détection d'anomalies a échoué pour une raison technique. Contactez votre administrateur si le problème persiste."


def _make_progress_callback(db, job: AnomalyJob):
    def callback(step: str, percent: int) -> None:
        job.progress_step = step
        job.progress_percent = percent
        job.progress_updated_at = datetime.now(timezone.utc)
        db.commit()

    return callback


def run_anomaly_job(job_id: int) -> None:
    """Point d'entrée RQ — enfilé par `POST /anomalies/jobs`."""
    db = SessionLocal()
    try:
        job = db.query(AnomalyJob).filter(AnomalyJob.id == job_id).first()
        if job is None:
            logger.error("[Anomaly] Job %s introuvable", job_id)
            return

        job.status = "running"
        job.started_at = datetime.now(timezone.utc)
        job.progress_updated_at = job.started_at
        db.commit()

        try:
            dataset = db.query(Dataset).filter(Dataset.id == job.dataset_id).first()
            if dataset is None or dataset.status != "ready":
                raise TrainingAbortedError("Dataset introuvable ou non prêt")

            df = read_dataset_dataframe(Path(dataset.file_path), Path(dataset.file_path).suffix)
            feature_columns = json.loads(job.feature_columns_json)
            missing = [c for c in feature_columns if c not in df.columns]
            if missing:
                raise TrainingAbortedError(f"Colonne(s) absente(s) du dataset : {', '.join(missing)}")

            config_dict = json.loads(job.config_json)
            config = AnomalyConfig(**config_dict)

            progress_cb = _make_progress_callback(db, job)
            result = train_and_evaluate_anomalies(df[feature_columns], config, progress_cb)

            artifact_path = anomaly_model_file_path(job.organization_id, job.id)
            joblib.dump(result.pipeline_bundle, artifact_path)

            anomaly_model = AnomalyModel(
                organization_id=job.organization_id,
                anomaly_job_id=job.id,
                n_samples_total=result.n_samples_total,
                n_samples_used=result.n_samples_used,
                sampled=result.sampled,
                n_anomalies_isolation_forest=result.n_anomalies_isolation_forest,
                n_anomalies_lof=result.n_anomalies_lof,
                n_anomalies_consensus=result.n_anomalies_consensus,
                anomaly_rate_isolation_forest=result.anomaly_rate_isolation_forest,
                anomaly_rate_lof=result.anomaly_rate_lof,
                anomaly_rate_consensus=result.anomaly_rate_consensus,
                score_histogram_json=json.dumps(result.score_histogram),
                feature_columns_json=json.dumps(result.feature_columns),
                model_card_json=json.dumps(result.model_card),
                file_path=str(artifact_path),
            )
            db.add(anomaly_model)

            for observation in result.top_observations:
                db.add(
                    AnomalyObservationRecord(
                        organization_id=job.organization_id,
                        anomaly_job_id=job.id,
                        row_index=observation.row_index,
                        rank=observation.rank,
                        consensus_score=observation.consensus_score,
                        score_isolation_forest=observation.score_isolation_forest,
                        score_lof=observation.score_lof,
                        is_anomaly_isolation_forest=observation.is_anomaly_isolation_forest,
                        is_anomaly_lof=observation.is_anomaly_lof,
                        agreement=observation.agreement,
                        numeric_deviations_json=json.dumps(observation.numeric_deviations),
                        categorical_flags_json=json.dumps(observation.categorical_flags),
                    )
                )

            job.status = "completed"
            job.progress_step = "Terminé"
            job.progress_percent = 100
            job.finished_at = datetime.now(timezone.utc)
            db.commit()
            logger.info(
                "[Anomaly] Job %s terminé — %d observations en consensus", job_id, result.n_anomalies_consensus
            )

        except TrainingAbortedError as exc:
            job.status = "failed"
            job.error_message = str(exc)
            job.finished_at = datetime.now(timezone.utc)
            db.commit()
            logger.warning("[Anomaly] Job %s — échec diagnostiqué : %s", job_id, exc)

        except Exception as exc:  # toute erreur ne doit jamais faire planter le worker
            job.status = "failed"
            job.error_message = _user_safe_error_message(exc)
            job.finished_at = datetime.now(timezone.utc)
            db.commit()
            logger.error("[Anomaly] Job %s échoué : %s\n%s", job_id, exc, traceback.format_exc())

    finally:
        db.close()
