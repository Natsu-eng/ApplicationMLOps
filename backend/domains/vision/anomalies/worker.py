"""Fonction exécutée par le worker RQ pour un job de détection d'anomalies
visuelles (structure normal/défaut) — même process worker que `vision_classification_worker.py`,
enfilée sur `vision_queue` (Lot 4, correctif I6 — voir
`api/core/job_queue.py`). Mêmes conventions : session DB propre,
progression persistée à chaque étape, `_user_safe_error_message` copié
(pas importé)."""
from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

from api.core.database import SessionLocal
from api.core.models import VisionAnomalyExampleRecord, VisionAnomalyJob, VisionAnomalyModel, VisionDataset
from api.core.storage import vision_anomaly_model_file_path
from domains.shared.ml_preprocessing import TrainingAbortedError
from domains.vision.anomalies.services.engine import AnomalyVisionConfig, train_and_evaluate_anomaly_vision

logger = logging.getLogger("datalab.vision_anomaly_worker")


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
            "L'entraînement a dépassé la mémoire disponible. Essayez de réduire la taille du lot "
            "(batch_size) ou le nombre d'images du dataset."
        )
    return "La détection d'anomalies visuelles a échoué pour une raison technique. Contactez votre administrateur si le problème persiste."


def _make_progress_callback(db, job: VisionAnomalyJob):
    def callback(step: str, percent: int) -> None:
        job.progress_step = step
        job.progress_percent = percent
        job.progress_updated_at = datetime.now(timezone.utc)
        db.commit()

    return callback


def run_vision_anomaly_job(job_id: int) -> None:
    """Point d'entrée RQ — enfilé par `POST /vision/anomalies/jobs`."""
    db = SessionLocal()
    try:
        job = db.query(VisionAnomalyJob).filter(VisionAnomalyJob.id == job_id).first()
        if job is None:
            logger.error("[VisionAnomaly] Job %s introuvable", job_id)
            return

        job.status = "running"
        job.started_at = datetime.now(timezone.utc)
        job.progress_updated_at = job.started_at
        db.commit()

        try:
            dataset = db.query(VisionDataset).filter(VisionDataset.id == job.vision_dataset_id).first()
            if dataset is None or dataset.status != "ready":
                raise TrainingAbortedError("Dataset d'images introuvable ou non prêt")
            if dataset.structure_type != "mvtec_ad":
                raise TrainingAbortedError(
                    "Ce dataset n'a pas une structure normal/défaut (train/good + test/good + test/<defaut>) — "
                    "un dataset de classification ne peut pas être utilisé pour la détection d'anomalies visuelles"
                )

            config = AnomalyVisionConfig(**json.loads(job.config_json))
            progress_cb = _make_progress_callback(db, job)
            result = train_and_evaluate_anomaly_vision(Path(dataset.storage_dir), config, progress_cb)

            artifact_path = vision_anomaly_model_file_path(job.organization_id, job.id)
            torch.save(result.model_artifact, artifact_path)

            vision_model = VisionAnomalyModel(
                organization_id=job.organization_id,
                vision_anomaly_job_id=job.id,
                model_id=result.model_id,
                n_train=result.n_train,
                n_val=result.n_val,
                n_test=result.n_test,
                n_calibration=result.n_calibration,
                n_evaluation=result.n_evaluation,
                history_json=json.dumps([vars(m) for m in result.history]),
                threshold=result.threshold,
                roc_auc=result.roc_auc,
                test_accuracy=result.test_accuracy,
                test_precision=result.test_precision,
                test_recall=result.test_recall,
                test_f1=result.test_f1,
                confusion_matrix_json=json.dumps(result.confusion_matrix),
                model_card_json=json.dumps(result.model_card),
                file_path=str(artifact_path),
            )
            db.add(vision_model)
            db.flush()

            for example in result.examples:
                db.add(
                    VisionAnomalyExampleRecord(
                        organization_id=job.organization_id,
                        vision_anomaly_model_id=vision_model.id,
                        relative_path=example.relative_path,
                        defect_category=example.defect_category,
                        true_label=example.true_label,
                        predicted_label=example.predicted_label,
                        anomaly_score=example.anomaly_score,
                        heatmap_png=example.heatmap_png,
                        mask_png=example.mask_png,
                    )
                )

            job.status = "completed"
            job.progress_step = "Terminé"
            job.progress_percent = 100
            job.finished_at = datetime.now(timezone.utc)
            db.commit()
            logger.info(
                "[VisionAnomaly] Job %s terminé — ROC-AUC test %.3f", job_id, result.roc_auc
            )

        except TrainingAbortedError as exc:
            job.status = "failed"
            job.error_message = str(exc)
            job.finished_at = datetime.now(timezone.utc)
            db.commit()
            logger.warning("[VisionAnomaly] Job %s — échec diagnostiqué : %s", job_id, exc)

        except Exception as exc:  # toute erreur ne doit jamais faire planter le worker
            job.status = "failed"
            job.error_message = _user_safe_error_message(exc)
            job.finished_at = datetime.now(timezone.utc)
            db.commit()
            logger.error("[VisionAnomaly] Job %s échoué : %s\n%s", job_id, exc, traceback.format_exc())

    finally:
        db.close()
