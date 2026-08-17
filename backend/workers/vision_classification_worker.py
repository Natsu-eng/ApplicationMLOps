"""Fonction exécutée par le worker RQ pour un job de classification
d'images — même process worker que `training_worker.py`/`anomaly_worker.py`
(une seule file `training_queue`, un seul worker physique CPU). Mêmes
conventions : session DB propre, progression persistée à chaque étape,
`_user_safe_error_message` copié (pas importé)."""
from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

from api.core.database import SessionLocal
from api.core.models import VisionClassificationJob, VisionClassificationModel, VisionDataset
from api.core.storage import vision_classification_model_file_path
from services.ml_preprocessing import TrainingAbortedError
from services.vision_classification_training import ClassificationConfig, train_and_evaluate_classification

logger = logging.getLogger("datalab.vision_classification_worker")


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
    return "L'entraînement de classification a échoué pour une raison technique. Contactez votre administrateur si le problème persiste."


def _make_progress_callback(db, job: VisionClassificationJob):
    def callback(step: str, percent: int) -> None:
        job.progress_step = step
        job.progress_percent = percent
        job.progress_updated_at = datetime.now(timezone.utc)
        db.commit()

    return callback


def run_vision_classification_job(job_id: int) -> None:
    """Point d'entrée RQ — enfilé par `POST /vision/classification/jobs`."""
    db = SessionLocal()
    try:
        job = db.query(VisionClassificationJob).filter(VisionClassificationJob.id == job_id).first()
        if job is None:
            logger.error("[VisionClassification] Job %s introuvable", job_id)
            return

        job.status = "running"
        job.started_at = datetime.now(timezone.utc)
        job.progress_updated_at = job.started_at
        db.commit()

        try:
            dataset = db.query(VisionDataset).filter(VisionDataset.id == job.vision_dataset_id).first()
            if dataset is None or dataset.status != "ready":
                raise TrainingAbortedError("Dataset d'images introuvable ou non prêt")
            if dataset.structure_type != "classification":
                raise TrainingAbortedError(
                    "Ce dataset n'a pas une structure de classification (dossiers de classes) — "
                    "un dataset normal/défaut ne peut pas être utilisé pour la classification"
                )

            config = ClassificationConfig(**json.loads(job.config_json))
            progress_cb = _make_progress_callback(db, job)
            result = train_and_evaluate_classification(Path(dataset.storage_dir), config, progress_cb)

            artifact_path = vision_classification_model_file_path(job.organization_id, job.id)
            torch.save(result.model_artifact, artifact_path)

            vision_model = VisionClassificationModel(
                organization_id=job.organization_id,
                vision_classification_job_id=job.id,
                backbone_id=result.backbone_id,
                class_names_json=json.dumps(result.class_names),
                n_train=result.n_train,
                n_val=result.n_val,
                n_test=result.n_test,
                history_json=json.dumps([vars(m) for m in result.history]),
                test_accuracy=result.test_accuracy,
                test_precision_macro=result.test_precision_macro,
                test_recall_macro=result.test_recall_macro,
                test_f1_macro=result.test_f1_macro,
                confusion_matrix_json=json.dumps(result.confusion_matrix),
                examples_json=json.dumps([vars(e) for e in result.examples]),
                model_card_json=json.dumps(result.model_card),
                file_path=str(artifact_path),
            )
            db.add(vision_model)

            job.status = "completed"
            job.progress_step = "Terminé"
            job.progress_percent = 100
            job.finished_at = datetime.now(timezone.utc)
            db.commit()
            logger.info(
                "[VisionClassification] Job %s terminé — accuracy test %.3f", job_id, result.test_accuracy
            )

        except TrainingAbortedError as exc:
            job.status = "failed"
            job.error_message = str(exc)
            job.finished_at = datetime.now(timezone.utc)
            db.commit()
            logger.warning("[VisionClassification] Job %s — échec diagnostiqué : %s", job_id, exc)

        except Exception as exc:  # toute erreur ne doit jamais faire planter le worker
            job.status = "failed"
            job.error_message = _user_safe_error_message(exc)
            job.finished_at = datetime.now(timezone.utc)
            db.commit()
            logger.error("[VisionClassification] Job %s échoué : %s\n%s", job_id, exc, traceback.format_exc())

    finally:
        db.close()
