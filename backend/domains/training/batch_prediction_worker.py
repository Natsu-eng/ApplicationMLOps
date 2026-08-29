"""Fonction exécutée par le worker RQ pour un job de prédiction en lot
(retour utilisateur : "batch prediction — upload d'un fichier, prédictions
pour toutes les lignes") — même conventions que `worker.py` (session DB
propre, progression persistée à chaque étape, `_user_safe_error_message`
copié, pas importé — même raison : chaque worker de job reste lisible
isolément, sans dépendre d'un module partagé qui changerait son comportement
pour tous les jobs à la fois si modifié pour un seul).

Enfilé sur `analysis_queue` (voir `api/core/job_queue.py`) — pas de
recherche d'hyperparamètres ici (le modèle est déjà entraîné), coût
comparable au clustering/à la réduction de dimension, jamais aussi long
qu'un entraînement complet."""
from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path

from api.core.database import SessionLocal
from api.core.models import BatchPredictionJob
from api.core.observability import request_id_var
from api.core.storage import batch_prediction_output_file_path
from domains.shared.dataset_io import DatasetParsingError, read_dataframe
from domains.shared.model_bundle import InferenceError, load_bundle
from domains.training.services.inference import predict_batch

logger = logging.getLogger("datalab.batch_prediction_worker")


def _user_safe_error_message(exc: Exception) -> str:
    text = str(exc).lower()
    is_memory_error = (
        isinstance(exc, MemoryError)
        or "bad allocation" in text
        or "unable to allocate" in text
        or "out of memory" in text
    )
    if is_memory_error:
        return "La prédiction en lot a dépassé la mémoire disponible. Essayez avec un fichier plus petit."
    return (
        "La prédiction en lot a échoué pour une raison technique. "
        "Contactez votre administrateur si le problème persiste."
    )


def _make_progress_callback(db, job: BatchPredictionJob):
    def callback(step: str, percent: int) -> None:
        job.progress_step = step
        job.progress_percent = percent
        job.progress_updated_at = datetime.now(timezone.utc)
        db.commit()

    return callback


def run_batch_prediction_job(batch_job_id: int) -> None:
    """Point d'entrée RQ — enfilé par `POST /training/jobs/{id}/predict-batch`."""
    db = SessionLocal()
    request_id_token = request_id_var.set("-")
    try:
        job = db.query(BatchPredictionJob).filter(BatchPredictionJob.id == batch_job_id).first()
        if job is None:
            logger.error("[BatchPrediction] Job %s introuvable", batch_job_id)
            return
        request_id_var.set(job.request_id or "-")

        job.status = "running"
        job.started_at = datetime.now(timezone.utc)
        job.progress_updated_at = job.started_at
        db.commit()

        progress_cb = _make_progress_callback(db, job)
        try:
            training_job = job.training_job
            model = training_job.model if training_job else None
            if model is None:
                raise InferenceError("Le modèle source n'est plus disponible (entraînement supprimé ?)")

            progress_cb("Lecture du fichier", 10)
            extension = Path(job.input_file_path).suffix
            df = read_dataframe(Path(job.input_file_path), extension)
            feature_columns = json.loads(model.feature_columns_json)

            progress_cb("Chargement du modèle", 30)
            bundle = load_bundle(model.file_path)

            progress_cb(f"Prédiction sur {len(df)} lignes", 50)
            result_df = predict_batch(bundle, feature_columns, df)

            progress_cb("Écriture du résultat", 90)
            output_path = batch_prediction_output_file_path(job.organization_id, job.id)
            result_df.to_csv(output_path, index=False)
            job.output_file_path = str(output_path)
            job.n_rows = int(len(result_df))

            job.status = "completed"
            job.progress_step = "Terminé"
            job.progress_percent = 100
            job.finished_at = datetime.now(timezone.utc)
            db.commit()
            logger.info("[BatchPrediction] Job %s terminé — %s lignes prédites", batch_job_id, job.n_rows)

        except (InferenceError, DatasetParsingError) as exc:
            job.status = "failed"
            job.error_message = str(exc)
            job.finished_at = datetime.now(timezone.utc)
            db.commit()
            logger.warning("[BatchPrediction] Job %s — échec diagnostiqué : %s", batch_job_id, exc)

        except Exception as exc:  # toute erreur ne doit jamais faire planter le worker
            job.status = "failed"
            job.error_message = _user_safe_error_message(exc)
            job.finished_at = datetime.now(timezone.utc)
            db.commit()
            logger.error("[BatchPrediction] Job %s échoué : %s\n%s", batch_job_id, exc, traceback.format_exc())

    finally:
        request_id_var.reset(request_id_token)
        db.close()
