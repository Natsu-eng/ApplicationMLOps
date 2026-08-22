"""Fonction exécutée par le worker RQ pour un job de réduction de dimension —
même process worker que `training_worker.py`/`clustering_worker.py`,
enfilée sur `analysis_queue` (Lot 4, correctif I6 — voir
`api/core/job_queue.py`), fonction dédiée.

Mêmes conventions que `clustering_worker.py` : session DB propre, progression
persistée à chaque étape, `_user_safe_error_message` copié (pas importé) —
indépendance volontaire des traductions si elles divergent un jour."""
from __future__ import annotations

import json
import logging
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib

from api.core.database import SessionLocal
from api.core.models import Dataset, DimensionalityJob, DimensionalityModel, DimensionalityPoint
from api.core.storage import dimensionality_model_file_path
from domains.shared.dataset_io import read_dataset_dataframe
from domains.dimensionality.services.engine import DimensionalityConfig, train_and_evaluate_dimensionality
from domains.shared.ml_preprocessing import TrainingAbortedError

logger = logging.getLogger("datalab.dimensionality_worker")


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
            "Le calcul de projection a dépassé la mémoire disponible. Essayez de réduire le nombre de "
            "variables sélectionnées, ou la taille du jeu de données."
        )
    return "Le calcul de projection a échoué pour une raison technique. Contactez votre administrateur si le problème persiste."


def _make_progress_callback(db, job: DimensionalityJob):
    def callback(step: str, percent: int) -> None:
        job.progress_step = step
        job.progress_percent = percent
        job.progress_updated_at = datetime.now(timezone.utc)
        db.commit()

    return callback


def run_dimensionality_job(job_id: int) -> None:
    """Point d'entrée RQ — enfilé par `POST /dimensionality/jobs`."""
    db = SessionLocal()
    try:
        job = db.query(DimensionalityJob).filter(DimensionalityJob.id == job_id).first()
        if job is None:
            logger.error("[Dimensionality] Job %s introuvable", job_id)
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
            config = DimensionalityConfig(**config_dict)

            progress_cb = _make_progress_callback(db, job)
            result = train_and_evaluate_dimensionality(df[feature_columns], config, progress_cb)

            artifact_path = dimensionality_model_file_path(job.organization_id, job.id)
            joblib.dump(result.pipeline_bundle, artifact_path)

            dimensionality_model = DimensionalityModel(
                organization_id=job.organization_id,
                dimensionality_job_id=job.id,
                algorithm=result.algorithm_label,
                algorithm_id=result.algorithm_id,
                n_samples_total=result.n_samples_total,
                n_samples_used=result.n_samples_used,
                sampled=result.sampled,
                trustworthiness_primary=result.trustworthiness_primary,
                trustworthiness_pca=result.trustworthiness_pca,
                variance_explained_json=json.dumps(result.variance_explained),
                total_variance_explained=result.total_variance_explained,
                loadings_json=json.dumps([asdict(loading) for loading in result.loadings]),
                distance_fidelity_note=result.distance_fidelity_note,
                feature_columns_json=json.dumps(result.feature_columns),
                model_card_json=json.dumps(result.model_card),
                file_path=str(artifact_path),
            )
            db.add(dimensionality_model)

            # Points 2D — même transaction que le modèle/`job.status`, garantit
            # qu'un résultat "completed" a toujours ses points associés.
            for point in result.points:
                db.add(
                    DimensionalityPoint(
                        organization_id=job.organization_id,
                        dimensionality_job_id=job.id,
                        row_index=point.row_index,
                        x=point.x,
                        y=point.y,
                    )
                )

            job.status = "completed"
            job.progress_step = "Terminé"
            job.progress_percent = 100
            job.finished_at = datetime.now(timezone.utc)
            db.commit()
            logger.info("[Dimensionality] Job %s terminé — %s", job_id, result.algorithm_label)

        except TrainingAbortedError as exc:
            job.status = "failed"
            job.error_message = str(exc)
            job.finished_at = datetime.now(timezone.utc)
            db.commit()
            logger.warning("[Dimensionality] Job %s — échec diagnostiqué : %s", job_id, exc)

        except Exception as exc:  # toute erreur ne doit jamais faire planter le worker
            job.status = "failed"
            job.error_message = _user_safe_error_message(exc)
            job.finished_at = datetime.now(timezone.utc)
            db.commit()
            logger.error("[Dimensionality] Job %s échoué : %s\n%s", job_id, exc, traceback.format_exc())

    finally:
        db.close()
