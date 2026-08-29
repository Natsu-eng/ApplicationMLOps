"""Fonction exécutée par le worker RQ pour un job de clustering — même
process worker que `training_worker.py`, enfilée sur `analysis_queue`
(Lot 4, correctif I6 — voir `api/core/job_queue.py`), fonction dédiée.

Mêmes conventions que `training_worker.py` : session DB propre (pas de
`Depends`), progression persistée directement en base à chaque étape,
messages d'erreur jamais bruts (voir `_user_safe_error_message`, copié —
pas importé — pour rester indépendant si les deux traductions divergent un
jour, même choix que la séparation des modules `ml_registry`/
`clustering_registry`)."""
from __future__ import annotations

import json
import logging
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib

from api.core.database import SessionLocal
from api.core.models import ClusterCandidateRecord, ClusteringJob, ClusterModel, Dataset
from api.core.observability import request_id_var
from api.core.storage import cluster_model_file_path
from domains.clustering.services.engine import ClusteringConfig, train_and_evaluate_clustering
from domains.shared.dataset_io import read_dataset_dataframe
from domains.shared.ml_preprocessing import TrainingAbortedError
from domains.shared.notifications import notify_job_terminal

logger = logging.getLogger("datalab.clustering_worker")


def _user_safe_error_message(exc: Exception) -> str:
    """Traduction générique de secours — voir
    `training_worker.py::_user_safe_error_message` pour le raisonnement
    complet (whitelist volontairement restreinte, jamais de détail
    technique/chemin de fichier affiché)."""
    text = str(exc).lower()
    is_memory_error = (
        isinstance(exc, MemoryError)
        or "bad allocation" in text
        or "unable to allocate" in text
        or "out of memory" in text
    )
    if is_memory_error:
        return (
            "Le clustering a dépassé la mémoire disponible. Essayez de réduire le nombre de "
            "variables sélectionnées, ou la taille du jeu de données."
        )
    return "Le clustering a échoué pour une raison technique. Contactez votre administrateur si le problème persiste."


def _make_progress_callback(db, job: ClusteringJob):
    def callback(step: str, percent: int) -> None:
        job.progress_step = step
        job.progress_percent = percent
        job.progress_updated_at = datetime.now(timezone.utc)
        db.commit()

    return callback


def run_clustering_job(job_id: int) -> None:
    """Point d'entrée RQ — enfilé par `POST /clustering/jobs`."""
    db = SessionLocal()
    request_id_token = request_id_var.set("-")
    try:
        job = db.query(ClusteringJob).filter(ClusteringJob.id == job_id).first()
        if job is None:
            logger.error("[Clustering] Job %s introuvable", job_id)
            return
        request_id_var.set(job.request_id or "-")

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
            config = ClusteringConfig(**config_dict)

            progress_cb = _make_progress_callback(db, job)
            result = train_and_evaluate_clustering(df[feature_columns], config, progress_cb)

            artifact_path = cluster_model_file_path(job.organization_id, job.id)
            joblib.dump(result.pipeline_bundle, artifact_path)

            cluster_model = ClusterModel(
                organization_id=job.organization_id,
                clustering_job_id=job.id,
                algorithm=result.winning_label,
                n_clusters=result.model_card["n_clusters"],
                feature_columns_json=json.dumps(result.feature_columns),
                metrics_json=json.dumps(
                    {
                        "silhouette": result.model_card["silhouette"],
                        "davies_bouldin": result.model_card["davies_bouldin"],
                        "calinski_harabasz": result.model_card["calinski_harabasz"],
                        "noise_ratio": result.model_card["noise_ratio"],
                    }
                ),
                profiles_json=json.dumps(
                    [
                        {
                            "cluster_id": p.cluster_id,
                            "size": p.size,
                            "size_pct": p.size_pct,
                            "numeric_summary": p.numeric_summary,
                            "categorical_summary": p.categorical_summary,
                            "differentiating_variables": p.differentiating_variables,
                        }
                        for p in result.cluster_profiles
                    ]
                ),
                labels_json=json.dumps(result.labels),
                model_card_json=json.dumps(result.model_card),
                file_path=str(artifact_path),
            )
            db.add(cluster_model)

            # Leaderboard (même transaction que `cluster_model`/`job.status` —
            # garantit que le gagnant et `is_winner=True` désignent toujours
            # le même candidat, même principe que `ModelCandidate`, Lot D).
            for candidate in result.all_candidates:
                db.add(
                    ClusterCandidateRecord(
                        organization_id=job.organization_id,
                        clustering_job_id=job.id,
                        algorithm=candidate.label,
                        family=candidate.family,
                        params_json=json.dumps(candidate.params),
                        n_clusters=candidate.n_clusters,
                        silhouette=candidate.silhouette,
                        davies_bouldin=candidate.davies_bouldin,
                        calinski_harabasz=candidate.calinski_harabasz,
                        noise_ratio=candidate.noise_ratio,
                        is_winner=candidate.is_winner,
                        rank=candidate.rank,
                        rank_silhouette=candidate.rank_silhouette,
                        rank_davies_bouldin=candidate.rank_davies_bouldin,
                        rank_calinski_harabasz=candidate.rank_calinski_harabasz,
                        composite_rank=candidate.composite_rank,
                        # Top 3 seulement (voir engine.py::TOP_N_WITH_FULL_RESULTS)
                        # — `None` au-delà, jamais une liste vide qui se
                        # confondrait avec "calculé, zéro segment".
                        cluster_profiles_json=(
                            json.dumps([asdict(p) for p in candidate.cluster_profiles])
                            if candidate.cluster_profiles is not None
                            else None
                        ),
                        noise_count=candidate.noise_count,
                    )
                )

            job.status = "completed"
            job.progress_step = "Terminé"
            job.progress_percent = 100
            job.finished_at = datetime.now(timezone.utc)
            notify_job_terminal(
                db, job.organization_id, job.created_by_id, "clustering", job.id, "completed", dataset.name
            )
            db.commit()
            logger.info("[Clustering] Job %s terminé — %s retenu", job_id, result.winning_label)

        except TrainingAbortedError as exc:
            job.status = "failed"
            job.error_message = str(exc)
            job.finished_at = datetime.now(timezone.utc)
            notify_job_terminal(
                db, job.organization_id, job.created_by_id, "clustering", job.id, "failed", f"Dataset #{job.dataset_id}"
            )
            db.commit()
            logger.warning("[Clustering] Job %s — échec diagnostiqué : %s", job_id, exc)

        except Exception as exc:  # toute erreur ne doit jamais faire planter le worker
            job.status = "failed"
            job.error_message = _user_safe_error_message(exc)
            job.finished_at = datetime.now(timezone.utc)
            notify_job_terminal(
                db, job.organization_id, job.created_by_id, "clustering", job.id, "failed", f"Dataset #{job.dataset_id}"
            )
            db.commit()
            logger.error("[Clustering] Job %s échoué : %s\n%s", job_id, exc, traceback.format_exc())

    finally:
        request_id_var.reset(request_id_token)
        db.close()
