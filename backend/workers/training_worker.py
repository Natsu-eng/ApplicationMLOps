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
from api.core.models import Dataset, MLModel, ModelCandidate, TrainingJob
from api.core.storage import model_file_path
from services.datasets import read_dataframe
from services.feature_engineering import apply_upstream_feature_engineering
from services.ml_preprocessing import DataLeakageError, split_dataset
from services.ml_training import TrainingConfig, train_and_evaluate

logger = logging.getLogger("datalab.training_worker")


def _user_safe_error_message(exc: Exception) -> str:
    """Traduit une exception technique en message actionnable, langage clair
    — jamais une trace brute ni un chemin de fichier interne affiché à
    l'utilisateur (diagnostic "bad allocation" affiché tel quel en
    production). Le détail technique complet (type, message d'origine,
    traceback) reste dans les logs serveur, jamais perdu — voir l'appel à
    `logger.error` juste après, qui ne change pas.

    Whitelist volontairement restreinte aux causes déjà observées en usage
    réel : mémoire insuffisante (message générique sûr sinon, pas de
    tentative de deviner toutes les causes possibles)."""
    text = str(exc).lower()
    is_memory_error = (
        isinstance(exc, MemoryError)
        or "bad allocation" in text
        or "unable to allocate" in text
        or "out of memory" in text
    )
    if is_memory_error:
        return (
            "L'entraînement a dépassé la mémoire disponible. Cela arrive souvent avec "
            "une colonne à très grand nombre de valeurs différentes (comme un "
            "identifiant). Essayez de retirer cette colonne des variables utilisées, "
            "ou de réduire la taille du jeu de données."
        )
    return "L'entraînement a échoué pour une raison technique. Contactez votre administrateur si le problème persiste."


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

            # `raw_feature_columns` : colonnes SAISIES par l'utilisateur (formulaire
            # de prédiction, feature_columns_json de MLModel) — distinctes des
            # colonnes VUES par le préprocesseur une fois la spec 4c appliquée
            # (ex. "date" saisie, mais "date_annee"/"date_mois"/... vues par le
            # modèle). Ne jamais mélanger les deux listes (Lot 4c, précision 3).
            raw_feature_columns = json.loads(job.feature_columns_json)
            config_dict = json.loads(job.config_json)
            config = TrainingConfig(**config_dict)

            # Spec de feature engineering (Lot 4c) — absente/NULL : comportement
            # strictement inchangé (rétrocompatibilité totale).
            feature_engineering_spec = (
                json.loads(job.feature_engineering_json) if job.feature_engineering_json else None
            )
            if feature_engineering_spec:
                # Transformations déterministes (datetime, ratio) appliquées UNE
                # SEULE FOIS ici, en amont du split — même fonction que celle
                # rejouée à l'inférence (services/ml_inference.py), pour garantir
                # des colonnes identiques dans les deux contextes.
                df, effective_feature_columns = apply_upstream_feature_engineering(
                    df, raw_feature_columns, feature_engineering_spec
                )
                pipeline_feature_engineering_config = feature_engineering_spec.get("pipeline")
            else:
                effective_feature_columns = raw_feature_columns
                pipeline_feature_engineering_config = None

            progress_cb = _make_progress_callback(db, job)
            split = split_dataset(
                df=df,
                target=job.target_column,
                feature_columns=effective_feature_columns,
                task_type=job.task_type,
                group_column=job.group_column,
                test_size=config.test_size,
                seed=config.seed,
            )

            result = train_and_evaluate(
                split, job.task_type, config, progress_cb,
                feature_engineering_config=pipeline_feature_engineering_config,
            )

            if feature_engineering_spec:
                # Persisté dans le bundle joblib — c'est lui, pas la base, que
                # `ml_inference.py` charge pour rejouer la partie amont à la
                # prédiction (voir `load_bundle`/`predict_one`).
                result.pipeline_bundle["feature_engineering_spec"] = feature_engineering_spec

            artifact_path = model_file_path(job.organization_id, job.id)
            joblib.dump(result.pipeline_bundle, artifact_path)

            # Schéma des colonnes d'entrée (nom + type) — dérivé du schéma déjà
            # calculé à l'upload du dataset (Lot 2), pour que le frontend puisse
            # générer un formulaire de prédiction sans redemander le dataset.
            # Basé sur `raw_feature_columns` : le formulaire de prédiction
            # demande toujours les colonnes saisies, jamais les dérivées.
            dataset_columns = json.loads(dataset.columns_json or "[]")
            feature_schema = [c for c in dataset_columns if c["name"] in raw_feature_columns]

            ml_model = MLModel(
                organization_id=job.organization_id,
                training_job_id=job.id,
                algorithm=result.algorithm,
                task_type=job.task_type,
                target_column=job.target_column,
                feature_columns_json=json.dumps(raw_feature_columns),
                feature_schema_json=json.dumps(feature_schema),
                file_path=str(artifact_path),
                metrics_json=json.dumps(result.metrics),
                shap_summary_json=json.dumps(result.shap_summary),
                cqr_json=json.dumps(result.cqr) if result.cqr else None,
                model_card_json=json.dumps(result.model_card),
                evaluation_json=json.dumps(result.evaluation),
                feature_engineering_json=json.dumps(feature_engineering_spec) if feature_engineering_spec else None,
            )
            db.add(ml_model)

            # Leaderboard (Lot D) — TOUS les candidats comparés par ce job,
            # pas seulement le gagnant (déjà ajouté ci-dessus via `ml_model`).
            # Même transaction que `ml_model`/`job.status` (un seul
            # `db.commit()` plus bas) : garantit que le candidat
            # `is_winner=True` et `ml_model` désignent TOUJOURS le même
            # modèle, par construction (les deux dérivent de `result.algorithm`
            # / `result.metrics["cv_score"]`, jamais recalculés séparément).
            for candidate in result.all_candidates:
                db.add(ModelCandidate(
                    organization_id=job.organization_id,
                    training_job_id=job.id,
                    algorithm=candidate["algorithm"],
                    family=candidate["family"],
                    selection_score=candidate["selection_score"],
                    is_winner=candidate["is_winner"],
                    rank=candidate["rank"],
                    fold_scores_json=json.dumps(candidate["fold_scores"]) if candidate["fold_scores"] is not None else None,
                    secondary_metric=candidate["secondary_metric"],
                    secondary_metric_label=candidate["secondary_metric_label"],
                ))

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
            job.error_message = _user_safe_error_message(exc)
            job.finished_at = datetime.now(timezone.utc)
            db.commit()
            # Détail technique complet (type, message brut, traceback) — JAMAIS
            # renvoyé à l'utilisateur (voir job.error_message ci-dessus), utile
            # uniquement en journal serveur pour diagnostiquer la cause réelle.
            logger.error("[Training] Job %s échoué : %s\n%s", job_id, exc, traceback.format_exc())

    finally:
        db.close()
