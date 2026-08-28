"""Router classification d'images — pilier Vision, Lot 15 sous-lots B et D.

Mêmes principes que `api/routers/anomalies.py` : isolation systématique par
`organization_id`, tâche de fond obligatoire (RQ, `vision_queue` — file
dédiée aux jobs vision (torch), séparée de `training_queue`/
`analysis_queue` depuis le correctif I6, voir `api/core/job_queue.py`)
pour l'entraînement, jamais de calcul ML dans la requête HTTP pour un
job. Le dataset source doit être un `VisionDataset` de
structure "classification" — vérifié ici ET dans le worker (défense en
profondeur, même principe que la validation dataset des autres routers).

L'endpoint `/explain` (sous-lot D, Grad-CAM) fait exception à "jamais de
calcul ML dans la requête HTTP" : une seule image, un seul forward+backward
pass sur un modèle déjà entraîné — de l'ordre de la seconde, même
raisonnement que `POST /training/jobs/{id}/predict` (inférence + SHAP local,
`api/routers/training.py`), synchrone lui aussi."""
from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

import torch
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session, joinedload

from api.core.config import get_settings
from api.core.database import SessionLocal, get_db
from api.core.job_queue import redis_conn, vision_queue
from api.core.models import User, VisionClassificationJob, VisionDataset
from api.core.pagination import paginate_by_id
from api.core.rate_limit import rate_limit_dependency
from domains.auth.router import get_current_user
from domains.shared.audit import log_action
from domains.shared.job_creation import enqueue_or_mark_failed, remember_idempotent_job_id, resolve_idempotent_job_id
from domains.shared.job_events import stream_job_updates
from domains.shared.job_lifecycle import ACTIVE_STATUSES, CANCELLED_MESSAGE, try_cancel_rq_job
from domains.shared.job_quota import ALL_JOB_MODELS, raise_if_quota_exceeded
from domains.shared.job_watchdog import reconcile_stale_jobs
from domains.vision.classification.services.engine import (
    AUGMENTATION_PRESET_IDS,
    DEFAULT_AUGMENTATION_PRESET,
    IMAGE_SIZE,
    MAX_BACKBONES_PER_COMPARISON,
)
from domains.vision.shared import ALLOWED_IMAGE_SIZES
from domains.vision.classification.services.gradcam import (
    GradCamError,
    explain_classification_prediction,
    explain_classification_predictions_batch,
)
from domains.vision.classification.services.registry import CLASSIFICATION_BACKBONE_REGISTRY, DEFAULT_BACKBONE_ID

router = APIRouter(prefix="/vision/classification", tags=["vision"])
_settings = get_settings()

_VALID_BACKBONE_IDS = {s.id for s in CLASSIFICATION_BACKBONE_REGISTRY}


# ── Schémas ──────────────────────────────────────────────────────────────


class VisionClassificationJobCreate(BaseModel):
    vision_dataset_id: int
    backbone_id: str = DEFAULT_BACKBONE_ID
    # Mode expert (retour utilisateur direct : "tavais dit taller fire aussi
    # pareil" côté vision — parité avec `model_ids` du ML tabulaire) — quand
    # fourni (≥ 2 entrées), remplace `backbone_id` : chaque backbone listé
    # est entraîné avec les mêmes autres réglages, le meilleur sur la
    # validation est automatiquement retenu (voir
    # `services/engine.py::train_and_compare_backbones`). None (défaut) :
    # comportement historique inchangé, un seul backbone (`backbone_id`).
    backbone_ids: Optional[List[str]] = None
    # Mode expert (retour utilisateur direct : "vision n'offre pas de
    # réduire/augmenter la taille des images") — voir
    # domains/vision/shared.py::ALLOWED_IMAGE_SIZES.
    image_size: int = IMAGE_SIZE
    num_epochs: int = Field(default=8, ge=1, le=30)
    batch_size: int = Field(default=16, ge=1, le=128)
    learning_rate: float = Field(default=1e-3, gt=0, le=1)
    # Régularisation L2 (mode expert) — voir
    # services/engine.py::ClassificationConfig.weight_decay.
    weight_decay: float = Field(default=0.0, ge=0, le=0.1)
    dropout_rate: float = Field(default=0.3, ge=0, le=0.9)
    freeze_backbone: bool = True
    unfreeze_after_epoch: Optional[int] = Field(default=None, ge=0)
    seed: Optional[int] = None
    # Lot 6A (correctif I8, AUDIT_DATALAB_2026-08-16.md §I8) — voir
    # services/vision_classification_training.py::ClassificationConfig
    # pour la justification de chaque défaut.
    class_weighting: bool = True
    early_stopping_patience: Optional[int] = Field(default=3, ge=1, le=10)
    use_lr_scheduler: bool = True
    # Lot 6A (correctif I9) — voir AUGMENTATION_PRESET_IDS.
    augmentation_preset: str = DEFAULT_AUGMENTATION_PRESET
    # Répartition personnalisée (Lot 6A, étape "Répartition") — voir
    # services/vision_classification_training.py::ClassificationConfig pour
    # la validation complète (>=10 % train, bornes ici juste pour rejeter
    # une valeur absurde avant même d'enfiler le job).
    val_ratio: float = Field(default=0.15, gt=0, lt=0.9)
    test_ratio: float = Field(default=0.15, gt=0, lt=0.9)


class BackboneOut(BaseModel):
    id: str
    label: str


class VisionClassificationJobSummary(BaseModel):
    id: int
    vision_dataset_id: int
    vision_dataset_name: Optional[str] = None
    backbone_id: str
    status: str
    progress_step: Optional[str] = None
    progress_percent: int
    error_message: Optional[str] = None
    created_by: Optional[str] = None
    created_at: Any
    started_at: Optional[Any] = None
    finished_at: Optional[Any] = None
    test_accuracy: Optional[float] = None


class EpochMetricsOut(BaseModel):
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float


class PredictionExampleOut(BaseModel):
    relative_path: str
    true_label: str
    predicted_label: str
    confidence: float
    correct: bool


class GradCamExplanationOut(BaseModel):
    predicted_label: str
    probabilities: dict[str, float]
    target_label: str
    heatmap_png: str


# Retour utilisateur direct : "Grad-CAM devrait supporter le batch, pas une
# image à la fois" — plafond volontairement modeste (voir
# `_explain_rate_limit` : "l'endpoint le plus coûteux du backend") pour
# qu'un seul appel batch ne puisse pas à lui seul consommer tout le quota
# horaire (`explain_rate_limit_max_attempts`, 20/heure par défaut) sans
# supprimer l'intérêt du plafond.
MAX_EXPLAIN_BATCH_SIZE = 12


class ExplainDatasetExamplesRequest(BaseModel):
    relative_paths: List[str] = Field(min_length=1, max_length=MAX_EXPLAIN_BATCH_SIZE)


class GradCamBatchItemOut(BaseModel):
    relative_path: str
    predicted_label: Optional[str] = None
    probabilities: Optional[dict[str, float]] = None
    target_label: Optional[str] = None
    heatmap_png: Optional[str] = None
    # `None` = expliquée avec succès ; sinon message actionnable — jamais un
    # batch tout-ou-rien silencieux sur les images qui ont échoué.
    error: Optional[str] = None


class ExplainDatasetExamplesResponse(BaseModel):
    results: List[GradCamBatchItemOut]


class VisionClassificationResultOut(BaseModel):
    backbone_id: str
    class_names: List[str]
    n_train: int
    n_val: int
    n_test: int
    history: List[EpochMetricsOut]
    test_accuracy: float
    test_precision_macro: float
    test_recall_macro: float
    test_f1_macro: float
    confusion_matrix: List[List[int]]
    examples: List[PredictionExampleOut]
    model_card: dict[str, Any]
    # Lot 6A (correctif 16G) — None sur les modèles entraînés avant ce
    # correctif (rétrocompatibilité par absence).
    roc_curves: Optional[dict[str, dict[str, List[float]]]] = None
    pr_curves: Optional[dict[str, dict[str, List[float]]]] = None
    test_roc_auc: Optional[float] = None
    # Onglet "Fiabilité" — None sur les modèles entraînés avant ce correctif
    # (rétrocompatibilité par absence, même motif que roc_curves ci-dessus).
    calibration: Optional[dict[str, dict[str, List[float]]]] = None


# ── Aides ────────────────────────────────────────────────────────────────


def _get_org_job(job_id: int, current_user: User, db: Session) -> VisionClassificationJob:
    job = (
        db.query(VisionClassificationJob)
        .filter(
            VisionClassificationJob.id == job_id,
            VisionClassificationJob.organization_id == current_user.organization_id,
        )
        .first()
    )
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "VISION_CLASSIFICATION_JOB_INTROUVABLE", "message": "Entraînement de classification introuvable"},
        )
    return job


def to_summary(job: VisionClassificationJob) -> VisionClassificationJobSummary:
    config = json.loads(job.config_json)
    result = job.result
    return VisionClassificationJobSummary(
        id=job.id,
        vision_dataset_id=job.vision_dataset_id,
        vision_dataset_name=job.vision_dataset.name if job.vision_dataset else None,
        backbone_id=config.get("backbone_id", DEFAULT_BACKBONE_ID),
        status=job.status,
        progress_step=job.progress_step,
        progress_percent=job.progress_percent,
        error_message=job.error_message,
        created_by=job.created_by.nom if job.created_by else None,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        test_accuracy=result.test_accuracy if result else None,
    )


# ── Endpoints ────────────────────────────────────────────────────────────


@router.get("/backbones", response_model=List[BackboneOut])
def list_backbones():
    return [BackboneOut(id=s.id, label=s.label) for s in CLASSIFICATION_BACKBONE_REGISTRY]


@router.post("/jobs", response_model=VisionClassificationJobSummary, status_code=status.HTTP_201_CREATED)
def create_vision_classification_job(
    body: VisionClassificationJobCreate,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Idempotence (Phase 2, AUDIT_BACKEND_2026-08-23.md §F4) — voir
    # domains/training/router.py::create_training_job pour la justification.
    existing_job_id = resolve_idempotent_job_id(redis_conn, current_user.organization_id, request)
    if existing_job_id is not None:
        existing = (
            db.query(VisionClassificationJob)
            .filter(
                VisionClassificationJob.id == existing_job_id,
                VisionClassificationJob.organization_id == current_user.organization_id,
            )
            .first()
        )
        if existing is not None:
            return to_summary(existing)

    if body.backbone_id not in _VALID_BACKBONE_IDS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "BACKBONE_INCONNU", "message": f"Backbone inconnu : {body.backbone_id}"},
        )
    if body.backbone_ids is not None:
        if len(body.backbone_ids) < 2 or len(body.backbone_ids) > MAX_BACKBONES_PER_COMPARISON:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": "COMPARATIF_BACKBONES_INVALIDE",
                    "message": f"Comparatif : entre 2 et {MAX_BACKBONES_PER_COMPARISON} backbones requis",
                },
            )
        if len(set(body.backbone_ids)) != len(body.backbone_ids):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "COMPARATIF_BACKBONES_INVALIDE", "message": "Backbones en double dans le comparatif"},
            )
        unknown = [b for b in body.backbone_ids if b not in _VALID_BACKBONE_IDS]
        if unknown:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "BACKBONE_INCONNU", "message": f"Backbone(s) inconnu(s) : {', '.join(unknown)}"},
            )
    if body.augmentation_preset not in AUGMENTATION_PRESET_IDS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "AUGMENTATION_PRESET_INCONNU",
                "message": f"Preset d'augmentation inconnu : {body.augmentation_preset!r}",
            },
        )
    if body.image_size not in ALLOWED_IMAGE_SIZES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "TAILLE_IMAGE_INCONNUE",
                "message": f"Taille d'image invalide : {body.image_size} (voir {ALLOWED_IMAGE_SIZES})",
            },
        )
    if body.val_ratio + body.test_ratio >= 0.9:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "REPARTITION_INVALIDE",
                "message": "Validation et test doivent laisser au moins 10 % des images pour l'entraînement",
            },
        )

    reconcile_stale_jobs(
        db, current_user.organization_id, _settings.stale_job_timeout_minutes, model=VisionClassificationJob
    )
    raise_if_quota_exceeded(db, current_user.organization_id, ALL_JOB_MODELS, _settings.max_concurrent_jobs_per_org)

    dataset = (
        db.query(VisionDataset)
        .filter(VisionDataset.id == body.vision_dataset_id, VisionDataset.organization_id == current_user.organization_id)
        .first()
    )
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "VISION_DATASET_INTROUVABLE", "message": "Dataset d'images introuvable"},
        )
    if dataset.status != "ready":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "VISION_DATASET_NON_PRET", "message": "Ce dataset n'a pas pu être validé"},
        )
    if dataset.structure_type != "classification":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "VISION_DATASET_STRUCTURE_INVALIDE",
                "message": "Ce dataset n'a pas une structure de classification (dossiers de classes)",
            },
        )

    config = {
        "backbone_id": body.backbone_id,
        "image_size": body.image_size,
        "num_epochs": body.num_epochs,
        "batch_size": body.batch_size,
        "learning_rate": body.learning_rate,
        "weight_decay": body.weight_decay,
        "dropout_rate": body.dropout_rate,
        "freeze_backbone": body.freeze_backbone,
        "unfreeze_after_epoch": body.unfreeze_after_epoch,
        "seed": body.seed if body.seed is not None else _settings.model_seed,
        "class_weighting": body.class_weighting,
        "early_stopping_patience": body.early_stopping_patience,
        "use_lr_scheduler": body.use_lr_scheduler,
        "augmentation_preset": body.augmentation_preset,
        "val_ratio": body.val_ratio,
        "test_ratio": body.test_ratio,
    }
    if body.backbone_ids is not None:
        config["backbone_ids"] = body.backbone_ids

    job = VisionClassificationJob(
        organization_id=current_user.organization_id,
        vision_dataset_id=dataset.id,
        created_by_id=current_user.id,
        config_json=json.dumps(config),
        status="queued",
        request_id=request.state.request_id,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    remember_idempotent_job_id(redis_conn, current_user.organization_id, request, job.id)
    log_action(
        db, current_user.organization_id, current_user.id, "vision_classification_job.created",
        target_type="vision_classification_job", target_id=job.id,
    )

    from domains.vision.classification.worker import run_vision_classification_job

    enqueue_or_mark_failed(db, job, vision_queue, run_vision_classification_job, 1800)

    return to_summary(job)


@router.get("/jobs", response_model=List[VisionClassificationJobSummary])
def list_vision_classification_jobs(
    response: Response,
    limit: Optional[int] = Query(None, ge=1, le=500),
    cursor: Optional[int] = Query(None, description="id de la dernière ligne de la page précédente"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # joinedload (Lot 4, correctif I3) — voir training.py::list_training_jobs.
    query = (
        db.query(VisionClassificationJob)
        .options(
            joinedload(VisionClassificationJob.vision_dataset),
            joinedload(VisionClassificationJob.created_by),
            joinedload(VisionClassificationJob.result),
        )
        .filter(VisionClassificationJob.organization_id == current_user.organization_id)
        # id DESC, pas created_at DESC — voir anomalies.py::list_anomaly_jobs.
        .order_by(VisionClassificationJob.id.desc())
    )
    jobs = paginate_by_id(query, VisionClassificationJob.id, response, cursor, limit)
    return [to_summary(j) for j in jobs]


@router.get("/jobs/{job_id}", response_model=VisionClassificationJobSummary)
def get_vision_classification_job(job_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return to_summary(_get_org_job(job_id, current_user, db))


@router.get("/jobs/{job_id}/events")
async def stream_vision_classification_job_events(job_id: int, current_user: User = Depends(get_current_user)):
    """Notifications de fin de job par SSE (Lot 7, §J.2) — voir
    `training.py::stream_training_job_events` pour le raisonnement complet."""
    organization_id = current_user.organization_id
    db = SessionLocal()
    try:
        job = (
            db.query(VisionClassificationJob)
            .filter(VisionClassificationJob.id == job_id, VisionClassificationJob.organization_id == organization_id)
            .first()
        )
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "VISION_CLASSIFICATION_JOB_INTROUVABLE", "message": "Entraînement de classification introuvable"},
            )
    finally:
        db.close()

    def fetch_snapshot():
        session = SessionLocal()
        try:
            row = (
                session.query(VisionClassificationJob)
                .filter(VisionClassificationJob.id == job_id, VisionClassificationJob.organization_id == organization_id)
                .first()
            )
            if row is None:
                return None
            return {
                "status": row.status,
                "progress_percent": row.progress_percent,
                "progress_step": row.progress_step,
                "error_message": row.error_message,
            }
        finally:
            session.close()

    return StreamingResponse(stream_job_updates(fetch_snapshot), media_type="text/event-stream")


@router.get("/jobs/{job_id}/result", response_model=VisionClassificationResultOut)
def get_vision_classification_result(job_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    job = _get_org_job(job_id, current_user, db)
    if job.result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "RESULTAT_INDISPONIBLE", "message": "Cet entraînement n'a pas encore de résultat"},
        )
    result = job.result
    return VisionClassificationResultOut(
        backbone_id=result.backbone_id,
        class_names=json.loads(result.class_names_json),
        n_train=result.n_train,
        n_val=result.n_val,
        n_test=result.n_test,
        history=[EpochMetricsOut(**m) for m in json.loads(result.history_json)],
        test_accuracy=result.test_accuracy,
        test_precision_macro=result.test_precision_macro,
        test_recall_macro=result.test_recall_macro,
        test_f1_macro=result.test_f1_macro,
        confusion_matrix=json.loads(result.confusion_matrix_json),
        examples=[PredictionExampleOut(**e) for e in json.loads(result.examples_json)],
        model_card=json.loads(result.model_card_json or "{}"),
        roc_curves=json.loads(result.roc_curves_json) if result.roc_curves_json else None,
        pr_curves=json.loads(result.pr_curves_json) if result.pr_curves_json else None,
        test_roc_auc=result.test_roc_auc,
        calibration=json.loads(result.calibration_json) if result.calibration_json else None,
    )


@router.get("/jobs/{job_id}/model/export")
def export_vision_classification_model(job_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Export de l'artefact (Lot 10, retour utilisateur direct — parité avec
    `GET /training/jobs/{job_id}/model/export`) : les poids du réseau entraîné,
    pour un déploiement hors de la plateforme (`torch.load` avec la même
    architecture de backbone)."""
    job = _get_org_job(job_id, current_user, db)
    if job.status != "completed" or job.result is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "MODELE_NON_DISPONIBLE", "message": "Cet entraînement n'a pas encore produit de modèle"},
        )
    artifact_path = Path(job.result.file_path)
    if not artifact_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "ARTEFACT_INTROUVABLE", "message": "Artefact du modèle introuvable sur le serveur"},
        )
    filename = f"vision_classification_job{job.id}.pt"
    return FileResponse(path=artifact_path, filename=filename, media_type="application/octet-stream")


_explain_rate_limit = rate_limit_dependency(
    "vision_explain", _settings.explain_rate_limit_max_attempts, _settings.explain_rate_limit_window_seconds,
    # Échec FERMÉ (Phase 1, AUDIT_BACKEND_2026-08-23.md §4, constat déjà
    # identifié dans l'audit initial) — charge un modèle torch complet à
    # chaque appel, l'endpoint le plus coûteux du backend : qui met Redis à
    # genoux ne doit jamais le déverrouiller.
    fail_open=False,
)


@router.post(
    "/jobs/{job_id}/explain", response_model=GradCamExplanationOut, dependencies=[Depends(_explain_rate_limit)]
)
async def explain_vision_classification_prediction(
    job_id: int,
    file: UploadFile = File(...),
    target_label: Optional[str] = Form(default=None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Grad-CAM (sous-lot D) — pourquoi ce modèle prédit cette classe pour
    cette image précise. Synchrone (voir docstring du module) : un seul
    forward+backward pass sur un modèle déjà entraîné."""
    job = _get_org_job(job_id, current_user, db)
    if job.result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "RESULTAT_INDISPONIBLE", "message": "Cet entraînement n'a pas encore de résultat"},
        )

    content = await file.read()
    try:
        image = Image.open(io.BytesIO(content))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "IMAGE_INVALIDE", "message": "Impossible de lire cette image"},
        )

    # Lot 1.4 (§C.2.7/R11, AUDIT_DATALAB_2026-08-16.md) — weights_only=True
    # (au lieu de False) : le fichier n'est aujourd'hui écrit que par notre
    # propre worker, jamais par un import utilisateur, mais un pickle non
    # restreint reste une bombe à retardement le jour où ça changerait.
    # Vérifié empiriquement : l'artefact ({"backbone_id": str, "class_names":
    # list[str], "dropout_rate": float, "state_dict": OrderedDict[str,
    # Tensor]} — voir vision_classification_training.py) ne contient que des
    # types couverts par l'allowlist par défaut de weights_only=True (torch
    # 2.13), donc PAS besoin de séparer state_dict et métadonnées en deux
    # fichiers — juste ce changement suffit à fermer le risque.
    artifact = torch.load(job.result.file_path, weights_only=True)
    try:
        explanation = explain_classification_prediction(artifact, image, target_label)
    except GradCamError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "CLASSE_INCONNUE", "message": str(exc)},
        )

    return GradCamExplanationOut(
        predicted_label=explanation.predicted_label,
        probabilities=explanation.probabilities,
        target_label=explanation.target_label,
        heatmap_png=explanation.heatmap_png,
    )


@router.post(
    "/jobs/{job_id}/explain-dataset-examples",
    response_model=ExplainDatasetExamplesResponse,
    dependencies=[Depends(_explain_rate_limit)],
)
def explain_vision_classification_dataset_examples(
    job_id: int,
    body: ExplainDatasetExamplesRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Grad-CAM en LOT sur des images déjà présentes dans le dataset
    d'entraînement (retour utilisateur direct : "Grad-CAM devrait supporter
    le batch, pas une image à la fois") — typiquement les exemples mal
    classés déjà affichés dans l'onglet "Exemples" : zéro ré-upload, un clic
    "Expliquer" directement sur une erreur, ou plusieurs à la fois. Modèle
    chargé UNE SEULE FOIS pour tout le lot (voir
    `services/vision_gradcam.py::explain_classification_predictions_batch`),
    jamais une fois par image.

    Complémentaire de `/explain` (upload libre, ex. tester une image externe
    au dataset) — celui-ci reste inchangé pour ce cas d'usage."""
    job = _get_org_job(job_id, current_user, db)
    if job.result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "RESULTAT_INDISPONIBLE", "message": "Cet entraînement n'a pas encore de résultat"},
        )

    # Même garde-fou anti-traversée de répertoire que
    # `vision/datasets/router.py::get_vision_dataset_image` — un chemin ne
    # peut désigner qu'un fichier réellement SOUS le dossier de CE dataset.
    base_dir = Path(job.vision_dataset.storage_dir).resolve()
    images: list[tuple[str, Image.Image]] = []
    missing: list[str] = []
    for relative_path in body.relative_paths:
        target = (base_dir / relative_path).resolve()
        if base_dir not in target.parents or not target.is_file():
            missing.append(relative_path)
            continue
        try:
            image = Image.open(target)
            image.load()
            images.append((relative_path, image))
        except (UnidentifiedImageError, OSError):
            missing.append(relative_path)

    artifact = torch.load(job.result.file_path, weights_only=True)
    try:
        batch_results = explain_classification_predictions_batch(artifact, images)
    finally:
        for _, opened_image in images:
            opened_image.close()

    by_key = {r.key: r for r in batch_results}
    results: list[GradCamBatchItemOut] = []
    for relative_path in body.relative_paths:
        if relative_path in missing:
            results.append(GradCamBatchItemOut(relative_path=relative_path, error="Image introuvable dans ce dataset"))
            continue
        item = by_key[relative_path]
        if item.error is not None or item.result is None:
            results.append(
                GradCamBatchItemOut(relative_path=relative_path, error=item.error or "Explication indisponible")
            )
        else:
            results.append(
                GradCamBatchItemOut(
                    relative_path=relative_path,
                    predicted_label=item.result.predicted_label,
                    probabilities=item.result.probabilities,
                    target_label=item.result.target_label,
                    heatmap_png=item.result.heatmap_png,
                )
            )
    return ExplainDatasetExamplesResponse(results=results)


@router.post("/jobs/{job_id}/cancel", response_model=VisionClassificationJobSummary)
def cancel_vision_classification_job(job_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Annule un entraînement Vision (classification) en attente ou en
    cours (Lot 7, §J.2) — garde une trace consultable
    (`status="cancelled"`)."""
    job = _get_org_job(job_id, current_user, db)
    if job.status not in ACTIVE_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "JOB_NON_ANNULABLE", "message": "Cet entraînement n'est plus en attente ni en cours"},
        )
    try_cancel_rq_job(job.rq_job_id, vision_queue)
    job.status = "cancelled"
    job.error_message = CANCELLED_MESSAGE
    job.finished_at = datetime.now(timezone.utc)
    log_action(
        db, current_user.organization_id, current_user.id, "vision_classification_job.cancelled",
        target_type="vision_classification_job", target_id=job.id, details={"vision_dataset_id": job.vision_dataset_id},
    )
    db.commit()
    db.refresh(job)
    return to_summary(job)


@router.post("/jobs/{job_id}/rerun", response_model=VisionClassificationJobSummary, status_code=status.HTTP_201_CREATED)
def rerun_vision_classification_job(
    job_id: int, request: Request, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """Relance un entraînement Vision (classification) avec EXACTEMENT la
    même configuration (Lot 7, §J.2) — `config_json` reprend ici tous les
    champs de `VisionClassificationJobCreate` à l'identique (voir sa
    construction dans `create_vision_classification_job`), dépaquetage
    direct sans reconstruction champ par champ."""
    job = _get_org_job(job_id, current_user, db)
    config = json.loads(job.config_json)
    body = VisionClassificationJobCreate(vision_dataset_id=job.vision_dataset_id, **config)
    # Arguments nommés (Phase 2, AUDIT_BACKEND_2026-08-23.md) — voir
    # domains/training/router.py::rerun_training_job pour le bug réel que
    # cela corrige.
    return create_vision_classification_job(body=body, request=request, current_user=current_user, db=db)


@router.delete("/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_vision_classification_job(job_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    job = _get_org_job(job_id, current_user, db)

    if job.status in ACTIVE_STATUSES:
        try_cancel_rq_job(job.rq_job_id, vision_queue)

    if job.result:
        if job.result.file_path:
            Path(job.result.file_path).unlink(missing_ok=True)
        db.delete(job.result)

    log_action(
        db, current_user.organization_id, current_user.id, "vision_classification_job.deleted",
        target_type="vision_classification_job", target_id=job.id,
        details={"vision_dataset_id": job.vision_dataset_id, "status": job.status},
    )
    db.delete(job)
    db.commit()
