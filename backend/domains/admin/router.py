"""Router d'administration de la PLATEFORME — vue globale de l'éditeur.

Ce module est la SEULE porte du projet autorisée à lire au-delà d'une
organisation. Tout le reste de l'API filtre systématiquement par
`organization_id` (voir `domains/auth/router.py::list_team_members` et
chaque routeur métier) : deux organisations ne se voient jamais.

Cette exception est délibérée et confinée ici, dans un domaine séparé,
plutôt que dispersée en conditions `if user.is_platform_admin` au fil des
endpoints existants — une frontière de sécurité doit se voir dans
l'arborescence, pas se deviner en relisant vingt fichiers. Corollaire :
aucun endpoint métier n'a été assoupli pour ce besoin.

Toutes les routes sont en LECTURE SEULE. Un administrateur plateforme
observe ; il ne modifie ni les données ni les comptes d'une organisation
cliente. Agir sur une organisation resterait le geste de SON propriétaire
— ce qui évite qu'un accès de supervision devienne un passe-partout.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from api.core.database import get_db
from api.core.error_codes import ErrorCode
from api.core.models import (
    AuditLog,
    Dataset,
    MLModel,
    Organization,
    Prediction,
    User,
    VisionDataset,
)
from domains.auth.router import get_current_user
from domains.shared.job_quota import ALL_JOB_MODELS

router = APIRouter(prefix="/admin", tags=["administration plateforme"])


#: Libellé lisible par pilier, pour ne pas exposer des noms de classes
#: Python dans une interface. L'ordre fixe celui de l'affichage.
_JOB_LABELS = {
    "TrainingJob": "Entraînement supervisé",
    "ClusteringJob": "Clustering",
    "DimensionalityJob": "Réduction de dimension",
    "AnomalyJob": "Détection d'anomalies",
    "VisionClassificationJob": "Vision — classification",
    "VisionAnomalyJob": "Vision — anomalies",
    "BatchPredictionJob": "Prédiction en lot",
}

#: Statuts terminaux d'échec, communs aux 7 modèles de job du projet.
_FAILED_STATUSES = ("failed", "error")


def require_platform_admin(current_user: User = Depends(get_current_user)) -> User:
    """Réservé aux administrateurs de la plateforme (l'éditeur).

    403 et non 404 : contrairement aux ressources d'une autre organisation
    — dont on ne révèle jamais l'existence — l'existence d'un espace
    d'administration n'est pas un secret, et un 404 rendrait le diagnostic
    incompréhensible pour l'administrateur légitime qui aurait perdu son
    droit."""
    if not current_user.is_platform_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": ErrorCode.AUTH_ADMIN_PLATEFORME_REQUIS,
                "message": "Espace réservé à l'administration de la plateforme",
            },
        )
    return current_user


# ── Schémas ──────────────────────────────────────────────────────────────────


class PlatformCounters(BaseModel):
    organizations: int
    users_total: int
    users_active: int
    users_revoked: int
    users_anonymized: int
    users_pending_password: int
    datasets: int
    vision_datasets: int
    datasets_bytes: int
    models: int
    predictions: int


class JobsByPillar(BaseModel):
    pillar: str
    label: str
    total: int
    running: int
    queued: int
    failed: int
    completed: int


class TimeseriesPoint(BaseModel):
    date: str  # AAAA-MM-JJ
    count: int


class PlatformOverview(BaseModel):
    counters: PlatformCounters
    jobs_by_pillar: List[JobsByPillar]
    jobs_total: int
    jobs_failed: int
    #: Part des jobs terminés en échec, sur l'ensemble des jobs terminés.
    #: `None` si aucun job terminé — jamais 0, qui se lirait à tort comme
    #: « aucune panne » alors qu'il n'y a simplement rien à mesurer.
    failure_rate: Optional[float]
    jobs_per_day: List[TimeseriesPoint]
    signups_per_day: List[TimeseriesPoint]
    window_days: int


class OrganizationRow(BaseModel):
    id: int
    name: str
    created_at: datetime
    members: int
    active_members: int
    datasets: int
    jobs: int
    last_activity_at: Optional[datetime]


class PlatformUserRow(BaseModel):
    id: int
    email: str
    nom: str
    role: str
    organization_id: int
    organization_name: str
    actif: bool
    is_platform_admin: bool
    must_change_password: bool
    created_at: datetime
    last_login: Optional[datetime]
    deactivated_at: Optional[datetime]
    anonymized_at: Optional[datetime]


class PlatformAuditRow(BaseModel):
    id: int
    action: str
    organization_id: int
    organization_name: str
    actor_name: Optional[str]
    target_type: Optional[str]
    target_id: Optional[int]
    details: Optional[dict]
    created_at: datetime


# ── Aides ────────────────────────────────────────────────────────────────────


def _job_status_counts(db: Session) -> tuple[List[JobsByPillar], int, int, int]:
    """Compte les jobs par pilier ET par statut, en 7 requêtes groupées —
    une par modèle, jamais une par (modèle × statut), qui en ferait 35.

    Réutilise `ALL_JOB_MODELS` (`domains/shared/job_quota.py`) plutôt que de
    redresser la liste ici : un 8ᵉ pilier ajouté au projet apparaîtra
    automatiquement dans cette vue, sans qu'on ait à y penser."""
    rows: List[JobsByPillar] = []
    total = failed_total = finished_total = 0

    for model in ALL_JOB_MODELS:
        name = model.__name__
        counts: dict[str, int] = defaultdict(int)
        for status_value, count in db.query(model.status, func.count(model.id)).group_by(model.status).all():
            counts[status_value] = count

        model_total = sum(counts.values())
        model_failed = sum(counts[s] for s in _FAILED_STATUSES)
        model_completed = counts["completed"]
        total += model_total
        failed_total += model_failed
        finished_total += model_failed + model_completed

        rows.append(JobsByPillar(
            pillar=name,
            label=_JOB_LABELS.get(name, name),
            total=model_total,
            running=counts["running"],
            queued=counts["queued"],
            failed=model_failed,
            completed=model_completed,
        ))
    return rows, total, failed_total, finished_total


def _daily_counts(db: Session, model: Any, since: datetime, days: int) -> List[TimeseriesPoint]:
    """Série journalière sur `days` jours, TROUS INCLUS.

    Un `GROUP BY date` ne renvoie que les jours où il s'est passé quelque
    chose : un graphique construit dessus relierait deux points distants
    d'une semaine par une ligne droite, laissant croire à une activité
    continue. Les jours vides sont donc explicitement remplis à zéro."""
    per_day: dict[str, int] = defaultdict(int)
    for (created_at,) in db.query(model.created_at).filter(model.created_at >= since).all():
        if created_at is not None:
            per_day[created_at.strftime("%Y-%m-%d")] += 1

    start = since.date()
    return [
        TimeseriesPoint(
            date=(start + timedelta(days=offset)).strftime("%Y-%m-%d"),
            count=per_day.get((start + timedelta(days=offset)).strftime("%Y-%m-%d"), 0),
        )
        for offset in range(days)
    ]


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.get("/overview", response_model=PlatformOverview)
def get_platform_overview(
    window_days: int = Query(30, ge=7, le=180),
    _admin: User = Depends(require_platform_admin),
    db: Session = Depends(get_db),
):
    """Vue d'ensemble chiffrée de toute la plateforme, toutes organisations
    confondues."""
    since = datetime.now(timezone.utc) - timedelta(days=window_days - 1)
    since = since.replace(hour=0, minute=0, second=0, microsecond=0)

    jobs_by_pillar, jobs_total, jobs_failed, jobs_finished = _job_status_counts(db)

    counters = PlatformCounters(
        organizations=db.query(func.count(Organization.id)).scalar() or 0,
        users_total=db.query(func.count(User.id)).scalar() or 0,
        users_active=db.query(func.count(User.id)).filter(User.actif.is_(True)).scalar() or 0,
        users_revoked=db.query(func.count(User.id)).filter(
            User.actif.is_(False), User.anonymized_at.is_(None)
        ).scalar() or 0,
        users_anonymized=db.query(func.count(User.id)).filter(User.anonymized_at.isnot(None)).scalar() or 0,
        users_pending_password=db.query(func.count(User.id)).filter(
            User.must_change_password.is_(True), User.actif.is_(True)
        ).scalar() or 0,
        datasets=db.query(func.count(Dataset.id)).scalar() or 0,
        vision_datasets=db.query(func.count(VisionDataset.id)).scalar() or 0,
        datasets_bytes=int(db.query(func.coalesce(func.sum(Dataset.file_size_bytes), 0)).scalar() or 0),
        models=db.query(func.count(MLModel.id)).scalar() or 0,
        predictions=db.query(func.count(Prediction.id)).scalar() or 0,
    )

    # Activité quotidienne : les entrées d'audit couvrent TOUS les piliers
    # d'un seul tenant, là où additionner 7 séries de jobs donnerait la même
    # courbe au prix de 7 requêtes.
    return PlatformOverview(
        counters=counters,
        jobs_by_pillar=jobs_by_pillar,
        jobs_total=jobs_total,
        jobs_failed=jobs_failed,
        failure_rate=(jobs_failed / jobs_finished) if jobs_finished else None,
        jobs_per_day=_daily_counts(db, AuditLog, since, window_days),
        signups_per_day=_daily_counts(db, User, since, window_days),
        window_days=window_days,
    )


@router.get("/organizations", response_model=List[OrganizationRow])
def list_all_organizations(
    _admin: User = Depends(require_platform_admin),
    db: Session = Depends(get_db),
):
    """Toutes les organisations, avec leur volumétrie et leur dernière
    activité connue — de quoi repérer d'un coup d'œil une organisation
    dormante ou au contraire très active."""
    # Dictionnaires construits explicitement plutôt que par `dict(query.all())` :
    # SQLAlchemy renvoie des `Row`, que le typage ne sait pas réduire à un
    # couple clé/valeur — l'annotation rend l'intention lisible et vérifiable.
    members: dict[int, int] = {
        org_id: count
        for org_id, count in db.query(User.organization_id, func.count(User.id))
        .group_by(User.organization_id)
        .all()
    }
    active_members: dict[int, int] = {
        org_id: count
        for org_id, count in db.query(User.organization_id, func.count(User.id))
        .filter(User.actif.is_(True))
        .group_by(User.organization_id)
        .all()
    }
    datasets: dict[int, int] = {
        org_id: count
        for org_id, count in db.query(Dataset.organization_id, func.count(Dataset.id))
        .group_by(Dataset.organization_id)
        .all()
    }
    last_activity: dict[int, datetime] = {
        org_id: seen_at
        for org_id, seen_at in db.query(AuditLog.organization_id, func.max(AuditLog.created_at))
        .group_by(AuditLog.organization_id)
        .all()
    }

    jobs: dict[int, int] = defaultdict(int)
    for model in ALL_JOB_MODELS:
        for org_id, count in db.query(model.organization_id, func.count(model.id)).group_by(
            model.organization_id
        ).all():
            jobs[org_id] += count

    return [
        OrganizationRow(
            id=org.id,
            name=org.name,
            created_at=org.created_at,
            members=members.get(org.id, 0),
            active_members=active_members.get(org.id, 0),
            datasets=datasets.get(org.id, 0),
            jobs=jobs.get(org.id, 0),
            last_activity_at=last_activity.get(org.id),
        )
        for org in db.query(Organization).order_by(Organization.created_at.desc()).all()
    ]


@router.get("/users", response_model=List[PlatformUserRow])
def list_all_users(
    _admin: User = Depends(require_platform_admin),
    db: Session = Depends(get_db),
):
    """Tous les comptes de la plateforme, avec leur état complet.

    Les comptes anonymisés apparaissent tels quels (« Utilisateur
    supprimé ») : leur ligne existe toujours puisqu'elle porte des
    productions, et la masquer donnerait un décompte incohérent avec la vue
    d'ensemble."""
    rows = (
        db.query(User, Organization.name)
        .join(Organization, Organization.id == User.organization_id)
        .order_by(User.created_at.desc())
        .all()
    )
    return [
        PlatformUserRow(
            id=user.id,
            email=user.email,
            nom=user.nom,
            role=user.role,
            organization_id=user.organization_id,
            organization_name=org_name,
            actif=user.actif,
            is_platform_admin=user.is_platform_admin,
            must_change_password=user.must_change_password,
            created_at=user.created_at,
            last_login=user.last_login,
            deactivated_at=user.deactivated_at,
            anonymized_at=user.anonymized_at,
        )
        for user, org_name in rows
    ]


@router.get("/activity", response_model=List[PlatformAuditRow])
def list_platform_activity(
    limit: int = Query(100, ge=1, le=500),
    _admin: User = Depends(require_platform_admin),
    db: Session = Depends(get_db),
):
    """Journal d'audit de TOUTES les organisations, du plus récent au plus
    ancien. C'est le fil d'activité réel de la plateforme : il couvre les
    7 piliers et les gestes d'équipe, là où compter les jobs ne montrerait
    que l'entraînement."""
    rows = (
        db.query(AuditLog, Organization.name, User.nom)
        .join(Organization, Organization.id == AuditLog.organization_id)
        .outerjoin(User, User.id == AuditLog.actor_id)
        .order_by(AuditLog.created_at.desc())
        .limit(limit)
        .all()
    )

    entries: List[PlatformAuditRow] = []
    for entry, org_name, actor_name in rows:
        details: Optional[dict] = None
        if entry.details_json:
            try:
                parsed = json.loads(entry.details_json)
                details = parsed if isinstance(parsed, dict) else None
            except (ValueError, TypeError):
                details = None  # entrée illisible : jamais devinée
        entries.append(PlatformAuditRow(
            id=entry.id,
            action=entry.action,
            organization_id=entry.organization_id,
            organization_name=org_name,
            actor_name=actor_name,
            target_type=entry.target_type,
            target_id=entry.target_id,
            details=details,
            created_at=entry.created_at,
        ))
    return entries
