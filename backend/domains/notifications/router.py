"""Router notifications (retour utilisateur : "notifications de fin de job
— email/navigateur") — personnelles (scopées par `user_id`, pas seulement
`organization_id` : un job lancé par un collègue ne doit jamais notifier
tout le monde). Créées par les workers RQ (voir
`domains/shared/notifications.py::notify_job_terminal`), jamais dans ce
router — ici, seulement la consultation et le marquage lu.

Pas de SSE ici (contrairement aux jobs individuels, `job_events.py`) —
volontairement un simple polling côté frontend (`GET /unread-count` toutes
les ~30s) : une notification n'est jamais urgente à la seconde près comme
une barre de progression qu'on regarde activement pendant un entraînement,
et un polling léger reste plus simple à maintenir qu'un flux global "toute
notification, tous types de job confondus" (contrairement au flux existant,
scopé à UN job précis)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.core.database import get_db
from api.core.models import Notification, User
from domains.auth.router import get_current_user

router = APIRouter(prefix="/notifications", tags=["notifications"])


class NotificationOut(BaseModel):
    id: int
    job_type: str
    job_id: int
    status: str
    title: str
    message: str
    link_path: str
    read_at: Optional[datetime] = None
    created_at: datetime


class UnreadCountOut(BaseModel):
    count: int


def _get_own_notification(notification_id: int, current_user: User, db: Session) -> Notification:
    notification = (
        db.query(Notification)
        .filter(Notification.id == notification_id, Notification.user_id == current_user.id)
        .first()
    )
    if notification is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOTIFICATION_INTROUVABLE", "message": "Notification introuvable"},
        )
    return notification


@router.get("", response_model=List[NotificationOut])
def list_notifications(
    limit: int = Query(30, ge=1, le=100),
    unread_only: bool = Query(False),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(Notification).filter(Notification.user_id == current_user.id)
    if unread_only:
        query = query.filter(Notification.read_at.is_(None))
    rows = query.order_by(Notification.id.desc()).limit(limit).all()
    return rows


@router.get("/unread-count", response_model=UnreadCountOut)
def get_unread_count(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    count = (
        db.query(Notification)
        .filter(Notification.user_id == current_user.id, Notification.read_at.is_(None))
        .count()
    )
    return UnreadCountOut(count=count)


@router.post("/{notification_id}/read", response_model=NotificationOut)
def mark_notification_read(
    notification_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    notification = _get_own_notification(notification_id, current_user, db)
    if notification.read_at is None:
        notification.read_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(notification)
    return notification


@router.post("/read-all", status_code=status.HTTP_204_NO_CONTENT)
def mark_all_notifications_read(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    db.query(Notification).filter(
        Notification.user_id == current_user.id, Notification.read_at.is_(None)
    ).update({"read_at": datetime.now(timezone.utc)})
    db.commit()
