"""Router retour utilisateur (Lot 10, refonte UI).

Retour utilisateur direct pendant la mission : « ajoute un formulaire pour
renseigner ce problème » plutôt qu'un simple lien mailto vers un support qui
n'existe pas pour cette app. Stocké tel quel (table `Feedback`), jamais
traité automatiquement — consultable par les administrateurs de LEUR
organisation uniquement (même isolation que le reste de l'app).

Extrait de `router.py` lors du découpage du router auth : ce routeur y
était déjà distinct (son propre `APIRouter`, son propre préfixe), mais
vivait au milieu des endpoints `/auth` — intercalé entre les préférences
d'interface et les endpoints d'équipe. Il réutilise `require_owner` de
`router.py` (import à sens unique) plutôt que de dupliquer la dépendance.
"""
from __future__ import annotations

from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.core.database import get_db
from api.core.models import Feedback, User
from domains.auth.router import get_current_user, require_owner

router = APIRouter(prefix="/feedback", tags=["retour utilisateur"])


# ── Schémas ──────────────────────────────────────────────────────────────────

class FeedbackCreate(BaseModel):
    page: str = Field(..., min_length=1, max_length=300)
    message: str = Field(..., min_length=1, max_length=4000)


class FeedbackOut(BaseModel):
    id: int
    page: str
    message: str
    author_name: str
    created_at: datetime

    # `model_config` plutôt que `class Config` (déprécié depuis Pydantic V2,
    # supprimé en V3) — aligne ce schéma sur tous les autres du projet
    # (UserProfile, TeamMemberProfile, UserPreferences...).
    model_config = {"from_attributes": True}


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("", response_model=FeedbackOut, status_code=status.HTTP_201_CREATED)
def create_feedback(
    body: FeedbackCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    entry = Feedback(
        organization_id=current_user.organization_id,
        user_id=current_user.id,
        page=body.page,
        message=body.message,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return FeedbackOut(
        id=entry.id, page=entry.page, message=entry.message,
        author_name=current_user.nom, created_at=entry.created_at,
    )


@router.get("", response_model=List[FeedbackOut])
def list_feedback(
    owner: User = Depends(require_owner),
    db: Session = Depends(get_db),
):
    """Retours de MON organisation uniquement — jamais ceux d'une autre.

    Correctif (Phase 1, AUDIT_BACKEND_2026-08-23.md, Axe B) : le commentaire
    de section ci-dessus affirme un accès réservé aux administrateurs
    depuis le Lot 10, mais la route utilisait `get_current_user` — tout
    membre ordinaire pouvait donc lire les retours de ses collègues.
    Aucun IDOR cross-tenant (le filtre `organization_id` était déjà
    correct), seulement un rôle mal appliqué."""
    entries = (
        db.query(Feedback)
        .filter(Feedback.organization_id == owner.organization_id)
        .order_by(Feedback.created_at.desc())
        .limit(200)
        .all()
    )
    return [
        FeedbackOut(id=e.id, page=e.page, message=e.message, author_name=e.author.nom, created_at=e.created_at)
        for e in entries
    ]
