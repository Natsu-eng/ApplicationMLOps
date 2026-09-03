"""Router préférences d'interface (Lot UI — refonte visuelle).

Préfixe `/users` plutôt que `/auth` : le thème n'est pas une information
d'identité/authentification, et le chemin est fixé par la mission
(GET/PATCH /api/users/me/preferences).

Extrait de `router.py` lors du découpage du router auth : ce routeur y
était déjà distinct (son propre `APIRouter`, son propre préfixe), mais
vivait au milieu des endpoints `/auth` — intercalé entre la
réinitialisation de mot de passe et les endpoints d'équipe. Il réutilise
`get_current_user` de `router.py` (import à sens unique, comme tous les
autres domaines) plutôt que de dupliquer la dépendance.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.orm import Session

from api.core.database import get_db
from api.core.models import User
from domains.auth.router import get_current_user

router = APIRouter(prefix="/users", tags=["préférences"])


# ── Schémas ──────────────────────────────────────────────────────────────────

VALID_UI_THEMES = {"graphite", "ivoire", "minuit", "ardoise", "porcelaine"}


class UserPreferences(BaseModel):
    ui_theme: str

    model_config = {"from_attributes": True}


class UserPreferencesUpdate(BaseModel):
    ui_theme: str = Field(..., description="graphite | ivoire | minuit | ardoise | porcelaine")

    @model_validator(mode="after")
    def _valid_theme(self) -> "UserPreferencesUpdate":
        if self.ui_theme not in VALID_UI_THEMES:
            raise ValueError(f"Thème inconnu : {self.ui_theme!r} (attendu : {', '.join(sorted(VALID_UI_THEMES))})")
        return self


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/me/preferences", response_model=UserPreferences)
def get_preferences(current_user: User = Depends(get_current_user)):
    return current_user


@router.patch("/me/preferences", response_model=UserPreferences)
def update_preferences(
    body: UserPreferencesUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    current_user.ui_theme = body.ui_theme
    db.commit()
    db.refresh(current_user)
    return current_user
