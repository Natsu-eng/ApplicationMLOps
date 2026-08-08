"""Router d'authentification — inscription, connexion, profil, équipe.

Modèle multi-tenant (Lot 1, décidé dans le diagnostic de migration) :
`POST /auth/register` crée une Organisation ET son premier utilisateur
("owner") en une seule opération — un bureau d'études = une organisation.
L'owner peut ensuite ajouter des membres à SA organisation via
`POST /auth/team/members` ; les données de deux organisations ne se
recoupent jamais (voir `list_team_members`, filtré par `organization_id`).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field, model_validator
from sqlalchemy.orm import Session

from api.core.database import get_db
from api.core.models import Organization, User
from api.core.security import create_access_token, decode_token, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["authentification"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


# ── Schémas ──────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    nom: str = Field(..., min_length=2, max_length=100)
    password: str = Field(..., min_length=8)
    organization_name: str = Field(..., min_length=2, max_length=150)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    nom: str
    organization_name: str


class UserProfile(BaseModel):
    id: int
    email: str
    nom: str
    role: str
    organization_id: int
    organization_name: str
    actif: bool
    created_at: datetime
    last_login: Optional[datetime] = None

    model_config = {"from_attributes": True}


class UserSelfUpdate(BaseModel):
    nom: Optional[str] = Field(None, min_length=2, max_length=100)


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=8, max_length=100)
    new_password_confirm: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def _check_passwords(self) -> "ChangePasswordRequest":
        if self.new_password != self.new_password_confirm:
            raise ValueError("Le nouveau mot de passe et sa confirmation ne correspondent pas")
        if self.new_password == self.current_password:
            raise ValueError("Le nouveau mot de passe doit être différent de l'ancien")
        return self


class TeamMemberCreate(BaseModel):
    email: EmailStr
    nom: str = Field(..., min_length=2, max_length=100)
    password: str = Field(..., min_length=8)


class TeamMemberProfile(BaseModel):
    id: int
    email: str
    nom: str
    role: str
    actif: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Dépendances ──────────────────────────────────────────────────────────────

def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    payload = decode_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "AUTH_TOKEN_INVALIDE", "message": "Token invalide ou expiré"},
        )
    try:
        user_id = int(payload.get("sub", 0))
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "AUTH_TOKEN_INVALIDE", "message": "Token invalide ou expiré"},
        )
    user = db.query(User).filter(User.id == user_id).first()
    if user is None or not user.actif:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "AUTH_UTILISATEUR_INTROUVABLE_OU_DESACTIVE", "message": "Utilisateur introuvable ou désactivé"},
        )
    return user


def require_owner(current_user: User = Depends(get_current_user)) -> User:
    """Réservé au owner de l'organisation — seul rôle autorisé à gérer l'équipe (Lot 1)."""
    if not current_user.is_owner:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "AUTH_OWNER_REQUIS", "message": "Action réservée au propriétaire de l'organisation"},
        )
    return current_user


# ── Endpoints — compte personnel ──────────────────────────────────────────────

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    """Crée une nouvelle organisation et son premier utilisateur (owner)."""
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "AUTH_EMAIL_DEJA_UTILISE", "message": "Email déjà utilisé"},
        )

    organization = Organization(name=body.organization_name)
    db.add(organization)
    db.flush()  # organization.id disponible sans committer déjà l'utilisateur

    user = User(
        email=body.email,
        nom=body.nom,
        hashed_password=hash_password(body.password),
        role="owner",
        organization_id=organization.id,
        actif=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token({"sub": user.id, "role": user.role, "org": organization.id})
    return TokenResponse(access_token=token, role=user.role, nom=user.nom, organization_name=organization.name)


@router.post("/login", response_model=TokenResponse)
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Connexion — retourne un JWT Bearer (username = email)."""
    user = db.query(User).filter(User.email == form.username).first()
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "AUTH_IDENTIFIANTS_INCORRECTS", "message": "Email ou mot de passe incorrect"},
        )
    if not user.actif:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "AUTH_COMPTE_DESACTIVE", "message": "Compte désactivé — contacter le propriétaire de votre organisation"},
        )

    user.last_login = datetime.now(timezone.utc)
    db.commit()

    token = create_access_token({"sub": user.id, "role": user.role, "org": user.organization_id})
    return TokenResponse(access_token=token, role=user.role, nom=user.nom, organization_name=user.organization_name)


@router.get("/me", response_model=UserProfile)
def me(current_user: User = Depends(get_current_user)):
    """Profil de l'utilisateur connecté."""
    return current_user


@router.patch("/me", response_model=UserProfile)
def update_me(
    body: UserSelfUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if body.nom is not None:
        current_user.nom = body.nom
        db.commit()
        db.refresh(current_user)
    return current_user


@router.patch("/me/password", status_code=status.HTTP_204_NO_CONTENT)
def change_own_password(
    body: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Change le mot de passe du compte connecté — jamais celui d'un autre utilisateur."""
    if not verify_password(body.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "AUTH_MDP_ACTUEL_INCORRECT", "message": "Mot de passe actuel incorrect"},
        )
    current_user.hashed_password = hash_password(body.new_password)
    db.commit()


@router.post("/logout")
def logout():
    """JWT est stateless : la déconnexion consiste à supprimer le token côté client."""
    return {"message": "Déconnexion effectuée (supprimer le token côté client)"}


# ── Endpoints — équipe (organisation) ────────────────────────────────────────

@router.get("/team/members", response_model=List[TeamMemberProfile])
def list_team_members(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Liste les membres de MA organisation uniquement — jamais ceux d'une autre."""
    return (
        db.query(User)
        .filter(User.organization_id == current_user.organization_id)
        .order_by(User.created_at)
        .all()
    )


@router.post("/team/members", response_model=TeamMemberProfile, status_code=status.HTTP_201_CREATED)
def add_team_member(
    body: TeamMemberCreate,
    owner: User = Depends(require_owner),
    db: Session = Depends(get_db),
):
    """[Owner] Ajoute un membre directement à SON organisation."""
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "AUTH_EMAIL_DEJA_UTILISE", "message": "Email déjà utilisé"},
        )
    member = User(
        email=body.email,
        nom=body.nom,
        hashed_password=hash_password(body.password),
        role="member",
        organization_id=owner.organization_id,
        actif=True,
    )
    db.add(member)
    db.commit()
    db.refresh(member)
    return member
