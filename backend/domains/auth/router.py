"""Router d'authentification — inscription, connexion, profil, équipe.

Modèle multi-tenant (Lot 1, décidé dans le diagnostic de migration) :
`POST /auth/register` crée une Organisation ET son premier utilisateur
("owner") en une seule opération — un bureau d'études = une organisation.
L'owner peut ensuite ajouter des membres à SA organisation via
`POST /auth/team/members` ; les données de deux organisations ne se
recoupent jamais (voir `list_team_members`, filtré par `organization_id`).
"""
from __future__ import annotations

import hashlib
import json
import logging
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from urllib.parse import quote

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field, model_validator
from sqlalchemy.orm import Session

from api.core.config import get_settings
from api.core.database import get_db
from api.core.job_queue import redis_conn
from api.core.mailer import (
    mailer_configured,
    send_password_changed_notification_email,
    send_password_reset_email,
)
from api.core.models import AuditLog, Feedback, Organization, PasswordResetToken, User
from api.core.password_policy import validate_password_strength
from api.core.rate_limit import get_client_ip, is_rate_limited, rate_limit_dependency, reset_rate_limit
from api.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    refresh_token_ttl_seconds,
    verify_password,
)
from api.core.token_store import (
    get_refresh_jti_owner,
    is_access_jti_revoked,
    revoke_access_jti,
    revoke_all_refresh_tokens,
    revoke_refresh_jti,
    store_refresh_jti,
)
from domains.shared.audit import log_action

router = APIRouter(prefix="/auth", tags=["authentification"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
_settings = get_settings()
logger = logging.getLogger("datalab.auth")


# ── Schémas ──────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    nom: str = Field(..., min_length=2, max_length=100)
    password: str = Field(..., min_length=8)
    organization_name: str = Field(..., min_length=2, max_length=150)

    @model_validator(mode="after")
    def _password_strength(self) -> "RegisterRequest":
        validate_password_strength(self.password, self.email)
        return self


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    role: str
    nom: str
    organization_name: str


class RefreshRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    # Optionnel : sans lui, seul le jeton access de la requête est révoqué —
    # le refresh token resté valide permettrait de regénérer un accès. Le
    # frontend l'envoie toujours (voir api/client.ts), optionnel côté schéma
    # pour ne pas casser un appelant qui n'aurait plus de refresh en main.
    refresh_token: Optional[str] = None


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

    @model_validator(mode="after")
    def _password_strength(self) -> "TeamMemberCreate":
        validate_password_strength(self.password, self.email)
        return self


class TeamMemberProfile(BaseModel):
    id: int
    email: str
    nom: str
    role: str
    actif: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Dépendances ──────────────────────────────────────────────────────────────

def _as_aware_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """SQLite (dev) ne conserve pas le fuseau horaire des colonnes
    `DateTime(timezone=True)` — une valeur relue est naïve alors qu'elle a
    toujours été écrite en UTC (`datetime.now(timezone.utc)`, voir
    `change_own_password` ci-dessous). PostgreSQL (prod) conserve le fuseau :
    `dt.tzinfo` est déjà renseigné, ne rien changer. Même correctif déjà
    appliqué à `job_watchdog.py`/`prediction_retention.py` — dupliqué ici
    plutôt qu'importé : `domains/auth/router.py` n'a sinon aucune raison de
    dépendre de `domains/shared/job_watchdog.py` (modèles `TrainingJob`/
    `ClusteringJob` sans rapport)."""
    if dt is None or dt.tzinfo is not None:
        return dt
    return dt.replace(tzinfo=timezone.utc)


_TOKEN_INVALIDE = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail={"code": "AUTH_TOKEN_INVALIDE", "message": "Token invalide ou expiré"},
)


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    payload = decode_token(token, expected_type="access")
    if payload is None:
        raise _TOKEN_INVALIDE
    jti = payload.get("jti")
    if not jti or is_access_jti_revoked(redis_conn, jti):
        raise _TOKEN_INVALIDE
    try:
        user_id = int(payload.get("sub", 0))
    except (TypeError, ValueError):
        raise _TOKEN_INVALIDE
    user = db.query(User).filter(User.id == user_id).first()
    if user is None or not user.actif:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "AUTH_UTILISATEUR_INTROUVABLE_OU_DESACTIVE", "message": "Utilisateur introuvable ou désactivé"},
        )
    # Correctif Phase 1 (AUDIT_BACKEND_2026-08-23.md §A.4) — un jeton émis
    # AVANT la dernière révocation en masse (changement de mot de passe,
    # réinitialisation) est rejeté même s'il n'a pas encore expiré et même
    # si son `jti` individuel n'a jamais été révoqué explicitement.
    valid_after = _as_aware_utc(user.token_valid_after)
    if valid_after is not None:
        iat = payload.get("iat")
        if iat is None or iat < valid_after.timestamp():
            raise _TOKEN_INVALIDE
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

_register_rate_limit = rate_limit_dependency(
    "register", _settings.register_rate_limit_max_attempts, _settings.register_rate_limit_window_seconds
)


@router.post(
    "/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(_register_rate_limit)],
)
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

    access_token, _ = create_access_token(user.id, user.role, organization.id)
    refresh_token, refresh_jti = create_refresh_token(user.id)
    store_refresh_jti(redis_conn, user.id, refresh_jti, refresh_token_ttl_seconds())
    return TokenResponse(
        access_token=access_token, refresh_token=refresh_token,
        role=user.role, nom=user.nom, organization_name=organization.name,
    )


@router.post("/login", response_model=TokenResponse)
def login(request: Request, form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Connexion — retourne un JWT Bearer (username = email).

    Limité par IP cliente (H11, AUDIT_ROADMAP.md) : au-delà de
    `login_rate_limit_max_attempts` tentatives ÉCHOUÉES dans la fenêtre
    glissante, 429 avant même de consulter la base — brute force sur mot de
    passe rendu impraticable sans pénaliser un utilisateur qui se trompe une
    ou deux fois."""
    client_ip = get_client_ip(request)
    rate_limit_key = f"login_attempts:{client_ip}"
    if is_rate_limited(
        redis_conn, rate_limit_key, _settings.login_rate_limit_max_attempts, _settings.login_rate_limit_window_seconds
    ):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "code": "AUTH_TROP_DE_TENTATIVES",
                "message": "Trop de tentatives de connexion — réessayez dans quelques minutes.",
            },
        )

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

    reset_rate_limit(redis_conn, rate_limit_key)
    user.last_login = datetime.now(timezone.utc)
    log_action(db, user.organization_id, user.id, "auth.login", details={"ip": client_ip})
    db.commit()

    access_token, _ = create_access_token(user.id, user.role, user.organization_id)
    refresh_token, refresh_jti = create_refresh_token(user.id)
    store_refresh_jti(redis_conn, user.id, refresh_jti, refresh_token_ttl_seconds())
    return TokenResponse(
        access_token=access_token, refresh_token=refresh_token,
        role=user.role, nom=user.nom, organization_name=user.organization_name,
    )


@router.post("/refresh", response_model=TokenResponse)
def refresh_tokens(body: RefreshRequest, db: Session = Depends(get_db)):
    """Renouvellement transparent (Phase 1, AUDIT_BACKEND_2026-08-23.md §A.1)
    — le frontend appelle cet endpoint quand un jeton access a expiré (20
    min) plutôt que de renvoyer l'utilisateur au formulaire de connexion.
    Rotatif : l'ancien refresh token est immédiatement invalidé, qu'il soit
    réutilisé ou non — une seconde présentation du MÊME refresh token (vol,
    rejeu) échoue toujours, elle ne prolonge jamais une session volée."""
    payload = decode_token(body.refresh_token, expected_type="refresh")
    if payload is None:
        raise _TOKEN_INVALIDE
    jti = payload.get("jti")
    try:
        user_id = int(payload.get("sub", 0))
    except (TypeError, ValueError) as exc:
        raise _TOKEN_INVALIDE from exc
    owner_id = get_refresh_jti_owner(redis_conn, jti) if jti else None
    if owner_id is None or owner_id != user_id:
        raise _TOKEN_INVALIDE
    user = db.query(User).filter(User.id == user_id).first()
    if user is None or not user.actif:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "AUTH_UTILISATEUR_INTROUVABLE_OU_DESACTIVE",
                "message": "Utilisateur introuvable ou désactivé",
            },
        )
    revoke_refresh_jti(redis_conn, user_id, jti)  # usage unique — avant d'émettre le remplaçant

    access_token, _ = create_access_token(user.id, user.role, user.organization_id)
    new_refresh_token, new_refresh_jti = create_refresh_token(user.id)
    store_refresh_jti(redis_conn, user.id, new_refresh_jti, refresh_token_ttl_seconds())
    return TokenResponse(
        access_token=access_token, refresh_token=new_refresh_token,
        role=user.role, nom=user.nom, organization_name=user.organization_name,
    )


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
    request: Request,
    body: ChangePasswordRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Change le mot de passe du compte connecté — jamais celui d'un autre
    utilisateur. Révoque TOUTES les sessions existantes (Phase 1,
    AUDIT_BACKEND_2026-08-23.md §A.4) : un utilisateur qui change son mot de
    passe parce qu'il se sait/croit compromis doit pouvoir chasser
    quiconque détiendrait déjà un jeton — sans ça, un jeton volé restait
    valide jusqu'à 24h après ce geste de protection."""
    if not verify_password(body.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "AUTH_MDP_ACTUEL_INCORRECT", "message": "Mot de passe actuel incorrect"},
        )
    # `ChangePasswordRequest` n'a pas l'email en champ (pas de raison de le
    # redemander) — vérifié ici plutôt que dans un `@model_validator` du
    # schéma, avec l'email du compte connecté.
    try:
        validate_password_strength(body.new_password, current_user.email)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "AUTH_MDP_TROP_FAIBLE", "message": str(exc)},
        ) from exc
    client_ip = get_client_ip(request)
    current_user.hashed_password = hash_password(body.new_password)
    current_user.token_valid_after = datetime.now(timezone.utc)
    log_action(
        db, current_user.organization_id, current_user.id, "auth.password_changed",
        details={"ip": client_ip},
    )
    db.commit()
    revoke_all_refresh_tokens(redis_conn, current_user.id)
    # Phase 1B, point 6 — souvent le seul signal qu'une victime reçoit si
    # le changement n'était pas d'elle (session déjà volée + mot de passe
    # changé par l'attaquant).
    if mailer_configured():
        background_tasks.add_task(_send_password_changed_notification_task, current_user.email, client_ip)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(
    request: Request,
    body: LogoutRequest,
    token: str = Depends(oauth2_scheme),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Révoque réellement le jeton access de CETTE session (Phase 1,
    AUDIT_BACKEND_2026-08-23.md §A.2) — avant ce correctif, `/logout` ne
    faisait rien côté serveur, un jeton "déconnecté" restait valide jusqu'à
    expiration naturelle. Révoque aussi le refresh token fourni, s'il y en
    a un (voir `LogoutRequest`) — sans ça, le refresh resté valide
    permettrait de regénérer un nouvel access token juste après."""
    payload = decode_token(token, expected_type="access")
    if payload is not None:
        jti = payload.get("jti")
        exp = payload.get("exp")
        if jti and exp:
            revoke_access_jti(redis_conn, jti, int(exp - time.time()))
    if body.refresh_token:
        refresh_payload = decode_token(body.refresh_token, expected_type="refresh")
        if refresh_payload is not None:
            refresh_jti = refresh_payload.get("jti")
            if refresh_jti:
                revoke_refresh_jti(redis_conn, current_user.id, refresh_jti)
    log_action(db, current_user.organization_id, current_user.id, "auth.logout", details={"ip": get_client_ip(request)})
    db.commit()


# ── Réinitialisation de mot de passe (Phase 1B, AUDIT_BACKEND_2026-08-23.md) ─
# Mécanisme repris de CIAM (`E:\concrete-ai-platform`, déjà éprouvé) : jeton
# `secrets.token_urlsafe(32)` cryptographiquement sûr, jamais stocké en
# clair (SHA-256), usage unique, un seul jeton actif à la fois par
# utilisateur, réponse `request` strictement neutre (204 sans corps, que
# l'email existe ou non — y compris rate-limité, pour ne jamais donner de
# signal exploitable), envoi mail en tâche de fond (la latence SMTP ne doit
# jamais dépendre de l'existence du compte).
#
# Durcissements au-delà de CIAM (voir _backend/JOURNAL.md, Décision Phase
# 1B pour le détail) : révocation de TOUTES les sessions à la confirmation
# (CIAM ne peut pas, JWT stateless sans jti) ; limite par COMPTE en plus de
# l'IP ; journalisé dans `AuditLog`, pas seulement `logger.info` ; purge des
# jetons expirés ; robustesse du mot de passe appliquée ici aussi ; mail
# avec date/IP de la demande + second mail de notification après
# changement effectif.

class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=8, max_length=100)
    new_password_confirm: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def _check_passwords(self) -> "ResetPasswordRequest":
        if self.new_password != self.new_password_confirm:
            raise ValueError("Le nouveau mot de passe et sa confirmation ne correspondent pas")
        return self


def _purge_expired_password_reset_tokens(db: Session) -> int:
    """Même approche que `services/prediction_retention.py`/`job_watchdog.py`
    — pas de scheduler dédié, purge à la demande au moment où elle a du
    sens (juste avant d'émettre un nouveau jeton). Comparaison en PYTHON,
    pas en filtre SQL (même raison que `prediction_retention.py` : écart de
    fuseau horaire SQLite en dev/test, invisible en Postgres/prod)."""
    now = datetime.now(timezone.utc)
    rows = db.query(PasswordResetToken.id, PasswordResetToken.expires_at).all()
    stale_ids = [row.id for row in rows if _as_aware_utc(row.expires_at) < now]
    if stale_ids:
        db.query(PasswordResetToken).filter(PasswordResetToken.id.in_(stale_ids)).delete(synchronize_session=False)
    return len(stale_ids)


def _issue_password_reset_token(db: Session, user: User, requested_from_ip: str) -> str:
    """Invalide les jetons non utilisés existants de l'utilisateur, en émet
    un nouveau (un seul actif à la fois), le persiste HASHÉ et retourne le
    jeton EN CLAIR — à insérer uniquement dans le lien envoyé par e-mail,
    jamais journalisé ni renvoyé dans une réponse API."""
    now = datetime.now(timezone.utc)
    for previous in db.query(PasswordResetToken).filter(
        PasswordResetToken.user_id == user.id,
        PasswordResetToken.used_at.is_(None),
    ).all():
        previous.used_at = now

    raw_token = secrets.token_urlsafe(32)
    db.add(PasswordResetToken(
        user_id=user.id,
        token_hash=hashlib.sha256(raw_token.encode()).hexdigest(),
        expires_at=now + timedelta(minutes=_settings.password_reset_expire_minutes),
        requested_from_ip=requested_from_ip,
    ))
    return raw_token


def _build_reset_link(raw_token: str) -> str:
    return f"{_settings.frontend_url}/reset-password?token={quote(raw_token)}"


def _send_password_reset_email_task(to_email: str, reset_link: str, requested_from_ip: str) -> None:
    """Tâche de fond (`BackgroundTasks`) — un échec SMTP ne doit jamais
    faire échouer la requête HTTP qui l'a déclenchée (déjà répondue 204 à
    ce stade)."""
    try:
        send_password_reset_email(
            to_email, reset_link, _settings.password_reset_expire_minutes, requested_from_ip
        )
    except Exception:
        logger.exception("[PasswordReset] Échec de l'envoi du mail de réinitialisation")


def _send_password_changed_notification_task(to_email: str, requested_from_ip: str) -> None:
    try:
        send_password_changed_notification_email(to_email, requested_from_ip)
    except Exception:
        logger.exception("[PasswordReset] Échec de l'envoi de la notification de changement")


@router.post("/password-reset/request", status_code=status.HTTP_204_NO_CONTENT)
def request_password_reset(
    body: ForgotPasswordRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Demande de réinitialisation. Réponse STRICTEMENT identique (204,
    aucun corps) que l'email corresponde à un compte actif ou non — ne
    jamais permettre l'énumération de comptes existants via cette route,
    ni via son temps de réponse (l'envoi mail est toujours en tâche de
    fond, jamais dans le chemin de réponse)."""
    client_ip = get_client_ip(request)
    ip_key = f"rate_limit:password_reset_ip:{client_ip}"
    email_key = f"rate_limit:password_reset_email:{body.email.lower()}"
    # Durcissement au-delà de CIAM (Phase 1B, point 2) : limite AUSSI par
    # compte, pas seulement par IP — un attaquant qui changerait d'IP
    # inonderait sinon la boîte mail de la victime sans jamais être bloqué.
    ip_limited = is_rate_limited(
        redis_conn, ip_key, _settings.password_reset_rate_limit_max_attempts_per_ip,
        _settings.password_reset_rate_limit_window_seconds,
    )
    email_limited = is_rate_limited(
        redis_conn, email_key, _settings.password_reset_rate_limit_max_attempts_per_email,
        _settings.password_reset_rate_limit_window_seconds,
    )
    if ip_limited or email_limited:
        # Pas de 429 distinct : la réponse reste la même 204 neutre,
        # throttlée ou non — sinon la réponse elle-même devient un oracle.
        return

    _purge_expired_password_reset_tokens(db)

    # Pas de .lower() sur la comparaison elle-même : suit exactement le
    # même critère que /auth/login (User.email == ...), pour ne jamais
    # désynchroniser les deux routes sur la casse.
    user = db.query(User).filter(User.email == body.email).first()
    if user and user.actif:
        raw_token = _issue_password_reset_token(db, user, client_ip)
        reset_link = _build_reset_link(raw_token)
        log_action(
            db, user.organization_id, user.id, "auth.password_reset_requested",
            details={"ip": client_ip},
        )
        db.commit()
        if mailer_configured():
            background_tasks.add_task(_send_password_reset_email_task, user.email, reset_link, client_ip)
        else:
            logger.info("[PasswordReset] Canal mail non configuré — lien non envoyé pour %s", user.email)
    else:
        db.commit()  # persiste la purge même si aucun jeton n'est émis
    # Aucune branche qui changerait la réponse : 204 sans corps, toujours.


@router.post("/password-reset/confirm", status_code=status.HTTP_204_NO_CONTENT)
def confirm_password_reset(
    body: ResetPasswordRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Valide le jeton (existant, non expiré, non utilisé) et change le mot
    de passe. Message d'erreur volontairement générique — ne distingue pas
    jeton inconnu / expiré / déjà utilisé, même logique de non-divulgation
    que la route `request`."""
    now = datetime.now(timezone.utc)
    token_hash = hashlib.sha256(body.token.encode()).hexdigest()
    reset = db.query(PasswordResetToken).filter(
        PasswordResetToken.token_hash == token_hash,
        PasswordResetToken.used_at.is_(None),
    ).first()

    expires_at = _as_aware_utc(reset.expires_at) if reset else None
    if not reset or expires_at is None or expires_at <= now:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "AUTH_RESET_TOKEN_INVALIDE", "message": "Lien de réinitialisation invalide ou expiré"},
        )

    user = db.query(User).filter(User.id == reset.user_id).first()
    if not user or not user.actif:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "AUTH_RESET_TOKEN_INVALIDE", "message": "Lien de réinitialisation invalide ou expiré"},
        )

    try:
        validate_password_strength(body.new_password, user.email)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "AUTH_MDP_TROP_FAIBLE", "message": str(exc)},
        ) from exc

    client_ip = get_client_ip(request)
    user.hashed_password = hash_password(body.new_password)
    # Mieux que CIAM (Phase 1B, point 1) : CIAM ne peut pas révoquer de
    # sessions (JWT stateless sans jti) — DataLab le peut depuis la Phase 1.
    # Un mot de passe réinitialisé qui laisse vivre les jetons émis avant
    # serait une correction à moitié faite.
    user.token_valid_after = now
    reset.used_at = now
    # Un seul lien actif par utilisateur (voir _issue_password_reset_token) :
    # invalide aussi tout autre jeton resté non utilisé (demandes
    # successives sans clic sur les liens précédents).
    for other in db.query(PasswordResetToken).filter(
        PasswordResetToken.user_id == user.id,
        PasswordResetToken.used_at.is_(None),
    ).all():
        other.used_at = now
    log_action(
        db, user.organization_id, user.id, "auth.password_reset_confirmed",
        details={"ip": client_ip},
    )
    db.commit()
    revoke_all_refresh_tokens(redis_conn, user.id)
    if mailer_configured():
        background_tasks.add_task(_send_password_changed_notification_task, user.email, client_ip)


# ── Préférences d'interface (Lot UI — refonte visuelle) ─────────────────────
# Routeur séparé (préfixe /users plutôt que /auth) : le thème n'est pas une
# information d'identité/authentification, et le chemin est fixé par la
# mission (GET/PATCH /api/users/me/preferences). Réutilise `get_current_user`
# et le modèle `User` déjà importés dans ce fichier plutôt que de dupliquer
# un domaine entier pour deux endpoints.
users_router = APIRouter(prefix="/users", tags=["préférences"])


@users_router.get("/me/preferences", response_model=UserPreferences)
def get_preferences(current_user: User = Depends(get_current_user)):
    return current_user


@users_router.patch("/me/preferences", response_model=UserPreferences)
def update_preferences(
    body: UserPreferencesUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    current_user.ui_theme = body.ui_theme
    db.commit()
    db.refresh(current_user)
    return current_user


# ── Retour utilisateur (Lot 10, refonte UI) ──────────────────────────────────
# Retour utilisateur direct pendant la mission : « ajoute un formulaire pour
# renseigner ce problème » plutôt qu'un simple lien mailto vers un support qui
# n'existe pas pour cette app. Stocké tel quel (table `Feedback`), jamais
# traité automatiquement — consultable par les administrateurs de LEUR
# organisation uniquement (même isolation que le reste de l'app).

class FeedbackCreate(BaseModel):
    page: str = Field(..., min_length=1, max_length=300)
    message: str = Field(..., min_length=1, max_length=4000)


class FeedbackOut(BaseModel):
    id: int
    page: str
    message: str
    author_name: str
    created_at: datetime

    class Config:
        from_attributes = True


feedback_router = APIRouter(prefix="/feedback", tags=["retour utilisateur"])


@feedback_router.post("", response_model=FeedbackOut, status_code=status.HTTP_201_CREATED)
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


@feedback_router.get("", response_model=List[FeedbackOut])
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
    db.flush()  # obtient member.id avant l'écriture du journal, même transaction
    log_action(
        db, owner.organization_id, owner.id, "member.added",
        target_type="user", target_id=member.id, details={"email": member.email, "nom": member.nom},
    )
    db.commit()
    db.refresh(member)
    return member


class AuditLogEntry(BaseModel):
    id: int
    action: str
    target_type: Optional[str] = None
    target_id: Optional[int] = None
    details: Optional[dict] = None
    actor_name: Optional[str] = None
    created_at: datetime


@router.get("/team/audit-log", response_model=List[AuditLogEntry])
def list_audit_log(
    owner: User = Depends(require_owner),
    db: Session = Depends(get_db),
    limit: int = 100,
):
    """[Owner] Journal des actions sensibles de MON organisation (Lot 10) —
    suppression de dataset/entraînement, ajout de membre, promotion de
    modèle. Réservé au owner, même règle que le reste de la gestion
    d'équipe : un membre ordinaire ne consulte pas ce journal."""
    entries = (
        db.query(AuditLog)
        .filter(AuditLog.organization_id == owner.organization_id)
        .order_by(AuditLog.created_at.desc())
        .limit(max(1, min(limit, 500)))
        .all()
    )
    return [
        AuditLogEntry(
            id=e.id,
            action=e.action,
            target_type=e.target_type,
            target_id=e.target_id,
            details=json.loads(e.details_json) if e.details_json else None,
            actor_name=e.actor.nom if e.actor else None,
            created_at=e.created_at,
        )
        for e in entries
    ]
