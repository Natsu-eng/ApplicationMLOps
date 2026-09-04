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
from typing import List, Literal, Optional
from urllib.parse import quote

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field, model_validator
from sqlalchemy.orm import Session

from api.core.config import get_settings
from api.core.database import get_db
from api.core.error_codes import ErrorCode
from api.core.job_queue import redis_conn
from api.core.mailer import (
    mailer_configured,
    send_password_changed_notification_email,
    send_password_reset_email,
)
from api.core.models import AuditLog, Organization, PasswordResetToken, User
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
    # Permet à l'interface d'imposer l'écran de changement de mot de passe :
    # l'API refuse déjà tout le reste (voir `get_current_user`), autant le
    # dire clairement plutôt que de laisser l'utilisateur se heurter à des
    # 403 sans comprendre.
    must_change_password: bool = False

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
    # Depuis quand l'accès est coupé (None = compte actif, ou désactivé
    # avant que cette date ne soit conservée).
    deactivated_at: Optional[datetime] = None
    # Le membre n'a pas encore remplacé le mot de passe provisoire : le
    # propriétaire qui l'a créé le connaît toujours. Affiché pour qu'il
    # puisse relancer l'intéressé.
    must_change_password: bool = False

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


# Seuls chemins accessibles tant qu'un mot de passe PROVISOIRE n'a pas été
# remplacé : savoir qui l'on est (l'interface a besoin de l'indicateur pour
# afficher le bon écran), changer justement ce mot de passe, et se
# déconnecter. Tout le reste de l'API est fermé — l'enforcement est ici,
# côté serveur, et non dans l'interface : un membre qui ignorerait l'écran
# laisserait sinon le mot de passe connu de son propriétaire valable
# indéfiniment, ce qui viderait le correctif de son sens.
_PASSWORD_CHANGE_EXEMPT_PATHS = frozenset({
    "/api/auth/me",
    "/api/auth/me/password",
    "/api/auth/logout",
})


def get_current_user(
    request: Request,
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
            detail={"code": ErrorCode.AUTH_UTILISATEUR_INTROUVABLE_OU_DESACTIVE, "message": "Utilisateur introuvable ou désactivé"},
        )
    # Correctif Phase 1 (AUDIT_BACKEND_2026-08-23.md §A.4) — un jeton émis
    # AVANT la dernière révocation en masse (changement de mot de passe,
    # réinitialisation) est rejeté même s'il n'a pas encore expiré et même
    # si son `jti` individuel n'a jamais été révoqué explicitement.
    valid_after = _as_aware_utc(user.token_valid_after)
    if valid_after is not None:
        iat = payload.get("iat")
        # Comparaison STRICTE, seuil non tronqué — volontaire. `iat` est un
        # entier de secondes (spec JWT) alors que `token_valid_after` porte
        # des microsecondes : dans la seconde de la révocation, un jeton
        # émis AVANT et un jeton émis APRÈS ont le même `iat` et sont donc
        # indiscernables. Il faut choisir lequel des deux on sacrifie.
        # Tronquer le seuil ferait survivre les anciens jetons — essayé,
        # et `test_confirm_reset_with_valid_token_changes_password_and_
        # revokes_sessions` l'a immédiatement rattrapé : une session volée
        # survivait au geste censé la chasser. On garde donc le rejet
        # strict, qui se trompe du côté sûr. Conséquence acceptée : se
        # reconnecter dans la même seconde qu'un changement de mot de passe
        # donne un jeton refusé, il faut recommencer une seconde plus tard.
        if iat is None or iat < valid_after.timestamp():
            raise _TOKEN_INVALIDE
    if user.must_change_password and request.url.path not in _PASSWORD_CHANGE_EXEMPT_PATHS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": ErrorCode.AUTH_MDP_PROVISOIRE,
                "message": (
                    "Mot de passe provisoire : choisissez le vôtre avant d'utiliser la plateforme — "
                    "celui qui vous a été communiqué est connu de la personne qui a créé votre compte"
                ),
            },
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
            detail={"code": ErrorCode.AUTH_EMAIL_DEJA_UTILISE, "message": "Email déjà utilisé"},
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
                "code": ErrorCode.AUTH_UTILISATEUR_INTROUVABLE_OU_DESACTIVE,
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
    # Limite AVANT la vérification, jamais après : appliquée seulement en
    # cas d'échec, elle laisserait passer autant d'essais que voulu tant que
    # le compteur n'a pas été incrémenté. Clé sur l'identifiant du COMPTE et
    # non sur l'IP — l'appelant est déjà authentifié ici, changer d'IP ne
    # doit rien lui redonner. Échec ouvert (comme /login) : si Redis tombe,
    # on ne bloque pas quelqu'un qui change légitimement son mot de passe.
    password_rate_limit_key = f"rate_limit:password_change:{current_user.id}"
    if is_rate_limited(
        redis_conn,
        password_rate_limit_key,
        _settings.password_change_rate_limit_max_attempts,
        _settings.password_change_rate_limit_window_seconds,
    ):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "code": ErrorCode.AUTH_TROP_DE_TENTATIVES,
                "message": "Trop de tentatives de changement de mot de passe — réessayez dans quelques minutes.",
            },
        )
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
            detail={"code": ErrorCode.AUTH_MDP_TROP_FAIBLE, "message": str(exc)},
        ) from exc
    client_ip = get_client_ip(request)
    # Changement réussi : on efface le compteur, pour qu'un utilisateur
    # légitime qui s'est trompé deux fois avant de réussir ne reste pas
    # pénalisé (même geste qu'après une connexion réussie).
    reset_rate_limit(redis_conn, password_rate_limit_key)
    current_user.hashed_password = hash_password(body.new_password)
    current_user.token_valid_after = datetime.now(timezone.utc)
    # Le mot de passe provisoire choisi par le propriétaire vient d'être
    # remplacé par celui de l'intéressé : la connaissance qu'en avait le
    # propriétaire devient sans valeur, le compte est débloqué.
    current_user.must_change_password = False
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
            detail={
                "code": ErrorCode.AUTH_RESET_TOKEN_INVALIDE,
                "message": "Lien de réinitialisation invalide ou expiré",
            },
        )

    user = db.query(User).filter(User.id == reset.user_id).first()
    if not user or not user.actif:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": ErrorCode.AUTH_RESET_TOKEN_INVALIDE,
                "message": "Lien de réinitialisation invalide ou expiré",
            },
        )

    try:
        validate_password_strength(body.new_password, user.email)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": ErrorCode.AUTH_MDP_TROP_FAIBLE, "message": str(exc)},
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
            detail={"code": ErrorCode.AUTH_EMAIL_DEJA_UTILISE, "message": "Email déjà utilisé"},
        )
    member = User(
        email=body.email,
        nom=body.nom,
        hashed_password=hash_password(body.password),
        role="member",
        organization_id=owner.organization_id,
        actif=True,
        # Le propriétaire choisit ce mot de passe, il le connaît donc. Sans
        # cet indicateur il le connaissait indéfiniment et pouvait se
        # connecter au compte de son collaborateur : l'intéressé doit en
        # choisir un autre avant tout usage de la plateforme (voir
        # `get_current_user`).
        must_change_password=True,
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


def _count_other_active_owners(db: Session, organization_id: int, excluding_user_id: int) -> int:
    """Nombre de propriétaires ACTIFS de l'organisation, hors celui visé.

    Sert d'invariant unique à deux endroits (révocation d'accès et
    rétrogradation) : une organisation doit conserver en permanence au
    moins un propriétaire actif. Sans cette garde, deux chemins mènent au
    même cul-de-sac — plus personne pour gérer l'équipe, lire les retours
    ou promouvoir qui que ce soit, état irréversible sans intervention
    directe en base. Un propriétaire désactivé ne compte pas : il ne peut
    plus se connecter, donc il ne gère plus rien."""
    return (
        db.query(User)
        .filter(
            User.organization_id == organization_id,
            User.role == "owner",
            User.actif.is_(True),
            User.id != excluding_user_id,
        )
        .count()
    )


class TeamMemberStatusUpdate(BaseModel):
    actif: bool


class TeamMemberRoleUpdate(BaseModel):
    role: Literal["owner", "member"]


@router.patch("/team/members/{member_id}/role", response_model=TeamMemberProfile)
def set_team_member_role(
    member_id: int,
    body: TeamMemberRoleUpdate,
    owner: User = Depends(require_owner),
    db: Session = Depends(get_db),
):
    """[Owner] Promeut un membre propriétaire, ou rétrograde un propriétaire.

    Répond à un blocage total : jusqu'ici `register` créait l'UNIQUE
    propriétaire d'une organisation, `add_team_member` ne créait que des
    `member`, et aucun endpoint ne changeait un rôle. Si ce propriétaire
    quittait l'entreprise ou perdait son accès, plus personne ne pouvait
    gérer l'équipe ni lire les retours — organisation définitivement
    bloquée, seule une écriture directe en base la débloquait.

    La succession se fait donc en deux temps, tous deux couverts ici :
    promouvoir son successeur, puis se rétrograder soi-même (autorisé —
    c'est précisément le scénario du départ) ou se faire révoquer par lui.

    Invariant protégé : il reste TOUJOURS au moins un propriétaire actif
    (voir `_count_other_active_owners`). Promouvoir est donc toujours sûr ;
    seule la rétrogradation peut être refusée.

    Promouvoir un membre désactivé est permis : ça ne casse pas
    l'invariant (il n'est pas compté comme actif) et ça laisse préparer
    une succession avant de réactiver le compte."""
    member = (
        db.query(User)
        .filter(User.id == member_id, User.organization_id == owner.organization_id)
        .first()
    )
    if member is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": ErrorCode.AUTH_MEMBRE_INTROUVABLE, "message": "Membre introuvable"},
        )
    if member.role == body.role:
        return member  # déjà dans cet état — pas d'entrée d'audit pour un non-changement

    if body.role == "member" and _count_other_active_owners(db, owner.organization_id, member.id) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": ErrorCode.AUTH_DERNIER_PROPRIETAIRE,
                "message": (
                    "Impossible de rétrograder le dernier propriétaire actif — promouvez d'abord "
                    "un autre membre, sinon plus personne ne pourrait gérer cette organisation"
                ),
            },
        )

    member.role = body.role
    log_action(
        db, owner.organization_id, owner.id,
        "member.promoted" if body.role == "owner" else "member.demoted",
        target_type="user", target_id=member.id, details={"email": member.email, "nom": member.nom},
    )
    db.commit()
    db.refresh(member)
    return member


@router.patch("/team/members/{member_id}", response_model=TeamMemberProfile)
def set_team_member_status(
    member_id: int,
    body: TeamMemberStatusUpdate,
    owner: User = Depends(require_owner),
    db: Session = Depends(get_db),
):
    """[Owner] Désactive ou réactive un membre de SON organisation.

    Comble un vrai trou d'offboarding : jusqu'ici `User.actif` n'était
    écrit qu'à la création (toujours `True`) et jamais repassé à `False`.
    Toute la moitié LECTURE du mécanisme existait pourtant déjà —
    `get_current_user` refuse un compte inactif, et un message dédié
    invite l'utilisateur à « contacter le propriétaire de votre
    organisation » — mais aucun endpoint ne permettait au propriétaire
    d'agir. Concrètement, un collaborateur parti conservait un accès
    complet aux datasets, modèles et prédictions de l'organisation, sans
    autre recours qu'une modification directe en base.

    DÉSACTIVATION plutôt que suppression, délibérément : datasets,
    entraînements et journal d'audit référencent l'utilisateur
    (`Dataset.uploaded_by`, `created_by` des jobs, `AuditLog.user_id`).
    Le supprimer détruirait en cascade du travail dont l'organisation a
    encore besoin, et surtout la trace d'audit — dont c'est précisément
    la raison d'être. Le compte reste, sans accès ; l'action est
    réversible (`actif: true`).

    Coupure IMMÉDIATE, pas à l'expiration du jeton : `token_valid_after`
    invalide les jetons d'accès déjà émis (voir `get_current_user`) et
    `revoke_all_refresh_tokens` empêche d'en regénérer — même mécanisme
    que le changement de mot de passe. Sans cela l'accès survivrait
    jusqu'à la fin de validité du jeton en cours, ce qui viderait la
    désactivation de son sens le jour où elle sert vraiment.

    La réactivation ne touche PAS `token_valid_after` : les anciens
    jetons, émis avant la désactivation, restent définitivement
    invalides — la personne se reconnecte."""
    member = (
        db.query(User)
        .filter(User.id == member_id, User.organization_id == owner.organization_id)
        .first()
    )
    if member is None:
        # 404 (et non 403) pour un compte d'une AUTRE organisation :
        # ne jamais révéler l'existence d'un utilisateur hors de la sienne.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": ErrorCode.AUTH_MEMBRE_INTROUVABLE, "message": "Membre introuvable"},
        )
    if member.id == owner.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": ErrorCode.AUTH_AUTO_DESACTIVATION_INTERDITE,
                "message": "Vous ne pouvez pas modifier l'accès de votre propre compte de propriétaire",
            },
        )
    # Pas de garde "dernier propriétaire actif" ICI, contrairement à
    # `set_team_member_role` — et ce n'est pas un oubli. Révoquer ne peut
    # PAS orpheliner l'organisation, par construction : `require_owner`
    # impose que l'auteur soit un propriétaire ACTIF (`get_current_user`
    # refuse un compte inactif), et le garde-fou ci-dessus impose que la
    # cible soit quelqu'un d'autre. Il reste donc toujours au moins un
    # propriétaire actif après l'opération : l'auteur lui-même. Une garde
    # supplémentaire ici serait du code mort déguisé en protection —
    # rassurant à la lecture, jamais exécuté. La rétrogradation, elle, EST
    # concernée : un propriétaire peut se rétrograder lui-même.
    member.actif = body.actif
    if not body.actif:
        member.token_valid_after = datetime.now(timezone.utc)
        member.deactivated_at = datetime.now(timezone.utc)
    else:
        # Réactivation : la date de révocation n'a plus de sens. On ne
        # conserve pas l'historique des révocations successives ici — le
        # journal d'audit le fait déjà, avec l'auteur de chaque geste.
        member.deactivated_at = None

    log_action(
        db, owner.organization_id, owner.id,
        "member.reactivated" if body.actif else "member.deactivated",
        target_type="user", target_id=member.id, details={"email": member.email, "nom": member.nom},
    )
    db.commit()
    db.refresh(member)

    # APRÈS le commit : ne jamais révoquer les jetons d'un changement qui
    # n'a pas été persisté (une transaction annulée laisserait la personne
    # déconnectée alors qu'elle est toujours active).
    if not body.actif:
        revoke_all_refresh_tokens(redis_conn, member.id)
    return member


class AuditLogEntry(BaseModel):
    id: int
    action: str
    target_type: Optional[str] = None
    target_id: Optional[int] = None
    details: Optional[dict] = None
    actor_name: Optional[str] = None
    # Phase 3 (AUDIT_BACKEND_2026-08-23.md, Axe I) — permet à un owner qui
    # investigue un incident de retrouver TOUTES les entrées d'audit
    # produites par la même requête HTTP (rare mais réel : un rollback
    # partiel, ou une action qui en déclenche une autre côté serveur).
    request_id: Optional[str] = None
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
            request_id=e.request_id,
            created_at=e.created_at,
        )
        for e in entries
    ]
