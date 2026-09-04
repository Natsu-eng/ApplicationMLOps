"""Réinitialisation de mot de passe — Phase 1B (AUDIT_BACKEND_2026-08-23.md).

Le canal SMTP n'est pas configuré dans l'environnement de test
(`mailer_configured()` retourne False) — ces tests vérifient donc le
comportement HTTP/DB/audit, pas l'envoi mail réel lui-même (`api/core/mailer.py`
n'a pas de dépendance testée ici, `smtplib` n'est jamais appelé). C'est
volontaire : le jeton en clair reste accessible directement en base pour ces
tests (`token_hash` est déterministe depuis le jeton), sans avoir besoin
d'intercepter un mail."""
from __future__ import annotations

import time

from api.core.models import PasswordResetToken


def _register(client, email="reset@bureau.fr", password="ancien-mdp-123"):
    resp = client.post(
        "/api/auth/register",
        json={"email": email, "nom": "Reset", "password": password, "organization_name": "Bureau"},
    )
    return resp.json()


def _request_reset(client, email):
    return client.post("/api/auth/password-reset/request", json={"email": email})


def _extract_raw_token_from_db(db_session, monkeypatch) -> str:
    """Le jeton en clair n'est jamais persisté ni renvoyé par l'API (voir
    `_issue_password_reset_token`) — pour le test, on intercepte
    `secrets.token_urlsafe` afin de connaître la valeur émise, plutôt que de
    contourner la fonction de hachage elle-même (qui doit rester testée
    telle qu'elle s'exécute en production)."""
    captured = {}
    import domains.auth.router as auth_router

    original = auth_router.secrets.token_urlsafe

    def _spy(*args, **kwargs):
        raw = original(*args, **kwargs)
        captured["token"] = raw
        return raw

    monkeypatch.setattr(auth_router.secrets, "token_urlsafe", _spy)
    return captured


def test_request_reset_returns_204_for_known_and_unknown_email_identically(client):
    _register(client, "known@bureau.fr")
    known_resp = _request_reset(client, "known@bureau.fr")
    unknown_resp = _request_reset(client, "personne@nulle-part.fr")
    assert known_resp.status_code == unknown_resp.status_code == 204
    assert known_resp.content == unknown_resp.content == b""


def test_request_reset_response_time_does_not_leak_account_existence(client):
    """Point 7 de la mission — la seule façon de prouver l'absence d'oracle :
    même ordre de grandeur de temps de réponse, existant ou non. Marge large
    (10x) : ce test vérifie l'ABSENCE d'un envoi SMTP synchrone bloquant
    (déjà garanti par BackgroundTasks), pas une micro-optimisation."""
    _register(client, "timing@bureau.fr")

    start_known = time.perf_counter()
    _request_reset(client, "timing@bureau.fr")
    duration_known = time.perf_counter() - start_known

    start_unknown = time.perf_counter()
    _request_reset(client, "personne-timing@nulle-part.fr")
    duration_unknown = time.perf_counter() - start_unknown

    slower, faster = max(duration_known, duration_unknown), max(min(duration_known, duration_unknown), 1e-6)
    assert slower / faster < 10


def test_request_reset_creates_token_only_for_existing_active_user(client, db_session):
    _register(client, "exists@bureau.fr")
    _request_reset(client, "exists@bureau.fr")
    _request_reset(client, "nexistepas@bureau.fr")

    tokens = db_session.query(PasswordResetToken).all()
    assert len(tokens) == 1
    assert tokens[0].user.email == "exists@bureau.fr"
    assert tokens[0].requested_from_ip  # renseigné, jamais vide


def test_confirm_reset_with_valid_token_changes_password_and_revokes_sessions(client, db_session, monkeypatch):
    tokens_before = _register(client, "confirm@bureau.fr", password="ancien-mdp-123")
    old_access_headers = {"Authorization": f"Bearer {tokens_before['access_token']}"}

    captured = _extract_raw_token_from_db(db_session, monkeypatch)
    _request_reset(client, "confirm@bureau.fr")
    raw_token = captured["token"]

    confirm_resp = client.post(
        "/api/auth/password-reset/confirm",
        json={"token": raw_token, "new_password": "nouveau-mdp-789", "new_password_confirm": "nouveau-mdp-789"},
    )
    assert confirm_resp.status_code == 204

    # Toutes les sessions précédentes sont fermées (Phase 1B, point 1).
    old_token_check = client.get("/api/auth/me", headers=old_access_headers)
    assert old_token_check.status_code == 401

    # Le nouveau mot de passe fonctionne réellement.
    fresh_login = client.post(
        "/api/auth/login", data={"username": "confirm@bureau.fr", "password": "nouveau-mdp-789"}
    )
    assert fresh_login.status_code == 200


def test_confirm_reset_token_is_single_use(client, db_session, monkeypatch):
    _register(client, "singleuse@bureau.fr")
    captured = _extract_raw_token_from_db(db_session, monkeypatch)
    _request_reset(client, "singleuse@bureau.fr")
    raw_token = captured["token"]

    first = client.post(
        "/api/auth/password-reset/confirm",
        json={"token": raw_token, "new_password": "un-mot-de-passe-1", "new_password_confirm": "un-mot-de-passe-1"},
    )
    assert first.status_code == 204

    reused = client.post(
        "/api/auth/password-reset/confirm",
        json={"token": raw_token, "new_password": "un-mot-de-passe-2", "new_password_confirm": "un-mot-de-passe-2"},
    )
    assert reused.status_code == 400
    assert reused.json()["detail"]["code"] == "AUTH_RESET_TOKEN_INVALIDE"


def test_confirm_reset_unknown_token_generic_error(client):
    resp = client.post(
        "/api/auth/password-reset/confirm",
        json={"token": "jeton-invente", "new_password": "un-mot-de-passe-1", "new_password_confirm": "un-mot-de-passe-1"},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "AUTH_RESET_TOKEN_INVALIDE"


def test_confirm_reset_rejects_weak_password(client, db_session, monkeypatch):
    _register(client, "weakreset@bureau.fr")
    captured = _extract_raw_token_from_db(db_session, monkeypatch)
    _request_reset(client, "weakreset@bureau.fr")
    raw_token = captured["token"]

    resp = client.post(
        "/api/auth/password-reset/confirm",
        json={"token": raw_token, "new_password": "password1", "new_password_confirm": "password1"},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "AUTH_MDP_TROP_FAIBLE"


def test_new_reset_request_invalidates_previous_unused_token(client, db_session):
    _register(client, "double@bureau.fr")
    _request_reset(client, "double@bureau.fr")
    first_token = db_session.query(PasswordResetToken).filter(PasswordResetToken.used_at.is_(None)).one()

    _request_reset(client, "double@bureau.fr")
    db_session.refresh(first_token)
    assert first_token.used_at is not None  # invalidé par la nouvelle demande


def test_request_reset_rate_limited_by_email_stays_neutral(client):
    """Durcissement au-delà de CIAM (point 2) — la limite par compte
    répond aussi 204, jamais un signal distinct."""
    from api.core.config import get_settings

    _register(client, "flood@bureau.fr")
    limit = get_settings().password_reset_rate_limit_max_attempts_per_email
    responses = [_request_reset(client, "flood@bureau.fr") for _ in range(limit + 2)]
    assert all(r.status_code == 204 for r in responses)


def test_voluntary_password_change_still_works_after_phase1b(client):
    """Non-régression : le changement volontaire (Phase 1) n'a pas été
    cassé par l'ajout de BackgroundTasks à sa signature."""
    tokens = _register(client, "voluntary@bureau.fr", password="ancien-mdp-123")
    resp = client.patch(
        "/api/auth/me/password",
        headers={"Authorization": f"Bearer {tokens['access_token']}"},
        json={
            "current_password": "ancien-mdp-123",
            "new_password": "nouveau-mdp-999",
            "new_password_confirm": "nouveau-mdp-999",
        },
    )
    assert resp.status_code == 204


def test_password_reset_clears_the_provisional_password_flag(client, db_session, monkeypatch):
    """Un membre au mot de passe provisoire qui passe par « mot de passe
    oublié » choisit bien SON mot de passe, via un lien envoyé à son
    adresse : le propriétaire ne le connaît plus. L'indicateur doit donc
    tomber là aussi, sinon l'intéressé reste bloqué derrière l'écran de
    changement obligatoire et doit en choisir un second sans comprendre."""
    from api.core.models import User

    owner = client.post(
        "/api/auth/register",
        json={"email": "chef@oubli.fr", "nom": "Chef", "password": "motdepasse123", "organization_name": "Bureau"},
    ).json()
    client.post(
        "/api/auth/team/members",
        headers={"Authorization": f"Bearer {owner['access_token']}"},
        json={"email": "oublieux@oubli.fr", "nom": "Oublieux", "password": "motdepasse123"},
    )
    assert db_session.query(User).filter(User.email == "oublieux@oubli.fr").first().must_change_password is True

    captured = _extract_raw_token_from_db(db_session, monkeypatch)
    _request_reset(client, "oublieux@oubli.fr")
    resp = client.post(
        "/api/auth/password-reset/confirm",
        json={
            "token": captured["token"],
            "new_password": "sonpropremdp456",
            "new_password_confirm": "sonpropremdp456",
        },
    )
    assert resp.status_code == 204

    db_session.expire_all()
    assert db_session.query(User).filter(User.email == "oublieux@oubli.fr").first().must_change_password is False

    # Et concrètement : il accède à l'API sans repasser par un second
    # changement de mot de passe.
    time.sleep(1.05)
    login = client.post("/api/auth/login", data={"username": "oublieux@oubli.fr", "password": "sonpropremdp456"})
    assert login.status_code == 200
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    assert client.get("/api/datasets", headers=headers).status_code == 200
