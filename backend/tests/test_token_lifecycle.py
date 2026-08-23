"""Cycle de vie des jetons — révocation réelle (Phase 1,
AUDIT_BACKEND_2026-08-23.md, Axe A).

Avant ce correctif : un seul JWT stateless de 24h, `POST /auth/logout` ne
faisait rien côté serveur, changer de mot de passe n'invalidait aucune
session existante. Ces tests échouent tous sans les correctifs de
`api/core/security.py`/`api/core/token_store.py`/`domains/auth/router.py`.
"""
from __future__ import annotations


def _register(client, email="user@bureau.fr", password="motdepasse123"):
    resp = client.post(
        "/api/auth/register",
        json={"email": email, "nom": "User", "password": password, "organization_name": "Bureau"},
    )
    assert resp.status_code == 201
    return resp.json()


def test_login_returns_both_tokens(client):
    _register(client, "a@bureau.fr")
    resp = client.post("/api/auth/login", data={"username": "a@bureau.fr", "password": "motdepasse123"})
    body = resp.json()
    assert body["access_token"]
    assert body["refresh_token"]
    assert body["access_token"] != body["refresh_token"]


def test_refresh_issues_new_working_access_token(client):
    tokens = _register(client, "b@bureau.fr")
    resp = client.post("/api/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert resp.status_code == 200
    new_tokens = resp.json()
    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {new_tokens['access_token']}"})
    assert me.status_code == 200
    assert me.json()["email"] == "b@bureau.fr"


def test_refresh_token_is_single_use_rotation(client):
    """C'est le coeur de la rotation : réutiliser un refresh token déjà
    échangé doit échouer, même s'il n'a pas expiré — sinon un refresh token
    volé une seule fois reste exploitable indéfiniment en parallèle du
    légitime."""
    tokens = _register(client, "c@bureau.fr")
    first = client.post("/api/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert first.status_code == 200

    reused = client.post("/api/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert reused.status_code == 401
    assert reused.json()["detail"]["code"] == "AUTH_TOKEN_INVALIDE"


def test_refresh_token_cannot_be_used_as_access_token(client):
    tokens = _register(client, "d@bureau.fr")
    resp = client.get("/api/auth/me", headers={"Authorization": f"Bearer {tokens['refresh_token']}"})
    assert resp.status_code == 401


def test_access_token_cannot_be_used_on_refresh_endpoint(client):
    tokens = _register(client, "e@bureau.fr")
    resp = client.post("/api/auth/refresh", json={"refresh_token": tokens["access_token"]})
    assert resp.status_code == 401


def test_logout_revokes_the_access_token_immediately(client):
    tokens = _register(client, "f@bureau.fr")
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}

    logout_resp = client.post("/api/auth/logout", headers=headers, json={})
    assert logout_resp.status_code == 204

    still_using_it = client.get("/api/auth/me", headers=headers)
    assert still_using_it.status_code == 401
    assert still_using_it.json()["detail"]["code"] == "AUTH_TOKEN_INVALIDE"


def test_logout_with_refresh_token_revokes_it_too(client):
    tokens = _register(client, "g@bureau.fr")
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}

    logout_resp = client.post("/api/auth/logout", headers=headers, json={"refresh_token": tokens["refresh_token"]})
    assert logout_resp.status_code == 204

    refresh_after_logout = client.post("/api/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert refresh_after_logout.status_code == 401


def test_password_change_revokes_all_existing_sessions(client):
    """Le scénario que la Phase 1 vise explicitement : un utilisateur qui
    change son mot de passe parce qu'il se croit compromis doit chasser
    quiconque détiendrait déjà un jeton — access ET refresh."""
    tokens = _register(client, "h@bureau.fr", password="ancien-mdp-123")
    old_access_headers = {"Authorization": f"Bearer {tokens['access_token']}"}

    change_resp = client.patch(
        "/api/auth/me/password",
        headers=old_access_headers,
        json={
            "current_password": "ancien-mdp-123",
            "new_password": "nouveau-mdp-456",
            "new_password_confirm": "nouveau-mdp-456",
        },
    )
    assert change_resp.status_code == 204

    old_token_still_works = client.get("/api/auth/me", headers=old_access_headers)
    assert old_token_still_works.status_code == 401
    assert old_token_still_works.json()["detail"]["code"] == "AUTH_TOKEN_INVALIDE"

    old_refresh_still_works = client.post("/api/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert old_refresh_still_works.status_code == 401

    fresh_login = client.post(
        "/api/auth/login", data={"username": "h@bureau.fr", "password": "nouveau-mdp-456"}
    )
    assert fresh_login.status_code == 200


def test_unknown_or_garbage_refresh_token_rejected(client):
    resp = client.post("/api/auth/refresh", json={"refresh_token": "not-a-real-jwt"})
    assert resp.status_code == 401
