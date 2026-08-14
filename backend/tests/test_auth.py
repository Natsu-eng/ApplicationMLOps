"""Tests du router auth (Lot 1) — inscription, connexion, équipe, isolation."""
from __future__ import annotations


def test_register_creates_organization_and_owner(client):
    resp = client.post(
        "/auth/register",
        json={
            "email": "alice@bureau-a.fr",
            "nom": "Alice",
            "password": "motdepasse123",
            "organization_name": "Bureau A",
        },
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["role"] == "owner"
    assert body["organization_name"] == "Bureau A"
    assert "access_token" in body


def test_register_duplicate_email_rejected(client):
    payload = {
        "email": "bob@bureau.fr",
        "nom": "Bob",
        "password": "motdepasse123",
        "organization_name": "Bureau",
    }
    assert client.post("/auth/register", json=payload).status_code == 201

    resp = client.post("/auth/register", json=payload)
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "AUTH_EMAIL_DEJA_UTILISE"


def test_login_wrong_password_rejected(client):
    client.post(
        "/auth/register",
        json={"email": "carla@bureau.fr", "nom": "Carla", "password": "motdepasse123", "organization_name": "Bureau"},
    )
    resp = client.post("/auth/login", data={"username": "carla@bureau.fr", "password": "faux"})
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "AUTH_IDENTIFIANTS_INCORRECTS"


# ── H11 (AUDIT_ROADMAP.md) — rate-limiting sur /auth/login ──────────────


def test_login_blocked_after_too_many_failed_attempts(client):
    from api.core.config import get_settings

    client.post(
        "/auth/register",
        json={"email": "dana@bureau.fr", "nom": "Dana", "password": "motdepasse123", "organization_name": "Bureau"},
    )
    limit = get_settings().login_rate_limit_max_attempts

    responses = [
        client.post("/auth/login", data={"username": "dana@bureau.fr", "password": "faux"}) for _ in range(limit)
    ]
    assert all(r.status_code == 400 for r in responses)

    blocked = client.post("/auth/login", data={"username": "dana@bureau.fr", "password": "faux"})
    assert blocked.status_code == 429
    assert blocked.json()["detail"]["code"] == "AUTH_TROP_DE_TENTATIVES"

    # Même avec le bon mot de passe : la limite s'applique par IP, avant
    # toute vérification des identifiants.
    still_blocked = client.post("/auth/login", data={"username": "dana@bureau.fr", "password": "motdepasse123"})
    assert still_blocked.status_code == 429


def test_login_success_resets_rate_limit_counter(client):
    client.post(
        "/auth/register",
        json={"email": "eva@bureau.fr", "nom": "Eva", "password": "motdepasse123", "organization_name": "Bureau"},
    )
    # Deux échecs, bien sous la limite, puis un succès.
    client.post("/auth/login", data={"username": "eva@bureau.fr", "password": "faux"})
    client.post("/auth/login", data={"username": "eva@bureau.fr", "password": "faux"})
    ok = client.post("/auth/login", data={"username": "eva@bureau.fr", "password": "motdepasse123"})
    assert ok.status_code == 200

    # Le compteur a été remis à zéro par le succès — un nouvel échec isolé
    # ne doit pas être proche d'atteindre la limite.
    resp = client.post("/auth/login", data={"username": "eva@bureau.fr", "password": "faux"})
    assert resp.status_code == 400  # pas 429


def test_login_rate_limit_isolated_by_client_and_never_blocks_registration(client):
    """La limite porte sur /auth/login (brute force de mot de passe), jamais
    sur /auth/register — un pic d'inscriptions légitimes ne doit jamais être
    confondu avec une attaque par force brute sur un mot de passe."""
    for i in range(15):
        resp = client.post(
            "/auth/register",
            json={
                "email": f"user{i}@bureau.fr",
                "nom": f"User {i}",
                "password": "motdepasse123",
                "organization_name": f"Bureau {i}",
            },
        )
        assert resp.status_code == 201


def test_me_requires_token(client):
    assert client.get("/auth/me").status_code == 401


def test_owner_can_add_member_but_member_cannot(client):
    owner = client.post(
        "/auth/register",
        json={"email": "owner@bureau.fr", "nom": "Owner", "password": "motdepasse123", "organization_name": "Bureau"},
    ).json()
    owner_headers = {"Authorization": f"Bearer {owner['access_token']}"}

    add_resp = client.post(
        "/auth/team/members",
        headers=owner_headers,
        json={"email": "membre@bureau.fr", "nom": "Membre", "password": "motdepasse123"},
    )
    assert add_resp.status_code == 201

    member_login = client.post(
        "/auth/login", data={"username": "membre@bureau.fr", "password": "motdepasse123"}
    ).json()
    member_headers = {"Authorization": f"Bearer {member_login['access_token']}"}

    forbidden = client.post(
        "/auth/team/members",
        headers=member_headers,
        json={"email": "autre@bureau.fr", "nom": "Autre", "password": "motdepasse123"},
    )
    assert forbidden.status_code == 403
    assert forbidden.json()["detail"]["code"] == "AUTH_OWNER_REQUIS"


def test_team_isolation_between_organizations(client):
    client.post(
        "/auth/register",
        json={"email": "a@bureau-a.fr", "nom": "Aa", "password": "motdepasse123", "organization_name": "Bureau A"},
    )
    org_b = client.post(
        "/auth/register",
        json={"email": "b@bureau-b.fr", "nom": "Bb", "password": "motdepasse123", "organization_name": "Bureau B"},
    ).json()

    headers_b = {"Authorization": f"Bearer {org_b['access_token']}"}
    members = client.get("/auth/team/members", headers=headers_b).json()
    assert len(members) == 1
    assert members[0]["email"] == "b@bureau-b.fr"
