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
