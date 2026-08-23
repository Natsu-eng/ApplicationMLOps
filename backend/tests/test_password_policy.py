"""Robustesse du mot de passe — un seul point de vérité côté serveur
(Phase 1B, AUDIT_BACKEND_2026-08-23.md, point 5)."""
from __future__ import annotations

import pytest

from api.core.password_policy import validate_password_strength


def test_short_password_rejected():
    with pytest.raises(ValueError, match="8 caractères"):
        validate_password_strength("court1")


def test_common_password_rejected():
    with pytest.raises(ValueError, match="trop courant"):
        validate_password_strength("password1")


def test_password_containing_email_rejected():
    with pytest.raises(ValueError, match="adresse e-mail"):
        validate_password_strength("marie.dupont-secure99", "marie.dupont@bureau.fr")


def test_reasonable_password_accepted():
    validate_password_strength("un-mot-de-passe-correct-42", "alice@bureau.fr")  # ne lève pas


def test_register_rejects_common_password(client):
    resp = client.post(
        "/api/auth/register",
        json={"email": "z@bureau.fr", "nom": "Zoe", "password": "password1", "organization_name": "Bureau"},
    )
    assert resp.status_code == 422
    assert "courant" in resp.json()["detail"]["message"]


def test_change_password_rejects_weak_new_password(client):
    tokens = client.post(
        "/api/auth/register",
        json={"email": "weak@bureau.fr", "nom": "Weak", "password": "motdepasse123", "organization_name": "Bureau"},
    ).json()
    resp = client.patch(
        "/api/auth/me/password",
        headers={"Authorization": f"Bearer {tokens['access_token']}"},
        json={
            "current_password": "motdepasse123",
            "new_password": "password1",
            "new_password_confirm": "password1",
        },
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "AUTH_MDP_TROP_FAIBLE"
