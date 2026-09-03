"""Tests des routeurs `preferences_router` et `feedback_router`.

Ces 4 routes (GET/PATCH /users/me/preferences, POST/GET /feedback)
n'avaient AUCUN test automatisé — trou de couverture constaté en les
extrayant de `domains/auth/router.py` vers leurs modules dédiés. Le
découpage lui-même a été vérifié autrement (inventaire des routes
comparé une à une, mypy, fumée manuelle), mais l'absence de test
laissait sans filet deux comportements de sécurité explicitement
documentés dans le code :

- `GET /feedback` est réservé au owner. Le commentaire de section
  affirmait cet accès réservé depuis le Lot 10, alors que la route
  utilisait `get_current_user` — tout membre ordinaire pouvait lire les
  retours de ses collègues (corrigé en Phase 1, AUDIT_BACKEND_2026-08-23,
  Axe B). Rien ne testait le correctif : une régression le réintroduirait
  silencieusement.
- L'isolation par `organization_id` sur les retours, jamais vérifiée non
  plus alors que c'est l'invariant multi-tenant de toute l'application.

Le validateur de thème est testé pour la même raison : il rejette les
valeurs hors liste, et ce rejet ne doit pas laisser la préférence dans un
état partiellement modifié.
"""
from __future__ import annotations

import time

MDP = "motdepasse123"
MDP_FINAL = "monmotdepasse456"


def _register(client, email: str, org: str, nom: str = "Test") -> dict:
    """Crée une organisation + son owner, renvoie les en-têtes d'auth."""
    resp = client.post(
        "/api/auth/register",
        json={"email": email, "nom": nom, "password": MDP, "organization_name": org},
    )
    assert resp.status_code == 201, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _add_member(client, owner_headers: dict, email: str, nom: str = "Membre") -> dict:
    """Ajoute un membre ordinaire (non-owner), lui fait remplacer le mot de
    passe provisoire fixé par le propriétaire, et renvoie ses en-têtes.

    Ce remplacement n'est pas cosmétique : tant qu'il n'a pas eu lieu, l'API
    refuse tout appel avec AUTH_MDP_PROVISOIRE (voir `get_current_user`) —
    un membre non intégré ne pourrait donc ni lire ses préférences ni
    déposer un retour."""
    resp = client.post(
        "/api/auth/team/members",
        headers=owner_headers,
        json={"email": email, "nom": nom, "password": MDP},
    )
    assert resp.status_code == 201, resp.text
    first = client.post("/api/auth/login", data={"username": email, "password": MDP})
    assert first.status_code == 200, first.text
    changed = client.patch(
        "/api/auth/me/password",
        headers={"Authorization": f"Bearer {first.json()['access_token']}"},
        json={"current_password": MDP, "new_password": MDP_FINAL, "new_password_confirm": MDP_FINAL},
    )
    assert changed.status_code == 204, changed.text
    # Le changement révoque toutes les sessions : reconnexion obligatoire.
    # Pause d'une seconde : un jeton émis dans la même seconde que la
    # révocation est rejeté (granularité de `iat`, voir get_current_user).
    time.sleep(1.05)
    again = client.post("/api/auth/login", data={"username": email, "password": MDP_FINAL})
    assert again.status_code == 200, again.text
    return {"Authorization": f"Bearer {again.json()['access_token']}"}


# ── Préférences d'interface ──────────────────────────────────────────────────


def test_preferences_require_token(client):
    assert client.get("/api/users/me/preferences").status_code == 401
    assert client.patch("/api/users/me/preferences", json={"ui_theme": "minuit"}).status_code == 401


def test_default_theme_then_update_persists(client):
    headers = _register(client, "owner@pref.fr", "Bureau Pref")

    initial = client.get("/api/users/me/preferences", headers=headers)
    assert initial.status_code == 200
    assert initial.json()["ui_theme"] == "graphite"  # server_default du modèle User

    updated = client.patch("/api/users/me/preferences", headers=headers, json={"ui_theme": "minuit"})
    assert updated.status_code == 200
    assert updated.json()["ui_theme"] == "minuit"

    # Relu dans une requête distincte : la valeur est bien persistée, pas
    # seulement renvoyée par l'écriture.
    assert client.get("/api/users/me/preferences", headers=headers).json()["ui_theme"] == "minuit"


def test_invalid_theme_rejected_and_leaves_preference_untouched(client):
    headers = _register(client, "owner@theme.fr", "Bureau Theme")
    client.patch("/api/users/me/preferences", headers=headers, json={"ui_theme": "ardoise"})

    refused = client.patch("/api/users/me/preferences", headers=headers, json={"ui_theme": "inexistant"})
    assert refused.status_code == 422

    # Le rejet ne doit pas avoir écrasé la préférence précédente.
    assert client.get("/api/users/me/preferences", headers=headers).json()["ui_theme"] == "ardoise"


def test_preferences_are_per_user(client):
    owner_headers = _register(client, "owner@iso.fr", "Bureau Iso")
    member_headers = _add_member(client, owner_headers, "membre@iso.fr")

    client.patch("/api/users/me/preferences", headers=member_headers, json={"ui_theme": "porcelaine"})

    # Le thème du membre ne déborde pas sur celui de l'owner.
    assert client.get("/api/users/me/preferences", headers=owner_headers).json()["ui_theme"] == "graphite"
    assert client.get("/api/users/me/preferences", headers=member_headers).json()["ui_theme"] == "porcelaine"


# ── Retour utilisateur ───────────────────────────────────────────────────────


def test_feedback_requires_token(client):
    assert client.get("/api/feedback").status_code == 401
    assert client.post("/api/feedback", json={"page": "/x", "message": "y"}).status_code == 401


def test_owner_creates_and_lists_feedback(client):
    headers = _register(client, "owner@fb.fr", "Bureau FB", nom="Le Owner")

    created = client.post("/api/feedback", headers=headers, json={"page": "/datasets", "message": "Souci ici"})
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["page"] == "/datasets"
    assert body["message"] == "Souci ici"
    assert body["author_name"] == "Le Owner"

    listed = client.get("/api/feedback", headers=headers)
    assert listed.status_code == 200
    assert [(e["page"], e["author_name"]) for e in listed.json()] == [("/datasets", "Le Owner")]


def test_member_can_post_feedback_but_never_list_it(client):
    """Correctif Phase 1 (Axe B) : la lecture est réservée au owner.

    Le membre doit pouvoir SIGNALER un problème (c'est l'objet de la
    fonctionnalité) sans pouvoir lire les retours de ses collègues.
    """
    owner_headers = _register(client, "owner@role.fr", "Bureau Role")
    member_headers = _add_member(client, owner_headers, "membre@role.fr", nom="Le Membre")

    posted = client.post("/api/feedback", headers=member_headers, json={"page": "/vision", "message": "Bug vision"})
    assert posted.status_code == 201

    refused = client.get("/api/feedback", headers=member_headers)
    assert refused.status_code == 403
    assert refused.json()["detail"]["code"] == "AUTH_OWNER_REQUIS"

    # L'owner, lui, voit bien le retour déposé par son membre.
    assert [e["author_name"] for e in client.get("/api/feedback", headers=owner_headers).json()] == ["Le Membre"]


def test_feedback_isolated_between_organizations(client):
    """Invariant multi-tenant : un owner ne voit QUE les retours de son org."""
    headers_a = _register(client, "a@org-a.fr", "Org A", nom="Aa")
    headers_b = _register(client, "b@org-b.fr", "Org B", nom="Bb")

    client.post("/api/feedback", headers=headers_a, json={"page": "/a", "message": "retour de A"})
    client.post("/api/feedback", headers=headers_b, json={"page": "/b", "message": "retour de B"})

    seen_by_b = client.get("/api/feedback", headers=headers_b).json()
    assert [e["message"] for e in seen_by_b] == ["retour de B"]

    seen_by_a = client.get("/api/feedback", headers=headers_a).json()
    assert [e["message"] for e in seen_by_a] == ["retour de A"]


def test_feedback_rejects_empty_and_oversized_input(client):
    headers = _register(client, "owner@valid.fr", "Bureau Valid")

    assert client.post("/api/feedback", headers=headers, json={"page": "", "message": "ok"}).status_code == 422
    assert client.post("/api/feedback", headers=headers, json={"page": "/x", "message": ""}).status_code == 422
    trop_long = {"page": "/x", "message": "a" * 4001}
    assert client.post("/api/feedback", headers=headers, json=trop_long).status_code == 422

    # Aucune de ces tentatives n'a été enregistrée.
    assert client.get("/api/feedback", headers=headers).json() == []
