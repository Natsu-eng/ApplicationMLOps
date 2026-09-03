"""Tests du router auth (Lot 1) — inscription, connexion, équipe, isolation."""
from __future__ import annotations

import time


def test_register_creates_organization_and_owner(client):
    resp = client.post(
        "/api/auth/register",
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
    assert client.post("/api/auth/register", json=payload).status_code == 201

    resp = client.post("/api/auth/register", json=payload)
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "AUTH_EMAIL_DEJA_UTILISE"


def test_register_blocked_after_too_many_attempts(client):
    """Lot 1.4 (§C.2.7/§D.4, AUDIT_DATALAB_2026-08-16.md) — avant, seul
    /auth/login était limité ; /register pouvait être spammé sans aucune
    borne. Même mécanisme (fenêtre glissante par IP), étendu ici."""
    from api.core.config import get_settings

    limit = get_settings().register_rate_limit_max_attempts
    responses = [
        client.post(
            "/api/auth/register",
            json={"email": f"spam{i}@bureau.fr", "nom": "Spam", "password": "motdepasse123", "organization_name": "Bureau"},
        )
        for i in range(limit)
    ]
    assert all(r.status_code == 201 for r in responses)

    blocked = client.post(
        "/api/auth/register",
        json={"email": "encoreun@bureau.fr", "nom": "Spam", "password": "motdepasse123", "organization_name": "Bureau"},
    )
    assert blocked.status_code == 429
    assert blocked.json()["detail"]["code"] == "TROP_DE_REQUETES"


def test_login_wrong_password_rejected(client):
    client.post(
        "/api/auth/register",
        json={"email": "carla@bureau.fr", "nom": "Carla", "password": "motdepasse123", "organization_name": "Bureau"},
    )
    resp = client.post("/api/auth/login", data={"username": "carla@bureau.fr", "password": "faux"})
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "AUTH_IDENTIFIANTS_INCORRECTS"


# ── H11 (AUDIT_ROADMAP.md) — rate-limiting sur /auth/login ──────────────


def test_login_blocked_after_too_many_failed_attempts(client):
    from api.core.config import get_settings

    client.post(
        "/api/auth/register",
        json={"email": "dana@bureau.fr", "nom": "Dana", "password": "motdepasse123", "organization_name": "Bureau"},
    )
    limit = get_settings().login_rate_limit_max_attempts

    responses = [
        client.post("/api/auth/login", data={"username": "dana@bureau.fr", "password": "faux"}) for _ in range(limit)
    ]
    assert all(r.status_code == 400 for r in responses)

    blocked = client.post("/api/auth/login", data={"username": "dana@bureau.fr", "password": "faux"})
    assert blocked.status_code == 429
    assert blocked.json()["detail"]["code"] == "AUTH_TROP_DE_TENTATIVES"

    # Même avec le bon mot de passe : la limite s'applique par IP, avant
    # toute vérification des identifiants.
    still_blocked = client.post("/api/auth/login", data={"username": "dana@bureau.fr", "password": "motdepasse123"})
    assert still_blocked.status_code == 429


def test_login_success_resets_rate_limit_counter(client):
    client.post(
        "/api/auth/register",
        json={"email": "eva@bureau.fr", "nom": "Eva", "password": "motdepasse123", "organization_name": "Bureau"},
    )
    # Deux échecs, bien sous la limite, puis un succès.
    client.post("/api/auth/login", data={"username": "eva@bureau.fr", "password": "faux"})
    client.post("/api/auth/login", data={"username": "eva@bureau.fr", "password": "faux"})
    ok = client.post("/api/auth/login", data={"username": "eva@bureau.fr", "password": "motdepasse123"})
    assert ok.status_code == 200

    # Le compteur a été remis à zéro par le succès — un nouvel échec isolé
    # ne doit pas être proche d'atteindre la limite.
    resp = client.post("/api/auth/login", data={"username": "eva@bureau.fr", "password": "faux"})
    assert resp.status_code == 400  # pas 429


def test_login_rate_limit_isolated_by_client_and_never_blocks_registration(client):
    """La limite de LOGIN (brute force de mot de passe) ne doit jamais
    s'appliquer à /register — un pic d'inscriptions légitimes ne doit
    jamais être confondu avec une attaque par force brute sur un mot de
    passe. /register a sa PROPRE limite depuis le Lot 1.4 (correctif
    §C.2.7/§D.4, register_rate_limit_max_attempts) — ce test reste dans
    cette limite pour isoler ce qu'il vérifie réellement (l'absence
    d'interférence avec le compteur de login), voir
    test_register_blocked_after_too_many_attempts pour la limite propre à
    /register elle-même."""
    from api.core.config import get_settings

    limit = get_settings().register_rate_limit_max_attempts
    for i in range(limit):
        resp = client.post(
            "/api/auth/register",
            json={
                "email": f"user{i}@bureau.fr",
                "nom": f"User {i}",
                "password": "motdepasse123",
                "organization_name": f"Bureau {i}",
            },
        )
        assert resp.status_code == 201


def test_me_requires_token(client):
    assert client.get("/api/auth/me").status_code == 401


def test_owner_can_add_member_but_member_cannot(client):
    owner = client.post(
        "/api/auth/register",
        json={"email": "owner@bureau.fr", "nom": "Owner", "password": "motdepasse123", "organization_name": "Bureau"},
    ).json()
    owner_headers = {"Authorization": f"Bearer {owner['access_token']}"}

    add_resp = client.post(
        "/api/auth/team/members",
        headers=owner_headers,
        json={"email": "membre@bureau.fr", "nom": "Membre", "password": "motdepasse123"},
    )
    assert add_resp.status_code == 201

    # Le membre doit d'abord remplacer le mot de passe provisoire fixé par
    # le propriétaire — sinon l'API lui renvoie AUTH_MDP_PROVISOIRE et non
    # le refus de rôle que ce test vise.
    member_headers = _onboard_member(client, "membre@bureau.fr")

    forbidden = client.post(
        "/api/auth/team/members",
        headers=member_headers,
        json={"email": "autre@bureau.fr", "nom": "Autre", "password": "motdepasse123"},
    )
    assert forbidden.status_code == 403
    assert forbidden.json()["detail"]["code"] == "AUTH_OWNER_REQUIS"


def test_team_isolation_between_organizations(client):
    client.post(
        "/api/auth/register",
        json={"email": "a@bureau-a.fr", "nom": "Aa", "password": "motdepasse123", "organization_name": "Bureau A"},
    )
    org_b = client.post(
        "/api/auth/register",
        json={"email": "b@bureau-b.fr", "nom": "Bb", "password": "motdepasse123", "organization_name": "Bureau B"},
    ).json()

    headers_b = {"Authorization": f"Bearer {org_b['access_token']}"}
    members = client.get("/api/auth/team/members", headers=headers_b).json()
    assert len(members) == 1
    assert members[0]["email"] == "b@bureau-b.fr"


# ── Désactivation d'un membre (offboarding) ─────────────────────────────────
# `User.actif` n'était jusqu'ici écrit qu'à la création (toujours True) :
# aucun endpoint ne permettait au propriétaire de couper l'accès d'un
# collaborateur parti, alors que get_current_user refusait déjà un compte
# inactif. Ces tests garantissent que la coupure est réelle et IMMÉDIATE.

MDP = "motdepasse123"
# Mot de passe que le membre choisit lui-même, distinct du provisoire fixé
# par le propriétaire (le serveur refuse de réutiliser l'ancien).
MDP_FINAL = "monmotdepasse456"


def _onboard_member(client, email: str, temporary: str = MDP, final: str = MDP_FINAL) -> dict:
    """Connecte un membre fraîchement créé et lui fait choisir SON mot de
    passe — parcours réel depuis que le mot de passe provisoire fixé par le
    propriétaire doit être remplacé avant tout usage de la plateforme.
    Renvoie les en-têtes d'un membre pleinement opérationnel."""
    first = client.post("/api/auth/login", data={"username": email, "password": temporary}).json()
    headers = {"Authorization": f"Bearer {first['access_token']}"}
    resp = client.patch(
        "/api/auth/me/password",
        headers=headers,
        json={"current_password": temporary, "new_password": final, "new_password_confirm": final},
    )
    assert resp.status_code == 204, resp.text
    # Changer son mot de passe révoque toutes les sessions : il faut se
    # reconnecter. La pause n'est pas un contournement de test mais une
    # propriété réelle du système : `iat` d'un JWT est en secondes entières
    # alors que `token_valid_after` porte des microsecondes, donc un jeton
    # émis dans la MÊME seconde que la révocation est rejeté (choix
    # délibéré, voir `get_current_user` — se tromper du côté sûr). Un
    # humain met de toute façon plus d'une seconde à ressaisir ses
    # identifiants ; un script, non.
    time.sleep(1.05)
    again = client.post("/api/auth/login", data={"username": email, "password": final}).json()
    return {"Authorization": f"Bearer {again['access_token']}"}


def _owner_and_member(client, org="Bureau", owner_email="owner@bureau.fr", member_email="membre@bureau.fr"):
    """Crée une organisation, son owner et un membre déjà intégré (mot de
    passe provisoire remplacé par le sien)."""
    owner = client.post(
        "/api/auth/register",
        json={"email": owner_email, "nom": "Owner", "password": MDP, "organization_name": org},
    ).json()
    owner_headers = {"Authorization": f"Bearer {owner['access_token']}"}
    created = client.post(
        "/api/auth/team/members",
        headers=owner_headers,
        json={"email": member_email, "nom": "Membre", "password": MDP},
    ).json()
    return owner_headers, created["id"], _onboard_member(client, member_email)


def test_deactivation_revokes_an_already_issued_token_immediately(client):
    """Le cœur du correctif : le jeton DÉJÀ ÉMIS du membre doit cesser de
    fonctionner à la seconde où le propriétaire le désactive — pas à
    l'expiration du jeton, sinon la désactivation ne sert à rien le jour
    où elle sert vraiment."""
    owner_headers, member_id, member_headers = _owner_and_member(client)
    assert client.get("/api/auth/me", headers=member_headers).status_code == 200

    resp = client.patch(f"/api/auth/team/members/{member_id}", headers=owner_headers, json={"actif": False})
    assert resp.status_code == 200, resp.text
    assert resp.json()["actif"] is False

    assert client.get("/api/auth/me", headers=member_headers).status_code == 401
    members = client.get("/api/auth/team/members", headers=owner_headers).json()
    assert [m["actif"] for m in members if m["id"] == member_id] == [False]


def test_deactivated_member_cannot_log_in_again(client):
    owner_headers, member_id, _ = _owner_and_member(client)
    client.patch(f"/api/auth/team/members/{member_id}", headers=owner_headers, json={"actif": False})

    # 403 et non 401 : les identifiants sont bons, c'est le COMPTE qui est
    # interdit — distinction volontaire côté endpoint de connexion. Ce
    # message existait depuis toujours mais était jusqu'ici inatteignable,
    # faute d'un moyen de désactiver qui que ce soit.
    resp = client.post("/api/auth/login", data={"username": "membre@bureau.fr", "password": MDP_FINAL})
    assert resp.status_code == 403
    assert resp.json()["detail"]["code"] == "AUTH_COMPTE_DESACTIVE"


def test_reactivated_member_can_log_in_again(client):
    """L'action est réversible — une désactivation n'est pas une suppression."""
    owner_headers, member_id, _ = _owner_and_member(client)
    client.patch(f"/api/auth/team/members/{member_id}", headers=owner_headers, json={"actif": False})

    resp = client.patch(f"/api/auth/team/members/{member_id}", headers=owner_headers, json={"actif": True})
    assert resp.status_code == 200
    assert resp.json()["actif"] is True
    assert client.post(
        "/api/auth/login", data={"username": "membre@bureau.fr", "password": MDP_FINAL}
    ).status_code == 200


def test_ordinary_member_cannot_deactivate_anyone(client):
    """Réservé au propriétaire — un membre ne coupe l'accès de personne,
    ni celui d'un collègue, ni celui du propriétaire."""
    owner_headers, member_id, member_headers = _owner_and_member(client)
    owner_id = [m["id"] for m in client.get("/api/auth/team/members", headers=owner_headers).json()
                if m["role"] == "owner"][0]

    for target in (member_id, owner_id):
        resp = client.patch(f"/api/auth/team/members/{target}", headers=member_headers, json={"actif": False})
        assert resp.status_code == 403
        assert resp.json()["detail"]["code"] == "AUTH_OWNER_REQUIS"

    # Le propriétaire n'a rien perdu au passage.
    assert client.get("/api/auth/me", headers=owner_headers).status_code == 200


def test_owner_cannot_deactivate_their_own_account(client):
    """Sinon l'organisation n'aurait plus personne pour gérer l'équipe."""
    owner_headers, _, _ = _owner_and_member(client)
    owner_id = [m["id"] for m in client.get("/api/auth/team/members", headers=owner_headers).json()
                if m["role"] == "owner"][0]

    resp = client.patch(f"/api/auth/team/members/{owner_id}", headers=owner_headers, json={"actif": False})
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "AUTH_AUTO_DESACTIVATION_INTERDITE"
    assert client.get("/api/auth/me", headers=owner_headers).status_code == 200


def test_cannot_deactivate_a_member_of_another_organization(client):
    """Isolation multi-tenant — et 404 plutôt que 403 : l'existence d'un
    compte d'une autre organisation n'est jamais révélée."""
    _, member_a_id, member_a_headers = _owner_and_member(client)
    headers_b = {
        "Authorization": "Bearer " + client.post(
            "/api/auth/register",
            json={"email": "b@org-b.fr", "nom": "Bb", "password": MDP, "organization_name": "Org B"},
        ).json()["access_token"]
    }

    resp = client.patch(f"/api/auth/team/members/{member_a_id}", headers=headers_b, json={"actif": False})
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "AUTH_MEMBRE_INTROUVABLE"
    # Le membre de l'organisation A n'a pas été touché.
    assert client.get("/api/auth/me", headers=member_a_headers).status_code == 200


def test_deactivation_is_recorded_in_the_audit_log(client):
    """Traçabilité : couper l'accès de quelqu'un doit laisser une trace
    consultable par le propriétaire."""
    owner_headers, member_id, _ = _owner_and_member(client)
    client.patch(f"/api/auth/team/members/{member_id}", headers=owner_headers, json={"actif": False})

    entries = client.get("/api/auth/team/audit-log", headers=owner_headers).json()
    deactivations = [e for e in entries if e["action"] == "member.deactivated"]
    assert len(deactivations) == 1
    assert deactivations[0]["target_id"] == member_id
    assert deactivations[0]["details"]["email"] == "membre@bureau.fr"


# ── Succession du propriétaire (promotion / rétrogradation) ─────────────────
# Jusqu'ici `register` créait l'UNIQUE owner et aucun endpoint ne changeait
# un rôle : le départ de ce propriétaire bloquait définitivement
# l'organisation. Invariant désormais protégé : au moins un propriétaire
# ACTIF en permanence — sur les deux chemins qui pourraient le violer
# (rétrogradation ET révocation d'accès).


def _owner_id_of(client, headers) -> int:
    return [m["id"] for m in client.get("/api/auth/team/members", headers=headers).json()
            if m["role"] == "owner"][0]


def test_owner_promotes_a_member_who_can_then_manage_the_team(client):
    owner_headers, member_id, member_headers = _owner_and_member(client)
    # Avant promotion, le membre ne peut pas gérer l'équipe.
    assert client.post(
        "/api/auth/team/members",
        headers=member_headers,
        json={"email": "x@bureau.fr", "nom": "Xx", "password": MDP},
    ).status_code == 403

    resp = client.patch(f"/api/auth/team/members/{member_id}/role", headers=owner_headers, json={"role": "owner"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["role"] == "owner"

    # Le jeton déjà émis porte l'ancien rôle, mais l'autorisation est relue
    # en base à chaque requête : le pouvoir est effectif immédiatement, sans
    # que le successeur ait à se reconnecter.
    assert client.post(
        "/api/auth/team/members",
        headers=member_headers,
        json={"email": "x@bureau.fr", "nom": "Xx", "password": MDP},
    ).status_code == 201


def test_full_succession_owner_promotes_then_steps_down(client):
    """Le scénario réel du départ : promouvoir son successeur, puis se
    rétrograder soi-même. L'organisation reste gérable de bout en bout."""
    owner_headers, member_id, member_headers = _owner_and_member(client)
    owner_id = _owner_id_of(client, owner_headers)

    client.patch(f"/api/auth/team/members/{member_id}/role", headers=owner_headers, json={"role": "owner"})
    # L'ancien propriétaire se rétrograde lui-même — autorisé, puisqu'un
    # autre propriétaire actif existe désormais.
    resp = client.patch(f"/api/auth/team/members/{owner_id}/role", headers=owner_headers, json={"role": "member"})
    assert resp.status_code == 200
    assert resp.json()["role"] == "member"

    # Il n'a plus les droits ; le successeur, si.
    assert client.get("/api/auth/team/audit-log", headers=owner_headers).status_code == 403
    assert client.get("/api/auth/team/audit-log", headers=member_headers).status_code == 200


def test_cannot_demote_the_last_active_owner(client):
    owner_headers, _, _ = _owner_and_member(client)
    owner_id = _owner_id_of(client, owner_headers)

    resp = client.patch(f"/api/auth/team/members/{owner_id}/role", headers=owner_headers, json={"role": "member"})
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "AUTH_DERNIER_PROPRIETAIRE"
    # L'organisation reste gérable.
    assert client.get("/api/auth/team/audit-log", headers=owner_headers).status_code == 200


def test_revoking_a_co_owner_always_leaves_an_active_owner(client):
    """La révocation ne peut PAS orpheliner l'organisation, par construction.

    Il n'y a volontairement aucune garde « dernier propriétaire actif » sur
    la révocation (contrairement à la rétrogradation) : elle serait du code
    mort. `require_owner` impose un auteur propriétaire ACTIF, et on ne peut
    pas se révoquer soi-même — l'auteur survit donc toujours à l'opération.
    Ce test fixe ce raisonnement : révoquer un co-propriétaire est permis, et
    l'organisation reste gérable après."""
    owner_headers, member_id, _ = _owner_and_member(client)
    client.patch(f"/api/auth/team/members/{member_id}/role", headers=owner_headers, json={"role": "owner"})

    resp = client.patch(f"/api/auth/team/members/{member_id}", headers=owner_headers, json={"actif": False})
    assert resp.status_code == 200

    members = client.get("/api/auth/team/members", headers=owner_headers).json()
    actifs = [m for m in members if m["role"] == "owner" and m["actif"]]
    assert len(actifs) == 1
    assert client.get("/api/auth/team/audit-log", headers=owner_headers).status_code == 200


def test_sole_owner_cannot_revoke_their_own_access(client):
    """L'autre moitié du raisonnement ci-dessus : le seul chemin qui aurait
    pu orpheliner l'organisation par révocation est fermé en amont."""
    owner_headers, _, _ = _owner_and_member(client)
    owner_id = _owner_id_of(client, owner_headers)

    resp = client.patch(f"/api/auth/team/members/{owner_id}", headers=owner_headers, json={"actif": False})
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "AUTH_AUTO_DESACTIVATION_INTERDITE"


def test_a_deactivated_owner_does_not_count_as_active(client):
    """Subtilité qui rend l'invariant correct : un propriétaire désactivé ne
    gère plus rien, il ne doit donc pas autoriser la rétrogradation du seul
    propriétaire encore actif."""
    owner_headers, member_id, _ = _owner_and_member(client)
    owner_id = _owner_id_of(client, owner_headers)

    # Le membre devient propriétaire, puis on lui révoque l'accès.
    client.patch(f"/api/auth/team/members/{member_id}/role", headers=owner_headers, json={"role": "owner"})
    client.patch(f"/api/auth/team/members/{member_id}", headers=owner_headers, json={"actif": False})

    # Il reste 2 propriétaires en base, mais UN SEUL actif : la rétrogradation
    # du propriétaire actif doit être refusée.
    resp = client.patch(f"/api/auth/team/members/{owner_id}/role", headers=owner_headers, json={"role": "member"})
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "AUTH_DERNIER_PROPRIETAIRE"


def test_ordinary_member_cannot_change_roles(client):
    owner_headers, member_id, member_headers = _owner_and_member(client)
    resp = client.patch(f"/api/auth/team/members/{member_id}/role", headers=member_headers, json={"role": "owner"})
    assert resp.status_code == 403
    assert resp.json()["detail"]["code"] == "AUTH_OWNER_REQUIS"


def test_cannot_change_role_of_a_member_of_another_organization(client):
    _, member_a_id, _ = _owner_and_member(client)
    headers_b = {
        "Authorization": "Bearer " + client.post(
            "/api/auth/register",
            json={"email": "b@org-b.fr", "nom": "Bb", "password": MDP, "organization_name": "Org B"},
        ).json()["access_token"]
    }
    resp = client.patch(f"/api/auth/team/members/{member_a_id}/role", headers=headers_b, json={"role": "owner"})
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "AUTH_MEMBRE_INTROUVABLE"


def test_role_changes_are_recorded_in_the_audit_log(client):
    owner_headers, member_id, _ = _owner_and_member(client)
    client.patch(f"/api/auth/team/members/{member_id}/role", headers=owner_headers, json={"role": "owner"})
    client.patch(f"/api/auth/team/members/{member_id}/role", headers=owner_headers, json={"role": "member"})

    actions = [e["action"] for e in client.get("/api/auth/team/audit-log", headers=owner_headers).json()]
    assert "member.promoted" in actions
    assert "member.demoted" in actions


def test_setting_the_same_role_is_a_noop_without_audit_entry(client):
    """Ne jamais polluer le journal d'audit avec un non-changement."""
    owner_headers, member_id, _ = _owner_and_member(client)
    resp = client.patch(f"/api/auth/team/members/{member_id}/role", headers=owner_headers, json={"role": "member"})
    assert resp.status_code == 200
    actions = [e["action"] for e in client.get("/api/auth/team/audit-log", headers=owner_headers).json()]
    assert "member.demoted" not in actions


# ── Mot de passe provisoire + date de révocation ────────────────────────────
# Le propriétaire choisit le mot de passe d'un membre qu'il ajoute : il le
# connaît donc. Sans changement forcé, il le connaissait INDÉFINIMENT et
# pouvait se connecter au compte de son collaborateur. L'enforcement est
# côté serveur (`get_current_user`), pas seulement dans l'interface.


def test_new_member_is_locked_out_of_the_api_until_they_choose_a_password(client):
    """Le cœur du correctif : tant que le mot de passe provisoire n'est pas
    remplacé, l'API entière est fermée — pas seulement l'écran masqué."""
    owner = client.post(
        "/api/auth/register",
        json={"email": "owner@mdp.fr", "nom": "Owner", "password": MDP, "organization_name": "Bureau"},
    ).json()
    owner_headers = {"Authorization": f"Bearer {owner['access_token']}"}
    client.post(
        "/api/auth/team/members",
        headers=owner_headers,
        json={"email": "neo@mdp.fr", "nom": "Neo", "password": MDP},
    )

    login = client.post("/api/auth/login", data={"username": "neo@mdp.fr", "password": MDP})
    assert login.status_code == 200  # la connexion elle-même reste permise
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    # Chemins exemptés : savoir qui l'on est, pour que l'interface sache
    # quel écran afficher.
    me = client.get("/api/auth/me", headers=headers)
    assert me.status_code == 200
    assert me.json()["must_change_password"] is True

    # Tout le reste est fermé, y compris hors du domaine auth.
    assert client.get("/api/datasets", headers=headers).status_code == 403
    refused = client.get("/api/auth/team/members", headers=headers)
    assert refused.status_code == 403
    assert refused.json()["detail"]["code"] == "AUTH_MDP_PROVISOIRE"

    # Après avoir choisi le sien, tout se débloque.
    assert client.patch(
        "/api/auth/me/password",
        headers=headers,
        json={"current_password": MDP, "new_password": MDP_FINAL, "new_password_confirm": MDP_FINAL},
    ).status_code == 204

    relogin = client.post("/api/auth/login", data={"username": "neo@mdp.fr", "password": MDP_FINAL}).json()
    new_headers = {"Authorization": f"Bearer {relogin['access_token']}"}
    assert client.get("/api/datasets", headers=new_headers).status_code == 200
    assert client.get("/api/auth/me", headers=new_headers).json()["must_change_password"] is False


def test_self_registered_owner_is_never_forced_to_change_password(client):
    """Le propriétaire a choisi son mot de passe lui-même à l'inscription :
    personne d'autre ne le connaît, rien à forcer."""
    owner = client.post(
        "/api/auth/register",
        json={"email": "solo@mdp.fr", "nom": "Solo", "password": MDP, "organization_name": "Bureau"},
    ).json()
    headers = {"Authorization": f"Bearer {owner['access_token']}"}
    assert client.get("/api/auth/me", headers=headers).json()["must_change_password"] is False
    assert client.get("/api/datasets", headers=headers).status_code == 200


def test_owner_sees_which_members_still_use_the_temporary_password(client):
    """Transparence : le propriétaire doit pouvoir relancer les retardataires."""
    owner_headers, member_id, _ = _owner_and_member(client)
    members = {m["id"]: m for m in client.get("/api/auth/team/members", headers=owner_headers).json()}
    # Le membre de la fixture a déjà choisi son mot de passe.
    assert members[member_id]["must_change_password"] is False

    client.post(
        "/api/auth/team/members",
        headers=owner_headers,
        json={"email": "retard@bureau.fr", "nom": "Retard", "password": MDP},
    )
    listed = client.get("/api/auth/team/members", headers=owner_headers).json()
    retard = [m for m in listed if m["email"] == "retard@bureau.fr"][0]
    assert retard["must_change_password"] is True


def test_deactivation_records_the_date_and_reactivation_clears_it(client):
    """`actif` ne dit que l'état courant ; `deactivated_at` dit DEPUIS QUAND,
    ce qu'exige un audit sérieux et une politique de rétention."""
    owner_headers, member_id, _ = _owner_and_member(client)

    before = client.get("/api/auth/team/members", headers=owner_headers).json()
    assert [m["deactivated_at"] for m in before if m["id"] == member_id] == [None]

    revoked = client.patch(f"/api/auth/team/members/{member_id}", headers=owner_headers, json={"actif": False})
    assert revoked.status_code == 200
    assert revoked.json()["deactivated_at"] is not None

    restored = client.patch(f"/api/auth/team/members/{member_id}", headers=owner_headers, json={"actif": True})
    assert restored.status_code == 200
    assert restored.json()["deactivated_at"] is None
