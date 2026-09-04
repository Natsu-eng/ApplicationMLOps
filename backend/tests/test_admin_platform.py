"""Tests de l'espace d'administration de la plateforme (`domains/admin`).

C'est le SEUL périmètre du projet autorisé à lire au-delà d'une
organisation. Deux exigences s'opposent et sont testées ensemble :

- l'administrateur DOIT voir toutes les organisations (sinon la vue globale
  ne sert à rien) ;
- son existence ne DOIT rien changer pour les autres : un utilisateur
  ordinaire, propriétaire compris, reste strictement enfermé dans son
  organisation. Le risque réel d'un rôle de supervision est d'être obtenu
  par assouplissement des endpoints existants — vérifié ici que non.
"""
from __future__ import annotations

import io

from api.core.models import User

MDP = "motdepasse123"


def _register(client, email, org, nom="Chef"):
    resp = client.post(
        "/api/auth/register",
        json={"email": email, "nom": nom, "password": MDP, "organization_name": org},
    )
    assert resp.status_code == 201, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _make_platform_admin(db_session, email: str) -> None:
    """Promotion directe en base — c'est exactement ce que fait le script
    `scripts/grant_platform_admin.py`, seul chemin prévu : aucun endpoint
    ne permet de s'auto-promouvoir, et c'est délibéré."""
    user = db_session.query(User).filter(User.email == email).first()
    assert user is not None
    user.is_platform_admin = True
    db_session.commit()


def _upload_dataset(client, headers, name="d.csv"):
    rows = "\n".join(f"{i},{i * 2},{i * 3}" for i in range(30))
    content = f"x1,x2,cible\n{rows}\n".encode()
    return client.post("/api/datasets", headers=headers, files={"file": (name, io.BytesIO(content), "text/csv")})


# ── Contrôle d'accès ─────────────────────────────────────────────────────────


def test_every_admin_route_is_closed_to_ordinary_users(client):
    """Y compris à un PROPRIÉTAIRE : être administrateur de son organisation
    ne donne aucun droit sur la plateforme."""
    headers = _register(client, "simple@bureau.fr", "Bureau")
    for path in ("/api/admin/overview", "/api/admin/organizations", "/api/admin/users", "/api/admin/activity"):
        resp = client.get(path, headers=headers)
        assert resp.status_code == 403, (path, resp.status_code)
        assert resp.json()["detail"]["code"] == "AUTH_ADMIN_PLATEFORME_REQUIS"


def test_admin_routes_require_authentication(client):
    for path in ("/api/admin/overview", "/api/admin/organizations", "/api/admin/users", "/api/admin/activity"):
        assert client.get(path).status_code == 401, path


def test_nobody_can_self_promote_through_the_api(client, db_session):
    """Aucun endpoint n'expose `is_platform_admin` en écriture. Le champ est
    exposé en LECTURE dans la liste des comptes de l'admin, jamais ailleurs :
    un propriétaire ne doit pas pouvoir se hisser au-dessus de son
    organisation par une requête bien choisie."""
    headers = _register(client, "ambitieux@bureau.fr", "Bureau")

    # Les deux seuls endpoints qui modifient un compte : profil et rôle.
    client.patch("/api/auth/me", headers=headers, json={"nom": "Nouveau", "is_platform_admin": True})
    member_id = db_session.query(User).filter(User.email == "ambitieux@bureau.fr").first().id
    client.patch(
        f"/api/auth/team/members/{member_id}/role",
        headers=headers,
        json={"role": "owner", "is_platform_admin": True},
    )

    db_session.expire_all()
    assert db_session.query(User).filter(User.email == "ambitieux@bureau.fr").first().is_platform_admin is False
    assert client.get("/api/admin/overview", headers=headers).status_code == 403


# ── Vue globale ──────────────────────────────────────────────────────────────


def test_overview_aggregates_every_organization(client, db_session):
    headers_a = _register(client, "a@org-a.fr", "Org A")
    _register(client, "b@org-b.fr", "Org B")
    _upload_dataset(client, headers_a)
    _make_platform_admin(db_session, "a@org-a.fr")

    resp = client.get("/api/admin/overview", headers=headers_a)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["counters"]["organizations"] == 2
    assert body["counters"]["users_total"] == 2
    assert body["counters"]["users_active"] == 2
    assert body["counters"]["datasets"] == 1
    assert body["counters"]["datasets_bytes"] > 0

    # Les 7 piliers sont présents même à zéro : une vue de supervision qui
    # masque les piliers inactifs laisse croire qu'ils n'existent pas.
    assert len(body["jobs_by_pillar"]) == 7
    assert {p["pillar"] for p in body["jobs_by_pillar"]} >= {"TrainingJob", "VisionAnomalyJob", "BatchPredictionJob"}
    assert all(p["label"] and p["label"] != p["pillar"] for p in body["jobs_by_pillar"])


def test_failure_rate_is_null_rather_than_zero_when_nothing_finished(client, db_session):
    """0 % se lirait comme « aucune panne » alors qu'il n'y a rien à
    mesurer — deux situations très différentes pour qui supervise."""
    headers = _register(client, "vide@org.fr", "Org Vide")
    _make_platform_admin(db_session, "vide@org.fr")

    body = client.get("/api/admin/overview", headers=headers).json()
    assert body["jobs_total"] == 0
    assert body["failure_rate"] is None


def test_timeseries_fill_empty_days_instead_of_skipping_them(client, db_session):
    """Un `GROUP BY date` ne renvoie que les jours actifs : un graphique
    construit dessus relierait deux points distants d'une semaine par une
    ligne droite, suggérant une activité continue qui n'a pas eu lieu."""
    headers = _register(client, "series@org.fr", "Org Series")
    _make_platform_admin(db_session, "series@org.fr")

    body = client.get("/api/admin/overview?window_days=14", headers=headers).json()
    assert body["window_days"] == 14
    assert len(body["signups_per_day"]) == 14
    assert len(body["jobs_per_day"]) == 14
    # Dates strictement croissantes et sans trou.
    dates = [p["date"] for p in body["signups_per_day"]]
    assert dates == sorted(dates)
    assert len(set(dates)) == 14
    # L'inscription du jour est comptée quelque part dans la fenêtre.
    assert sum(p["count"] for p in body["signups_per_day"]) >= 1


def test_organizations_view_lists_all_with_their_volumetry(client, db_session):
    headers_a = _register(client, "a@org-a.fr", "Org A")
    _register(client, "b@org-b.fr", "Org B")
    _upload_dataset(client, headers_a)
    client.post(
        "/api/auth/team/members", headers=headers_a,
        json={"email": "membre@org-a.fr", "nom": "Membre", "password": MDP},
    )
    _make_platform_admin(db_session, "a@org-a.fr")

    rows = client.get("/api/admin/organizations", headers=headers_a).json()
    by_name = {r["name"]: r for r in rows}
    assert set(by_name) == {"Org A", "Org B"}
    assert by_name["Org A"]["members"] == 2
    assert by_name["Org A"]["datasets"] == 1
    assert by_name["Org A"]["last_activity_at"] is not None
    assert by_name["Org B"]["members"] == 1
    assert by_name["Org B"]["datasets"] == 0


def test_users_view_exposes_account_state_across_organizations(client, db_session):
    headers_a = _register(client, "a@org-a.fr", "Org A")
    _register(client, "b@org-b.fr", "Org B")
    client.post(
        "/api/auth/team/members", headers=headers_a,
        json={"email": "attente@org-a.fr", "nom": "Attente", "password": MDP},
    )
    _make_platform_admin(db_session, "a@org-a.fr")

    rows = client.get("/api/admin/users", headers=headers_a).json()
    by_email = {r["email"]: r for r in rows}
    assert set(by_email) >= {"a@org-a.fr", "b@org-b.fr", "attente@org-a.fr"}
    # Chaque compte porte le nom de SON organisation : sans cela la vue
    # serait ininterprétable dès que deux organisations ont des homonymes.
    assert by_email["b@org-b.fr"]["organization_name"] == "Org B"
    # L'état complet est visible, y compris le compte encore en attente.
    assert by_email["attente@org-a.fr"]["must_change_password"] is True
    assert by_email["a@org-a.fr"]["is_platform_admin"] is True
    assert by_email["b@org-b.fr"]["is_platform_admin"] is False


def test_activity_view_spans_organizations_and_names_the_actor(client, db_session):
    headers_a = _register(client, "a@org-a.fr", "Org A")
    headers_b = _register(client, "b@org-b.fr", "Org B", nom="Bee")
    _upload_dataset(client, headers_b, "b.csv")
    dataset_b = client.get("/api/datasets", headers=headers_b).json()[0]
    client.delete(f"/api/datasets/{dataset_b['id']}", headers=headers_b)
    _make_platform_admin(db_session, "a@org-a.fr")

    rows = client.get("/api/admin/activity?limit=50", headers=headers_a).json()
    orgs = {r["organization_name"] for r in rows}
    assert "Org B" in orgs  # activité d'une AUTRE organisation, bien visible
    deletion = [r for r in rows if r["action"] == "dataset.deleted"]
    assert deletion and deletion[0]["actor_name"] == "Bee"
    assert deletion[0]["details"]["name"] == "b.csv"


# ── Non-régression de l'isolation ────────────────────────────────────────────


def test_platform_admin_does_not_widen_the_ordinary_endpoints(client, db_session):
    """LE test qui compte. L'administrateur voit tout par `/admin`, mais les
    endpoints métier restent filtrés par organisation POUR LUI AUSSI : la
    supervision ne doit jamais devenir un passe-partout sur les données
    clientes."""
    headers_a = _register(client, "a@org-a.fr", "Org A")
    headers_b = _register(client, "b@org-b.fr", "Org B")
    _upload_dataset(client, headers_b, "prive.csv")
    dataset_b = client.get("/api/datasets", headers=headers_b).json()[0]
    _make_platform_admin(db_session, "a@org-a.fr")

    # Il voit bien Org B dans la vue globale…
    assert any(r["name"] == "Org B" for r in client.get("/api/admin/organizations", headers=headers_a).json())

    # …mais pas ses datasets par les routes normales.
    assert client.get("/api/datasets", headers=headers_a).json() == []
    assert client.get(f"/api/datasets/{dataset_b['id']}", headers=headers_a).status_code == 404
    # Ni son équipe.
    emails = {m["email"] for m in client.get("/api/auth/team/members", headers=headers_a).json()}
    assert emails == {"a@org-a.fr"}


def test_admin_space_is_read_only(client):
    """Aucune route d'écriture n'existe sous /admin — vérifié sur la table
    de routage elle-même plutôt que sur une liste tenue à la main, qui
    cesserait d'être exacte au premier ajout."""
    from api.main import app

    write_routes = [
        (sorted(r.methods or []), r.path)
        for r in app.routes
        if getattr(r, "path", "").startswith("/api/admin")
        and {"POST", "PUT", "PATCH", "DELETE"} & (getattr(r, "methods", None) or set())
    ]
    assert write_routes == [], write_routes
