"""Tests des notifications de fin de job (retour utilisateur : "notifications
de fin de job — email/navigateur") — le helper partagé
(domains/shared/notifications.py::notify_job_terminal) en isolation, puis le
router de consultation (liste, compteur, marquage lu, isolation par
utilisateur)."""
from __future__ import annotations

from unittest.mock import MagicMock

from api.core.models import Notification, User
from domains.shared.notifications import notify_job_terminal


def _register(client, email="owner@bureau.fr", org="Bureau"):
    resp = client.post(
        "/api/auth/register",
        json={"email": email, "nom": "Owner", "password": "motdepasse123", "organization_name": org},
    ).json()
    return {"Authorization": f"Bearer {resp['access_token']}"}


def _current_user(client, headers) -> User:
    resp = client.get("/api/auth/me", headers=headers)
    return resp.json()


# ── Helper (domains/shared/notifications.py) ────────────────────────────


def test_notify_job_terminal_creates_a_notification(client, db_session):
    headers = _register(client)
    me = _current_user(client, headers)

    notify_job_terminal(
        db_session, organization_id=1, user_id=me["id"], job_type="training", job_id=42, status="completed",
        subtitle="mon_dataset.csv → cible",
    )
    db_session.commit()

    row = db_session.query(Notification).filter(Notification.user_id == me["id"]).first()
    assert row is not None
    assert row.job_type == "training"
    assert row.job_id == 42
    assert row.status == "completed"
    assert "mon_dataset.csv" in row.message
    assert row.link_path == "/training?job=42"
    assert row.read_at is None


def test_notify_job_terminal_uses_link_job_id_when_provided():
    """Prédiction en lot (retour utilisateur direct) — pas de page dédiée,
    le lien doit pointer vers l'entraînement SOURCE, pas vers l'id du lot
    lui-même."""
    db = MagicMock()
    notify_job_terminal(
        db, organization_id=1, user_id=7, job_type="batch_prediction", job_id=99, status="completed",
        subtitle="a_predire.csv", link_job_id=12,
    )
    added = db.add.call_args[0][0]
    assert added.link_path == "/training?job=12"
    assert added.job_id == 99  # l'id du LOT reste 99, seul le lien change


def test_notify_job_terminal_skips_silently_when_user_id_is_none(db_session):
    """`created_by_id` est nullable sur toutes les tables de job (compte
    supprimé après coup) — jamais une erreur, jamais de notification créée."""
    before = db_session.query(Notification).count()

    notify_job_terminal(
        db_session, organization_id=1, user_id=None, job_type="training", job_id=1, status="failed", subtitle="x"
    )
    db_session.commit()

    assert db_session.query(Notification).count() == before


# ── Router ───────────────────────────────────────────────────────────────


def test_list_notifications_returns_most_recent_first(client, db_session):
    headers = _register(client)
    me = _current_user(client, headers)
    for i in range(3):
        notify_job_terminal(
            db_session, organization_id=1, user_id=me["id"], job_type="training", job_id=i, status="completed",
            subtitle=f"job {i}",
        )
    db_session.commit()

    resp = client.get("/api/notifications", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 3
    assert [n["job_id"] for n in body] == [2, 1, 0]


def test_unread_count_reflects_only_unread(client, db_session):
    headers = _register(client)
    me = _current_user(client, headers)
    notify_job_terminal(
        db_session, organization_id=1, user_id=me["id"], job_type="training", job_id=1, status="completed",
        subtitle="a",
    )
    notify_job_terminal(
        db_session, organization_id=1, user_id=me["id"], job_type="training", job_id=2, status="failed", subtitle="b"
    )
    db_session.commit()

    assert client.get("/api/notifications/unread-count", headers=headers).json()["count"] == 2

    notification_id = client.get("/api/notifications", headers=headers).json()[0]["id"]
    client.post(f"/api/notifications/{notification_id}/read", headers=headers)

    assert client.get("/api/notifications/unread-count", headers=headers).json()["count"] == 1


def test_mark_notification_read_is_idempotent(client, db_session):
    headers = _register(client)
    me = _current_user(client, headers)
    notify_job_terminal(
        db_session, organization_id=1, user_id=me["id"], job_type="training", job_id=1, status="completed",
        subtitle="a",
    )
    db_session.commit()
    notification_id = client.get("/api/notifications", headers=headers).json()[0]["id"]

    first = client.post(f"/api/notifications/{notification_id}/read", headers=headers).json()
    second = client.post(f"/api/notifications/{notification_id}/read", headers=headers).json()
    assert first["read_at"] == second["read_at"]


def test_mark_all_read_clears_the_unread_count(client, db_session):
    headers = _register(client)
    me = _current_user(client, headers)
    for i in range(5):
        notify_job_terminal(
            db_session, organization_id=1, user_id=me["id"], job_type="training", job_id=i, status="completed",
            subtitle=f"job {i}",
        )
    db_session.commit()

    resp = client.post("/api/notifications/read-all", headers=headers)
    assert resp.status_code == 204
    assert client.get("/api/notifications/unread-count", headers=headers).json()["count"] == 0


def test_notifications_are_isolated_between_users(client, db_session):
    """Personnelles, jamais partagées à toute l'organisation (retour
    utilisateur direct) — un collègue de la MÊME organisation ne doit
    jamais voir les notifications d'un autre membre."""
    headers_a = _register(client, "a@bureau.fr", "Bureau")
    me_a = _current_user(client, headers_a)
    headers_b = client.post(
        "/api/auth/register",
        json={"email": "b@bureau.fr", "nom": "Membre B", "password": "motdepasse123", "organization_name": "Bureau 2"},
    ).json()
    headers_b = {"Authorization": f"Bearer {headers_b['access_token']}"}

    notify_job_terminal(
        db_session, organization_id=1, user_id=me_a["id"], job_type="training", job_id=1, status="completed",
        subtitle="a",
    )
    db_session.commit()

    assert client.get("/api/notifications", headers=headers_a).json() != []
    assert client.get("/api/notifications", headers=headers_b).json() == []


def test_mark_read_404_for_another_users_notification(client, db_session):
    headers_a = _register(client, "a@bureau.fr", "Bureau")
    me_a = _current_user(client, headers_a)
    headers_b = client.post(
        "/api/auth/register",
        json={"email": "b@bureau.fr", "nom": "Membre B", "password": "motdepasse123", "organization_name": "Bureau 2"},
    ).json()
    headers_b = {"Authorization": f"Bearer {headers_b['access_token']}"}

    notify_job_terminal(
        db_session, organization_id=1, user_id=me_a["id"], job_type="training", job_id=1, status="completed",
        subtitle="a",
    )
    db_session.commit()
    notification_id = client.get("/api/notifications", headers=headers_a).json()[0]["id"]

    resp = client.post(f"/api/notifications/{notification_id}/read", headers=headers_b)
    assert resp.status_code == 404


def test_unread_only_filter(client, db_session):
    headers = _register(client)
    me = _current_user(client, headers)
    notify_job_terminal(
        db_session, organization_id=1, user_id=me["id"], job_type="training", job_id=1, status="completed",
        subtitle="a",
    )
    notify_job_terminal(
        db_session, organization_id=1, user_id=me["id"], job_type="training", job_id=2, status="failed", subtitle="b"
    )
    db_session.commit()
    notification_id = client.get("/api/notifications", headers=headers).json()[-1]["id"]
    client.post(f"/api/notifications/{notification_id}/read", headers=headers)

    resp = client.get("/api/notifications?unread_only=true", headers=headers)
    assert len(resp.json()) == 1
