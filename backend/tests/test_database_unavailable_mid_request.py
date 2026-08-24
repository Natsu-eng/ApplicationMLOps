"""Comportement explicite quand PostgreSQL devient indisponible EN COURS DE
REQUÊTE (Phase 2, AUDIT_BACKEND_2026-08-23.md, Axe F) — distinct du cas déjà
couvert par `test_database_startup.py` (Postgres down AU DÉMARRAGE, l'API
démarre quand même en mode dégradé). Ici, la connexion tombe pendant qu'une
requête HTTP est en cours de traitement : le client ne doit jamais recevoir
une trace Python brute (le gestionnaire d'erreur global de la Phase 1,
`api/main.py::unhandled_exception_handler`, doit intercepter proprement)."""

from __future__ import annotations

from sqlalchemy.exc import OperationalError
from starlette.testclient import TestClient

from api.core.database import get_db
from api.main import app


def test_db_failure_mid_request_returns_clean_500_not_a_raw_traceback(client):
    """Simule une session DB qui lève `OperationalError` (Postgres
    injoignable) au moment où un endpoint tente de l'utiliser — le
    gestionnaire d'erreur global doit produire l'enveloppe standard du
    projet (code, message français, request_id), jamais `str(exc)` brut ni
    une trace de la pile."""

    class _BrokenSession:
        def query(self, *args, **kwargs):
            raise OperationalError("SELECT 1", {}, Exception("connexion refusée (simulation)"))

        def close(self):
            pass

    def _override_broken_db():
        db = _BrokenSession()
        try:
            yield db
        finally:
            db.close()

    # Nécessite un utilisateur authentifié : `get_current_user` lui-même
    # interroge la DB (`db.query(User)...`) — la panne se manifeste dès la
    # résolution de l'authentification, avant même d'atteindre le corps de
    # `list_datasets`. Représentatif : TOUT endpoint authentifié de l'API
    # passe par cette même dépendance en premier.
    tokens = client.post(
        "/api/auth/register",
        json={"email": "dbdown@bureau.fr", "nom": "Test", "password": "motdepasse123", "organization_name": "Bureau"},
    ).json()
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}

    # `ServerErrorMiddleware` (Starlette) envoie la réponse au client PUIS
    # relève quand même l'exception (pour qu'un vrai serveur ASGI la
    # journalise) — un vrai client HTTP ne voit jamais cette relève (la
    # réponse est déjà partie sur le socket), mais le `client` partagé du
    # dépôt (`TestClient` par défaut) la fait remonter dans le process de
    # test, ce qui casserait ce test alors que le comportement serveur est
    # correct. `raise_server_exceptions=False` fait se comporter ce client
    # de test comme le ferait un vrai navigateur/nginx : seule la réponse
    # HTTP compte, jamais l'exception Python interne au serveur.
    app.dependency_overrides[get_db] = _override_broken_db
    try:
        with TestClient(app, raise_server_exceptions=False) as broken_client:
            resp = broken_client.get("/api/datasets", headers=headers)
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 500
    body = resp.json()["detail"]
    assert body["code"] == "ERREUR_INTERNE"
    assert "request_id" in body
    # Jamais la trace brute ni le texte de l'exception SQL exposés au client.
    assert "OperationalError" not in body["message"]
    assert "connexion refusée" not in body["message"]
