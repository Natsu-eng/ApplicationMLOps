"""Test de fumée contre la stack Docker réelle — correctif (retour utilisateur,
incident de déploiement) : une suite de 525 tests backend passait au vert
pendant que l'application ne démarrait pas en production. La cause (routers
backend non préfixés `/api`, nginx qui ne proxifie que `location /api/`,
`POST /auth/login` recevant du HTML au lieu du backend) n'était détectable
par AUCUN test existant, puisque `tests/*.py` appelle l'application FastAPI
directement (`TestClient(app)`, voir `tests/conftest.py`) — jamais au
travers de nginx, jamais avec les vraies images Docker construites.

Ce script fait ce que les 525 tests ne peuvent pas faire par construction :
il suppose `docker compose up -d --build` déjà lancé (voir CI et
`docker-compose.yml`), attend que la stack soit prête, puis rejoue un
scénario utilisateur réel contre le conteneur nginx exposé (port 80 par
défaut) — jamais contre `uvicorn --reload` ni contre le backend en direct.

Usage :
    docker compose up -d --build
    cd backend && python -m scripts.smoke_test_docker
    docker compose down

Variable d'environnement optionnelle :
    SMOKE_TEST_BASE_URL   URL de base à tester (défaut : http://localhost)
"""
from __future__ import annotations

import os
import sys
import time
import uuid

import httpx

BASE_URL = os.environ.get("SMOKE_TEST_BASE_URL", "http://localhost")
READINESS_TIMEOUT_SECONDS = 90
READINESS_POLL_INTERVAL_SECONDS = 3


def _fail(message: str) -> None:
    print(f"[ÉCHEC] {message}", file=sys.stderr)
    sys.exit(1)


def wait_for_readiness() -> None:
    """Attend que `GET /api/health` réponde 200 avec la base de données
    connectée — pas seulement que le conteneur nginx écoute (le healthcheck
    Docker peut passer avant que le backend n'ait fini son initialisation)."""
    deadline = time.monotonic() + READINESS_TIMEOUT_SECONDS
    last_error: str | None = None
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"{BASE_URL}/api/health", timeout=5)
            if resp.status_code == 200 and resp.json().get("database") == "up":
                print(f"[OK] API prête ({BASE_URL}/api/health)")
                return
            last_error = f"statut {resp.status_code} : {resp.text[:200]}"
        except httpx.HTTPError as exc:
            last_error = str(exc)
        time.sleep(READINESS_POLL_INTERVAL_SECONDS)
    _fail(f"L'API n'est jamais devenue prête en {READINESS_TIMEOUT_SECONDS}s — dernière erreur : {last_error}")


def check_spa_served_at_root() -> None:
    """`GET /` doit renvoyer le HTML de la SPA (pas une erreur nginx, pas du
    JSON) — vérifie que le conteneur frontend sert bien les fichiers
    statiques construits, distinct de la vérification API ci-dessous."""
    resp = httpx.get(BASE_URL, timeout=10, follow_redirects=True)
    content_type = resp.headers.get("content-type", "")
    if resp.status_code != 200 or "text/html" not in content_type:
        _fail(f"GET / attendu 200 text/html, reçu {resp.status_code} {content_type!r}")
    print("[OK] GET / sert bien le HTML de la SPA")


def run_scenario() -> None:
    """Inscription → connexion → liste des datasets, exactement comme le
    ferait le frontend réel (`frontend/src/api/client.ts`) : même préfixe
    `/api`, même endpoint de login form-encoded (`OAuth2PasswordRequestForm`),
    à travers nginx — c'est précisément le chemin que l'incident cassait."""
    unique = uuid.uuid4().hex[:12]
    email = f"smoke-{unique}@datalab-test.local"
    password = "motdepasse123"

    # 1. Inscription — POST /api/auth/register (JSON)
    register_resp = httpx.post(
        f"{BASE_URL}/api/auth/register",
        json={
            "email": email,
            "nom": "Smoke Test",
            "password": password,
            "organization_name": f"Bureau smoke {unique}",
        },
        timeout=15,
    )
    if register_resp.status_code != 201:
        _fail(
            f"POST /api/auth/register attendu 201, reçu {register_resp.status_code} "
            f"({register_resp.headers.get('content-type')!r}) : {register_resp.text[:300]}"
        )
    print("[OK] Inscription (POST /api/auth/register)")

    # 2. Connexion — POST /api/auth/login (x-www-form-urlencoded, PAS JSON :
    # c'est exactement le chemin qui recevait du HTML pendant l'incident,
    # provoquant une SyntaxError sur res.json() côté frontend).
    login_resp = httpx.post(
        f"{BASE_URL}/api/auth/login",
        data={"username": email, "password": password},
        timeout=15,
    )
    if login_resp.status_code != 200:
        _fail(
            f"POST /api/auth/login attendu 200, reçu {login_resp.status_code} "
            f"({login_resp.headers.get('content-type')!r}) : {login_resp.text[:300]}"
        )
    content_type = login_resp.headers.get("content-type", "")
    if "application/json" not in content_type:
        _fail(f"POST /api/auth/login n'a pas renvoyé du JSON (Content-Type: {content_type!r}) — reproduit l'incident.")
    token = login_resp.json().get("access_token")
    if not token:
        _fail("POST /api/auth/login : réponse JSON sans access_token")
    print("[OK] Connexion (POST /api/auth/login)")

    # 3. Liste des datasets — GET /api/datasets, avec le token obtenu
    datasets_resp = httpx.get(
        f"{BASE_URL}/api/datasets",
        headers={"Authorization": f"Bearer {token}"},
        timeout=15,
    )
    if datasets_resp.status_code != 200:
        _fail(f"GET /api/datasets attendu 200, reçu {datasets_resp.status_code} : {datasets_resp.text[:300]}")
    if not isinstance(datasets_resp.json(), list):
        _fail("GET /api/datasets n'a pas renvoyé une liste JSON")
    print("[OK] Liste des datasets (GET /api/datasets) — organisation neuve, liste vide attendue")


def main() -> None:
    print(f"[INFO] Test de fumée contre {BASE_URL}")
    wait_for_readiness()
    check_spa_served_at_root()
    run_scenario()
    print("[OK] Scénario de fumée complet réussi.")


if __name__ == "__main__":
    main()
