"""Fixtures pytest partagées.

Base de données de test isolée (SQLite fichier temporaire, jamais la base de
développement/production) recréée avant chaque test, et client FastAPI de
test. `DATABASE_URL`/`JWT_SECRET_KEY` sont surchargées AVANT tout import de
`api.core.*` : `get_settings()` est mis en cache (`lru_cache`) dès le premier
import, donc l'ordre des imports ci-dessous est important.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

_TEST_DB_PATH = Path(tempfile.gettempdir()) / "datalab_test.db"
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB_PATH}"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-not-for-production"
# Bug réel trouvé en écrivant les tests de la Phase 1B
# (test_password_reset.py) — `backend/.env` local contient de VRAIES
# identifiants SMTP (Gmail, pour tester la fonctionnalité manuellement).
# Sans cette surcharge, la suite de tests déclenchait de véritables
# tentatives de connexion à smtp.gmail.com (échouées, mais lentes et
# dépendantes du réseau) à chaque test qui déclenche l'envoi d'un mail en
# tâche de fond — une suite de tests ne doit JAMAIS dépendre d'un service
# tiers réel, quel que soit le `.env` local du poste qui l'exécute.
os.environ["SMTP_HOST"] = ""
os.environ["SMTP_USER"] = ""
os.environ["SMTP_PASSWORD"] = ""

from api.core import models  # noqa: E402,F401 — enregistre les tables sur Base.metadata
from api.core.database import Base, get_db  # noqa: E402
from api.main import app  # noqa: E402

# NullPool : chaque session ouvre puis referme sa propre connexion SQLite,
# plutôt que de la garder en pool. Sur une suite complète (257 tests, des
# dizaines d'entraînements réels, ~30 min), le pool par défaut de SQLAlchemy
# peut accumuler des connexions SQLite non recyclées jusqu'à des symptômes
# qui n'apparaissent qu'en run complet (jamais un fichier isolé) — constaté
# lors de l'audit du 2026-08-14 (voir AUDIT_ROADMAP.md, H1) : 2 tests de
# tests/test_inference.py échouaient uniquement en suite complète, jamais
# isolés ni sur des sous-groupes de fichiers testés explicitement.
_test_engine = create_engine(
    f"sqlite:///{_TEST_DB_PATH}", connect_args={"check_same_thread": False}, poolclass=NullPool
)
_TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_test_engine)


def _override_get_db() -> Iterator:
    db = _TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def _fresh_database() -> Iterator[None]:
    """Schéma vierge avant CHAQUE test — isolation totale, aucun état ne fuit
    d'un test à l'autre (contrairement à une base partagée réutilisée)."""
    Base.metadata.drop_all(bind=_test_engine)
    Base.metadata.create_all(bind=_test_engine)
    app.dependency_overrides[get_db] = _override_get_db
    yield
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def _fresh_rate_limit_counters() -> Iterator[None]:
    """Nettoie les compteurs Redis de rate-limiting (H11, AUDIT_ROADMAP.md ;
    généralisé Lot 1.4, `core/rate_limit.py::rate_limit_dependency`) avant
    CHAQUE test — sans ça, `TestClient` rapporte toujours la même IP
    simulée ("testclient"), donc un compteur comme `login_attempts:testclient`
    ou `rate_limit:register:testclient` s'accumulerait silencieusement d'un
    test à l'autre sur toute la suite (même catégorie de risque que
    l'isolation DB ci-dessus, voir H1 — constaté en pratique au Lot 1.4 :
    `rate_limit:register:*` non nettoyé faisait échouer en cascade tout
    test appelant `_register()` après le 10ᵉ de la suite). `login_attempts:*`
    et `rate_limit:*` sont deux préfixes distincts (l'historique H11
    précède la généralisation Lot 1.4), les deux doivent être nettoyés.
    Échec ouvert si Redis est indisponible pendant les tests — la suite ne
    doit jamais dépendre d'un Redis local pour s'exécuter."""
    from api.core.job_queue import redis_conn

    try:
        for pattern in ("login_attempts:*", "rate_limit:*"):
            for key in redis_conn.scan_iter(pattern):
                redis_conn.delete(key)
    except Exception:
        pass
    yield


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def db_session() -> Iterator:
    """Session DB directe — pour les tests qui doivent insérer des données
    comme le ferait un worker (hors requête HTTP), ex : test_inference.py."""
    db = _TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
