"""Garde-fou de non-régression pour `backend/Dockerfile` (Phase 5,
AUDIT_BACKEND_2026-08-23.md, Axe J) -- deux bugs réels déjà trouvés dans ce
fichier au cours de ce chantier, tous deux invisibles à la suite pytest
normale (qui ne construit jamais l'image Docker) :

1. Décision 1 (Phase 1) -- `domains/` jamais copié après le Lot 8
   (monolithe modulaire) : `ModuleNotFoundError` au démarrage, chaque
   déploiement cassé jusqu'à correction.
2. Phase 5 -- `alembic.ini`/`alembic/` jamais copiés : `init_db()` (appelé
   au démarrage par `api/main.py::lifespan`) échoue silencieusement (le
   `try/except` de `lifespan` avale l'exception et démarre en mode
   dégradé) -- aucune migration n'est jamais appliquée en production.

Ce test analyse le texte du Dockerfile plutôt que de construire l'image
(trop lent pour la suite normale, déjà couvert bout-en-bout par le job
`smoke` de la CI) -- suffisant pour empêcher qu'un futur retrait
accidentel d'une ligne `COPY` régresse silencieusement l'un de ces deux
bugs déjà vécus."""

from __future__ import annotations

from pathlib import Path

_DOCKERFILE = Path(__file__).resolve().parent.parent / "Dockerfile"


def test_dockerfile_copies_domains_and_alembic():
    text = _DOCKERFILE.read_text(encoding="utf-8")
    assert "COPY domains/ ./domains/" in text, "régression du bug de la Décision 1 (Phase 1)"
    assert "COPY alembic.ini" in text, "régression -- init_db()/run_migrations() échouera silencieusement"
    assert "COPY alembic/ ./alembic/" in text, "régression -- init_db()/run_migrations() échouera silencieusement"
