"""Preuve, contre un VRAI PostgreSQL peuplé, qu'une migration Alembic
n'efface ni ne corrompt des données existantes (Phase 5,
AUDIT_BACKEND_2026-08-23.md, Axe J) — appelé par `.github/workflows/ci.yml`.

Pourquoi un script séparé plutôt qu'un test pytest de plus : les tests
existants (`tests/test_alembic_migration.py`) prouvent déjà ce
comportement, mais UNIQUEMENT contre SQLite (bases isolées `tmp_path`,
jamais de service Postgres démarré pour la suite pytest en CI — voir
`.github/workflows/ci.yml`, commentaire du job `backend`). Or c'est
justement l'écart SQLite/Postgres qui a déjà causé un incident réel sur
ce dépôt (`alembic stamp head` sur une base peuplée a un jour cassé
`GET /vision/anomalies/jobs` — voir `_backend/JOURNAL.md`, Décision 1) :
un DDL qui se comporte correctement sur SQLite peut échouer ou se
comporter différemment sur PostgreSQL. Ce script ferme cet écart sans
risquer de modifier `tests/test_alembic_migration.py` (861 tests déjà
verts, aucune raison d'y introduire un risque de régression pour ajouter
une vérification qui peut vivre à côté).

Scénario :
1. Migre une base Postgres neuve jusqu'à la révision JUSTE AVANT la tête
   actuelle (donc AVANT le dernier ajout de colonnes).
2. Peuple : une organisation, un utilisateur, un dataset, un job
   d'entraînement — des lignes RÉELLES, pas un schéma vide.
3. Migre jusqu'à `head` (applique la dernière migration par-dessus les
   données existantes).
4. Vérifie qu'aucune ligne n'a été perdue et que les nouvelles colonnes
   existent avec la valeur attendue (NULL, jamais appliquées
   rétroactivement à des lignes préexistantes).

Usage :
    DATABASE_URL=postgresql://... python -m scripts.verify_migration_on_populated_db
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

from sqlalchemy import create_engine, text

from alembic import command
from api.core.database import _alembic_config

# Révision juste avant la tête actuelle (Phase 3, request_id) -- mettre à
# jour si une nouvelle migration est ajoutée en tête après celle-ci, comme
# les tests équivalents de tests/test_alembic_migration.py.
_PREVIOUS_HEAD = "c274f8e19a3b"


def main() -> int:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("DATABASE_URL requis (pointer vers un Postgres réel, jamais SQLite ici).", file=sys.stderr)
        return 1

    # `Config.set_main_option` passe par `ConfigParser`, qui interprète `%`
    # comme un début d'interpolation -- un `%3D` (encodage URL de `=`, ex.
    # dans `?options=-csearch_path%3D...`) y casse sans un `%` doublé.
    # `create_engine` (SQLAlchemy pur, appelé plus bas) ne doit PAS recevoir
    # cette version doublée -- seul `_alembic_config` (ConfigParser) en a
    # besoin. Bug réel trouvé en testant ce script contre un vrai Postgres
    # local (schéma isolé avec `?options=-csearch_path%3D...`).
    cfg = _alembic_config(db_url.replace("%", "%%"))

    print(f"[1/4] Migration vers la révision pré-existante {_PREVIOUS_HEAD}...")
    command.upgrade(cfg, _PREVIOUS_HEAD)

    print("[2/4] Peuplement avec des données réelles...")
    engine = create_engine(db_url)
    now = datetime.now(timezone.utc)
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO organizations (id, name, created_at) VALUES (1, 'Org CI peuplée', :now)"), {"now": now}
        )
        conn.execute(
            text(
                "INSERT INTO users (id, organization_id, email, nom, hashed_password, role, actif, created_at) "
                "VALUES (1, 1, 'ci@bureau.fr', 'CI', 'hash-factice', 'owner', true, :now)"
            ),
            {"now": now},
        )
        conn.execute(
            text(
                "INSERT INTO datasets (id, organization_id, name, file_path, file_size_bytes, status, created_at) "
                "VALUES (1, 1, 'ci.csv', 'unused', 1, 'ready', :now)"
            ),
            {"now": now},
        )
        conn.execute(
            text(
                "INSERT INTO training_jobs "
                "(id, organization_id, dataset_id, task_type, target_column, feature_columns_json, "
                "config_json, status, progress_percent, created_at) "
                "VALUES (1, 1, 1, 'regression', 'cible', '[]', '{}', 'completed', 100, :now)"
            ),
            {"now": now},
        )
    engine.dispose()

    print("[3/4] Migration vers head (applique la dernière migration sur la base peuplée)...")
    command.upgrade(cfg, "head")

    print("[4/4] Vérification : données intactes, nouvelles colonnes correctes...")
    engine = create_engine(db_url)
    with engine.connect() as conn:
        org_name = conn.execute(text("SELECT name FROM organizations WHERE id = 1")).scalar()
        user_email = conn.execute(text("SELECT email FROM users WHERE id = 1")).scalar()
        dataset_name = conn.execute(text("SELECT name FROM datasets WHERE id = 1")).scalar()
        job_status = conn.execute(text("SELECT status FROM training_jobs WHERE id = 1")).scalar()
        # Colonne ajoutée par la dernière migration -- doit exister et
        # valoir NULL pour une ligne créée AVANT son ajout (jamais
        # rétro-appliquée à une valeur inventée).
        job_request_id = conn.execute(text("SELECT request_id FROM training_jobs WHERE id = 1")).scalar()
    engine.dispose()

    assert org_name == "Org CI peuplée", f"organisation perdue ou corrompue : {org_name!r}"
    assert user_email == "ci@bureau.fr", f"utilisateur perdu ou corrompu : {user_email!r}"
    assert dataset_name == "ci.csv", f"dataset perdu ou corrompu : {dataset_name!r}"
    assert job_status == "completed", f"job perdu ou corrompu : {job_status!r}"
    assert job_request_id is None, f"request_id aurait dû rester NULL sur une ligne préexistante : {job_request_id!r}"

    print("OK -- migration appliquée sur une base peuplée sans perte ni corruption de données.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
