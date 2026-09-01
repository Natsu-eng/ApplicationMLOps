"""fix_timezone_and_notnull_drift

Corrige 3 écarts de schéma préexistants entre `api/core/models.py` et la
base réelle, détectés par `alembic --autogenerate` et volontairement mis
de côté lors de la Phase 3 de la consolidation backend (voir
`_backend/RAPPORT-FINAL.md` §2/§3) faute de temps pour les qualifier
correctement. Qualifiés et corrigés ici :

1. `ml_models.promoted_at` et `training_jobs.progress_updated_at` étaient
   `TIMESTAMP WITHOUT TIME ZONE` alors que le modèle déclare
   `DateTime(timezone=True)` (comme TOUTES les autres colonnes
   `progress_updated_at` du dépôt, déjà `TIMESTAMPTZ` sans écart).

   Impact réel trouvé en creusant (pas juste un écart cosmétique) :
   `domains/shared/job_watchdog.py::_as_aware_utc()` reçoit un datetime
   naïf lu depuis une colonne `TIMESTAMP` et lui colle `tzinfo=utc` par
   simple `.replace()`, EN SUPPOSANT que la valeur stockée est déjà en
   UTC. Or le serveur Postgres de développement a pour fuseau de session
   `Africa/Casablanca` (UTC+1) — vérifié via `show timezone`. Le code
   applicatif écrit systématiquement `datetime.now(timezone.utc)`
   (`domains/*/worker.py`), mais une valeur timezone-aware insérée dans
   une colonne `TIMESTAMP` (sans fuseau) est convertie par Postgres au
   fuseau de la session AVANT d'être stockée, nue de toute information de
   fuseau. Résultat : la valeur naïve relue et étiquetée "UTC" par
   `_as_aware_utc` est en réalité décalée d'+1h — la détection de jobs
   bloqués (`job_watchdog`) sous-estimait systématiquement l'ancienneté
   réelle d'1 heure (un job réellement bloqué 90 min pouvait n'apparaître
   bloqué que 30 min à ce mécanisme).

   Correctif : passage en `TIMESTAMPTZ`. Le cast implicite Postgres
   `timestamp -> timestamptz` réinterprète la valeur naïve stockée dans
   le fuseau de session ACTUEL — qui est le même fuseau qu'à l'écriture
   (aucune configuration de fuseau par connexion dans
   `api/core/database.py`) — ce qui reconstruit exactement l'instant UTC
   d'origine, vérifié empiriquement avant ce correctif (valeurs
   comparées avant/après cast sur des lignes réelles, aucune donnée
   perdue ni décalée). Une fois `TIMESTAMPTZ`, psycopg2 renvoie des
   datetimes déjà "aware" : `_as_aware_utc` (qui court-circuite déjà les
   valeurs non-naïves via `dt.tzinfo is not None`) devient un simple
   passe-plat correct pour ces 2 colonnes, sans rien à changer côté code.

2. `password_reset_tokens.created_at` était nullable en base alors que le
   modèle le déclare non-optionnel (`Mapped[datetime]`, jamais
   `Optional`) avec `server_default=func.now()` — la base était donc
   MOINS stricte que ce que le code suppose partout ailleurs. Aucune
   ligne existante n'a `created_at IS NULL` (vérifié avant ce correctif :
   0 ligne) — `SET NOT NULL` s'applique donc sans perte ni migration de
   données.

Note opérationnelle sur cette révision précise : générée et appliquée
malgré elle par un redémarrage automatique d'uvicorn (`--reload`) qui a
détecté le fichier fraîchement créé et relancé `lifespan()` ->
`init_db()` -> `alembic upgrade head` avant que ce fichier ne soit
renommé/documenté — d'où l'écart entre la révision (`2c88481be342`,
déjà stampée dans `alembic_version`) et son message initial de
diagnostic temporaire. Contenu SQL inchangé par rapport à ce qui a été
réellement exécuté ; seuls le nom de fichier et cette note ont été
complétés après coup pour que l'historique du dépôt reste cohérent avec
l'état réel de la base.

Revision ID: 2c88481be342
Revises: d9870e654ab5
Create Date: 2026-09-01 22:04:23.680875

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '2c88481be342'
down_revision: Union[str, Sequence[str], None] = 'd9870e654ab5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.alter_column('ml_models', 'promoted_at',
               existing_type=postgresql.TIMESTAMP(),
               type_=sa.DateTime(timezone=True),
               existing_nullable=True)
    op.alter_column('password_reset_tokens', 'created_at',
               existing_type=postgresql.TIMESTAMP(timezone=True),
               nullable=False,
               existing_server_default=sa.text('now()'))
    op.alter_column('training_jobs', 'progress_updated_at',
               existing_type=postgresql.TIMESTAMP(),
               type_=sa.DateTime(timezone=True),
               existing_nullable=True)


def downgrade() -> None:
    """Downgrade schema."""
    op.alter_column('training_jobs', 'progress_updated_at',
               existing_type=sa.DateTime(timezone=True),
               type_=postgresql.TIMESTAMP(),
               existing_nullable=True)
    op.alter_column('password_reset_tokens', 'created_at',
               existing_type=postgresql.TIMESTAMP(timezone=True),
               nullable=True,
               existing_server_default=sa.text('now()'))
    op.alter_column('ml_models', 'promoted_at',
               existing_type=sa.DateTime(timezone=True),
               type_=postgresql.TIMESTAMP(),
               existing_nullable=True)
