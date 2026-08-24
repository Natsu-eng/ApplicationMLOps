"""phase3 request_id sur jobs et audit_logs

Revision ID: 0ecc0331cbd1
Revises: c274f8e19a3b
Create Date: 2026-08-24 11:51:42.164862

Ajoute `request_id` (Phase 3, AUDIT_BACKEND_2026-08-23.md, Axe I) sur les 6
tables de job + `audit_logs` — corrèle un job/une entrée d'audit à la
requête HTTP qui l'a produit (voir api/core/observability.py).

Note : l'autogénération Alembic a aussi détecté 3 écarts de schéma SANS
RAPPORT avec cette phase (`ml_models.promoted_at` et
`training_jobs.progress_updated_at` : TIMESTAMP sans fuseau en base vs
`DateTime(timezone=True)` dans models.py ; `password_reset_tokens.created_at` :
NOT NULL en base mais nullable dans le modèle) — dérive préexistante,
volontairement PAS incluse ici (hors périmètre traçabilité, mériterait
son propre correctif et sa propre validation dédiée) — voir
_backend/JOURNAL.md, Phase 3, et RAPPORT-FINAL.md "ce qui a été laissé de
côté"."""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '0ecc0331cbd1'
down_revision: Union[str, Sequence[str], None] = 'c274f8e19a3b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('anomaly_jobs', sa.Column('request_id', sa.String(length=36), nullable=True))
    op.add_column('audit_logs', sa.Column('request_id', sa.String(length=36), nullable=True))
    op.add_column('clustering_jobs', sa.Column('request_id', sa.String(length=36), nullable=True))
    op.add_column('dimensionality_jobs', sa.Column('request_id', sa.String(length=36), nullable=True))
    op.add_column('training_jobs', sa.Column('request_id', sa.String(length=36), nullable=True))
    op.add_column('vision_anomaly_jobs', sa.Column('request_id', sa.String(length=36), nullable=True))
    op.add_column('vision_classification_jobs', sa.Column('request_id', sa.String(length=36), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('vision_classification_jobs', 'request_id')
    op.drop_column('vision_anomaly_jobs', 'request_id')
    op.drop_column('training_jobs', 'request_id')
    op.drop_column('dimensionality_jobs', 'request_id')
    op.drop_column('clustering_jobs', 'request_id')
    op.drop_column('audit_logs', 'request_id')
    op.drop_column('anomaly_jobs', 'request_id')
