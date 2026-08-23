"""ajoute token_valid_after sur users

Revision ID: a1c9d4e27b6f
Revises: f1b7fa244aeb
Create Date: 2026-08-23 21:15:00.000000

Phase 1 (AUDIT_BACKEND_2026-08-23.md, Axe A) — cycle de vie des jetons :
colonne NULLABLE, aucun server_default (NULL = « jamais révoqué en masse »,
comportement historique préservé pour toutes les lignes existantes, aucun
backfill nécessaire — contrairement à `ui_theme` qui avait besoin d'une
valeur non NULL pour toutes les lignes).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1c9d4e27b6f'
down_revision: Union[str, Sequence[str], None] = 'f1b7fa244aeb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema — colonne nullable, pas de backfill requis."""
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.add_column(sa.Column('token_valid_after', sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.drop_column('token_valid_after')
