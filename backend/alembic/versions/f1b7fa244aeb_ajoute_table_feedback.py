"""ajoute table feedback

Revision ID: f1b7fa244aeb
Revises: 55bc1e62c303
Create Date: 2026-08-22 17:29:22.578801

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'f1b7fa244aeb'
down_revision: Union[str, Sequence[str], None] = '55bc1e62c303'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.

    Ne crée QUE la table `feedback` — l'autogenerate d'Alembic avait aussi
    détecté un écart préexistant, sans rapport (TIMESTAMP -> DateTime(tz)
    sur ml_models.promoted_at/training_jobs.progress_updated_at), retiré
    volontairement de cette migration : un chantier séparé, pas dans le
    périmètre de ce lot (refonte UI, formulaire de retour du centre
    d'aide)."""
    op.create_table('feedback',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('organization_id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('page', sa.String(length=300), nullable=False),
    sa.Column('message', sa.Text(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_feedback_id'), 'feedback', ['id'], unique=False)
    op.create_index(op.f('ix_feedback_organization_id'), 'feedback', ['organization_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_feedback_organization_id'), table_name='feedback')
    op.drop_index(op.f('ix_feedback_id'), table_name='feedback')
    op.drop_table('feedback')
