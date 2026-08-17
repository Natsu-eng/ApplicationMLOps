"""ajout table predictions

Revision ID: 0860f4355873
Revises: 2744196bc3c7
Create Date: 2026-08-17 21:15:01.699750

Lot 5 (correctif I2, AUDIT_DATALAB_2026-08-16.md §I2) — persiste chaque
prédiction individuelle (`POST /training/jobs/{id}/predict`), jusqu'ici
perdue sitôt la réponse HTTP envoyée. Table neuve, aucune donnée
existante à migrer.

Généré par `alembic revision --autogenerate` contre une base neuve
(vide, à jour de 2744196bc3c7) : seule la table `predictions` a été
détectée — aucune dérive de schéma pré-existante recapturée au passage
(contrairement à 2744196bc3c7, un correctif de rattrapage).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0860f4355873'
down_revision: Union[str, Sequence[str], None] = '2744196bc3c7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema — crée la table `predictions`."""
    op.create_table('predictions',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('organization_id', sa.Integer(), nullable=False),
    sa.Column('ml_model_id', sa.Integer(), nullable=False),
    sa.Column('requested_by_id', sa.Integer(), nullable=True),
    sa.Column('input_json', sa.Text(), nullable=False),
    sa.Column('output_json', sa.Text(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.ForeignKeyConstraint(['ml_model_id'], ['ml_models.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['requested_by_id'], ['users.id'], ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('predictions', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_predictions_created_at'), ['created_at'], unique=False)
        batch_op.create_index(batch_op.f('ix_predictions_id'), ['id'], unique=False)
        batch_op.create_index(batch_op.f('ix_predictions_ml_model_id'), ['ml_model_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_predictions_organization_id'), ['organization_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema — supprime la table `predictions` (et son
    historique de traçabilité avec elle — irréversible, comme toute
    suppression de table)."""
    with op.batch_alter_table('predictions', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_predictions_organization_id'))
        batch_op.drop_index(batch_op.f('ix_predictions_ml_model_id'))
        batch_op.drop_index(batch_op.f('ix_predictions_id'))
        batch_op.drop_index(batch_op.f('ix_predictions_created_at'))

    op.drop_table('predictions')
