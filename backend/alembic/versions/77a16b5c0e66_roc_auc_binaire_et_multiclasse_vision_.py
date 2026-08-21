"""roc auc binaire et multiclasse vision 16G

Revision ID: 77a16b5c0e66
Revises: d6231c7548cf
Create Date: 2026-08-21 13:23:51.675090

Lot 6A (correctif 16G, AUDIT_DATALAB_2026-08-16.md §16G) — ajoute
`roc_curves_json`/`pr_curves_json`/`test_roc_auc` a `vision_classification_models`.
Les trois colonnes sont NULLABLE des le depart (contrairement a P1/dataset_id
qui exigeaient un backfill) : aucune donnee existante ne permet de calculer
retroactivement une courbe ROC (les probabilites par classe du jeu de test
n'ont jamais ete persistees) — retrocompatibilite par absence, comme
`model_card_json` deja present sur cette meme table.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '77a16b5c0e66'
down_revision: Union[str, Sequence[str], None] = 'd6231c7548cf'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema — 3 colonnes nullable, aucun backfill possible ni necessaire."""
    with op.batch_alter_table('vision_classification_models', schema=None) as batch_op:
        batch_op.add_column(sa.Column('roc_curves_json', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('pr_curves_json', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('test_roc_auc', sa.Float(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('vision_classification_models', schema=None) as batch_op:
        batch_op.drop_column('test_roc_auc')
        batch_op.drop_column('pr_curves_json')
        batch_op.drop_column('roc_curves_json')
