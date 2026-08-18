"""hash et doublon de dataset P2

Revision ID: d6231c7548cf
Revises: f983e8e87dcb
Create Date: 2026-08-17 23:14:17.963291

Lot 5 (correctif P2, AUDIT_DATALAB_2026-08-16.md §P2) — ajoute
`content_hash` (SHA-256 du fichier, calculé à l'upload) et
`duplicate_of_dataset_id` (renseigné si un dataset de la même
organisation partage déjà ce hash) à `datasets`.

Les deux colonnes sont NULLABLE, sans rétropeuplement : contrairement à
P1 (Lot 5, migration f983e8e87dcb), il n'existe aucune donnée à
recalculer à partir des colonnes déjà en base — le hash exige de relire
le fichier source sur disque, hors périmètre d'une migration de schéma
(chemins de stockage non garantis identiques entre environnements, gros
fichiers potentiellement lents à hasher en masse). Un dataset uploadé
avant ce lot garde `content_hash IS NULL` pour toujours — dégradation
honnête, cohérente avec le reste du projet (voir `stage`/`promoted_at`
sur `ml_models`, jamais rétropeuplés non plus).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd6231c7548cf'
down_revision: Union[str, Sequence[str], None] = 'f983e8e87dcb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema — ajoute content_hash/duplicate_of_dataset_id."""
    with op.batch_alter_table('datasets', schema=None) as batch_op:
        batch_op.add_column(sa.Column('content_hash', sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column('duplicate_of_dataset_id', sa.Integer(), nullable=True))
        batch_op.create_index(batch_op.f('ix_datasets_content_hash'), ['content_hash'], unique=False)
        batch_op.create_foreign_key(
            'fk_datasets_duplicate_of_dataset_id_datasets',
            'datasets', ['duplicate_of_dataset_id'], ['id'], ondelete='SET NULL',
        )


def downgrade() -> None:
    """Downgrade schema — supprime content_hash/duplicate_of_dataset_id."""
    with op.batch_alter_table('datasets', schema=None) as batch_op:
        batch_op.drop_constraint('fk_datasets_duplicate_of_dataset_id_datasets', type_='foreignkey')
        batch_op.drop_index(batch_op.f('ix_datasets_content_hash'))
        batch_op.drop_column('duplicate_of_dataset_id')
        batch_op.drop_column('content_hash')
