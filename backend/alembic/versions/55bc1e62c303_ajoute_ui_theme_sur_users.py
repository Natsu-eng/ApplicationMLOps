"""ajoute ui_theme sur users

Revision ID: 55bc1e62c303
Revises: 77a16b5c0e66
Create Date: 2026-08-22 10:05:00.000000

Lot UI (refonte visuelle, Fondations) — persistance serveur du thème choisi
(GET/PATCH /api/users/me/preferences). Colonne NOT NULL avec server_default
"graphite" : les lignes existantes reçoivent la valeur par défaut au moment
de l'ALTER TABLE, aucun backfill applicatif séparé n'est nécessaire — c'est
précisément le point qui avait cassé GET /vision/anomalies/jobs par le passé
(colonne ajoutée au modèle sans migration réellement appliquée sur une base
déjà peuplée) : cette migration a été vérifiée sur une copie de
database/datalab.db, pas seulement sur une base neuve.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '55bc1e62c303'
down_revision: Union[str, Sequence[str], None] = '77a16b5c0e66'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema — colonne NOT NULL avec server_default, aucun backfill séparé requis."""
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.add_column(sa.Column('ui_theme', sa.String(length=20), nullable=False, server_default='graphite'))


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.drop_column('ui_theme')
