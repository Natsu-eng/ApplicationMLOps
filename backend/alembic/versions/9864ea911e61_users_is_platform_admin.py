"""users: is_platform_admin

Revision ID: 9864ea911e61
Revises: c57a3d066f70
Create Date: 2026-09-04 13:06:03.597984

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9864ea911e61'
down_revision: Union[str, Sequence[str], None] = 'c57a3d066f70'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Colonne ADDITIVE — aucun compte n'est promu par cette migration.

    `server_default=false()` : TOUS les comptes existants restent de simples
    utilisateurs. Promouvoir quelqu'un est un geste explicite, fait par le
    script `scripts/grant_platform_admin.py` — jamais par une migration, qui
    devrait pour cela embarquer une adresse en dur et s'appliquerait
    aveuglement a n'importe quel environnement, production comprise.

    `sa.false()` et non la chaine 'false' : SQLite stockerait sinon le TEXTE
    'false', truthy en Python — chaque compte deviendrait administrateur de
    la plateforme en dev et en test. Meme piege que `must_change_password`
    (migration 1c2fb52e305f), evite ici des l'ecriture.
    """
    op.add_column('users', sa.Column('is_platform_admin', sa.Boolean(), server_default=sa.false(), nullable=False))


def downgrade() -> None:
    """Retire la colonne — toutes les promotions sont alors perdues."""
    op.drop_column('users', 'is_platform_admin')
