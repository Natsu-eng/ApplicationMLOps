"""users: anonymized_at

Revision ID: c57a3d066f70
Revises: 1c2fb52e305f
Create Date: 2026-09-04 12:40:43.773885

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c57a3d066f70'
down_revision: Union[str, Sequence[str], None] = '1c2fb52e305f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Colonne ADDITIVE sur `users` — aucun backfill.

    Horodate l'anonymisation d'un compte (NULL = compte nominatif). Le
    compte n'est jamais SUPPRIME : ses datasets, entrainements et entrees
    d'audit appartiennent a l'organisation et doivent lui survivre. Seules
    les donnees personnelles sont effacees sur place, la ligne et toutes
    ses cles etrangeres restant valides — d'ou une colonne plutot qu'un
    DELETE en cascade, qui detruirait des livrables encore utiles.
    """
    op.add_column('users', sa.Column('anonymized_at', sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    """Retire la colonne. Attention : les anonymisations deja effectuees ne
    sont PAS annulees — elles sont irreversibles par construction, seule la
    trace de leur date disparait."""
    op.drop_column('users', 'anonymized_at')
