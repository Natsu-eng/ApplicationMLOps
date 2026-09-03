"""users: must_change_password + deactivated_at

Revision ID: 1c2fb52e305f
Revises: ec93c856ac0b
Create Date: 2026-09-04 00:46:38.850797

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1c2fb52e305f'
down_revision: Union[str, Sequence[str], None] = 'ec93c856ac0b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Deux colonnes ADDITIVES sur `users` — aucun backfill nécessaire.

    `must_change_password` (server_default="false") : un membre ajouté par
    le propriétaire reçoit un mot de passe que celui-ci a choisi et connaît
    donc indéfiniment. Cet indicateur force l'intéressé à en choisir un
    autre avant toute autre action. La valeur par défaut `false` est
    délibérée : les comptes EXISTANTS ne sont pas forcés rétroactivement,
    leur mot de passe ayant déjà été choisi par eux ou avant ce correctif.

    `deactivated_at` (NULL) : date de révocation de l'accès, pour afficher
    depuis quand un compte est coupé. NULL sur tous les comptes existants,
    y compris ceux déjà inactifs — la date n'était pas conservée jusqu'ici,
    l'inventer serait mentir sur une donnée d'audit.
    """
    # `sa.false()` et non la chaîne 'false' : SQLite stockerait sinon le
    # TEXTE 'false', truthy en Python, bloquant tout nouveau compte en
    # "mot de passe provisoire". PostgreSQL n'est pas concerné, d'où un
    # bug qui ne serait apparu qu'en dev/test ou sur un déploiement SQLite.
    op.add_column('users', sa.Column('must_change_password', sa.Boolean(), server_default=sa.false(), nullable=False))
    op.add_column('users', sa.Column('deactivated_at', sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    """Retire les deux colonnes — réversible sans perte fonctionnelle
    (aucune autre table ne les référence)."""
    op.drop_column('users', 'deactivated_at')
    op.drop_column('users', 'must_change_password')
