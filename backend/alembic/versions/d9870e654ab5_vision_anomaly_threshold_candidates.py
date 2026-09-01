"""vision_anomaly_threshold_candidates

Revision ID: d9870e654ab5
Revises: c4995f4bee3d
Create Date: 2026-09-01 18:35:30.159401

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'd9870e654ab5'
down_revision: Union[str, Sequence[str], None] = 'c4995f4bee3d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Seuils candidats (retour utilisateur, maquette de refonte) — nullable,
    # rétrocompatibilité par absence pour les modèles entraînés avant ce
    # correctif (services/engine.py::_compute_threshold_candidates).
    #
    # Note : l'autogénération Alembic a aussi détecté 3 dérives préexistantes
    # sans rapport avec ce correctif (types timestamp/nullabilité sur
    # ml_models.promoted_at, password_reset_tokens.created_at,
    # training_jobs.progress_updated_at) — volontairement OMISES ici, à
    # traiter dans une migration dédiée si besoin.
    op.add_column('vision_anomaly_models', sa.Column('threshold_candidates_json', sa.Text(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('vision_anomaly_models', 'threshold_candidates_json')
