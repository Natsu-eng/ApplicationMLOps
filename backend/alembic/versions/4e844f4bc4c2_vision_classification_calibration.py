"""vision_classification_calibration

Revision ID: 4e844f4bc4c2
Revises: d6d6b99f43c1
Create Date: 2026-08-28 13:06:48.225033

Ajoute `vision_classification_models.calibration_json` — onglet "Fiabilité"
du pilier Vision (retour utilisateur : "d'autres fonctionnalités modernes
que les autres plateformes n'offrent pas") : courbe de calibration
(reliability diagram), portée depuis `ml_training.py::_compute_calibration`
et déjà exposée côté tabulaire (`ml_models.calibration_json`) — même motif
NULLABLE que `roc_curves_json`/`pr_curves_json` (rétrocompatibilité par
absence pour les modèles entraînés avant ce correctif).

`alembic revision --autogenerate` a, comme systématiquement (déjà repéré en
Phase 3 puis dans les migrations `0ecc0331cbd1`/`d6d6b99f43c1`), redétecté 3
dérives de schéma préexistantes SANS RAPPORT avec ce correctif
(`ml_models.promoted_at`, `training_jobs.progress_updated_at`,
`password_reset_tokens.created_at`) — retirées manuellement ci-dessous,
seule la nouvelle colonne est conservée.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '4e844f4bc4c2'
down_revision: Union[str, Sequence[str], None] = 'd6d6b99f43c1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('vision_classification_models', sa.Column('calibration_json', sa.Text(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('vision_classification_models', 'calibration_json')
