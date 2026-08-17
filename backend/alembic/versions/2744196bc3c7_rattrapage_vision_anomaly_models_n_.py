"""rattrapage vision_anomaly_models n_calibration n_evaluation

Revision ID: 2744196bc3c7
Revises: 594bce594adf
Create Date: 2026-08-17 02:22:24.022995

Migration de rattrapage (correctif — incident réel, AUDIT_DATALAB_2026-08-16.md).
`n_calibration`/`n_evaluation` ont été ajoutées au modèle `VisionAnomalyModel`
au Lot 0.2, avant l'introduction d'Alembic (Lot 1.1). La révision initiale
(594bce594adf, autogénérée après coup) les décrit correctement, mais la base
de développement existante a été STAMPÉE `head` par une version antérieure,
moins stricte, de `run_migrations()` sans que ces deux colonnes n'aient
jamais été créées physiquement — chaque appel touchant `vision_anomaly_models`
échouait en 500 (`UndefinedColumn`). Cette révision comble l'écart sans
toucher aux données existantes (colonnes nullables, valeur NULL sur les
lignes déjà en base — dégradation honnête déjà gérée côté API/UI, jamais de
valeur inventée).

Généré par `alembic revision --autogenerate` contre la base de
développement réelle, puis élagué à la main : l'autogénération a aussi
détecté un changement de type sur `ml_models.promoted_at` et
`training_jobs.progress_updated_at` (TIMESTAMP → DateTime(timezone=True)),
sans rapport avec cet incident — non traité ici (voir DECISIONS.md).

**Idempotente, volontairement** : 594bce594adf crée déjà ces deux colonnes
pour toute base NEUVE (elle décrit correctement le modèle actuel). Cette
révision ne sert qu'aux bases EXISTANTES retombées sur le schéma d'avant
Lot 0.2 à cause de l'incident décrit ci-dessus. Un `op.add_column`
inconditionnel casserait donc toute base neuve qui traverse la chaîne
complète (`duplicate column name` — constaté en test avant ce correctif) ;
`upgrade()`/`downgrade()` vérifient donc l'état réel avant d'agir, dans les
deux sens.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2744196bc3c7'
down_revision: Union[str, Sequence[str], None] = '594bce594adf'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_TABLE = "vision_anomaly_models"
_COLUMNS = ("n_calibration", "n_evaluation")


def upgrade() -> None:
    """Upgrade schema — ajoute uniquement les colonnes réellement absentes."""
    existing = {c["name"] for c in sa.inspect(op.get_bind()).get_columns(_TABLE)}
    for column_name in _COLUMNS:
        if column_name not in existing:
            op.add_column(_TABLE, sa.Column(column_name, sa.Integer(), nullable=True))


def downgrade() -> None:
    """Downgrade schema — ne retire que les colonnes réellement présentes."""
    existing = {c["name"] for c in sa.inspect(op.get_bind()).get_columns(_TABLE)}
    for column_name in reversed(_COLUMNS):
        if column_name in existing:
            op.drop_column(_TABLE, column_name)
