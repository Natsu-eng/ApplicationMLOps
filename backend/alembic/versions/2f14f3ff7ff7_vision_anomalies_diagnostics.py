"""vision_anomalies_diagnostics

Revision ID: 2f14f3ff7ff7
Revises: 4e844f4bc4c2
Create Date: 2026-08-28 13:35:00.000000

Ajoute 4 colonnes à `vision_anomaly_models` (retour utilisateur : "rendre
l'onglet détection d'anomalies aussi riche/transparent que la
classification" + "d'autres fonctionnalités modernes que les autres
plateformes n'offrent pas") :
- `roc_curves_json`/`pr_curves_json` : courbes ROC/PR sur l'évaluation, une
  seule clé "Défaut" (classe positive binaire) — même FORME EXACTE que
  `vision_classification_models.roc_curves_json`/`pr_curves_json`, pour
  réutiliser `EvaluationCharts.tsx` tel quel côté frontend.
- `score_histogram_json` : distribution des scores d'anomalie normal vs
  défaut — montre la séparabilité des deux populations.
- `category_breakdown_json` : taux de détection PAR catégorie de défaut,
  calculé sur la totalité de l'évaluation (pas seulement les exemples
  affichés) — un dataset multi-défauts peut cacher un type de défaut mal
  détecté derrière une bonne moyenne globale.

Toutes NULLABLE — NULL sur les modèles entraînés avant ce correctif
(rétrocompatibilité par absence, même motif que roc_curves_json côté
classification).

`alembic revision --autogenerate` a, comme systématiquement (déjà repéré en
Phase 3 puis dans les migrations `0ecc0331cbd1`/`d6d6b99f43c1`/
`4e844f4bc4c2`), redétecté 3 dérives de schéma préexistantes SANS RAPPORT
avec ce correctif (`ml_models.promoted_at`, `training_jobs.progress_updated_at`,
`password_reset_tokens.created_at`) — retirées manuellement ci-dessous,
seules les 4 nouvelles colonnes sont conservées.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '2f14f3ff7ff7'
down_revision: Union[str, Sequence[str], None] = '4e844f4bc4c2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('vision_anomaly_models', sa.Column('roc_curves_json', sa.Text(), nullable=True))
    op.add_column('vision_anomaly_models', sa.Column('pr_curves_json', sa.Text(), nullable=True))
    op.add_column('vision_anomaly_models', sa.Column('score_histogram_json', sa.Text(), nullable=True))
    op.add_column('vision_anomaly_models', sa.Column('category_breakdown_json', sa.Text(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('vision_anomaly_models', 'category_breakdown_json')
    op.drop_column('vision_anomaly_models', 'score_histogram_json')
    op.drop_column('vision_anomaly_models', 'pr_curves_json')
    op.drop_column('vision_anomaly_models', 'roc_curves_json')
