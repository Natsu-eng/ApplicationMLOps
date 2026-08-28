"""clustering top3 rangs composites et profils

Revision ID: d6d6b99f43c1
Revises: 0ecc0331cbd1
Create Date: 2026-08-28 11:19:17.542610

Retour utilisateur direct : la sélection du meilleur clustering au seul
score de silhouette pouvait élire une configuration nettement pire sur les
2 autres métriques (Davies-Bouldin, Calinski-Harabasz) pour un gain de
silhouette marginal — voir domains/clustering/services/engine.py::
_attach_composite_rank. Ajoute les rangs individuels + le rang composite
(transparence de la sélection, jamais une boîte noire) et le profil complet
(segments) pour le TOP 3 candidats (comparer plusieurs modèles au lieu d'un
seul, retour utilisateur direct).

NB : `alembic revision --autogenerate` a aussi détecté 3 écarts de schéma
SANS RAPPORT avec ce correctif (ml_models.promoted_at,
training_jobs.progress_updated_at : TIMESTAMP sans fuseau en base vs
DateTime(timezone=True) dans le modèle ; password_reset_tokens.created_at :
NOT NULL en base mais nullable dans le modèle — déjà repérés une première
fois en Phase 3, voir _backend/JOURNAL.md) — retirés manuellement de cette
migration, comme en Phase 3, pour ne jamais mélanger un correctif de dérive
de schéma non lié avec ce changement.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'd6d6b99f43c1'
down_revision: Union[str, Sequence[str], None] = '0ecc0331cbd1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('cluster_candidates', sa.Column('rank_silhouette', sa.Float(), nullable=True))
    op.add_column('cluster_candidates', sa.Column('rank_davies_bouldin', sa.Float(), nullable=True))
    op.add_column('cluster_candidates', sa.Column('rank_calinski_harabasz', sa.Float(), nullable=True))
    op.add_column('cluster_candidates', sa.Column('composite_rank', sa.Float(), nullable=True))
    op.add_column('cluster_candidates', sa.Column('cluster_profiles_json', sa.Text(), nullable=True))
    op.add_column('cluster_candidates', sa.Column('noise_count', sa.Integer(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('cluster_candidates', 'noise_count')
    op.drop_column('cluster_candidates', 'cluster_profiles_json')
    op.drop_column('cluster_candidates', 'composite_rank')
    op.drop_column('cluster_candidates', 'rank_calinski_harabasz')
    op.drop_column('cluster_candidates', 'rank_davies_bouldin')
    op.drop_column('cluster_candidates', 'rank_silhouette')
