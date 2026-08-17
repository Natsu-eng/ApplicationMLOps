"""versions de modele reelles P1

Revision ID: f983e8e87dcb
Revises: 0860f4355873
Create Date: 2026-08-17 22:19:35.709452

Lot 5 (correctif P1, AUDIT_DATALAB_2026-08-16.md §P1) — ajoute `dataset_id`
(dénormalisé depuis `training_jobs.dataset_id`, comme `target_column` déjà
présent) et `version` (numéro séquentiel au sein du "problème" organisation
+ dataset + cible) à `ml_models`.

Base autogénérée (`alembic revision --autogenerate`) PUIS retravaillée à la
main : l'autogénération produit des colonnes `NOT NULL` sans valeur par
défaut, ce qui échoue dès qu'une seule ligne `ml_models` existe déjà
(cas réel — cette table est peuplée dès le Lot 3). Colonnes ajoutées
NULLABLE, rétropeuplées, PUIS rendues `NOT NULL` :
- `dataset_id` : une seule requête corrélée depuis `training_jobs`
  (portable SQLite/PostgreSQL, jamais de UPDATE...FROM propriétaire).
- `version` : calculée en PYTHON (pas de fonction fenêtrée SQL — portable
  y compris sur une vieille version de SQLite sans support des fonctions
  fenêtrées), regroupée par (organization_id, dataset_id, target_column),
  numérotée par `id` croissant (jamais `created_at` — precision seconde de
  SQLite, egalites possibles entre lignes creees rapidement, meme lecon
  que le correctif I3 du Lot 4/pagination).

Testée : upgrade contre une base SQLite peuplée (plusieurs "problèmes",
plusieurs modèles par problème) → vérification que chaque lignée reçoit
bien 1, 2, 3... → downgrade → ré-upgrade, avant ce commit.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f983e8e87dcb'
down_revision: Union[str, Sequence[str], None] = '0860f4355873'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema — colonnes nullable, rétropeuplées, puis NOT NULL."""
    with op.batch_alter_table('ml_models', schema=None) as batch_op:
        batch_op.add_column(sa.Column('dataset_id', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('version', sa.Integer(), nullable=True))

    bind = op.get_bind()

    # dataset_id : dénormalisé depuis training_jobs.dataset_id, une seule
    # ligne training_jobs par ml_models.training_job_id (contrainte
    # UNIQUE déjà en place sur cette colonne).
    bind.execute(sa.text(
        "UPDATE ml_models SET dataset_id = ("
        "SELECT training_jobs.dataset_id FROM training_jobs "
        "WHERE training_jobs.id = ml_models.training_job_id"
        ")"
    ))

    # version : numérotation par lignée (organization_id, dataset_id,
    # target_column), par id croissant — calculée en Python plutôt qu'en
    # SQL (ROW_NUMBER()) pour rester portable sans supposer une version
    # récente de SQLite.
    rows = bind.execute(sa.text(
        "SELECT id, organization_id, dataset_id, target_column FROM ml_models ORDER BY id ASC"
    )).fetchall()
    counters: dict[tuple, int] = {}
    for row in rows:
        key = (row.organization_id, row.dataset_id, row.target_column)
        counters[key] = counters.get(key, 0) + 1
        bind.execute(
            sa.text("UPDATE ml_models SET version = :version WHERE id = :id"),
            {"version": counters[key], "id": row.id},
        )

    with op.batch_alter_table('ml_models', schema=None) as batch_op:
        batch_op.alter_column('dataset_id', existing_type=sa.Integer(), nullable=False)
        batch_op.alter_column('version', existing_type=sa.Integer(), nullable=False)
        batch_op.create_index(batch_op.f('ix_ml_models_dataset_id'), ['dataset_id'], unique=False)
        batch_op.create_unique_constraint(
            'uq_ml_models_version', ['organization_id', 'dataset_id', 'target_column', 'version']
        )
        batch_op.create_foreign_key(
            'fk_ml_models_dataset_id_datasets', 'datasets', ['dataset_id'], ['id'], ondelete='CASCADE'
        )


def downgrade() -> None:
    """Downgrade schema — supprime dataset_id/version (et avec eux, le
    numéro de version et la contrainte d'unicité — irréversible comme
    toute suppression de colonne)."""
    with op.batch_alter_table('ml_models', schema=None) as batch_op:
        batch_op.drop_constraint('uq_ml_models_version', type_='unique')
        batch_op.drop_constraint('fk_ml_models_dataset_id_datasets', type_='foreignkey')
        batch_op.drop_index(batch_op.f('ix_ml_models_dataset_id'))
        batch_op.drop_column('version')
        batch_op.drop_column('dataset_id')
