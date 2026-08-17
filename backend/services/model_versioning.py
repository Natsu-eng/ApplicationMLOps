"""Numérotation de version des modèles (Lot 5, correctif P1,
AUDIT_DATALAB_2026-08-16.md §P1) — un modèle "version 3" garde ce numéro
pour toujours : assigné une seule fois à la création (voir
`workers/training_worker.py`), jamais recalculé rétroactivement."""
from __future__ import annotations

from sqlalchemy import func
from sqlalchemy.orm import Session

from api.core.models import MLModel


def next_version(db: Session, organization_id: int, dataset_id: int, target_column: str) -> int:
    """Numéro de version pour un nouveau modèle sur ce "problème" (même
    dataset + même colonne cible, au sein de l'organisation) — 1 pour le
    tout premier, sinon le maximum existant + 1."""
    current_max = (
        db.query(func.max(MLModel.version))
        .filter(
            MLModel.organization_id == organization_id,
            MLModel.dataset_id == dataset_id,
            MLModel.target_column == target_column,
        )
        .scalar()
    )
    return (current_max or 0) + 1
