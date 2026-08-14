"""Journal d'audit (Lot 10 — durcissement SaaS) — logique pure d'écriture
d'une entrée, appelée depuis les routers pour les actions sensibles
(suppression de dataset/entraînement, ajout de membre, promotion de
modèle). Volontairement minimal : pas un système d'événements générique,
juste ce qu'un `owner` voudrait pouvoir auditer après coup.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from sqlalchemy.orm import Session

from api.core.models import AuditLog


def log_action(
    db: Session,
    organization_id: int,
    actor_id: Optional[int],
    action: str,
    target_type: Optional[str] = None,
    target_id: Optional[int] = None,
    details: Optional[dict[str, Any]] = None,
) -> None:
    """Ajoute une entrée au journal — ne COMMIT PAS elle-même : l'appelant
    l'ajoute à la même transaction que l'action auditée (ex. la suppression
    elle-même), pour qu'un rollback annule les deux ensemble plutôt que de
    risquer un journal incohérent avec ce qui s'est réellement passé."""
    db.add(AuditLog(
        organization_id=organization_id,
        actor_id=actor_id,
        action=action,
        target_type=target_type,
        target_id=target_id,
        details_json=json.dumps(details) if details else None,
    ))
