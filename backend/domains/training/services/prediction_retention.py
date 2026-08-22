"""Purge des prédictions trop anciennes (Lot 5, correctif I2,
AUDIT_DATALAB_2026-08-16.md §I2, "+ rétention").

Même approche que `services/job_watchdog.py` : pas de scheduler dédié
(Celery beat ou équivalent), une purge à la demande au moment où elle a du
sens — ici, juste avant d'enregistrer une nouvelle prédiction pour
l'organisation. Une prédiction jamais réutilisée est donc purgée au plus
tard à la prochaine prédiction de sa propre organisation, ce qui est
précisément le moment où sa présence a un coût (stockage, exposition de
données potentiellement sensibles dans `input_json`)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy.orm import Session

from api.core.models import Prediction


def _as_aware_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Copié depuis `services/job_watchdog.py` (pas importé — indépendance
    volontaire, même principe que `_user_safe_error_message` dans les
    workers vision) : SQLite (dev) ne conserve pas le fuseau horaire des
    colonnes `DateTime(timezone=True)`, une valeur relue est naïve alors
    qu'elle a toujours été écrite en UTC (`server_default=func.now()`).
    PostgreSQL (prod) conserve le fuseau, rien à changer."""
    if dt is None or dt.tzinfo is not None:
        return dt
    return dt.replace(tzinfo=timezone.utc)


def purge_old_predictions(db: Session, organization_id: int, retention_days: int) -> int:
    """Supprime les prédictions de l'organisation antérieures à
    `retention_days`. Retourne le nombre de lignes supprimées. Ne COMMIT
    PAS elle-même — l'appelant l'inclut dans la même transaction que
    l'action qui déclenche la purge (voir
    api/routers/training.py::predict_with_model).

    Comparaison faite en PYTHON, jamais par un filtre SQL sur
    `created_at` : le même écart de fuseau horaire SQLite que
    `job_watchdog.py` documente rendrait un filtre SQL direct peu fiable
    en dev/test (SQLite) tout en fonctionnant correctement en prod
    (PostgreSQL) — un bug qui ne se manifesterait jamais dans la suite de
    tests locale. Charge seulement `id`/`created_at` (pas les colonnes
    JSON, potentiellement volumineuses) pour cette comparaison."""
    threshold = datetime.now(timezone.utc) - timedelta(days=retention_days)
    rows = (
        db.query(Prediction.id, Prediction.created_at)
        .filter(Prediction.organization_id == organization_id)
        .all()
    )
    stale_ids = [row.id for row in rows if _as_aware_utc(row.created_at) < threshold]
    if stale_ids:
        db.query(Prediction).filter(Prediction.id.in_(stale_ids)).delete(synchronize_session=False)
    return len(stale_ids)
