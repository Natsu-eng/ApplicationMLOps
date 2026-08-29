"""Notifications de fin de job (retour utilisateur : "notifications de fin
de job — email/navigateur") — un seul point d'appel pour les 7 types de job
du projet, jamais une insertion `Notification(...)` répétée à la main dans
chaque worker (même principe que `job_lifecycle.py`/`job_quota.py`/
`job_watchdog.py` : un comportement partagé, un seul endroit à faire
évoluer).

Volontairement scopé aux deux transitions qui intéressent réellement
l'utilisateur qui attend un résultat — `completed` et `failed`. Jamais
`cancelled` : l'utilisateur qui annule un job SAIT déjà qu'il l'a annulé,
une notification serait du bruit, pas une information."""
from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from api.core.models import Notification

# Libellé humain + chemin frontend par type de job — seule source de vérité
# sur les `job_type` valides (voir Notification.job_type, api/core/models.py).
# Même convention `?job=` que Dashboard.tsx/AllHistory.tsx pour le
# deep-link, jamais reconstruite différemment ici.
_JOB_TYPE_INFO: dict[str, tuple[str, str]] = {
    "training": ("Entraînement", "/training"),
    "clustering": ("Clustering", "/clustering"),
    "dimensionality": ("Réduction de dimension", "/reduction-dimension"),
    "anomaly": ("Détection d'anomalies", "/anomalies"),
    "vision_classification": ("Classification d'images", "/vision/classification"),
    "vision_anomaly": ("Détection d'anomalies visuelles", "/vision/anomalies"),
    # Pas de page dédiée (vit dans l'onglet "Prédiction en lot" de la fiche
    # modèle) — le lien pointe vers l'entraînement source, `link_job_id`
    # doit alors être le `training_job_id`, pas l'id du lot lui-même.
    "batch_prediction": ("Prédiction en lot", "/training"),
}

_STATUS_LABELS = {"completed": "terminé avec succès", "failed": "en échec"}


def notify_job_terminal(
    db: Session,
    organization_id: int,
    user_id: Optional[int],
    job_type: str,
    job_id: int,
    status: str,
    subtitle: str,
    link_job_id: Optional[int] = None,
) -> None:
    """À appeler juste avant le `db.commit()` final d'un worker, une fois le
    job passé `completed`/`failed` (jamais pour `cancelled`, voir docstring
    du module) — ajoutée à la MÊME transaction, jamais un commit séparé.

    `user_id=None` (créateur du job supprimé depuis, `created_by_id` est
    nullable sur toutes les tables de job) : aucune notification à créer,
    silencieusement — jamais une erreur pour un cas qui n'a rien
    d'exceptionnel. `subtitle` : contexte court déjà lisible par l'utilisateur
    (ex. nom du dataset, "classe_a vs classe_b") — jamais un identifiant
    technique brut. `link_job_id` : id à utiliser dans le lien si différent
    de `job_id` (voir `batch_prediction` ci-dessus)."""
    if user_id is None:
        return
    label, path = _JOB_TYPE_INFO[job_type]
    db.add(Notification(
        organization_id=organization_id,
        user_id=user_id,
        job_type=job_type,
        job_id=job_id,
        status=status,
        title=f"{label} — {_STATUS_LABELS[status]}",
        message=subtitle,
        link_path=f"{path}?job={link_job_id if link_job_id is not None else job_id}",
    ))
