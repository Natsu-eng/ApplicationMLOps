"""Pagination par curseur (Lot 4, correctif I3, AUDIT_DATALAB_2026-08-16.md
§C.2.4/§P) — `GET /training/jobs` et les 5 endpoints équivalents (clustering,
dimensionality, anomalies, vision/classification, vision/anomalies)
renvoyaient TOUJOURS la totalité des jobs de l'organisation, sans limite.

Rétrocompatible par construction, pas par exception : `limit`/`cursor`
absents (défaut) = comportement STRICTEMENT inchangé (tout renvoyer) — aucun
appelant existant (Dashboard, pages Historique) n'est cassé par ce lot. La
capacité de paginer est disponible pour qui l'utilise explicitement ; la
migration des pages Historique vers une UI de pagination réelle est un lot
produit séparé (P6, refonte de `Table.tsx`), pas ce correctif de robustesse
backend.

Curseur = `id` de la dernière ligne de la page précédente, jamais un numéro
de page (stable même si des lignes sont insérées/supprimées entre deux
requêtes, contrairement à `OFFSET`). Suppose que la requête est déjà triée
par `created_at DESC` (ou tout ordre où `id` décroissant = ordre
d'insertion décroissant — vrai pour toutes les tables de jobs, `id` étant
auto-incrémenté) : `WHERE id < cursor` avance alors correctement dans le
même ordre."""
from __future__ import annotations

from typing import Optional, Sequence, TypeVar

from fastapi import Response
from sqlalchemy.orm import Query

_HasId = TypeVar("_HasId")


def paginate_by_id(
    query: "Query[_HasId]",
    id_column,
    response: Response,
    cursor: Optional[int],
    limit: Optional[int],
) -> Sequence[_HasId]:
    """Applique le curseur (`id_column < cursor`) et la limite à `query`
    (déjà triée par `created_at DESC` par l'appelant), et pose l'en-tête
    `X-Next-Cursor` si une page suivante existe — jamais dans le corps JSON,
    pour que la forme de la réponse (`List[XSummary]`) reste identique que
    la pagination soit utilisée ou non.

    `limit=None` : comportement d'avant ce lot, aucune limite, `cursor`
    ignoré (il n'y a qu'une seule page)."""
    if limit is None:
        return query.all()

    if cursor is not None:
        query = query.filter(id_column < cursor)

    # Une ligne de plus que demandé, jamais renvoyée : sert uniquement à
    # savoir s'il existe une page suivante, sans un second aller-retour
    # `COUNT(*)`.
    rows = query.limit(limit + 1).all()
    if len(rows) > limit:
        response.headers["X-Next-Cursor"] = str(rows[limit - 1].id)
        rows = rows[:limit]
    return rows
