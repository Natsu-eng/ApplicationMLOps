"""Notifications de fin de job par Server-Sent Events (SSE) — Lot 7, §J.2.

Remplace le polling client dupliqué (`setInterval(3000)`, quasi identique
sur 6 pages) par un flux serveur qui pousse une mise à jour SEULEMENT quand
elle change, et se ferme de lui-même à un statut terminal — jamais un
`setInterval` qui continue de tourner après que la page a déjà tout reçu.

Aucune nouvelle dépendance : `StreamingResponse` (FastAPI) + un générateur
`text/event-stream` écrit à la main suffisent, pas besoin de `sse-starlette`
pour un flux aussi simple (une ligne `data: {...}` par mise à jour).

Chaque tick ouvre une session DB de COURTE durée (`SessionLocal()`/
`db.close()`, même pattern que les workers RQ) plutôt que de garder la
session de la requête FastAPI ouverte pendant toute la durée du flux
(potentiellement plusieurs minutes pour un entraînement) — évite d'épuiser
le pool de connexions avec des flux SSE ouverts longtemps. La requête SQL
elle-même tourne dans un thread (`asyncio.to_thread`) pour ne jamais
bloquer la boucle asyncio le temps de l'aller-retour DB."""
from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncGenerator, Callable, Optional

POLL_INTERVAL_SECONDS = 1.5
# Filet de sécurité — ferme le flux après 1h même sans statut terminal
# (ex. un job resté "running" après un crash worker, voir job_watchdog.py :
# le watchdog le marquera "failed" en base tôt ou tard, mais ce flux-ci ne
# doit pas rester ouvert indéfiniment en attendant).
MAX_STREAM_SECONDS = 3600.0

TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


async def stream_job_updates(fetch_snapshot: Callable[[], Optional[dict[str, Any]]]) -> AsyncGenerator[str, None]:
    """`fetch_snapshot` : callback SYNCHRONE (ouvre sa propre session DB,
    la ferme avant de retourner) qui renvoie un petit dict JSON-sérialisable
    (`status`/`progress_percent`/`progress_step`/`error_message`...) ou
    `None` si le job a disparu (supprimé pendant que le flux était ouvert)."""
    last_payload: Optional[str] = None
    elapsed = 0.0
    while elapsed < MAX_STREAM_SECONDS:
        snapshot = await asyncio.to_thread(fetch_snapshot)
        if snapshot is None:
            yield f"event: error\ndata: {json.dumps({'code': 'JOB_INTROUVABLE'})}\n\n"
            return
        payload = json.dumps(snapshot)
        if payload != last_payload:
            yield f"data: {payload}\n\n"
            last_payload = payload
        if snapshot.get("status") in TERMINAL_STATUSES:
            return
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
        elapsed += POLL_INTERVAL_SECONDS
    yield f"event: timeout\ndata: {json.dumps({'code': 'FLUX_EXPIRE'})}\n\n"
