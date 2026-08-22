"""Tests de services/job_events.py (Lot 7, §J.2) — flux SSE de notifications
de fin de job.

`asyncio.run()` en synchrone plutôt que `@pytest.mark.anyio`/`asyncio` :
aucune infrastructure de test async n'existe encore dans ce dépôt (aucun
`async def test_` avant ce fichier) — évite d'introduire une dépendance à
une configuration pytest non vérifiée pour 3 tests aussi simples."""
from __future__ import annotations

import asyncio
import json

from domains.shared.job_events import stream_job_updates


async def _collect(fetch_snapshot):
    events = []
    async for chunk in stream_job_updates(fetch_snapshot):
        events.append(chunk)
    return events


def test_stream_closes_immediately_on_already_terminal_job():
    events = asyncio.run(
        _collect(lambda: {"status": "completed", "progress_percent": 100, "progress_step": "Terminé", "error_message": None})
    )
    assert len(events) == 1
    assert events[0].startswith("data: ")
    payload = json.loads(events[0][len("data: "):].strip())
    assert payload["status"] == "completed"


def test_stream_emits_error_event_and_stops_when_job_missing():
    events = asyncio.run(_collect(lambda: None))
    assert len(events) == 1
    assert events[0].startswith("event: error")
    assert "JOB_INTROUVABLE" in events[0]


def test_stream_deduplicates_unchanged_snapshots(monkeypatch):
    """Deux ticks identiques de suite ne doivent produire qu'UN seul
    événement — jamais un doublon pour rien."""
    import domains.shared.job_events as job_events_module

    monkeypatch.setattr(job_events_module, "POLL_INTERVAL_SECONDS", 0.01)

    calls = {"n": 0}

    def fetch():
        calls["n"] += 1
        if calls["n"] < 3:
            return {"status": "running", "progress_percent": 10, "progress_step": "En cours", "error_message": None}
        return {"status": "completed", "progress_percent": 100, "progress_step": "Terminé", "error_message": None}

    events = asyncio.run(_collect(fetch))

    # 1 événement "running" (les 2 premiers ticks identiques ne produisent
    # qu'un seul envoi) + 1 événement "completed" qui ferme le flux.
    assert len(events) == 2
    assert json.loads(events[0][len("data: "):].strip())["status"] == "running"
    assert json.loads(events[1][len("data: "):].strip())["status"] == "completed"
