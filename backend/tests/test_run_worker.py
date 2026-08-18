"""Tests de `workers/run_worker.py` — Lot 1.3 (correctif C7,
`_resolve_worker_mode`) et Lot 4 (correctif I6, `_resolve_queues`).
Logique pure, sans Redis ni worker réel : `job_queue.py` crée bien un
`Redis.from_url(...)` au chargement du module, mais c'est un client
paresseux qui ne se connecte qu'au premier usage — l'importer suffit ici,
jamais besoin d'un Redis réel."""
from __future__ import annotations

import pytest

from workers.run_worker import _resolve_queues, _resolve_worker_mode


def test_explicit_fork_override_wins_regardless_of_platform(monkeypatch):
    monkeypatch.setenv("RQ_WORKER_MODE", "fork")
    monkeypatch.setattr("workers.run_worker.sys.platform", "win32")
    assert _resolve_worker_mode() == "fork"


def test_explicit_simple_override_wins_regardless_of_platform(monkeypatch):
    monkeypatch.setenv("RQ_WORKER_MODE", "simple")
    monkeypatch.setattr("workers.run_worker.sys.platform", "linux")
    assert _resolve_worker_mode() == "simple"


def test_defaults_to_simple_on_windows_without_override(monkeypatch):
    monkeypatch.delenv("RQ_WORKER_MODE", raising=False)
    monkeypatch.setattr("workers.run_worker.sys.platform", "win32")
    assert _resolve_worker_mode() == "simple"


def test_defaults_to_fork_on_linux_without_override(monkeypatch):
    monkeypatch.delenv("RQ_WORKER_MODE", raising=False)
    monkeypatch.setattr("workers.run_worker.sys.platform", "linux")
    assert _resolve_worker_mode() == "fork"


def test_unknown_override_value_falls_back_to_platform_detection(monkeypatch):
    """Une valeur invalide ne doit jamais planter le worker au démarrage —
    repli sur la détection plateforme, comme si la variable était absente."""
    monkeypatch.setenv("RQ_WORKER_MODE", "n_importe_quoi")
    monkeypatch.setattr("workers.run_worker.sys.platform", "linux")
    assert _resolve_worker_mode() == "fork"


# ── _resolve_queues (Lot 4, correctif I6) ───────────────────────────────────

def test_no_rq_queues_env_listens_to_all_three(monkeypatch):
    monkeypatch.delenv("RQ_QUEUES", raising=False)
    names = {q.name for q in _resolve_queues()}
    assert names == {"training", "vision", "analysis"}


def test_rq_queues_env_selects_a_subset(monkeypatch):
    monkeypatch.setenv("RQ_QUEUES", "training,vision")
    names = [q.name for q in _resolve_queues()]
    assert names == ["training", "vision"]


def test_rq_queues_env_is_case_and_whitespace_insensitive(monkeypatch):
    monkeypatch.setenv("RQ_QUEUES", " Analysis ,TRAINING")
    names = [q.name for q in _resolve_queues()]
    assert names == ["analysis", "training"]


def test_unknown_queue_name_raises_rather_than_silently_listening_to_nothing(monkeypatch):
    """Une file inconnue ne doit jamais démarrer un worker qui écoute
    silencieusement zéro file (jobs jamais traités, sans erreur visible) —
    échec explicite au démarrage à la place."""
    monkeypatch.setenv("RQ_QUEUES", "training,bogus")
    with pytest.raises(ValueError):
        _resolve_queues()
