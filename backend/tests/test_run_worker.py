"""Tests de `workers/run_worker.py::_resolve_worker_mode` — Lot 1.3
(correctif C7, AUDIT_DATALAB_2026-08-16.md). Logique pure, sans Redis ni
worker réel : seule la décision "fork" vs "simple" est testée ici."""
from __future__ import annotations

from workers.run_worker import _resolve_worker_mode


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
