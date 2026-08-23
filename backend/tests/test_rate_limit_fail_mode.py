"""Mode d'échec du rate-limiting quand Redis est indisponible (Phase 1,
AUDIT_BACKEND_2026-08-23.md §4) — ouvert pour l'authentification, fermé
pour les endpoints coûteux (upload, /explain)."""
from __future__ import annotations

import pytest

from api.core.rate_limit import RateLimitBackendUnavailable, is_rate_limited


class _BrokenRedis:
    """Simule un Redis injoignable — `incr` lève systématiquement, comme le
    ferait `redis.exceptions.ConnectionError` en pratique."""

    def incr(self, key: str) -> int:
        raise ConnectionError("Redis indisponible (simulation)")


def test_fail_open_never_blocks_when_redis_is_down() -> None:
    assert is_rate_limited(_BrokenRedis(), "k", 10, 60, fail_open=True) is False


def test_fail_closed_refuses_when_redis_is_down() -> None:
    with pytest.raises(RateLimitBackendUnavailable):
        is_rate_limited(_BrokenRedis(), "k", 10, 60, fail_open=False)
