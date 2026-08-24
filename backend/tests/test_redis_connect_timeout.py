"""Bug réel signalé par l'opérateur (poste de dev sans Redis démarré) :
`Redis.from_url()` sans timeout explicite (api/core/job_queue.py) laissait
la connexion TCP initiale bloquer indéfiniment quand Redis est injoignable
— borné uniquement par le comportement par défaut de la pile TCP du
système (souvent 20 s ou plus sous Windows). Concrètement observé sur
`/login` : `is_rate_limited()` est bien "échec ouvert" (Phase 1, §4), mais
n'échoue "ouvert" qu'APRÈS l'expiration de cette tentative de connexion —
l'utilisateur voyait "Connexion…" bloqué de longues secondes.

Ce test prouve que la connexion à un hôte injoignable échoue maintenant
RAPIDEMENT (quelques secondes, borné par `socket_connect_timeout=3`),
jamais qu'elle réussit — `10.255.255.1` (bloc de test RFC 5737-like, non
routé) ne répond jamais, ce qui exerce réellement le délai de connexion
(contrairement à un port fermé sur localhost, où le refus est quasi
instantané quel que soit le timeout configuré — ne prouverait rien)."""

from __future__ import annotations

import time

import pytest
from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

# Adresse non routée (bloc réservé à la documentation/tests, RFC 5737-style)
# — ne répond jamais, contrairement à un port fermé qui refuserait la
# connexion quasi instantanément et ne testerait donc pas le timeout.
_UNREACHABLE_HOST = "10.255.255.1"
_UNREACHABLE_PORT = 6379

# Marge large (le timeout configuré est 3s) : borne le test lui-même sans
# le rendre fragile sur une machine lente, tout en prouvant sans ambiguïté
# que la connexion n'attend pas 20s+ comme avant ce correctif.
_MAX_ACCEPTABLE_SECONDS = 8


def test_connecting_to_an_unreachable_redis_fails_fast_not_after_20_plus_seconds():
    conn = Redis(host=_UNREACHABLE_HOST, port=_UNREACHABLE_PORT, socket_connect_timeout=3)
    start = time.monotonic()
    with pytest.raises((RedisConnectionError, RedisTimeoutError, OSError)):
        conn.incr("sonde-de-test")
    elapsed = time.monotonic() - start
    assert elapsed < _MAX_ACCEPTABLE_SECONDS, (
        f"la connexion a pris {elapsed:.1f}s -- socket_connect_timeout n'est manifestement pas appliqué"
    )


def test_job_queue_redis_client_has_the_connect_timeout_configured():
    """Vérifie directement la configuration du client partagé de l'app
    (`api/core/job_queue.py::redis_conn`), pas seulement le principe
    général ci-dessus — un futur changement qui retirerait
    `socket_connect_timeout` par erreur doit casser CE test précisément."""
    from api.core.job_queue import redis_conn

    assert redis_conn.connection_pool.connection_kwargs.get("socket_connect_timeout") == 3
