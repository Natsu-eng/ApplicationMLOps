"""IP cliente résistante à la topologie nginx→backend.

Correctif Phase 1 (AUDIT_BACKEND_2026-08-23.md §A.6) : avant ce correctif,
`request.client.host` valait l'IP du conteneur nginx pour TOUTE requête
passée par le reverse proxy — un seul attaquant pouvait donc verrouiller
`/login` (et tout endpoint limité en débit) pour l'application entière. Ces
tests échouent sans `api.core.rate_limit.get_client_ip`.
"""
from __future__ import annotations

from starlette.requests import Request

from api.core.rate_limit import get_client_ip


def _make_request(peer_host: str, headers: dict[str, str] | None = None) -> Request:
    raw_headers = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    scope = {
        "type": "http",
        "client": (peer_host, 12345),
        "headers": raw_headers,
    }
    return Request(scope)


def test_untrusted_peer_ignores_x_real_ip_header() -> None:
    """Un pair TCP HORS de la plage de confiance ne doit jamais pouvoir
    usurper une IP via X-Real-IP — sinon un client qui contournerait nginx
    en parlant directement au backend pourrait choisir sa propre clé de
    rate-limit."""
    request = _make_request("203.0.113.42", {"X-Real-IP": "1.2.3.4"})
    assert get_client_ip(request) == "203.0.113.42"


def test_trusted_proxy_peer_uses_x_real_ip_header() -> None:
    """Pair TCP dans `trusted_proxy_cidrs` (défaut 172.16.0.0/12, plage
    standard des réseaux bridge Docker où vit nginx) : X-Real-IP fait foi —
    c'est nginx qui l'a posé à `$remote_addr`, jamais dérivé d'un en-tête
    client (voir nginx/templates/default.conf.template)."""
    request = _make_request("172.20.0.5", {"X-Real-IP": "198.51.100.7"})
    assert get_client_ip(request) == "198.51.100.7"


def test_trusted_proxy_peer_without_header_falls_back_to_peer() -> None:
    request = _make_request("172.20.0.5")
    assert get_client_ip(request) == "172.20.0.5"


def test_two_distinct_real_clients_behind_same_proxy_get_distinct_keys() -> None:
    """C'est le coeur du bug corrigé : deux clients distincts derrière le
    MÊME conteneur nginx ne doivent jamais retomber sur la même clé de
    rate-limit (l'IP du proxy)."""
    victim = _make_request("172.20.0.5", {"X-Real-IP": "198.51.100.1"})
    attacker = _make_request("172.20.0.5", {"X-Real-IP": "198.51.100.2"})
    assert get_client_ip(victim) != get_client_ip(attacker)
    assert get_client_ip(victim) != "172.20.0.5"


def test_non_ip_peer_falls_back_to_raw_string() -> None:
    """`starlette.testclient.TestClient` utilise un pair `("testclient",
    50000)` par défaut — pas une IP. Ne doit jamais lever, juste retomber
    sur la chaîne brute (comportement historique de `request.client.host`,
    préservé pour ne pas casser la suite de tests existante)."""
    request = _make_request("testclient")
    assert get_client_ip(request) == "testclient"
