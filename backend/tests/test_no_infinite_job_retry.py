"""Message empoisonné qui ne tourne pas en boucle (Phase 2,
AUDIT_BACKEND_2026-08-23.md §F2) — l'audit a confirmé par lecture qu'aucun
`enqueue()` ne passe `retry=Retry(...)` nulle part dans le backend (RQ ne
retente donc jamais un job automatiquement ; un payload qui fait toujours
échouer la logique métier se termine "failed" une fois, jamais en boucle).

Garde-fou structurel plutôt qu'un test comportemental (simuler un vrai
retry RQ nécessiterait un worker réel — hors de portée d'un test unitaire
rapide) : scanne le code source des 6 routers à job pour l'absence de
`Retry(` — si un futur lot introduit une politique de retry RQ, ce test
échoue et force une revue explicite plutôt qu'une régression silencieuse
vers un job qui pourrait reboucler indéfiniment sur un payload empoisonné."""

from __future__ import annotations

from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_ROUTER_FILES = [
    "domains/training/router.py",
    "domains/clustering/router.py",
    "domains/dimensionality/router.py",
    "domains/anomalies/router.py",
    "domains/vision/classification/router.py",
    "domains/vision/anomalies/router.py",
]


def test_no_router_configures_rq_retry():
    offenders = []
    for rel_path in _ROUTER_FILES:
        content = (_BACKEND_DIR / rel_path).read_text(encoding="utf-8")
        if "Retry(" in content or "retry=Retry" in content:
            offenders.append(rel_path)
    assert not offenders, (
        f"Retry RQ introduit dans {offenders} — vérifier explicitement qu'un payload "
        "empoisonné ne peut pas reboucler indéfiniment (nombre de tentatives borné, "
        "délai entre tentatives) avant de considérer ce changement sûr."
    )
