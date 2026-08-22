"""Garde-fou de frontières entre domaines (Lot 8 — monolithe modulaire).

Vérifie par analyse statique (AST, aucune dépendance nouvelle) deux choses :

1. Aucun fichier du dépôt n'importe plus les anciens chemins plats
   (`services.X`, `workers.X` pour un module de domaine, `api.routers.X`)
   qui existaient avant le découpage en `domains/<domaine>/` — garde-fou de
   RÉGRESSION permanent, pas seulement une vérification ponctuelle de ce lot.

2. Un fichier de `domains/<domaine>/...` n'importe jamais les internes d'un
   AUTRE domaine — seulement son propre sous-arbre, `domains.shared` (socle
   partagé), `domains.auth.router` (dépendance FastAPI universelle
   `get_current_user`, seul import router→router légitime), et pour les
   trois sous-domaines Vision, `domains.vision.shared`/
   `domains.vision.localization` (socle partagé propre à Vision).

Un domaine qui a besoin des internes d'un autre domaine doit soit passer
par une interface publique explicite, soit voir le module concerné
promu vers `domains/shared/` — jamais un import direct qui recrée le
couplage que ce lot a justement corrigé (voir DECISIONS.md, D8.1)."""
from __future__ import annotations

import ast
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
DOMAINS_ROOT = BACKEND_ROOT / "domains"

# Domaines dont le nom de dossier N'EST PAS le domaine complet — cas des
# trois sous-domaines Vision (`domains/vision/classification/...` est le
# domaine "vision.classification", pas juste "vision").
_VISION_SUBDOMAINS = ("classification", "anomalies", "datasets")


def _iter_domain_python_files():
    for path in DOMAINS_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        yield path


def _domain_of(path: Path) -> str:
    """Nom de domaine à partir du chemin — ex.
    `domains/training/router.py` → "training",
    `domains/vision/classification/router.py` → "vision.classification",
    `domains/vision/shared.py` → "vision" (module vision-partagé, pas un sous-domaine),
    `domains/shared/audit.py` → "shared"."""
    rel_parts = path.relative_to(DOMAINS_ROOT).parts
    top = rel_parts[0]
    if top == "vision" and len(rel_parts) > 1 and rel_parts[1] in _VISION_SUBDOMAINS:
        return f"vision.{rel_parts[1]}"
    return top


def _imported_domain_modules(path: Path) -> list[str]:
    """Modules `domains.*` importés par CE fichier — chaîne complète après
    "domains." (ex. "training.services.engine"), une par import."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("domains."):
            modules.append(node.module[len("domains.") :])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("domains."):
                    modules.append(alias.name[len("domains.") :])
    return modules


def _is_allowed(own_domain: str, imported_module: str) -> bool:
    if imported_module.startswith("shared") or imported_module == "shared":
        return True
    if imported_module.startswith("auth.router") or imported_module == "auth.router":
        return True
    if own_domain.startswith("vision.") and (imported_module in ("vision.shared", "vision.localization")):
        return True
    # `dashboard` est structurellement un agrégateur cross-pilier (Lot 4,
    # correctif I3) — importer le `router.py` public de chaque domaine de
    # job (schémas *Summary + `to_summary()` PUBLIC, jamais un `_`-préfixé
    # depuis le correctif Phase 0) est son rôle documenté, pas une fuite.
    if own_domain == "dashboard" and imported_module.endswith(".router"):
        return True
    # Propre sous-arbre — le module importé commence par le nom du domaine
    # courant (ex. domaine "training" autorise "training.services.engine").
    return imported_module == own_domain or imported_module.startswith(f"{own_domain}.")


def test_no_domain_imports_another_domains_internals():
    violations = []
    for path in _iter_domain_python_files():
        own_domain = _domain_of(path)
        for imported_module in _imported_domain_modules(path):
            if not _is_allowed(own_domain, imported_module):
                violations.append(f"{path.relative_to(BACKEND_ROOT)} (domaine {own_domain!r}) importe domains.{imported_module}")
    assert not violations, "Imports inter-domaines interdits :\n" + "\n".join(sorted(violations))


# Modules de domaine qui vivaient à plat avant le Lot 8 — un import de ces
# chemins aujourd'hui ne peut être qu'un reliquat, jamais volontaire (les
# modules infra `api.core.*` et `workers.run_worker` restent légitimes,
# volontairement absents de cette liste).
_LEGACY_PREFIXES = ("services.", "workers.training_worker", "workers.clustering_worker",
                    "workers.dimensionality_worker", "workers.anomaly_worker",
                    "workers.vision_classification_worker", "workers.vision_anomaly_worker",
                    "api.routers.")


def test_no_file_references_pre_lot8_flat_paths():
    violations = []
    for path in BACKEND_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts or ".venv" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.ImportFrom) and node.module:
                module = node.module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(_LEGACY_PREFIXES):
                        violations.append(f"{path.relative_to(BACKEND_ROOT)}: import {alias.name}")
                continue
            if module and module.startswith(_LEGACY_PREFIXES):
                violations.append(f"{path.relative_to(BACKEND_ROOT)}: from {module} import ...")
    assert not violations, "Références à d'anciens chemins plats (pré-Lot 8) :\n" + "\n".join(sorted(violations))
