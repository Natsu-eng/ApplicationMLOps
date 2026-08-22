"""Chargement générique d'un artefact modèle sérialisé (joblib) — extrait de
`services/ml_inference.py` (Lot 8, correctif de frontières §Phase 0) : ce
chargement n'a rien de spécifique au supervisé, `clustering.py` en avait
besoin pour charger ses propres bundles KMeans/DBSCAN et devait jusqu'ici
importer un module training pour deux symboles génériques.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


class InferenceError(ValueError):
    """La donnée fournie ne peut pas être utilisée pour prédire (colonne
    manquante, valeur non convertible, artefact introuvable...)."""


def load_bundle(file_path: str) -> dict[str, Any]:
    # Lot 1.4 (§C.2.7/R11, AUDIT_DATALAB_2026-08-16.md) — même risque que
    # torch.load côté vision (api/routers/vision_classification.py) : ce
    # fichier n'est aujourd'hui écrit que par nos propres workers (training,
    # clustering...), jamais par un import utilisateur, mais `joblib.load`
    # reste un pickle non restreint (exécution de code arbitraire si le
    # fichier venait d'ailleurs). Contrairement à `torch.load`, joblib/pickle
    # n'a PAS d'équivalent `weights_only=True` — aucune restructuration ne
    # referme ce risque ici, seule une frontière de confiance (jamais de
    # fichier fourni par un utilisateur passé à cette fonction) le tient
    # aujourd'hui. Si l'import de modèle externe devient une fonctionnalité
    # réelle (hors périmètre actuel), ce point devra être retraité en
    # profondeur (format non-pickle, ou sandboxing).
    if not Path(file_path).exists():
        raise InferenceError("Artefact du modèle introuvable sur le serveur")
    return joblib.load(file_path)
