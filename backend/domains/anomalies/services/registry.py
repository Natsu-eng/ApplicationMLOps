"""Registre des algorithmes de détection d'anomalies — Lot 14 (ML non
supervisé).

Même esprit que `clustering_registry.py`/`dimensionality_registry.py` :
chaque entrée déclare tout ce qui distingue un algorithme du reste, ajouter
un algorithme = ajouter une entrée, jamais toucher le moteur
(`anomaly_training.py`).

Contrairement au clustering, il n'y a PAS de "meilleur candidat" à élire
(aucune vérité terrain disponible pour comparer objectivement Isolation
Forest et LOF) : les deux tournent TOUJOURS ensemble — jamais un seul essai
lancé à l'aveugle, principe déjà établi par le reste du produit — et le
moteur calcule un score de consensus plutôt qu'un classement gagnant/perdant.
Pas de sélection utilisateur d'algorithme non plus (voir
`api/routers/anomalies.py` : pas de `/algorithms-catalog`, rien à choisir)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Union

from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

Family = Literal["ensemble", "densite"]

# "auto" (défaut, dérivé de la formule du papier original de chaque
# algorithme) ou une fraction explicite (0, 0.5] — jamais 0 (aucune anomalie
# n'aurait de sens) ni au-delà de 0.5 (sklearn lève une ValueError, la
# "majorité" ne peut pas être l'anomalie par définition).
Contamination = Union[Literal["auto"], float]


@dataclass(frozen=True)
class AnomalySpec:
    id: str
    label: str
    family: Family
    build_estimator: Callable[[int, int, Contamination], BaseEstimator]  # (n_samples, seed, contamination) -> estimateur


def _build_isolation_forest(n_samples: int, seed: int, contamination: Contamination = "auto") -> BaseEstimator:
    # contamination="auto" par défaut (seuil dérivé de la formule du papier
    # original, pas une fraction arbitraire fixée à l'aveugle) — cohérent
    # avec le mode guidé du reste du produit. Réglable explicitement en mode
    # expert (Lot 6B, §F.2 — jusqu'ici codé en dur, jamais exposé).
    return IsolationForest(contamination=contamination, random_state=seed, n_estimators=200)


def _build_lof(n_samples: int, seed: int, contamination: Contamination = "auto") -> BaseEstimator:
    n_neighbors = max(2, min(20, n_samples - 1))
    # novelty=False : mode détection sur le jeu d'entraînement lui-même
    # (fit_predict) — reste le comportement d'ENTRAÎNEMENT, inchangé. La
    # notation d'une nouvelle observation (voir `services/inference.py`,
    # Lot 6B, §F.2) passe par une SECONDE instance dédiée (`build_lof_novelty_from`
    # ci-dessous) : sklearn interdit d'appeler `.predict()` sur de nouvelles
    # données avec une instance `novelty=False` (`AttributeError` explicite),
    # et changer ce réglage sur l'instance d'entraînement changerait
    # silencieusement la sémantique déjà testée de `fit_predict`.
    return LocalOutlierFactor(contamination=contamination, novelty=False, n_neighbors=n_neighbors)


def build_lof_novelty_from(fitted_lof: LocalOutlierFactor, X_processed: Any) -> LocalOutlierFactor:
    """Seconde instance LOF, `novelty=True`, MÊMES hyperparamètres que
    `fitted_lof` (déjà entraînée en `novelty=False` pour la détection sur le
    jeu d'entraînement) — réentraînée sur les MÊMES données transformées,
    uniquement pour noter de nouvelles observations (voir docstring de
    `_build_lof`). Ne remplace ni ne modifie `fitted_lof`."""
    params = fitted_lof.get_params()
    params["novelty"] = True
    novelty_estimator = LocalOutlierFactor(**params)
    novelty_estimator.fit(X_processed)
    return novelty_estimator


ANOMALY_REGISTRY: list[AnomalySpec] = [
    AnomalySpec(id="isolation_forest", label="Isolation Forest", family="ensemble", build_estimator=_build_isolation_forest),
    AnomalySpec(id="lof", label="Local Outlier Factor (LOF)", family="densite", build_estimator=_build_lof),
]
