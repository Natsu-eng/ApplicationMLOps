"""Registre des méthodes de réduction de dimension — Lot 13 (ML non
supervisé).

Même esprit que `services/clustering_registry.py` (Lot 11) : chaque entrée
déclare tout ce qui distingue une méthode du reste, ajouter une méthode =
ajouter une entrée, jamais toucher le moteur (`dimensionality_training.py`).

Module VOLONTAIREMENT séparé de `clustering_registry.py`/`ml_registry.py` —
PCA/t-SNE/UMAP ne partagent ni la même API sklearn (`fit_transform` sans
notion de "cluster" ni de score de sélection commun) ni les mêmes
hyperparamètres qu'un algorithme de clustering ou un modèle supervisé.

UMAP optionnelle et chargée en paresseux (`_umap_module`) : le catalogue
tombe silencieusement à PCA/t-SNE seules si `umap-learn` n'est pas
installable dans l'environnement, jamais un crash au démarrage de l'API
(voir requirements.txt pour le choix de version `umap-learn==0.5.6`, qui
évite de forcer une mise à niveau majeure de scikit-learn)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

Family = Literal["lineaire", "manifold"]

try:
    from umap import UMAP  # type: ignore[import-untyped]

    _UMAP_AVAILABLE = True
except ImportError:  # pragma: no cover — environnement sans umap-learn installé
    _UMAP_AVAILABLE = False


@dataclass(frozen=True)
class DimensionalitySpec:
    id: str
    label: str
    family: Family
    build_estimator: Callable[[int, int], BaseEstimator]  # (n_samples, seed) -> estimateur n_components=2


# Note de fidélité des distances — exigence du skill senior-ai-saas-engineer
# (data-science.md, section "Réduction de dimension") : t-SNE/UMAP préservent
# les voisinages locaux mais PAS les distances globales, jamais présentées
# comme une carte fidèle sans cette mise en garde explicite dans l'UI.
DISTANCE_FIDELITY_NOTES: dict[str, str] = {
    "pca": "Projection linéaire : les distances et échelles globales entre points restent fidèles à l'espace d'origine (aux 2 dimensions retenues près).",
    "tsne": "t-SNE préserve les voisinages locaux mais PAS les distances globales : deux groupes proches sur ce graphique ne sont pas forcément proches dans les données réelles, et la taille apparente d'un groupe n'est pas interprétable.",
    "umap": "UMAP préserve la structure locale mais les distances entre groupes éloignés et leur taille relative peuvent être trompeuses — comparez uniquement les voisinages proches.",
}


def _build_pca(n_samples: int, seed: int) -> BaseEstimator:
    n_components = max(1, min(2, n_samples - 1))
    return PCA(n_components=n_components, random_state=seed)


def _build_tsne(n_samples: int, seed: int) -> BaseEstimator:
    # sklearn exige perplexity < n_samples strictement — bornée entre 2 et 30
    # (défaut usuel), puis reclippée par sécurité sous n_samples.
    perplexity = max(2, min(30, (n_samples - 1) // 3))
    perplexity = min(perplexity, n_samples - 1)
    return TSNE(n_components=2, perplexity=perplexity, random_state=seed, init="pca", learning_rate="auto")


def _build_umap(n_samples: int, seed: int) -> BaseEstimator:
    n_neighbors = max(2, min(15, n_samples - 1))
    return UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=seed)


DIMENSIONALITY_REGISTRY: list[DimensionalitySpec] = [
    DimensionalitySpec(id="pca", label="PCA (analyse en composantes principales)", family="lineaire", build_estimator=_build_pca),
    DimensionalitySpec(id="tsne", label="t-SNE", family="manifold", build_estimator=_build_tsne),
]
if _UMAP_AVAILABLE:
    DIMENSIONALITY_REGISTRY.append(
        DimensionalitySpec(id="umap", label="UMAP", family="manifold", build_estimator=_build_umap)
    )

DEFAULT_ALGORITHM_ID = "pca"


def spec_for(algorithm_id: str) -> DimensionalitySpec:
    for spec in DIMENSIONALITY_REGISTRY:
        if spec.id == algorithm_id:
            return spec
    raise ValueError(f"Méthode de réduction de dimension inconnue : {algorithm_id}")
