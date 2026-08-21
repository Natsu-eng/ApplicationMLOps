"""Moteur de réduction de dimension — Lot 13 (ML non supervisé).

Module séparé de `clustering_training.py`/`ml_training.py` (voir
`dimensionality_registry.py` pour le raisonnement complet) — PCA/t-SNE/UMAP
n'ont pas de score de sélection commun entre familles d'algorithmes
(contrairement au clustering où la silhouette compare KMeans/DBSCAN/
hiérarchique) : pas de leaderboard ici, l'utilisateur choisit une méthode,
PCA est TOUJOURS calculée en plus (rapide, variance expliquée/loadings
réels) pour donner un repère de qualité objectif même quand t-SNE/UMAP est
la projection principale affichée."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness

from services.dimensionality_registry import DISTANCE_FIDELITY_NOTES, spec_for
from services.ml_preprocessing import TrainingAbortedError, build_preprocessor
from services.stats_utils import sample_if_large

ProgressCallback = Callable[[str, int], None]

# t-SNE est quasi-quadratique et le rendu d'un nuage de points SVG devient
# poussif au-delà de quelques milliers de points — dégradation transparente
# (échantillon signalé, jamais une erreur bloquante), pas une limite dure.
MAX_ROWS_FOR_EMBEDDING = 5000
MIN_ROWS_FOR_EMBEDDING = 5
_ROW_INDEX_COLUMN = "__dl_row_index__"


@dataclass
class DimensionalityConfig:
    algorithm_id: str = "pca"
    seed: int = 42


@dataclass
class DimensionalityPointResult:
    row_index: int  # position dans le dataset D'ORIGINE, préservée à travers l'échantillonnage
    x: float
    y: float


@dataclass
class LoadingResult:
    feature: str
    pc1: float
    pc2: float


@dataclass
class DimensionalityResult:
    algorithm_id: str
    algorithm_label: str
    n_samples_total: int
    n_samples_used: int
    sampled: bool
    points: list[DimensionalityPointResult]
    trustworthiness_primary: float
    trustworthiness_pca: float
    variance_explained: list[float]
    total_variance_explained: float
    loadings: list[LoadingResult]
    distance_fidelity_note: str
    feature_columns: list[str]
    model_card: dict[str, Any]
    pipeline_bundle: dict[str, Any] = field(repr=False)


def train_and_evaluate_dimensionality(
    X: pd.DataFrame,
    config: DimensionalityConfig,
    progress_cb: ProgressCallback,
) -> DimensionalityResult:
    """Point d'entrée principal. Calcule TOUJOURS une PCA (variance
    expliquée, loadings, mesure de fidélité de référence) en plus de la
    méthode choisie — jamais de texte inventé sur la qualité d'une
    projection, uniquement des statistiques réellement calculées (voir
    skill senior-ai-saas-engineer, data-science.md)."""
    progress_cb("Préparation des données", 5)
    n_samples_total = int(len(X))
    if n_samples_total < MIN_ROWS_FOR_EMBEDDING:
        raise TrainingAbortedError(
            f"Il faut au moins {MIN_ROWS_FOR_EMBEDDING} observations pour calculer une projection 2D "
            f"exploitable (dataset actuel : {n_samples_total})."
        )

    # Échantillonnage préservant l'index d'origine — une colonne technique
    # portant l'index avant tri/échantillonnage, retirée juste avant le
    # préprocesseur (jamais traitée comme une variable du clustering/de la
    # projection).
    X_indexed = X.reset_index(drop=True).copy()
    X_indexed[_ROW_INDEX_COLUMN] = np.arange(n_samples_total)
    sampled_flag = n_samples_total > MAX_ROWS_FOR_EMBEDDING
    X_sampled = sample_if_large(X_indexed, MAX_ROWS_FOR_EMBEDDING, config.seed)
    row_indices = X_sampled[_ROW_INDEX_COLUMN].to_numpy()
    X_used = X_sampled.drop(columns=[_ROW_INDEX_COLUMN])
    n_samples_used = int(len(X_used))

    progress_cb("Préparation des données", 15)
    preprocessor = build_preprocessor(X_used)
    X_processed = preprocessor.fit_transform(X_used)
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    if X_processed.shape[1] < 2 or n_samples_used < 3:
        raise TrainingAbortedError(
            "Impossible de calculer une projection 2D : il faut au moins 2 variables exploitables "
            "après préparation et 3 observations."
        )

    spec = spec_for(config.algorithm_id)

    progress_cb("Analyse de la variance expliquée (PCA)", 25)
    n_pca_components = min(2, X_processed.shape[1], n_samples_used - 1)
    pca = PCA(n_components=n_pca_components, random_state=config.seed)
    embedding_pca = pca.fit_transform(X_processed)
    if embedding_pca.shape[1] < 2:
        # Une seule composante exploitable (ex. une seule variable après
        # préprocessing) — deuxième axe à zéro plutôt qu'un plantage, signalé
        # honnêtement via une variance expliquée de 0 sur PC2.
        embedding_pca = np.pad(embedding_pca, ((0, 0), (0, 2 - embedding_pca.shape[1])))
    variance_ratio = list(pca.explained_variance_ratio_[:2])
    while len(variance_ratio) < 2:
        variance_ratio.append(0.0)

    feature_names = list(preprocessor.get_feature_names_out())

    # Transparence catégorielle (Lot 6B, §F.2) — un one-hot sur une colonne à
    # forte cardinalité peut dominer les distances calculées par la
    # projection sans que l'utilisateur en soit informé. `build_preprocessor`
    # nomme toujours ses transformateurs "num"/"cat" (appelé ici sans
    # `feature_engineering_config`, seule la branche simple s'applique — voir
    # `services/ml_preprocessing.py::build_preprocessor`), donc chaque colonne
    # encodée en one-hot produit un ou plusieurs noms préfixés "cat__".
    categorical_columns = [c for c in X_used.columns if not pd.api.types.is_numeric_dtype(X_used[c])]
    numeric_columns = [c for c in X_used.columns if c not in categorical_columns]
    n_categorical_dimensions = sum(1 for f in feature_names if f.startswith("cat__"))

    components = pca.components_
    loadings: list[LoadingResult] = []
    for i, feature in enumerate(feature_names):
        pc1 = float(components[0][i]) if components.shape[0] > 0 else 0.0
        pc2 = float(components[1][i]) if components.shape[0] > 1 else 0.0
        loadings.append(LoadingResult(feature=feature, pc1=pc1, pc2=pc2))
    loadings.sort(key=lambda entry: (entry.pc1**2 + entry.pc2**2) ** 0.5, reverse=True)
    loadings = loadings[:15]

    progress_cb(f"Calcul de la projection ({spec.label})", 45)
    if spec.id == "pca":
        embedding_primary = embedding_pca
    else:
        estimator = spec.build_estimator(n_samples_used, config.seed)
        embedding_primary = estimator.fit_transform(X_processed)

    progress_cb("Évaluation de la fidélité de la projection", 80)
    n_neighbors_trust = min(5, n_samples_used - 1)
    trust_pca = float(trustworthiness(X_processed, embedding_pca, n_neighbors=n_neighbors_trust))
    trust_primary = trust_pca if spec.id == "pca" else float(
        trustworthiness(X_processed, embedding_primary, n_neighbors=n_neighbors_trust)
    )

    points = [
        DimensionalityPointResult(row_index=int(row_indices[i]), x=float(embedding_primary[i, 0]), y=float(embedding_primary[i, 1]))
        for i in range(n_samples_used)
    ]

    progress_cb("Terminé", 100)

    total_variance_explained = float(sum(variance_ratio))
    model_card = {
        "algorithm": spec.label,
        "algorithm_id": spec.id,
        "n_samples_total": n_samples_total,
        "n_samples_used": n_samples_used,
        "sampled": sampled_flag,
        "total_variance_explained": total_variance_explained,
        "trustworthiness_primary": trust_primary,
        "trustworthiness_pca": trust_pca,
        "seed": config.seed,
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "n_dimensions_after_encoding": len(feature_names),
        "n_categorical_dimensions": n_categorical_dimensions,
    }

    return DimensionalityResult(
        algorithm_id=spec.id,
        algorithm_label=spec.label,
        n_samples_total=n_samples_total,
        n_samples_used=n_samples_used,
        sampled=sampled_flag,
        points=points,
        trustworthiness_primary=trust_primary,
        trustworthiness_pca=trust_pca,
        variance_explained=variance_ratio,
        total_variance_explained=total_variance_explained,
        loadings=loadings,
        distance_fidelity_note=DISTANCE_FIDELITY_NOTES[spec.id],
        feature_columns=list(X.columns),
        model_card=model_card,
        pipeline_bundle={"preprocessor": preprocessor, "pca_model": pca, "algorithm_id": spec.id},
    )
