"""Tests de services/clustering_training.py (Lot 11 — ML non supervisé)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from services.clustering_registry import CLUSTER_REGISTRY
from services.clustering_training import ClusteringConfig, train_and_evaluate_clustering
from services.ml_preprocessing import TrainingAbortedError

_NOOP = lambda step, pct: None  # noqa: E731


def _make_three_blobs_df(n_per_group: int = 50, seed: int = 42) -> pd.DataFrame:
    """3 groupes numériques bien séparés + une colonne catégorielle
    parfaitement corrélée au groupe — jeu de données jouet où le résultat
    attendu est connu à l'avance, pour vérifier le moteur, pas juste qu'il
    ne plante pas."""
    rng = np.random.default_rng(seed)
    group = np.repeat([0, 1, 2], n_per_group)
    centers = {0: (0, 0), 1: (12, 12), 2: (0, 12)}
    x1 = np.array([centers[g][0] for g in group]) + rng.normal(0, 0.7, len(group))
    x2 = np.array([centers[g][1] for g in group]) + rng.normal(0, 0.7, len(group))
    cat = np.where(group == 0, "A", np.where(group == 1, "B", "C"))
    return pd.DataFrame({"x1": x1, "x2": x2, "categorie": cat})


def test_finds_the_right_number_of_clusters_on_well_separated_data():
    df = _make_three_blobs_df()
    result = train_and_evaluate_clustering(df, ClusteringConfig(seed=42), _NOOP)
    assert result.model_card["n_clusters"] == 3
    assert result.model_card["silhouette"] > 0.7  # groupes très séparés : silhouette proche de 1 attendue


def test_all_candidates_ranked_by_silhouette_descending():
    df = _make_three_blobs_df()
    result = train_and_evaluate_clustering(df, ClusteringConfig(seed=42), _NOOP)
    scores = [c.silhouette for c in result.all_candidates if c.silhouette is not None]
    assert scores == sorted(scores, reverse=True)
    assert result.all_candidates[0].is_winner is True
    assert all(not c.is_winner for c in result.all_candidates[1:])


def test_cluster_profiles_correctly_identify_dominant_category():
    """Preuve la plus directe que les profils sont corrects, pas juste
    calculés : chaque cluster découvert doit être dominé par LA bonne
    catégorie d'origine (données construites pour que ce soit vérifiable)."""
    df = _make_three_blobs_df()
    result = train_and_evaluate_clustering(df, ClusteringConfig(seed=42), _NOOP)

    assert len(result.cluster_profiles) == 3
    dominant_categories = set()
    for profile in result.cluster_profiles:
        cat_summary = profile.categorical_summary.get("categorie")
        assert cat_summary is not None
        assert cat_summary["top_pct"] == 100.0  # groupes parfaitement purs par construction
        dominant_categories.add(cat_summary["top_category"])
    assert dominant_categories == {"A", "B", "C"}


def test_cluster_profiles_sizes_sum_to_total_minus_noise():
    df = _make_three_blobs_df()
    result = train_and_evaluate_clustering(df, ClusteringConfig(seed=42), _NOOP)
    total_in_profiles = sum(p.size for p in result.cluster_profiles)
    assert total_in_profiles + result.noise_count == len(df)


def test_differentiating_variables_identify_the_real_signal():
    """x1/x2 portent tout le signal de groupe par construction — doivent
    apparaître comme variables différenciantes, pas une colonne de bruit
    pur si on en ajoutait une (ici il n'y en a pas, donc x1/x2 doivent être
    en tête pour CHAQUE cluster)."""
    df = _make_three_blobs_df()
    result = train_and_evaluate_clustering(df, ClusteringConfig(seed=42), _NOOP)
    for profile in result.cluster_profiles:
        assert set(profile.differentiating_variables[:2]) == {"x1", "x2"}


def test_algorithm_ids_restricts_to_explicit_selection():
    df = _make_three_blobs_df()
    result = train_and_evaluate_clustering(df, ClusteringConfig(algorithm_ids=["kmeans"], seed=42), _NOOP)
    assert all(c.algorithm_id == "kmeans" for c in result.all_candidates)


def test_seed_is_actually_wired_to_the_fitted_estimator():
    """Preuve directe (pas une déduction statistique fragile) que
    `ClusteringConfig.seed` atteint bien l'estimateur construit — inspecte
    l'attribut `random_state` de l'objet KMeans réellement utilisé, plutôt
    que de déduire la propagation d'une éventuelle différence de résultat
    (K-Means avec `n_init=10` peut converger vers la même partition quel que
    soit le seed sur des données aussi séparées, ce qui ne prouverait rien)."""
    df = _make_three_blobs_df()
    result = train_and_evaluate_clustering(df, ClusteringConfig(algorithm_ids=["kmeans"], seed=777), _NOOP)
    assert result.pipeline_bundle["model"].random_state == 777


def test_progress_callback_called_with_increasing_percentages():
    df = _make_three_blobs_df(n_per_group=20)
    calls: list[int] = []
    train_and_evaluate_clustering(df, ClusteringConfig(seed=42), lambda step, pct: calls.append(pct))
    assert calls[0] < calls[-1]
    assert calls[-1] == 100
    assert all(0 <= p <= 100 for p in calls)


def test_too_few_rows_raises_actionable_error_not_a_crash():
    """En dessous du k minimum candidat (2), aucune configuration n'est
    même générée — doit lever une erreur diagnosticable
    (`TrainingAbortedError`, surfacée telle quelle depuis le correctif H7),
    jamais une exception technique brute de sklearn ou un plantage
    silencieux."""
    df = pd.DataFrame({"x1": [5.0], "x2": [3.0]})
    with pytest.raises(TrainingAbortedError):
        train_and_evaluate_clustering(df, ClusteringConfig(seed=42), _NOOP)


def test_all_identical_rows_still_degrades_gracefully():
    """Toutes les lignes strictement identiques : aucune vraie structure de
    groupes, mais certains algorithmes (le clustering hiérarchique en
    particulier, qui peut forcer une partition arbitraire même à distance
    nulle) peuvent techniquement renvoyer un résultat — jamais un plantage
    dans tous les cas, `eps` dégénéré (0.0) pour DBSCAN écarté proprement en
    amont plutôt que de lever une erreur sklearn brute."""
    df = pd.DataFrame({"x1": [5.0] * 30, "x2": [3.0] * 30})
    # Ne doit jamais lever une exception technique brute (InvalidParameterError,
    # etc.) — soit un résultat (même dégénéré), soit TrainingAbortedError.
    try:
        result = train_and_evaluate_clustering(df, ClusteringConfig(seed=42), _NOOP)
        assert result.winning_algorithm_id in {s.id for s in CLUSTER_REGISTRY}
    except TrainingAbortedError:
        pass


def test_pipeline_bundle_contains_fitted_preprocessor_and_model():
    """Base de la future inférence (prédire le cluster d'une nouvelle
    observation, prévu dans le plan) — le bundle doit contenir un
    préprocesseur DÉJÀ FIT (jamais refit à l'inférence, même principe que
    `services/ml_inference.py` côté supervisé)."""
    df = _make_three_blobs_df()
    result = train_and_evaluate_clustering(df, ClusteringConfig(seed=42), _NOOP)
    bundle = result.pipeline_bundle
    assert "preprocessor" in bundle and "model" in bundle
    # Un préprocesseur non fit lève AttributeError/NotFittedError à transform().
    transformed = bundle["preprocessor"].transform(df)
    assert transformed.shape[0] == len(df)


def test_only_default_subset_evaluated_when_algorithm_ids_is_none():
    from services.clustering_registry import DEFAULT_ALGORITHM_IDS

    df = _make_three_blobs_df()
    result = train_and_evaluate_clustering(df, ClusteringConfig(seed=42), _NOOP)
    assert {c.algorithm_id for c in result.all_candidates} <= DEFAULT_ALGORITHM_IDS
