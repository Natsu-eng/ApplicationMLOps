"""Tests de domains/clustering/services/engine.py (Lot 11 — ML non supervisé)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from domains.clustering.services import engine as clustering_training
from domains.clustering.services.registry import CLUSTER_REGISTRY
from domains.clustering.services.engine import (
    ClusteringConfig,
    MAX_SELECTABLE_NOISE_RATIO,
    _rank_candidates_with_noise_budget,
    train_and_evaluate_clustering,
)
from domains.shared.ml_preprocessing import TrainingAbortedError

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


# ── Garde-fou bruit (AUDIT_PILIER2_ET_REFONTE_UX.md, P2) ──────────────────
# Un DBSCAN qui ne rattache qu'une poignée de points très compacts à un
# cluster peut afficher une silhouette artificiellement haute (calculée
# uniquement sur les points non-bruit) — testé ici en fonction pure, sans
# recalculer un vrai clustering (le scénario "silhouette haute mais presque
# tout en bruit" est trivial à construire à la main, pas la peine de le
# faire émerger d'un vrai fit DBSCAN instable).


def _candidate(silhouette: float, noise_ratio: float, label: str) -> dict:
    return {"silhouette": silhouette, "noise_ratio": noise_ratio, "label": label}


def test_noisy_candidate_with_higher_silhouette_is_not_elected_winner():
    high_noise_but_tight = _candidate(silhouette=0.95, noise_ratio=0.9, label="dbscan_trop_strict")
    honest_full_coverage = _candidate(silhouette=0.55, noise_ratio=0.0, label="kmeans_honnete")

    ranked, exceeded_for_all = _rank_candidates_with_noise_budget([high_noise_but_tight, honest_full_coverage])

    assert exceeded_for_all is False
    assert ranked[0]["label"] == "kmeans_honnete"  # gagnant malgré une silhouette plus basse
    assert ranked[1]["label"] == "dbscan_trop_strict"  # toujours visible, relégué en second


def test_noise_budget_respected_candidates_still_ranked_by_silhouette():
    a = _candidate(silhouette=0.6, noise_ratio=0.1, label="a")
    b = _candidate(silhouette=0.8, noise_ratio=0.2, label="b")
    ranked, exceeded_for_all = _rank_candidates_with_noise_budget([a, b])
    assert exceeded_for_all is False
    assert [c["label"] for c in ranked] == ["b", "a"]


def test_all_candidates_exceeding_noise_budget_falls_back_without_crashing():
    only_noisy = [
        _candidate(silhouette=0.9, noise_ratio=0.95, label="x"),
        _candidate(silhouette=0.7, noise_ratio=0.99, label="y"),
    ]
    ranked, exceeded_for_all = _rank_candidates_with_noise_budget(only_noisy)
    assert exceeded_for_all is True
    assert [c["label"] for c in ranked] == ["x", "y"]  # retombe sur le tri brut par silhouette


def test_boundary_noise_ratio_exactly_at_threshold_is_selectable():
    at_threshold = _candidate(silhouette=0.5, noise_ratio=MAX_SELECTABLE_NOISE_RATIO, label="pile")
    ranked, exceeded_for_all = _rank_candidates_with_noise_budget([at_threshold])
    assert exceeded_for_all is False
    assert ranked[0]["label"] == "pile"


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
    from domains.clustering.services.registry import DEFAULT_ALGORITHM_IDS

    df = _make_three_blobs_df()
    result = train_and_evaluate_clustering(df, ClusteringConfig(seed=42), _NOOP)
    assert {c.algorithm_id for c in result.all_candidates} <= DEFAULT_ALGORITHM_IDS


# ── Lot 6B, §F.2 : transparence d'échantillonnage ──────────────────────────


def test_sampling_caps_at_limit_and_reports_transparency(monkeypatch):
    monkeypatch.setattr(clustering_training, "MAX_ROWS_FOR_CLUSTERING", 60)
    df = _make_three_blobs_df(n_per_group=50)  # 150 lignes, > cap de test (60)
    result = train_and_evaluate_clustering(df, ClusteringConfig(seed=42), _NOOP)
    assert result.model_card["sampled"] is True
    assert result.model_card["n_samples_total"] == 150
    assert result.model_card["n_samples_used"] == 60
    # `n_samples` (nom historique lu par le frontend) doit rester aligné sur
    # les données RÉELLEMENT clusterisées, jamais le total avant échantillon.
    assert result.model_card["n_samples"] == 60
    total_in_profiles = sum(p.size for p in result.cluster_profiles)
    assert total_in_profiles + result.noise_count == 60


def test_no_sampling_when_dataset_under_cap():
    df = _make_three_blobs_df(n_per_group=50)  # 150 lignes, sous le vrai cap (5000)
    result = train_and_evaluate_clustering(df, ClusteringConfig(seed=42), _NOOP)
    assert result.model_card["sampled"] is False
    assert result.model_card["n_samples_total"] == 150
    assert result.model_card["n_samples_used"] == 150


# ── Lot 6B, §F.2 : transparence catégorielle (référence population) ────────


def test_categorical_summary_exposes_population_baseline_and_lift():
    """Chaque groupe pèse 1/3 de la population par construction — un cluster
    pur (100 % d'une catégorie) sur une catégorie qui ne pèse qu'1/3
    globalement doit afficher une sur-représentation (`lift`) proche de 3."""
    df = _make_three_blobs_df()
    result = train_and_evaluate_clustering(df, ClusteringConfig(seed=42), _NOOP)
    for profile in result.cluster_profiles:
        cat = profile.categorical_summary["categorie"]
        assert cat["population_pct"] == pytest.approx(100 / 3, abs=1.0)
        assert cat["lift"] == pytest.approx(3.0, abs=0.3)


# ── Lot 6B, §F.2 : indicateur de stabilité de k ─────────────────────────────


def test_stability_ari_is_high_on_well_separated_groups():
    """3 groupes numériques très séparés : un sous-échantillonnage à 80 % ne
    doit presque jamais changer la structure retrouvée — stabilité proche de
    1 attendue, pas une valeur arbitraire."""
    df = _make_three_blobs_df()
    result = train_and_evaluate_clustering(df, ClusteringConfig(seed=42), _NOOP)
    assert result.model_card["stability_ari"] is not None
    assert result.model_card["stability_ari"] > 0.7


def test_stability_ari_is_none_below_minimum_rows(monkeypatch):
    monkeypatch.setattr(clustering_training, "MIN_ROWS_FOR_STABILITY", 10_000)
    df = _make_three_blobs_df()
    result = train_and_evaluate_clustering(df, ClusteringConfig(seed=42), _NOOP)
    assert result.model_card["stability_ari"] is None
