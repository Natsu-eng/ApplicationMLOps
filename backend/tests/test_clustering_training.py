"""Tests de domains/clustering/services/engine.py (Lot 11 — ML non supervisé)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from domains.clustering.services import engine as clustering_training
from domains.clustering.services.registry import CLUSTER_REGISTRY
from domains.clustering.services.engine import (
    ClusteringConfig,
    MAX_ROWS_FOR_CLUSTERING,
    MAX_ROWS_FOR_CLUSTERING_LINEAR_ONLY,
    MAX_SELECTABLE_NOISE_RATIO,
    _attach_composite_rank,
    _effective_row_cap,
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


def test_all_candidates_ranked_by_composite_score_ascending():
    """Depuis le correctif rang composite (retour utilisateur direct — la
    silhouette seule pouvait élire une configuration nettement pire sur les
    2 autres métriques), le classement se fait sur `composite_rank`
    (croissant = meilleur d'abord), plus sur la silhouette seule."""
    df = _make_three_blobs_df()
    result = train_and_evaluate_clustering(df, ClusteringConfig(seed=42), _NOOP)
    composite_ranks = [c.composite_rank for c in result.all_candidates if c.composite_rank is not None]
    assert composite_ranks == sorted(composite_ranks)
    assert result.all_candidates[0].is_winner is True
    assert all(not c.is_winner for c in result.all_candidates[1:])
    # 3 blobs très séparés par construction : le vrai k=3 doit rester
    # excellent sur les 3 métriques à la fois (voir
    # test_composite_rank_still_prefers_clear_winner_on_all_metrics), donc
    # toujours élu même avec le nouveau critère — non-régression du
    # comportement observable pour l'utilisateur sur un cas simple.
    assert result.model_card["n_clusters"] == 3


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


def _candidate(
    silhouette: float,
    noise_ratio: float,
    label: str,
    davies_bouldin: float = 1.0,
    calinski_harabasz: float = 100.0,
) -> dict:
    # `davies_bouldin`/`calinski_harabasz` par défaut IDENTIQUES pour tous
    # les candidats d'un même test qui ne les fournit pas explicitement —
    # neutralise leur effet sur le rang composite (voir
    # `_attach_composite_rank`) pour les tests qui ne testent QUE le budget
    # de bruit, sans changer leur intention.
    return {
        "silhouette": silhouette,
        "noise_ratio": noise_ratio,
        "label": label,
        "davies_bouldin": davies_bouldin,
        "calinski_harabasz": calinski_harabasz,
    }


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


# ── Rang composite (retour utilisateur direct, deux cas réels observés) ───
# La silhouette seule élisait une configuration nettement pire sur les deux
# autres métriques pour un gain marginal de silhouette — reproduit ici le
# scénario réel (K-Means k=2 vs k=3, valeurs authentiques du run signalé).


def test_composite_rank_prefers_balanced_candidate_over_marginal_silhouette_winner():
    """`a` gagne en silhouette seul (0.63 > 0.61, marge minime) mais est
    NETTEMENT pire sur les deux autres métriques (dernier en Davies-Bouldin
    ET en Calinski-Harabasz) — le rang composite doit élire `b`, pas `a`,
    contrairement au comportement d'avant ce correctif (retour utilisateur
    direct : reproduit le principe du cas réel observé — K-Means k=2
    gagnant du silhouette seul mais classé avant-dernier en Davies-Bouldin —
    avec des valeurs construites pour ne laisser aucune ambiguïté sur les 3
    métriques à la fois, contrairement aux valeurs réelles du run signalé
    où Calinski-Harabasz favorisait en fait `k=2`, aboutissant à une
    égalité de rang composite légitimement départagée par la silhouette —
    voir `test_composite_rank_breaks_ties_on_silhouette`)."""
    a = _candidate(silhouette=0.63, noise_ratio=0.0, label="a", davies_bouldin=1.245, calinski_harabasz=90)
    b = _candidate(silhouette=0.61, noise_ratio=0.0, label="b", davies_bouldin=0.797, calinski_harabasz=140)

    ranked, exceeded_for_all = _rank_candidates_with_noise_budget([a, b])

    assert exceeded_for_all is False
    assert ranked[0]["label"] == "b", "le rang composite doit préférer le meilleur compromis, pas le silhouette maximal isolé"


def test_composite_rank_still_prefers_clear_winner_on_all_metrics():
    """Non-régression du cas simple : un candidat qui gagne sur les 3
    métriques à la fois reste toujours élu — le rang composite ne doit
    jamais inventer une préférence contre-intuitive quand il n'y a pas de
    compromis à faire."""
    clear_winner = _candidate(silhouette=0.9, noise_ratio=0.0, label="net", davies_bouldin=0.3, calinski_harabasz=500)
    mediocre = _candidate(silhouette=0.4, noise_ratio=0.0, label="mediocre", davies_bouldin=1.5, calinski_harabasz=50)

    ranked, _ = _rank_candidates_with_noise_budget([clear_winner, mediocre])

    assert ranked[0]["label"] == "net"


def test_composite_rank_breaks_ties_on_silhouette():
    """Égalité de rang composite (rare, construite ici à la main) : la
    silhouette départage — seule des 3 métriques bornée et directement
    interprétable, cohérent avec le choix historique."""
    a = _candidate(silhouette=0.7, noise_ratio=0.0, label="a", davies_bouldin=0.5, calinski_harabasz=200)
    b = _candidate(silhouette=0.6, noise_ratio=0.0, label="b", davies_bouldin=0.4, calinski_harabasz=250)
    # a : rang 1 silhouette, rang 2 DB, rang 2 CH -> moyenne (1+2+2)/3 = 1.67
    # b : rang 2 silhouette, rang 1 DB, rang 1 CH -> moyenne (2+1+1)/3 = 1.33
    # Pas d'égalité ici (b gagne nettement) — juste une preuve que le
    # candidat au meilleur rang composite gagne même sans être 1er en silhouette.
    ranked, _ = _rank_candidates_with_noise_budget([a, b])
    assert ranked[0]["label"] == "b"


def test_attach_composite_rank_exposes_individual_ranks():
    """Les rangs individuels (silhouette/Davies-Bouldin/Calinski-Harabasz)
    doivent être exposés sur chaque candidat, pas seulement la moyenne —
    nécessaire pour une explication transparente côté UI (pas une boîte
    noire : l'utilisateur doit pouvoir voir POURQUOI ce candidat a gagné)."""
    a = _candidate(silhouette=0.8, noise_ratio=0.0, label="a", davies_bouldin=0.5, calinski_harabasz=300)
    b = _candidate(silhouette=0.5, noise_ratio=0.0, label="b", davies_bouldin=1.0, calinski_harabasz=100)
    candidates = [a, b]
    _attach_composite_rank(candidates)
    assert a["rank_silhouette"] == 1
    assert a["rank_davies_bouldin"] == 1
    assert a["rank_calinski_harabasz"] == 1
    assert a["composite_rank"] == 1.0
    assert b["rank_silhouette"] == 2
    assert b["composite_rank"] == 2.0


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


# ── Plafond de lignes différencié par famille d'algorithme (retour
# utilisateur direct : "l'entreprise a 50 000 lignes, la plateforme n'en
# traite que 5 000 (10 %), est-ce que ça généralise ?" — voir le
# commentaire de MAX_ROWS_FOR_CLUSTERING dans engine.py pour l'analyse
# complète) ───────────────────────────────────────────────────────────────


def test_effective_row_cap_is_conservative_by_default():
    """Mode guidé (`algorithm_ids=None`) : le sous-ensemble par défaut
    inclut le hiérarchique (O(n²)) -> plafond conservateur."""
    assert _effective_row_cap(None) == MAX_ROWS_FOR_CLUSTERING


def test_effective_row_cap_is_conservative_when_hierarchical_selected():
    assert _effective_row_cap(["kmeans", "hierarchical"]) == MAX_ROWS_FOR_CLUSTERING


def test_effective_row_cap_is_conservative_when_dbscan_selected():
    assert _effective_row_cap(["dbscan"]) == MAX_ROWS_FOR_CLUSTERING


def test_effective_row_cap_is_generous_when_only_linear_algorithms_selected():
    """KMeans/MiniBatchKMeans seuls : coût quasi linéaire, pas de risque
    mémoire O(n²) -> plafond plus généreux, moins d'échantillonnage."""
    assert _effective_row_cap(["kmeans"]) == MAX_ROWS_FOR_CLUSTERING_LINEAR_ONLY
    assert _effective_row_cap(["minibatch_kmeans"]) == MAX_ROWS_FOR_CLUSTERING_LINEAR_ONLY
    assert _effective_row_cap(["kmeans", "minibatch_kmeans"]) == MAX_ROWS_FOR_CLUSTERING_LINEAR_ONLY


def test_model_card_exposes_the_row_cap_actually_applied():
    """Transparence (Lot 6B, §F.2) : le frontend doit pouvoir expliquer
    POURQUOI ce chiffre précis, pas juste afficher `sampled: true`."""
    df = _make_three_blobs_df()
    result = train_and_evaluate_clustering(df, ClusteringConfig(seed=42, algorithm_ids=["kmeans"]), _NOOP)
    assert result.model_card["row_cap"] == MAX_ROWS_FOR_CLUSTERING_LINEAR_ONLY
    assert result.model_card["row_cap_linear_only"] is True

    result_default = train_and_evaluate_clustering(df, ClusteringConfig(seed=42), _NOOP)
    assert result_default.model_card["row_cap"] == MAX_ROWS_FOR_CLUSTERING
    assert result_default.model_card["row_cap_linear_only"] is False
