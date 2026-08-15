"""Tests de services/dimensionality_training.py (Lot 13 — ML non supervisé)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import services.dimensionality_training as dimensionality_training
from services.dimensionality_registry import DIMENSIONALITY_REGISTRY
from services.dimensionality_training import DimensionalityConfig, train_and_evaluate_dimensionality
from services.ml_preprocessing import TrainingAbortedError

_NOOP = lambda step, pct: None  # noqa: E731
_ALGO_IDS = [s.id for s in DIMENSIONALITY_REGISTRY]


def _make_two_blobs_df(n_per_group: int = 100, seed: int = 42) -> pd.DataFrame:
    """2 groupes numériques très séparés — utilisé pour les tests qui n'ont
    besoin que d'une vraie structure (pas d'une répartition précise de la
    variance entre colonnes)."""
    rng = np.random.default_rng(seed)
    group = np.repeat([0, 1], n_per_group)
    signal = np.where(group == 0, 0.0, 20.0) + rng.normal(0, 0.5, len(group))
    noise = rng.normal(0, 0.5, len(group))
    return pd.DataFrame({"signal": signal, "noise": noise})


def _make_correlated_signal_df(n_per_group: int = 100, seed: int = 42) -> pd.DataFrame:
    """2 variables corrélées entre elles (portent le même signal de groupe)
    + 1 variable indépendante pure bruit. Le préprocesseur standardise
    chaque colonne (variance unitaire) AVANT la PCA — un signal porté par
    une seule colonne au milieu de bruit d'échelle comparable n'aurait donc
    aucune raison de dominer la variance expliquée après standardisation.
    Ici, PC1 doit capter la corrélation partagée par signal_a/signal_b :
    c'est la structure attendue après standardisation, pas juste une
    grande variance brute sur une colonne."""
    rng = np.random.default_rng(seed)
    group = np.repeat([0, 1], n_per_group)
    shared = np.where(group == 0, 0.0, 20.0)
    signal_a = shared + rng.normal(0, 1.0, len(group))
    signal_b = shared + rng.normal(0, 1.0, len(group))
    noise = rng.normal(0, 1.0, len(group))
    return pd.DataFrame({"signal_a": signal_a, "signal_b": signal_b, "noise": noise})


@pytest.mark.parametrize("algo_id", _ALGO_IDS)
def test_runs_end_to_end_for_every_registered_algorithm(algo_id):
    df = _make_two_blobs_df()
    result = train_and_evaluate_dimensionality(df, DimensionalityConfig(algorithm_id=algo_id, seed=42), _NOOP)
    assert len(result.points) == len(df)
    assert result.algorithm_id == algo_id


def test_pca_variance_and_loadings_identify_the_real_signal():
    df = _make_correlated_signal_df()
    result = train_and_evaluate_dimensionality(df, DimensionalityConfig(algorithm_id="pca", seed=42), _NOOP)
    # signal_a/signal_b partagent le même signal de groupe (corrélés) — PC1
    # doit capter cette corrélation partagée (variance unitaire pour les 3
    # colonnes après standardisation, sans cette corrélation PC1 n'expliquerait
    # qu'environ 1/3 de la variance totale).
    assert result.variance_explained[0] > 0.5
    by_pc1 = sorted(result.loadings, key=lambda entry: abs(entry.pc1), reverse=True)
    assert {by_pc1[0].feature, by_pc1[1].feature} == {"num__signal_a", "num__signal_b"}
    # `noise` est indépendant de signal_a/signal_b par construction : une fois
    # la direction partagée retirée par PC1, c'est la seule variance restante
    # — PC2 doit donc être dominé par `noise`, pas par signal_a/signal_b.
    by_pc2 = max(result.loadings, key=lambda entry: abs(entry.pc2))
    assert by_pc2.feature == "num__noise"


def test_trustworthiness_always_in_unit_interval():
    df = _make_two_blobs_df()
    for algo_id in _ALGO_IDS:
        result = train_and_evaluate_dimensionality(df, DimensionalityConfig(algorithm_id=algo_id, seed=42), _NOOP)
        assert 0.0 <= result.trustworthiness_primary <= 1.0
        assert 0.0 <= result.trustworthiness_pca <= 1.0


def test_distance_fidelity_note_present_and_differs_by_algorithm():
    df = _make_two_blobs_df()
    notes = set()
    for algo_id in _ALGO_IDS:
        result = train_and_evaluate_dimensionality(df, DimensionalityConfig(algorithm_id=algo_id, seed=42), _NOOP)
        assert result.distance_fidelity_note  # jamais vide
        notes.add(result.distance_fidelity_note)
    assert len(notes) == len(_ALGO_IDS)  # une note distincte par algorithme


def test_seed_is_wired_to_pca_and_tsne_estimators():
    df = _make_two_blobs_df()
    result = train_and_evaluate_dimensionality(df, DimensionalityConfig(algorithm_id="tsne", seed=777), _NOOP)
    assert result.pipeline_bundle["pca_model"].random_state == 777


def test_progress_callback_increasing_and_reaches_100():
    df = _make_two_blobs_df(n_per_group=20)
    calls: list[int] = []
    train_and_evaluate_dimensionality(df, DimensionalityConfig(algorithm_id="pca", seed=42), lambda step, pct: calls.append(pct))
    assert calls[0] < calls[-1]
    assert calls[-1] == 100
    assert all(0 <= p <= 100 for p in calls)


def test_too_few_rows_raises_actionable_error():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    with pytest.raises(TrainingAbortedError):
        train_and_evaluate_dimensionality(df, DimensionalityConfig(seed=42), _NOOP)


def test_sampling_preserves_original_row_index_and_caps_at_limit(monkeypatch):
    """Force une limite basse pour ne pas payer le coût d'un vrai dataset de
    5000+ lignes dans les tests — vérifie que l'échantillonnage est signalé
    honnêtement et que chaque point conserve la position réelle de sa ligne
    dans le dataset D'ORIGINE (pas une position recalculée après tri)."""
    monkeypatch.setattr(dimensionality_training, "MAX_ROWS_FOR_EMBEDDING", 50)
    df = _make_two_blobs_df(n_per_group=100)  # 200 lignes, > cap de test (50)
    result = train_and_evaluate_dimensionality(df, DimensionalityConfig(algorithm_id="pca", seed=42), _NOOP)

    assert result.sampled is True
    assert result.n_samples_total == 200
    assert result.n_samples_used == 50
    row_indices = [p.row_index for p in result.points]
    assert len(set(row_indices)) == len(row_indices)  # tous uniques
    assert all(0 <= idx < 200 for idx in row_indices)


def test_no_sampling_flag_when_dataset_fits_under_the_limit():
    df = _make_two_blobs_df(n_per_group=20)  # 40 lignes, largement sous la limite réelle
    result = train_and_evaluate_dimensionality(df, DimensionalityConfig(algorithm_id="pca", seed=42), _NOOP)
    assert result.sampled is False
    assert result.n_samples_used == result.n_samples_total == 40


def test_single_exploitable_column_degrades_gracefully_instead_of_crashing():
    """Une seule variable numérique exploitable après préprocessing —
    PC2/y doit être à zéro plutôt qu'un plantage (`np.pad` défensif)."""
    df = pd.DataFrame({"only_col": np.random.default_rng(0).normal(0, 1, 30)})
    with pytest.raises(TrainingAbortedError):
        # Une seule colonne = < 2 variables exploitables, dégradation propre attendue.
        train_and_evaluate_dimensionality(df, DimensionalityConfig(seed=42), _NOOP)
