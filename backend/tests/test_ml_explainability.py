"""Tests de services/ml_explainability.py — normalisation partagée de la
sortie SHAP (Lot Explicabilité locale), réutilisée par ml_training.py
(résumé global) et ml_inference.py (explication locale)."""
from __future__ import annotations

import numpy as np

from domains.training.services.explainability import normalize_base_value, select_class_matrix, shap_values_per_class


# ── shap_values_per_class ────────────────────────────────────────────────


def test_shap_values_per_class_passes_through_list_unchanged():
    raw = [np.zeros((5, 3)), np.ones((5, 3))]
    result = shap_values_per_class(raw)
    assert isinstance(result, list)
    assert len(result) == 2
    assert np.array_equal(result[1], np.ones((5, 3)))


def test_shap_values_per_class_splits_3d_array_by_last_axis():
    raw = np.stack([np.full((4, 2), k) for k in range(3)], axis=-1)  # (4, 2, 3)
    result = shap_values_per_class(raw)
    assert isinstance(result, list)
    assert len(result) == 3
    for k, matrix in enumerate(result):
        assert matrix.shape == (4, 2)
        assert np.all(matrix == k)


def test_shap_values_per_class_keeps_2d_array_as_is():
    raw = np.random.default_rng(0).normal(size=(5, 4))
    result = shap_values_per_class(raw)
    assert isinstance(result, np.ndarray)
    assert result.shape == (5, 4)


# ── select_class_matrix ──────────────────────────────────────────────────


def test_select_class_matrix_picks_requested_index_from_list():
    matrices = [np.zeros((1, 2)), np.ones((1, 2)), np.full((1, 2), 2.0)]
    assert np.array_equal(select_class_matrix(matrices, 2), np.full((1, 2), 2.0))


def test_select_class_matrix_falls_back_to_last_when_index_out_of_range():
    matrices = [np.zeros((1, 2)), np.ones((1, 2))]
    assert np.array_equal(select_class_matrix(matrices, 99), np.ones((1, 2)))


def test_select_class_matrix_returns_2d_array_unchanged_regardless_of_index():
    arr = np.arange(6).reshape(1, 6)
    assert np.array_equal(select_class_matrix(arr, 0), arr)
    assert np.array_equal(select_class_matrix(arr, None), arr)


# ── normalize_base_value ─────────────────────────────────────────────────


def test_normalize_base_value_scalar_passthrough():
    assert normalize_base_value(0.42, None) == 0.42


def test_normalize_base_value_picks_class_index_from_list():
    assert normalize_base_value([0.1, 0.2, 0.3], 2) == 0.3


def test_normalize_base_value_falls_back_to_last_when_index_missing():
    assert normalize_base_value([0.1, 0.2], None) == 0.2
