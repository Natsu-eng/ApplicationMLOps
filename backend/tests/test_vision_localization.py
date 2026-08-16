"""Tests de `services/vision_localization.py` (pilier Vision, Lot 15
sous-lot C) — correctifs des bugs #10/#11/#14/#17 déjà documentés dans
docs/legacy/ANALYSE_COMPLETE_COMPUTER_VISION.md."""
from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image

from services.vision_localization import (
    encode_heatmap_png,
    encode_mask_png,
    generate_binary_mask,
    resize_map_to_original,
)


def test_generate_binary_mask_keeps_top_fraction_of_pixels():
    error_map = np.arange(100, dtype=np.float32).reshape(10, 10)
    mask = generate_binary_mask(error_map, percentile=0.9)
    assert mask.shape == (10, 10)
    assert set(np.unique(mask)).issubset({0, 1})
    # ~10% des pixels les plus élevés doivent être à 1
    assert 8 <= mask.sum() <= 12


def test_generate_binary_mask_rejects_invalid_percentile():
    error_map = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        generate_binary_mask(error_map, percentile=1.0)
    with pytest.raises(ValueError):
        generate_binary_mask(error_map, percentile=0.0)


def test_resize_map_to_original_changes_shape_correctly():
    """Correctif direct des bugs #10/#17 : la carte doit être réalignée à
    la taille RÉELLE de l'image source, jamais laissée à la résolution du
    modèle."""
    error_map = np.random.rand(32, 32).astype(np.float32)
    resized = resize_map_to_original(error_map, original_size=(256, 512))  # (largeur, hauteur)
    assert resized.shape == (512, 256)  # numpy = (hauteur, largeur)


def test_resize_map_to_original_preserves_relative_hot_zone():
    """Une zone d'erreur élevée dans le coin supérieur gauche doit rester
    dans le coin supérieur gauche après resize — pas de décalage introduit."""
    error_map = np.zeros((32, 32), dtype=np.float32)
    error_map[:8, :8] = 10.0  # zone chaude en haut à gauche
    resized = resize_map_to_original(error_map, original_size=(64, 64))
    top_left_mean = resized[:16, :16].mean()
    bottom_right_mean = resized[48:, 48:].mean()
    assert top_left_mean > bottom_right_mean


def test_encode_heatmap_png_is_valid_image():
    error_map = np.random.rand(16, 16).astype(np.float32)
    data_uri = encode_heatmap_png(error_map)
    assert data_uri.startswith("data:image/png;base64,")
    raw = base64.b64decode(data_uri.split(",", 1)[1])
    img = Image.open(io.BytesIO(raw))
    assert img.size == (16, 16)
    assert img.mode == "L"


def test_encode_heatmap_png_handles_constant_map_without_crashing():
    error_map = np.full((8, 8), 5.0, dtype=np.float32)
    data_uri = encode_heatmap_png(error_map)
    assert data_uri.startswith("data:image/png;base64,")


def test_encode_mask_png_is_black_and_white():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:5, 2:5] = 1
    data_uri = encode_mask_png(mask)
    raw = base64.b64decode(data_uri.split(",", 1)[1])
    img = Image.open(io.BytesIO(raw))
    arr = np.array(img)
    assert set(np.unique(arr)).issubset({0, 255})
    assert arr[3, 3] == 255
    assert arr[0, 0] == 0
