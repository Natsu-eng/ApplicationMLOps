"""Tests de `services/vision_anomaly_registry.py` (pilier Vision, Lot 15
sous-lot C)."""
from __future__ import annotations

import pytest
import torch

from services.vision_anomaly_registry import (
    ANOMALY_MODEL_REGISTRY,
    DEFAULT_ANOMALY_MODEL_ID,
    IMAGE_SIZE,
    get_anomaly_model_spec,
)


def test_default_model_is_registered():
    ids = {s.id for s in ANOMALY_MODEL_REGISTRY}
    assert DEFAULT_ANOMALY_MODEL_ID in ids


def test_get_anomaly_model_spec_rejects_unknown_id():
    with pytest.raises(ValueError):
        get_anomaly_model_spec("patchcore")  # volontairement pas dans le registre (trop lourd CPU)


@pytest.mark.parametrize("spec", ANOMALY_MODEL_REGISTRY, ids=lambda s: s.id)
def test_model_reconstructs_same_shape_in_zero_one_range(spec):
    model = spec.build_model()
    model.eval()
    x = torch.rand(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        out = model(x)
    assert out.shape == x.shape
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0
