"""Tests de `services/vision_classification_registry.py` (pilier Vision,
Lot 15 sous-lot B)."""
from __future__ import annotations

import pytest
import torch

from services.vision_classification_registry import (
    CLASSIFICATION_BACKBONE_REGISTRY,
    DEFAULT_BACKBONE_ID,
    freeze_backbone,
    get_backbone_spec,
    unfreeze_backbone,
)


def test_default_backbone_is_registered():
    ids = {s.id for s in CLASSIFICATION_BACKBONE_REGISTRY}
    assert DEFAULT_BACKBONE_ID in ids


def test_get_backbone_spec_rejects_unknown_id():
    with pytest.raises(ValueError):
        get_backbone_spec("resnet152")  # volontairement pas dans le registre (trop lourd pour du CPU)


@pytest.mark.parametrize("spec", CLASSIFICATION_BACKBONE_REGISTRY, ids=lambda s: s.id)
def test_backbone_forward_pass_matches_num_classes(spec):
    num_classes = 4
    model = spec.build_model(num_classes, 0.3)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, num_classes)


@pytest.mark.parametrize("spec", CLASSIFICATION_BACKBONE_REGISTRY, ids=lambda s: s.id)
def test_freeze_backbone_leaves_only_head_trainable(spec):
    model = spec.build_model(3, 0.3)
    total_params = sum(p.numel() for p in model.parameters())
    freeze_backbone(model, spec)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 0 < trainable < total_params  # la tête reste entraînable, le tronc non


@pytest.mark.parametrize("spec", CLASSIFICATION_BACKBONE_REGISTRY, ids=lambda s: s.id)
def test_unfreeze_backbone_makes_everything_trainable(spec):
    model = spec.build_model(3, 0.3)
    freeze_backbone(model, spec)
    unfreeze_backbone(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == total_params
