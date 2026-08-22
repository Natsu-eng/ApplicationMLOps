"""Tests de `services/vision_anomaly_registry.py` (pilier Vision, Lot 15
sous-lot C)."""
from __future__ import annotations

import pytest
import torch

from domains.vision.anomalies.services.registry import (
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


def test_registry_has_three_entries_from_legacy():
    # Lot 6A — parité avec la classification (élargissement du registre) :
    # denoising et VAE repris du legacy sur l'architecture propre existante,
    # PatchCore/Siamese volontairement absents (voir docstring du module).
    ids = {s.id for s in ANOMALY_MODEL_REGISTRY}
    assert ids == {"conv_autoencoder", "denoising_autoencoder", "conv_vae"}


def test_only_the_vae_declares_the_vae_loss_kind():
    for spec in ANOMALY_MODEL_REGISTRY:
        expected = "vae" if spec.id == "conv_vae" else "mse"
        assert spec.loss_kind == expected


def test_denoising_autoencoder_only_perturbs_input_in_training_mode():
    from domains.vision.anomalies.services.registry import DenoisingConvAutoEncoder

    torch.manual_seed(0)
    model = DenoisingConvAutoEncoder(noise_factor=0.5)
    x = torch.rand(2, 3, IMAGE_SIZE, IMAGE_SIZE)

    model.eval()
    with torch.no_grad():
        out_eval_1 = model(x)
        out_eval_2 = model(x)
    # Éval déterministe (pas de bruit) — deux passes sur la même entrée
    # donnent EXACTEMENT le même résultat.
    assert torch.equal(out_eval_1, out_eval_2)

    model.train()
    out_train_1 = model(x)
    out_train_2 = model(x)
    # Entraînement : bruit gaussien ré-échantillonné à chaque appel — les
    # deux sorties diffèrent (le bruit change l'entrée avant l'encodeur).
    assert not torch.equal(out_train_1, out_train_2)


def test_conv_vae_compute_loss_includes_kl_term_after_training_forward():
    from domains.vision.anomalies.services.registry import ConvVAE

    torch.manual_seed(0)
    model = ConvVAE()
    x = torch.rand(2, 3, IMAGE_SIZE, IMAGE_SIZE)

    model.train()
    reconstructed = model(x)
    loss_with_kl = model.compute_loss(x, reconstructed)

    import torch.nn.functional as F

    mse_only = F.mse_loss(reconstructed, x)
    # La loss totale (reconstruction + KL) ne doit jamais être égale à la
    # reconstruction seule — sans quoi mu/logvar ne recevraient aucun
    # gradient utile du terme KL (VAE mal entraîné, voir docstring ConvVAE).
    assert loss_with_kl.item() != pytest.approx(mse_only.item())


def test_get_anomaly_model_spec_accepts_all_registered_ids():
    for spec in ANOMALY_MODEL_REGISTRY:
        assert get_anomaly_model_spec(spec.id) is spec
