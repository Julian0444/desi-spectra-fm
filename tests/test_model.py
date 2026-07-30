import math

import torch

from desi_fm.data import SpectrumPreprocessConfig
from desi_fm.model import DESIFoundationModel, DESIFoundationModelConfig


def test_forward_shapes_and_losses():
    config = DESIFoundationModelConfig(
        n_pixels=128,
        n_tokens=16,
        d_model=48,
        n_layers=2,
        n_heads=4,
        dropout=0.0,
    )
    model = DESIFoundationModel(config)
    flux = torch.randn(3, config.n_pixels)
    valid = torch.ones_like(flux)
    z = torch.tensor([0.1, 0.3, 0.8])
    out = model(flux, valid, z=z, mask_ratio=0.5)
    assert out["patch_pred"].shape == (3, config.n_tokens, config.patch_size)
    assert out["z_pred"].shape == (3,)
    assert out["loss"].ndim == 0
    assert torch.isfinite(out["loss"])


def test_redshift_is_not_input_dependent_on_true_z():
    config = DESIFoundationModelConfig(
        n_pixels=64,
        n_tokens=8,
        d_model=32,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
    )
    model = DESIFoundationModel(config)
    flux = torch.randn(2, config.n_pixels)
    valid = torch.ones_like(flux)
    mask = torch.zeros(2, config.n_tokens, dtype=torch.bool)
    out_a = model(flux, valid, z=torch.tensor([0.1, 0.2]), spectrum_mask=mask)
    out_b = model(flux, valid, z=torch.tensor([2.0, 3.0]), spectrum_mask=mask)
    assert torch.allclose(out_a["z_pred"], out_b["z_pred"])


def test_classification_head_shapes_and_ranges():
    config = DESIFoundationModelConfig(n_pixels=128, n_tokens=16, d_model=48,
                                       n_layers=2, n_heads=4, dropout=0.0,
                                       n_z_bins=32, z_max=6.0)
    model = DESIFoundationModel(config)
    flux = torch.randn(3, config.n_pixels); valid = torch.ones_like(flux)
    out = model(flux, valid, z=torch.tensor([0.1, 0.8, 3.0]), mask_ratio=0.5)
    assert out["z_logits"].shape == (3, 32)
    assert torch.isfinite(out["loss"])
    for key in ("z_pred", "z_pred_map"):
        assert (out[key] >= 0).all() and (out[key] <= 6.0 + 1e-4).all()
    assert (out["z_confidence"] >= 0).all() and (out["z_confidence"] <= 1).all()


def test_scalar_head_remains_default():
    config = DESIFoundationModelConfig(n_pixels=64, n_tokens=8, d_model=32,
                                       n_layers=1, n_heads=4, dropout=0.0)
    out = DESIFoundationModel(config)(torch.randn(2, 64), torch.ones(2, 64),
                                      z=torch.tensor([0.1, 0.5]))
    assert "z_logits" not in out           # v1 intacta


def test_classification_loss_can_be_normalized_by_log_bins():
    config = DESIFoundationModelConfig(
        n_pixels=64,
        n_tokens=8,
        d_model=32,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
        n_z_bins=16,
        z_max=6.0,
        z_label_smoothing=0.0,
        normalize_redshift_ce=True,
        redshift_loss_weight=1.0,
    )
    model = DESIFoundationModel(config)
    with torch.no_grad():
        model.redshift_head[-1].weight.zero_()
        model.redshift_head[-1].bias.zero_()

    flux = torch.randn(4, config.n_pixels)
    valid = torch.ones_like(flux)
    mask = torch.zeros(4, config.n_tokens, dtype=torch.bool)
    out = model(
        flux,
        valid,
        z=torch.tensor([0.1, 0.4, 1.0, 2.0]),
        spectrum_mask=mask,
    )

    assert torch.allclose(
        out["redshift_loss_raw"],
        torch.tensor(math.log(config.n_z_bins)),
        atol=1e-5,
    )
    assert torch.allclose(out["redshift_loss"], torch.tensor(1.0), atol=1e-5)


def test_default_preprocess_grid_is_log_spaced():
    cfg = SpectrumPreprocessConfig(n_pixels=16)
    wave = cfg.target_wavelength
    ratios = wave[1:] / wave[:-1]
    assert abs(float(ratios.max() - ratios.min())) < 1e-5


def test_model_has_physical_wavelength_embedding():
    config = DESIFoundationModelConfig(
        n_pixels=128,
        n_tokens=16,
        d_model=48,
        n_layers=1,
        n_heads=4,
    )
    model = DESIFoundationModel(config)
    assert model.wavelength_pos_embed.shape == (1, config.n_tokens + 1, config.d_model)
    assert model.wavelength_pos_embed[:, : config.n_tokens].abs().sum() > 0
    assert model.wavelength_pos_embed[:, config.n_tokens].abs().sum() == 0
