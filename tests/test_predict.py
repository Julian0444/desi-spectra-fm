import numpy as np
import torch

from desi_fm.model import DESIFoundationModel, DESIFoundationModelConfig
from desi_fm.predict import embed_spectrum, predict_spectrum


def test_predict_spectrum_exposes_map_and_confidence_for_classification():
    config = DESIFoundationModelConfig(
        n_pixels=64,
        n_tokens=8,
        lambda_min=3600.0,
        lambda_max=9800.0,
        d_model=32,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
        n_z_bins=16,
        z_max=6.0,
    )
    model = DESIFoundationModel(config)
    wavelength = np.geomspace(3600.0, 9800.0, 64).astype(np.float32)
    flux = np.sin(wavelength / 500.0).astype(np.float32)
    result = predict_spectrum(flux=flux, wavelength=wavelength, model=model)

    assert 0.0 <= result["z_pred_map"] <= 6.0
    assert 0.0 <= result["z_confidence"] <= 1.0
    assert "z_pred" in result  # la esperanza se conserva por retrocompatibilidad


def _line_spectrum(z: float, seed: int, wavelength: np.ndarray) -> np.ndarray:
    """Continuum + strong emission lines redshifted by (1+z), light noise."""
    rng = np.random.default_rng(seed)
    flux = 0.6 + 0.1 * np.sin(wavelength / 900.0) + rng.normal(0.0, 0.02, wavelength.size)
    for rest in (3727.1, 4861.3, 5006.8, 6562.8):
        observed = rest * (1.0 + z)
        if wavelength[0] <= observed <= wavelength[-1]:
            flux += 1.5 * np.exp(-0.5 * ((wavelength - observed) / 40.0) ** 2)
    return flux.astype(np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_embed_spectrum_same_z_more_similar():
    # Sanity contract of the embedding: two spectra with the same lines (same
    # z, different noise) must be closer in cosine similarity than a spectrum
    # whose lines sit somewhere else entirely.
    torch.manual_seed(0)
    config = DESIFoundationModelConfig(
        n_pixels=512,
        n_tokens=16,
        d_model=32,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
        n_z_bins=16,
    )
    model = DESIFoundationModel(config).eval()
    wavelength = np.geomspace(3600.0, 9800.0, 512).astype(np.float32)

    emb_a = embed_spectrum(
        flux=_line_spectrum(0.0, seed=1, wavelength=wavelength),
        wavelength=wavelength, model=model,
    )
    emb_b = embed_spectrum(
        flux=_line_spectrum(0.0, seed=2, wavelength=wavelength),
        wavelength=wavelength, model=model,
    )
    emb_far = embed_spectrum(
        flux=_line_spectrum(0.7, seed=3, wavelength=wavelength),
        wavelength=wavelength, model=model,
    )

    assert emb_a.shape == (config.d_model,)
    assert emb_a.dtype == np.float32
    assert np.isfinite(emb_a).all()
    assert _cosine(emb_a, emb_b) > _cosine(emb_a, emb_far)


def test_embed_spectrum_is_deterministic():
    torch.manual_seed(0)
    config = DESIFoundationModelConfig(
        n_pixels=512, n_tokens=16, d_model=32, n_layers=1, n_heads=4, dropout=0.0
    )
    model = DESIFoundationModel(config).eval()
    wavelength = np.geomspace(3600.0, 9800.0, 512).astype(np.float32)
    flux = _line_spectrum(0.3, seed=4, wavelength=wavelength)
    first = embed_spectrum(flux=flux, wavelength=wavelength, model=model)
    second = embed_spectrum(flux=flux, wavelength=wavelength, model=model)
    np.testing.assert_array_equal(first, second)
