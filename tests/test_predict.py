import numpy as np

from desi_fm.model import DESIFoundationModel, DESIFoundationModelConfig
from desi_fm.predict import predict_spectrum


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
