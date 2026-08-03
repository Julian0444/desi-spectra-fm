import io

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

import desi_fm.api as api
from desi_fm.model import DESIFoundationModel, DESIFoundationModelConfig


def _small_model() -> DESIFoundationModel:
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
    model.eval()
    return model


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(api, "_model", _small_model())
    return TestClient(api.app)


def _npz_bytes(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def _spectrum(n_pixels: int = 64):
    wavelength = np.geomspace(3600.0, 9800.0, n_pixels).astype(np.float32)
    flux = np.sin(wavelength / 500.0).astype(np.float32)
    return flux, wavelength


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_npz_single_spectrum(client):
    flux, wavelength = _spectrum()
    r = client.post(
        "/predict",
        files={"file": ("spec.npz", _npz_bytes(flux=flux, wavelength=wavelength))},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 1
    assert len(body["z_pred"]) == 1
    assert 0.0 <= body["z_pred_map"][0] <= 6.0  # la predicción oficial
    assert 0.0 <= body["z_confidence"][0] <= 1.0


def test_predict_npz_batch(client):
    flux, wavelength = _spectrum()
    fluxes = np.stack([flux, flux * 2.0])
    r = client.post(
        "/predict",
        files={"file": ("spec.npz", _npz_bytes(flux=fluxes, wavelength=wavelength))},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 2
    assert len(body["z_pred_map"]) == 2


def test_predict_npz_missing_keys_is_422(client):
    flux, _ = _spectrum()
    r = client.post("/predict", files={"file": ("spec.npz", _npz_bytes(flux=flux))})
    assert r.status_code == 422
    assert "wavelength" in r.json()["detail"]


def test_predict_npz_garbage_file_is_422(client):
    r = client.post("/predict", files={"file": ("spec.npz", b"not an npz at all")})
    assert r.status_code == 422


def test_predict_npz_mask_ratio_out_of_range_is_422(client):
    flux, wavelength = _spectrum()
    r = client.post(
        "/predict?mask_ratio=1.5",
        files={"file": ("spec.npz", _npz_bytes(flux=flux, wavelength=wavelength))},
    )
    assert r.status_code == 422


def test_predict_json(client):
    flux, wavelength = _spectrum()
    r = client.post(
        "/predict_json",
        json={"flux": flux.tolist(), "wavelength": wavelength.tolist()},
    )
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["z_pred_map"] <= 6.0
    assert 0.0 <= body["z_confidence"] <= 1.0
    assert "z_pred" in body


def test_predict_json_length_mismatch_is_422(client):
    flux, wavelength = _spectrum()
    r = client.post(
        "/predict_json",
        json={"flux": flux.tolist(), "wavelength": wavelength.tolist()[:-4]},
    )
    assert r.status_code == 422
