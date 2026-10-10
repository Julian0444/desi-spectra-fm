"""REST API for the DESI foundation model (FastAPI).

Serves the public v2.1 checkpoint (`jirustaroure/desi-spectra-fm` on the
Hugging Face Hub) behind two endpoints:

    POST /predict       — multipart upload of a .npz with `flux` (P,) or (N, P)
                          and `wavelength` (P,) or (N, P) in Angstroms;
                          optional `ivar` / `mask`. Query param `mask_ratio`.
    POST /predict_json  — a single spectrum as JSON lists.

The official prediction of the v2.1 checkpoint is **`z_pred_map`** (argmax of
a 100-bin posterior over log(1+z)), reported together with `z_confidence`
(posterior concentration, 0-1). `z_pred` (expm1 of the posterior mean of log(1+z)) is kept as a
secondary output for backwards compatibility.

The checkpoint is downloaded lazily on the first request and cached; set
`DESI_FM_CKPT` to a local checkpoint path to skip the Hub download, and
`DESI_FM_DEVICE` to override the default `cpu`. Inference runs on CPU by
default so the API works on free hardware.

Run locally:

    uvicorn desi_fm.api:app --port 7860

Interactive docs (Swagger UI) at /docs.
"""

from __future__ import annotations

import io
import os
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .predict import load_model_from_checkpoint, predict_spectra_batch, predict_spectrum, model_identity
from .inputs import read_npz, SpectrumInputError
from .assets import download_asset

MODEL_REPO = "jirustaroure/desi-spectra-fm"
CHECKPOINT_FILE = "checkpoint_last.pt"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # free-tier containers have modest RAM
MAX_SPECTRA_PER_REQUEST = 32  # keep worst-case CPU latency of one request bounded

app = FastAPI(
    title="desi-fm API",
    version="0.2.0",
    description=(
        "Redshift prediction for DESI spectra "
        "(DESI foundation model v2.1). Official prediction: `z_pred_map` "
        "with its `z_confidence`; `z_pred` is expm1(E[log(1+z)]) from the same posterior. "
        "Model card: https://huggingface.co/jirustaroure/desi-spectra-fm — "
        "code: https://github.com/Julian0444/desi-spectra-fm"
    ),
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_model = None


def get_model():
    """Load the model once, lazily. `DESI_FM_CKPT` overrides the Hub download."""
    global _model
    if _model is None:
        ckpt = os.environ.get("DESI_FM_CKPT")
        if not ckpt:
            ckpt = download_asset(CHECKPOINT_FILE)
        device = torch.device(os.environ.get("DESI_FM_DEVICE", "cpu"))
        _model = load_model_from_checkpoint(ckpt, device)
    return _model


def _z_payload(result: dict) -> dict:
    """Order matters for readability: the official prediction goes first."""
    payload = {}
    if "z_pred_map" in result:
        payload["z_pred_map"] = np.asarray(result["z_pred_map"]).tolist()
        payload["z_confidence"] = np.asarray(result["z_confidence"]).tolist()
    payload["z_pred"] = np.asarray(result["z_pred"]).tolist()
    payload["coverage"] = result["coverage"]
    payload["model_identity"] = result["model_identity"]
    return payload


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_loaded": _model is not None,
            "model_identity": model_identity(_model) if _model is not None else None}


@app.post("/predict")
async def predict_npz(
    file: UploadFile,
    mask_ratio: float = Query(
        0.0,
        ge=0.0,
        le=0.9,
        description=(
            "Fraction of input tokens randomly hidden before the forward pass "
            "(0 = plain redshift prediction, deterministic)."
        ),
    ),
):
    """Predict redshifts for every spectrum in an uploaded .npz file."""
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large (>{MAX_UPLOAD_BYTES // 2**20} MB).")
    try:
        batch = read_npz(io.BytesIO(raw))
    except SpectrumInputError as exc:
        raise HTTPException(422, str(exc)) from exc
    fluxes, wavelengths, ivars, masks = batch.flux, batch.wavelength, batch.ivar, batch.mask
    if fluxes.shape[0] > MAX_SPECTRA_PER_REQUEST:
        raise HTTPException(413, "Too many spectra; limit is 32")

    try:
        result = predict_spectra_batch(
            fluxes=fluxes,
            wavelengths=wavelengths,
            ivars=ivars,
            masks=masks,
            model=get_model(),
            mask_ratio=mask_ratio,
        )
    except SpectrumInputError as exc:
        raise HTTPException(422, str(exc))

    payload = {"n": int(fluxes.shape[0])}
    payload.update(_z_payload(result))
    return payload


class SpectrumJSON(BaseModel):
    flux: list = Field(description="Flux values, any units.")
    wavelength: list = Field(description="Wavelengths in Angstroms.")
    ivar: Optional[list] = Field(None, description="Optional inverse variance.")
    mask: Optional[list] = Field(None, description="Optional bad-pixel mask.")
    mask_ratio: float = Field(0.0, ge=0.0, le=0.9, strict=True)
    wavelength_unit: str = "angstrom"


@app.post("/predict_json")
def predict_json(s: SpectrumJSON):
    """Predict the redshift of a single spectrum passed as JSON lists."""
    try:
        result = predict_spectrum(
            flux=s.flux,
            wavelength=s.wavelength,
            ivar=s.ivar,
            mask=s.mask,
            model=get_model(),
            mask_ratio=s.mask_ratio,
            wavelength_unit=s.wavelength_unit,
        )
    except SpectrumInputError as exc:
        raise HTTPException(422, str(exc))
    return _z_payload(result)
