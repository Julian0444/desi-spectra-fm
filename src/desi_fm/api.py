"""REST API for the DESI foundation model (FastAPI).

Serves the public v2.1 checkpoint (`jirustaroure/desi-spectra-fm` on the
Hugging Face Hub) behind two endpoints:

    POST /predict       — multipart upload of a .npz with `flux` (P,) or (N, P)
                          and `wavelength` (P,) or (N, P) in Angstroms;
                          optional `ivar` / `mask`. Query param `mask_ratio`.
    POST /predict_json  — a single spectrum as JSON lists.

The official prediction of the v2.1 checkpoint is **`z_pred_map`** (argmax of
a 100-bin posterior over log(1+z)), reported together with `z_confidence`
(posterior concentration, 0-1). `z_pred` (posterior expectation) is kept as a
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

from .predict import load_model_from_checkpoint, predict_spectra_batch, predict_spectrum

MODEL_REPO = "jirustaroure/desi-spectra-fm"
CHECKPOINT_FILE = "checkpoint_last.pt"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # free-tier containers have modest RAM
MAX_SPECTRA_PER_REQUEST = 32  # keep worst-case CPU latency of one request bounded

app = FastAPI(
    title="desi-fm API",
    version="1.0",
    description=(
        "Redshift prediction + masked reconstruction for galaxy spectra "
        "(DESI foundation model v2.1). Official prediction: `z_pred_map` "
        "with its `z_confidence`; `z_pred` is the secondary posterior mean. "
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
            from huggingface_hub import hf_hub_download

            ckpt = hf_hub_download(MODEL_REPO, CHECKPOINT_FILE)
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
    return payload


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_loaded": _model is not None}


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
        data = np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception as exc:
        raise HTTPException(422, f"Could not read the file as a .npz archive: {exc}")
    if "flux" not in data.files or "wavelength" not in data.files:
        raise HTTPException(
            422,
            f"The .npz must contain 'flux' and 'wavelength'; found {sorted(data.files)}.",
        )

    fluxes = np.asarray(data["flux"], dtype=np.float32)
    wavelengths = np.asarray(data["wavelength"], dtype=np.float32)
    ivars = np.asarray(data["ivar"], dtype=np.float32) if "ivar" in data.files else None
    masks = np.asarray(data["mask"], dtype=bool) if "mask" in data.files else None
    if fluxes.ndim == 1:
        fluxes = fluxes[None, :]
        if ivars is not None and ivars.ndim == 1:
            ivars = ivars[None, :]
        if masks is not None and masks.ndim == 1:
            masks = masks[None, :]
    if fluxes.ndim != 2:
        raise HTTPException(422, f"'flux' must be 1-D or 2-D, got shape {fluxes.shape}.")
    if fluxes.shape[0] > MAX_SPECTRA_PER_REQUEST:
        raise HTTPException(
            413,
            f"Too many spectra ({fluxes.shape[0]}); the limit is "
            f"{MAX_SPECTRA_PER_REQUEST} per request.",
        )
    if fluxes.shape[1] < 16:
        raise HTTPException(422, f"Spectra too short ({fluxes.shape[1]} pixels).")

    try:
        result = predict_spectra_batch(
            fluxes=fluxes,
            wavelengths=wavelengths,
            ivars=ivars,
            masks=masks,
            model=get_model(),
            mask_ratio=mask_ratio,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc))

    payload = {"n": int(fluxes.shape[0])}
    payload.update(_z_payload(result))
    return payload


class SpectrumJSON(BaseModel):
    flux: List[float] = Field(description="Flux values, any units.")
    wavelength: List[float] = Field(description="Wavelengths in Angstroms.")
    ivar: Optional[List[float]] = Field(None, description="Optional inverse variance.")
    mask: Optional[List[bool]] = Field(None, description="Optional bad-pixel mask.")
    mask_ratio: float = Field(0.0, ge=0.0, le=0.9)


@app.post("/predict_json")
def predict_json(s: SpectrumJSON):
    """Predict the redshift of a single spectrum passed as JSON lists."""
    flux = np.asarray(s.flux, dtype=np.float32)
    wavelength = np.asarray(s.wavelength, dtype=np.float32)
    if flux.size < 16:
        raise HTTPException(422, f"Spectrum too short ({flux.size} pixels).")
    try:
        result = predict_spectrum(
            flux=flux,
            wavelength=wavelength,
            ivar=np.asarray(s.ivar, dtype=np.float32) if s.ivar is not None else None,
            mask=np.asarray(s.mask, dtype=bool) if s.mask is not None else None,
            model=get_model(),
            mask_ratio=s.mask_ratio,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return _z_payload(result)
