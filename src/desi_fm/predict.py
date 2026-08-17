"""Instrument-agnostic inference for the DESI foundation model.

This module is the entry point intended for the held-out benchmark evaluation.
It accepts spectra from any instrument as raw NumPy arrays (wavelength + flux,
plus optional ivar and bad-pixel mask) and returns:

    * the predicted redshift `z`
    * the reconstructed spectrum, in the caller's wavelength grid and in
      original flux units

The model itself was trained on DESI spectra resampled onto a log-spaced
wavelength grid; the wavelength positional embedding is in log-lambda
coordinates, so non-DESI spectra are handled by interpolating them onto the
same internal grid before the forward pass and interpolating the
reconstruction back to the caller's grid afterwards.

Python API
----------
    from desi_fm.predict import predict_spectrum
    result = predict_spectrum(
        flux=flux_array,            # 1D NumPy, any flux units
        wavelength=wavelength_array,# 1D NumPy, Angstroms
        checkpoint_path="runs/desi_50k_zw10_mask50/checkpoint_last.pt",
    )
    z = result["z_pred"]
    recon = result["reconstruction_input_grid"]  # same length as `flux`

CLI
---
    python -m desi_fm.predict \
        --checkpoint runs/desi_50k_zw10_mask50/checkpoint_last.pt \
        --input  spectra.npz \
        --output predictions.npz

The input .npz must contain `flux` (N, P) and `wavelength` (N, P) or (P,).
Optional: `ivar` (N, P), `mask` (N, P) bool. A 1-D input is treated as a
single spectrum.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import SpectrumPreprocessConfig, preprocess_spectrum
from .model import DESIFoundationModel, DESIFoundationModelConfig
from .train import pick_device


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_model_from_checkpoint(
    checkpoint_path: Path | str, device: torch.device
) -> DESIFoundationModel:
    """Load a desi_fm checkpoint produced by `desi_fm.train` and put it on `device`.

    Tolerant of older checkpoints where the learned positional embedding was
    named `pos_embed` instead of `learned_pos_embed`.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = DESIFoundationModelConfig(**checkpoint["config"])
    model = DESIFoundationModel(config).to(device)
    state_dict = checkpoint["model"]
    if "pos_embed" in state_dict and "learned_pos_embed" not in state_dict:
        state_dict["learned_pos_embed"] = state_dict.pop("pos_embed")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def _denormalize_flux(
    norm_flux: np.ndarray, center: float, scale: float
) -> np.ndarray:
    """Invert the arcsinh normalization used in preprocess_spectrum."""
    return np.sinh(norm_flux) * scale + center


def _interpolate_to_grid(
    source_wavelength: np.ndarray,
    source_values: np.ndarray,
    target_wavelength: np.ndarray,
) -> np.ndarray:
    """Linear interpolation onto an arbitrary wavelength grid.

    Outside the source range we fill with zero. The caller is expected to know
    which target pixels lie inside the model's nominal range (3600-9800 A).
    """
    order = np.argsort(source_wavelength)
    sw = source_wavelength[order]
    sv = source_values[order]
    return np.interp(target_wavelength, sw, sv, left=0.0, right=0.0).astype(np.float32)


def _device_of(model: DESIFoundationModel) -> torch.device:
    return next(model.parameters()).device


def _prepare_inputs(
    model: DESIFoundationModel,
    flux: np.ndarray,
    wavelength: np.ndarray,
    ivar: np.ndarray | None,
    mask: np.ndarray | None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], SpectrumPreprocessConfig]:
    """Validate raw arrays and preprocess them onto the model's grid.

    Returns the (1, n_pixels) flux/valid tensors on the model's device plus
    the raw `preprocess_spectrum` output and the preprocess config used.
    """
    device_t = _device_of(model)
    config = model.config
    preprocess = SpectrumPreprocessConfig(
        n_pixels=config.n_pixels,
        lambda_min=config.lambda_min,
        lambda_max=config.lambda_max,
        wavelength_grid=config.wavelength_grid,
    )

    flux = np.asarray(flux, dtype=np.float32).reshape(-1)
    wavelength = np.asarray(wavelength, dtype=np.float32).reshape(-1)
    if flux.shape != wavelength.shape:
        raise ValueError(
            f"flux ({flux.shape}) and wavelength ({wavelength.shape}) must have the same shape."
        )
    if ivar is None:
        ivar_arr = np.ones_like(flux)
    else:
        ivar_arr = np.asarray(ivar, dtype=np.float32).reshape(-1)
    if mask is None:
        mask_arr = np.zeros_like(flux, dtype=bool)
    else:
        mask_arr = np.asarray(mask, dtype=bool).reshape(-1)

    processed = preprocess_spectrum(flux, ivar_arr, wavelength, mask_arr, preprocess)
    flux_t = torch.from_numpy(processed["flux"]).to(device_t).unsqueeze(0)
    valid_t = torch.from_numpy(processed["valid"]).to(device_t).unsqueeze(0)
    return flux_t, valid_t, processed, preprocess


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict_spectrum(
    flux: np.ndarray,
    wavelength: np.ndarray,
    *,
    checkpoint_path: Path | str | None = None,
    ivar: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    device: str = "auto",
    mask_ratio: float = 0.0,
    model: DESIFoundationModel | None = None,
) -> dict[str, Any]:
    """Run inference on a single spectrum from any instrument.

    Args:
        flux: 1D array of flux values (any units).
        wavelength: 1D array of wavelengths in Angstroms, same length as flux.
        checkpoint_path: path to a `desi_fm.train` checkpoint. Required if
            `model` is not provided.
        ivar: optional 1D inverse-variance array.
        mask: optional 1D bool array, True for bad pixels.
        device: 'auto', 'cpu', 'mps', or 'cuda'. Ignored if `model` is given.
        mask_ratio: fraction of token positions to randomly mask before the
            forward pass. Use 0.0 (default) for redshift prediction from the
            full spectrum; use >0 to evaluate masked-region reconstruction.
        model: optional preloaded model. If provided, `checkpoint_path` is
            ignored and the model's device is used.

    Returns:
        dict with keys:
            z_pred: float, predicted redshift z.
            reconstruction_input_grid: 1D float32, reconstruction interpolated
                back onto the caller's wavelength grid, in the same units as
                the input flux.
            reconstruction_model_grid: 1D float32, reconstruction on the
                model's internal wavelength grid, in input flux units.
            normalized_reconstruction: 1D float32, reconstruction on the
                model's grid in normalized (arcsinh) space.
            normalized_input: 1D float32, normalized input the model saw.
            model_wavelength: 1D float32, the model's internal wavelength grid.
            spectrum_mask: 1D bool, which of the n_tokens token positions
                were masked (length = config.n_tokens).
            mask_ratio_used: float.
            center: float, per-spectrum normalization center.
            scale: float, per-spectrum normalization scale.
    """
    if model is None:
        if checkpoint_path is None:
            raise ValueError("Either `model` or `checkpoint_path` must be provided.")
        model = load_model_from_checkpoint(checkpoint_path, pick_device(device))
    device_t = _device_of(model)

    config = model.config
    wavelength = np.asarray(wavelength, dtype=np.float32).reshape(-1)
    flux_t, valid_t, processed, preprocess = _prepare_inputs(
        model, flux, wavelength, ivar, mask
    )

    if mask_ratio <= 0.0:
        spectrum_mask = torch.zeros(
            1, config.n_tokens, dtype=torch.bool, device=device_t
        )
    else:
        spectrum_mask = (
            torch.rand(1, config.n_tokens, device=device_t) < float(mask_ratio)
        )

    out = model(flux_t, valid_t, z=None, spectrum_mask=spectrum_mask)
    z_pred = float(out["z_pred"].item())
    recon_normalized = (
        model.unpatchify(out["patch_pred"]).squeeze(0).cpu().numpy().astype(np.float32)
    )

    center = float(processed["center"])
    scale = float(processed["scale"])
    recon_native = _denormalize_flux(recon_normalized, center, scale).astype(np.float32)

    model_wavelength = preprocess.target_wavelength.astype(np.float32)
    recon_input_grid = _interpolate_to_grid(
        model_wavelength, recon_native, wavelength
    )

    result = {
        "z_pred": z_pred,
        "reconstruction_input_grid": recon_input_grid,
        "reconstruction_model_grid": recon_native,
        "normalized_reconstruction": recon_normalized,
        "normalized_input": processed["flux"].astype(np.float32),
        "model_wavelength": model_wavelength,
        "spectrum_mask": spectrum_mask.squeeze(0).cpu().numpy(),
        "mask_ratio_used": float(mask_ratio),
        "center": center,
        "scale": scale,
    }
    if "z_pred_map" in out:
        result["z_pred_map"] = float(out["z_pred_map"].item())
        result["z_confidence"] = float(out["z_confidence"].item())
    return result


@torch.no_grad()
def embed_spectrum(
    flux: np.ndarray,
    wavelength: np.ndarray,
    *,
    checkpoint_path: Path | str | None = None,
    ivar: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    device: str = "auto",
    model: DESIFoundationModel | None = None,
) -> np.ndarray:
    """Embed a single spectrum with the encoder (no masking, deterministic).

    Same preprocessing and argument semantics as `predict_spectrum`, but the
    output is the mean-pooled representation of the valid spectral tokens — a
    reusable embedding for similarity search, clustering or any downstream
    task, independent of the redshift heads.

    Returns:
        1D float32 array of length `config.d_model`.
    """
    if model is None:
        if checkpoint_path is None:
            raise ValueError("Either `model` or `checkpoint_path` must be provided.")
        model = load_model_from_checkpoint(checkpoint_path, pick_device(device))

    flux_t, valid_t, _, _ = _prepare_inputs(model, flux, wavelength, ivar, mask)
    return model.encode(flux_t, valid_t).squeeze(0).cpu().numpy().astype(np.float32)


def predict_spectra_batch(
    fluxes: np.ndarray,
    wavelengths: np.ndarray,
    *,
    checkpoint_path: Path | str | None = None,
    ivars: np.ndarray | None = None,
    masks: np.ndarray | None = None,
    device: str = "auto",
    mask_ratio: float = 0.0,
    model: DESIFoundationModel | None = None,
) -> dict[str, np.ndarray]:
    """Vectorized inference over many spectra.

    Args:
        fluxes: (N, P) array of flux values.
        wavelengths: (N, P) or (P,) array of wavelengths in Angstroms. If 1D,
            the same grid is reused for every spectrum.
        ivars: optional (N, P) or (P,) inverse variance.
        masks: optional (N, P) or (P,) bool array of bad pixels.
        Other args as in `predict_spectrum`.

    Returns:
        dict of stacked arrays. See `predict_spectrum` for field semantics.
        `model_wavelength` is shared across the batch and returned as 1D.
    """
    fluxes = np.asarray(fluxes, dtype=np.float32)
    if fluxes.ndim == 1:
        fluxes = fluxes[None, :]
    n = fluxes.shape[0]

    wavelengths = np.asarray(wavelengths, dtype=np.float32)
    if wavelengths.ndim == 1:
        wavelengths = np.broadcast_to(wavelengths, fluxes.shape).copy()
    elif wavelengths.shape != fluxes.shape:
        raise ValueError(
            f"wavelengths shape {wavelengths.shape} must match fluxes shape {fluxes.shape}."
        )

    if ivars is not None:
        ivars = np.asarray(ivars, dtype=np.float32)
        if ivars.ndim == 1:
            ivars = np.broadcast_to(ivars, fluxes.shape).copy()
    if masks is not None:
        masks = np.asarray(masks, dtype=bool)
        if masks.ndim == 1:
            masks = np.broadcast_to(masks, fluxes.shape).copy()

    if model is None:
        if checkpoint_path is None:
            raise ValueError("Either `model` or `checkpoint_path` must be provided.")
        device_t = pick_device(device)
        model = load_model_from_checkpoint(checkpoint_path, device_t)

    z_pred = np.zeros(n, dtype=np.float32)
    recon_input = np.zeros(fluxes.shape, dtype=np.float32)
    recon_model = np.zeros((n, model.config.n_pixels), dtype=np.float32)
    recon_norm = np.zeros((n, model.config.n_pixels), dtype=np.float32)
    norm_input = np.zeros((n, model.config.n_pixels), dtype=np.float32)
    centers = np.zeros(n, dtype=np.float32)
    scales = np.zeros(n, dtype=np.float32)
    masks_used = np.zeros((n, model.config.n_tokens), dtype=bool)
    has_map = model.config.n_z_bins > 0
    z_pred_map = np.zeros(n, dtype=np.float32) if has_map else None
    z_confidence = np.zeros(n, dtype=np.float32) if has_map else None

    model_wavelength: np.ndarray | None = None

    for i in range(n):
        r = predict_spectrum(
            fluxes[i],
            wavelengths[i],
            ivar=ivars[i] if ivars is not None else None,
            mask=masks[i] if masks is not None else None,
            mask_ratio=mask_ratio,
            model=model,
        )
        z_pred[i] = r["z_pred"]
        recon_input[i] = r["reconstruction_input_grid"]
        recon_model[i] = r["reconstruction_model_grid"]
        recon_norm[i] = r["normalized_reconstruction"]
        norm_input[i] = r["normalized_input"]
        centers[i] = r["center"]
        scales[i] = r["scale"]
        masks_used[i] = r["spectrum_mask"]
        if has_map:
            z_pred_map[i] = r["z_pred_map"]
            z_confidence[i] = r["z_confidence"]
        if model_wavelength is None:
            model_wavelength = r["model_wavelength"]

    assert model_wavelength is not None
    result = {
        "z_pred": z_pred,
        "reconstruction_input_grid": recon_input,
        "reconstruction_model_grid": recon_model,
        "normalized_reconstruction": recon_norm,
        "normalized_input": norm_input,
        "model_wavelength": model_wavelength,
        "spectrum_mask": masks_used,
        "center": centers,
        "scale": scales,
        "mask_ratio_used": np.float32(mask_ratio),
    }
    if has_map:
        result["z_pred_map"] = z_pred_map
        result["z_confidence"] = z_confidence
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Instrument-agnostic inference for desi_fm checkpoints."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a checkpoint produced by `python -m desi_fm.train ...`.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Input .npz containing at least 'flux' (N, P) and 'wavelength' "
            "(N, P) or (P,). Optional: 'ivar' (N, P), 'mask' (N, P) bool. "
            "A 1-D 'flux' is treated as a single spectrum."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output .npz path. See module docstring for layout.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=0.0,
        help=(
            "0.0 (default): predict z from the full spectrum. "
            ">0: randomly mask that fraction of token positions before the "
            "forward pass; useful for benchmarking masked reconstruction."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    data = np.load(args.input, allow_pickle=False)
    if "flux" not in data or "wavelength" not in data:
        raise KeyError(
            f"Input {args.input} must contain at least 'flux' and 'wavelength'."
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

    results = predict_spectra_batch(
        fluxes=fluxes,
        wavelengths=wavelengths,
        ivars=ivars,
        masks=masks,
        checkpoint_path=args.checkpoint,
        device=args.device,
        mask_ratio=args.mask_ratio,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **results)

    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(out_path),
                "num_spectra": int(fluxes.shape[0]),
                "checkpoint": str(args.checkpoint),
                "mask_ratio": float(args.mask_ratio),
                "first_z_pred": float(results["z_pred"][0]),
                "fields": sorted(results.keys()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
