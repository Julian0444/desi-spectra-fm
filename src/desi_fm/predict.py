"""Shared-contract inference for the DESI multitask transformer.

Resampling other wavelength grids does not validate accuracy on other instruments.

This module is the inference entry point. Accuracy outside DESI or under partial
coverage is unvalidated; historical validation is not an independent test.
It resamples wavelength grids in Angstroms from raw NumPy arrays (wavelength + flux,
plus optional ivar and bad-pixel mask) and returns:

    * the predicted redshift: `z_pred_map` (posterior argmax — the official
      prediction for classification checkpoints such as v2.1) with its
      `z_confidence` (posterior concentration in [0, 1]); `z_pred` (expm1 of
      the posterior mean in log(1+z)) is kept for backward compatibility and is the only
      prediction of scalar-head checkpoints (v1)
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
        # shipped v2.1 checkpoint; the same file is on the Hub as
        # jirustaroure/desi-spectra-fm / checkpoint_last.pt
        checkpoint_path="runs/desi_80k_classhead_v21/checkpoint_last.pt",
    )
    z = result["z_pred_map"]              # official prediction (v2.1)
    conf = result["z_confidence"]         # posterior concentration in [0, 1]
    recon = result["reconstruction_input_grid"]  # same length as `flux`

CLI
---
    python -m desi_fm.predict \
        --checkpoint runs/desi_80k_classhead_v21/checkpoint_last.pt \
        --input  spectra.npz \
        --output predictions.npz

The input .npz follows docs/INPUT_CONTRACT.md: `flux` (N, P),
`wavelength` (N, P) or shared (P,).
Optional: `ivar` (N, P), `mask` (N, P) bool. A 1-D input is treated as a
single spectrum.
"""

from __future__ import annotations

import argparse
import hashlib
from contextlib import contextmanager
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
from .inputs import parse_arrays, read_npz, bounded_real, SpectrumInputError


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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = DESIFoundationModelConfig(**checkpoint["config"])
    model = DESIFoundationModel(config).to(device)
    state_dict = dict(checkpoint["model"])
    if "pos_embed" in state_dict and "learned_pos_embed" not in state_dict:
        state_dict["learned_pos_embed"] = state_dict.pop("pos_embed")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    digest = hashlib.sha256()
    with open(checkpoint_path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(block)
    model.checkpoint_identity = {"sha256": digest.hexdigest(), "step": checkpoint.get("step"),
                                 "migration": "pos_embed rename" if "pos_embed" in checkpoint["model"] else None}
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

    flux, wavelength, ivar_arr, mask_arr = parse_arrays(flux, wavelength, ivar, mask).single()

    processed = preprocess_spectrum(flux, ivar_arr, wavelength, mask_arr, preprocess)
    flux_t = torch.from_numpy(processed["flux"]).to(device_t).unsqueeze(0)
    valid_t = torch.from_numpy(processed["valid"]).to(device_t).unsqueeze(0)
    return flux_t, valid_t, processed, preprocess


@contextmanager
def inference_mode(model):
    """Temporarily disable dropout; restore each module's prior training flag.

    Do not concurrently train and serve the same mutable model instance.
    """
    modes = [(m, m.training) for m in model.modules()]
    try:
        model.eval()
        with torch.inference_mode():
            yield
    finally:
        for module, training in modes:
            module.training = training


def model_identity(model):
    identity = {"config": model.config.to_dict(),
                "checkpoint": getattr(model, "checkpoint_identity", None),
                "source": "checkpoint" if hasattr(model, "checkpoint_identity") else "caller-supplied model"}
    if model.config.n_z_bins:
        centers = torch.expm1(model.z_bin_centers).detach().cpu()
        identity["redshift"] = {"configured_range": [0, model.config.z_max],
                               "bin_center_range": [float(centers[0]), float(centers[-1])],
                               "z_pred": "expm1(E[log(1+z)])", "z_confidence": "1 - H(p)/log(n_bins)",
                               "observed_max": None}
    return identity


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
    wavelength_unit: str = "angstrom",
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

    Supplied models run temporarily in eval mode; each prior module training
    flag is restored. Concurrent training on that same instance is unsupported.
    Input units default to Angstrom for legacy arrays, or declare wavelength_unit.

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
    mask_ratio = bounded_real(mask_ratio, "mask_ratio", 0, 0.9)
    flux, wavelength, ivar, mask = parse_arrays(flux, wavelength, ivar, mask, wavelength_unit=wavelength_unit).single()
    if model is None:
        if checkpoint_path is None:
            raise ValueError("Either `model` or `checkpoint_path` must be provided.")
        model = load_model_from_checkpoint(checkpoint_path, pick_device(device))
    device_t = _device_of(model)

    config = model.config
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

    with inference_mode(model):
        out = model(flux_t, valid_t, z=None, spectrum_mask=spectrum_mask)
    for key in ("z_pred", "z_pred_map", "z_confidence", "patch_pred"):
        if key in out and not torch.isfinite(out[key]).all():
            raise RuntimeError(f"Model produced nonfinite {key}")
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
        "coverage": processed["coverage"],
        "model_identity": model_identity(model),
        "valid_model_grid": processed["valid"].astype(bool),
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
    wavelength_unit: str = "angstrom",
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

    flux, wavelength, ivar, mask = parse_arrays(flux, wavelength, ivar, mask, wavelength_unit=wavelength_unit).single()
    flux_t, valid_t, _, _ = _prepare_inputs(model, flux, wavelength, ivar, mask)
    with inference_mode(model):
        emb = model.encode(flux_t, valid_t).squeeze(0).cpu().numpy().astype(np.float32)
    if not np.isfinite(emb).all() or np.linalg.norm(emb) == 0:
        raise RuntimeError("Model produced an unusable embedding")
    return emb


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
    wavelength_unit: str = "angstrom",
) -> dict[str, Any]:
    """Sequential inference over a validated batch.

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
    batch = parse_arrays(fluxes, wavelengths, ivars, masks, wavelength_unit=wavelength_unit)
    mask_ratio = bounded_real(mask_ratio, "mask_ratio", 0, 0.9)
    if model is None:
        if checkpoint_path is None:
            raise ValueError("Either model or checkpoint_path must be provided")
        model = load_model_from_checkpoint(checkpoint_path, pick_device(device))
    # Validate coverage for the entire batch before any forward pass.
    for f, w, v, m in zip(batch.flux, batch.wavelength, batch.ivar, batch.mask):
        _prepare_inputs(model, f, w, v, m)
    rows = [predict_spectrum(f, w, ivar=v, mask=m, model=model, mask_ratio=mask_ratio)
            for f, w, v, m in zip(batch.flux, batch.wavelength, batch.ivar, batch.mask)]
    shared = {"model_wavelength", "mask_ratio_used", "model_identity", "coverage"}
    result = {k: np.stack([row[k] for row in rows]) for k in rows[0] if k not in shared}
    result.update(model_wavelength=rows[0]["model_wavelength"], mask_ratio_used=float(mask_ratio),
                  model_identity=rows[0]["model_identity"], coverage=[r["coverage"] for r in rows])
    return result


def predict_file_single(source, *, model, mask_ratio=0):
    batch = read_npz(source)
    f, w, v, m = batch.single()
    return predict_spectrum(f, w, ivar=v, mask=m, model=model, mask_ratio=mask_ratio)


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

    batch = read_npz(args.input)
    fluxes, wavelengths, ivars, masks = batch.flux, batch.wavelength, batch.ivar, batch.mask

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
    arrays = {k: v for k, v in results.items() if k not in {"coverage", "model_identity"}}
    arrays["coverage_json"] = json.dumps(results["coverage"], allow_nan=False)
    arrays["model_identity_json"] = json.dumps(results["model_identity"], allow_nan=False)
    np.savez_compressed(out_path, **arrays)

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
    try:
        main()
    except SpectrumInputError as exc:
        raise SystemExit(f"SpectrumInputError: {exc}") from None
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
