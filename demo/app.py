"""Gradio demo for the DESI Spectra Foundation Model (v2.1 checkpoint).

Loads the public checkpoint from the Hugging Face Hub, lets the user pick a
real held-out DESI spectrum (or upload their own .npz), optionally mask a
fraction of the input tokens, and shows the official redshift prediction
(`z_pred_map`) plus the masked-region reconstruction.

Hosted on ZeroGPU (the free tier for Gradio Spaces): `@spaces.GPU` allocates a
GPU slice per call — mandatory on this hardware, an app without a decorated
function crashes at startup — and is a no-op when running locally. Visitors
spend a small amount of daily ZeroGPU quota per click (signing in to
huggingface.co raises the allowance). A call takes ~1-3 s.
"""

from __future__ import annotations

import os

import spaces  # must be imported before torch on ZeroGPU
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr
from huggingface_hub import hf_hub_download

from desi_fm.predict import load_model_from_checkpoint, predict_spectrum

MODEL_REPO = "jirustaroure/desi-spectra-fm"
CODE_URL = "https://github.com/Julian0444/desi-spectra-fm"

CKPT = hf_hub_download(MODEL_REPO, "checkpoint_last.pt")
# On ZeroGPU hosts CUDA is emulated at startup, so the model can sit on
# 'cuda' at module level (per the ZeroGPU docs); locally this resolves to CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = load_model_from_checkpoint(CKPT, DEVICE)
LAMBDA_MIN = MODEL.config.lambda_min
LAMBDA_MAX = MODEL.config.lambda_max


def _first_row(arr: np.ndarray, name: str) -> tuple[np.ndarray, bool]:
    """Return a 1-D spectrum from a 1-D or (N, P) array."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr, False
    if arr.ndim == 2:
        if arr.shape[0] == 0:
            raise gr.Error(f"'{name}' is empty (shape {arr.shape}).")
        return arr[0], arr.shape[0] > 1
    raise gr.Error(f"'{name}' must be 1-D or 2-D, got shape {arr.shape}.")


def _load_npz(npz_file) -> dict:
    path = npz_file if isinstance(npz_file, str) else getattr(npz_file, "name", None)
    if not path:
        raise gr.Error("Please select an example or upload a .npz file.")
    try:
        data = np.load(path, allow_pickle=False)
    except Exception as exc:  # zip/format errors from arbitrary uploads
        raise gr.Error(f"Could not read the file as a NumPy .npz archive: {exc}")
    if "flux" not in data.files or "wavelength" not in data.files:
        raise gr.Error(
            f"The .npz must contain 'flux' and 'wavelength' arrays; "
            f"found {sorted(data.files)}."
        )
    return data


def _masked_spans(spectrum_mask: np.ndarray, model_wavelength: np.ndarray,
                  n_pixels_per_token: int) -> list[tuple[float, float]]:
    """Merge masked tokens into contiguous wavelength spans for shading."""
    spans: list[tuple[float, float]] = []
    start = None
    for i, hidden in enumerate(spectrum_mask):
        if hidden and start is None:
            start = i
        elif not hidden and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(spectrum_mask)))
    out = []
    n = len(model_wavelength)
    for a, b in spans:
        lo = model_wavelength[min(a * n_pixels_per_token, n - 1)]
        hi = model_wavelength[min(b * n_pixels_per_token, n - 1)]
        out.append((float(lo), float(hi)))
    return out


@spaces.GPU(duration=8)
def analyze(npz_file, mask_ratio: float):
    plt.close("all")
    data = _load_npz(npz_file)

    flux, multi = _first_row(data["flux"], "flux")
    flux = flux.astype(np.float32)
    wavelength, _ = _first_row(data["wavelength"], "wavelength")
    wavelength = wavelength.astype(np.float32)
    if flux.shape != wavelength.shape:
        raise gr.Error(
            f"flux {flux.shape} and wavelength {wavelength.shape} must have "
            f"the same length."
        )
    if flux.size < 16:
        raise gr.Error(f"Spectrum too short ({flux.size} pixels).")

    ivar = mask = None
    if "ivar" in data.files:
        ivar = _first_row(data["ivar"], "ivar")[0].astype(np.float32)
        if ivar.shape != flux.shape:
            ivar = None
    if "mask" in data.files:
        mask = _first_row(data["mask"], "mask")[0].astype(bool)
        if mask.shape != flux.shape:
            mask = None

    result = predict_spectrum(
        flux=flux, wavelength=wavelength, ivar=ivar, mask=mask,
        model=MODEL, mask_ratio=float(mask_ratio),
    )

    # --- headline text: z_pred_map is the official prediction of v2.1 ---
    z_map = result.get("z_pred_map")
    conf = result.get("z_confidence")
    lines = []
    if z_map is not None:
        lines.append(f"z_pred_map = {z_map:.4f}   (official prediction, posterior argmax)")
        lines.append(f"z_confidence = {conf:.2f}   (posterior concentration, 0-1; "
                     f"low values mean the model itself is unsure)")
    lines.append(f"z_pred = {result['z_pred']:.4f}   (posterior expectation, secondary)")
    if "z_true" in data.files:
        z_true = float(np.asarray(data["z_true"]).reshape(-1)[0])
        lines.append(f"z_true = {z_true:.4f}   (DESI pipeline ground truth)")
        if z_map is not None:
            dz = abs(z_map - z_true) / (1.0 + z_true)
            flag = "catastrophic outlier (>0.15)" if dz > 0.15 else "ok"
            lines.append(f"|Δz|/(1+z) = {dz:.4f} → {flag}")
    if multi:
        lines.append("note: the file contains several spectra; showing the first one.")

    # --- plot: input + reconstruction, masked regions shaded ---
    in_range = (wavelength >= LAMBDA_MIN) & (wavelength <= LAMBDA_MAX)
    recon = result["reconstruction_input_grid"]

    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    finite = flux[np.isfinite(flux) & in_range]
    if finite.size:
        lo, hi = np.percentile(finite, [0.5, 99.5])
        pad = 0.15 * max(hi - lo, 1e-3)
        ax.set_ylim(lo - pad, hi + pad)
    ax.plot(wavelength[in_range], flux[in_range], lw=0.6, color="0.55",
            label="input flux")
    ax.plot(wavelength[in_range], recon[in_range], lw=1.1, color="#3D6FD6",
            label="model reconstruction")
    spans = _masked_spans(result["spectrum_mask"], result["model_wavelength"],
                          MODEL.config.patch_size)
    for i, (lo_w, hi_w) in enumerate(spans):
        ax.axvspan(lo_w, hi_w, color="#E8B84B", alpha=0.25, lw=0,
                   label="hidden from the model" if i == 0 else None)
    ax.set_xlabel("wavelength [Å]")
    ax.set_ylabel("flux (input units)")
    ax.set_title(
        f"mask_ratio = {float(mask_ratio):.2f} "
        f"({int(result['spectrum_mask'].sum())} of "
        f"{result['spectrum_mask'].size} tokens hidden)",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    return "\n".join(lines), fig


DESCRIPTION = f"""
26M-parameter encoder-only transformer trained with masked-token prediction on
DESI EDR/SV3 spectra, predicting redshift jointly (course project, USF). The
official prediction is **`z_pred_map`** (argmax of a 100-bin posterior over
`log(1+z)`); `z_confidence` says how concentrated that posterior is.

The four examples are **real DESI spectra from the canonical held-out split** —
never seen during training. Ground truth (`z_true`) is shown next to the
prediction, including one catastrophic outlier (z_true 1.57 → predicted 0.96,
flagged by `z_confidence` 0.18 vs ≥ 0.6 on the well-predicted examples): on the
full 2,000-spectrum held-out set the model reaches σ_NMAD = 0.030 with 14.95 %
catastrophic outliers — honest numbers for an 80k-spectra laptop run, ~3 orders
of magnitude away from the real DESI pipeline.

Move the slider to hide a fraction of the input before the forward pass and
watch the model fill the shaded regions in (the self-supervised task it was
trained on). Upload your own spectrum as a `.npz` with `flux` and `wavelength`
(Å) arrays — any instrument works, optional `ivar`/`mask` are used if present.

[Code + tests + evaluation notebook]({CODE_URL}) ·
[Model checkpoint](https://huggingface.co/{MODEL_REPO})
"""

demo = gr.Interface(
    fn=analyze,
    inputs=[
        gr.File(label="Spectrum .npz (flux + wavelength in Å)",
                file_types=[".npz"]),
        gr.Slider(0.0, 0.9, value=0.0, step=0.05,
                  label="fraction of input tokens masked"),
    ],
    outputs=[gr.Textbox(label="Redshift prediction", lines=5),
             gr.Plot(label="Spectrum + reconstruction")],
    examples=[
        ["examples/heldout_z020.npz", 0.0],
        ["examples/heldout_z083.npz", 0.35],
        ["examples/heldout_z287.npz", 0.0],
        ["examples/heldout_lowconf_z157.npz", 0.0],
    ],
    cache_examples=False,
    title="DESI Spectra Foundation Model — redshift + masked reconstruction",
    description=DESCRIPTION,
    article=(
        "**Input format** — `.npz` with `flux` (P,) or (N, P) and `wavelength` "
        "(P,) in Ångströms; optional `ivar`, `mask`, `z_true`. Spectra are "
        "interpolated onto the model's internal log-λ grid (3600-9800 Å), so "
        "instruments other than DESI work out of the box. "
        f"Details in the [README]({CODE_URL})."
    ),
)

demo.queue(max_size=8)

if __name__ == "__main__":
    # allowed_paths: example .npz files live in the repo, not in gradio's
    # upload cache — without this, clicking an example on Spaces raises
    # InvalidPathError once the files age out of the cache.
    demo.launch(
        allowed_paths=[os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "examples")]
    )
