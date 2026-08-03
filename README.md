# DESI Spectra Foundation Model

![ci](https://github.com/Julian0444/desi-spectra-fm/actions/workflows/ci.yml/badge.svg)

**[▶ Live demo](https://huggingface.co/spaces/jirustaroure/desi-spectra-fm-demo)** — pick a real held-out DESI spectrum (or upload your own), mask part of it, and watch the model predict the redshift and reconstruct the hidden regions in your browser.

A unimodal masked-token foundation model for astrophysical spectra, with the redshift mechanism redesigned per the project specification. Given a spectrum from **any instrument** (DESI or otherwise), the model predicts the redshift `z` and reconstructs masked spectral regions.

**Course:** PHYS303 / CS486 / CS686 — Final Project
**Student:** Julian Irusta Roure
**Due:** May 19, 2026 (11 PM)

> 📄 For full deliverable documentation (architecture, design decisions, metric progression), see [DELIVERABLE.md](DELIVERABLE.md).
> 📄 For the Spanish-language development walkthrough, see [README.es.md](README.es.md).
> 🤗 Model weights (v2.1 checkpoint) are hosted on Hugging Face Hub: [jirustaroure/desi-spectra-fm](https://huggingface.co/jirustaroure/desi-spectra-fm).
> 🔭 Live Gradio demo (HF Space): [jirustaroure/desi-spectra-fm-demo](https://huggingface.co/spaces/jirustaroure/desi-spectra-fm-demo) — app source in [`demo/`](demo/).
> 📡 Live REST API (FastAPI + Swagger): [jirustaroure-desi-fm-api.hf.space/api/docs](https://jirustaroure-desi-fm-api.hf.space/api/docs) — source in [`api/`](api/), see [REST API](#rest-api-fastapi).

---

## Quick start for graders (3 commands)

### 1. Install

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

Requirements: Python 3.9+, PyTorch 2.x (MPS / CUDA / CPU all supported).
Inference itself uses only NumPy + PyTorch; `huggingface_hub` (in
`requirements.txt`) is used once to download the shipped checkpoint.

### 2. Verify the build

```bash
python3 -m pytest tests/ -v
```

Expected output: **`24 passed`**. This confirms the model loads, the forward pass works, the redshift mechanism is information-leak-free, the train/held-out split is leak-free (disjointness, fixed membership, reproducible per-epoch order), and the REST API endpoints validate inputs correctly.

For the full evaluation with plots — metrics, bias/outlier analysis, reconstruction gallery, training curves, and a live inference demo — open the ready-to-run notebook [`notebooks/evaluation.ipynb`](notebooks/evaluation.ipynb). Sections 1–3 run offline from artifacts shipped in this repo; it ships pre-executed so the plots are visible without running anything.

### 3. Run inference on your benchmark spectra

Model weights are not tracked in git — the shipped v2.1 checkpoint lives on Hugging Face Hub at [`jirustaroure/desi-spectra-fm`](https://huggingface.co/jirustaroure/desi-spectra-fm). Download it once (it is cached afterwards):

```python
from huggingface_hub import hf_hub_download

ckpt = hf_hub_download(
    "jirustaroure/desi-spectra-fm",
    "checkpoint_last.pt",
)
```

Then run the CLI on your spectra:

```bash
CKPT=$(python3 -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('jirustaroure/desi-spectra-fm', 'checkpoint_last.pt'))")

python3 -m desi_fm.predict \
    --checkpoint "$CKPT" \
    --input  YOUR_SPECTRA.npz \
    --output predictions.npz
```

That's it. The script handles spectra from any instrument: it interpolates onto the model's internal log-λ grid, runs the forward pass, and interpolates the reconstruction back onto your input grid in your flux units. (If you trained the model yourself, a local `--checkpoint runs/desi_80k_classhead_v21/checkpoint_last.pt` works the same way.)

---

## Input format

`YOUR_SPECTRA.npz` is a NumPy `.npz` archive. **Required:**

| key | dtype | shape | meaning |
|---|---|---|---|
| `flux` | float32 | `(N, P)` or `(P,)` | observed flux, any units |
| `wavelength` | float32 | `(N, P)` or `(P,)` | wavelengths in Ångströms |

**Optional:**

| key | dtype | shape | meaning |
|---|---|---|---|
| `ivar` | float32 | `(N, P)` | inverse variance |
| `mask` | bool | `(N, P)` | `True` = bad / unusable pixel |

If `wavelength` is 1-D, the same grid is reused for every spectrum. A 1-D `flux` is treated as a single spectrum.

---

## Output format

`predictions.npz` contains:

| key | shape | meaning |
|---|---|---|
| `z_pred_map` | `(N,)` | **official predicted redshift** (posterior argmax; present for classification checkpoints like v2.1) |
| `z_confidence` | `(N,)` | posterior concentration in [0, 1] (classification checkpoints) |
| `z_pred` | `(N,)` | posterior-expectation redshift (kept for backward compatibility; the only prediction for v1) |
| `reconstruction_input_grid` | `(N, P)` | reconstruction on your wavelength grid, in your flux units |
| `reconstruction_model_grid` | `(N, 7081)` | reconstruction on the model's internal log-λ grid |
| `model_wavelength` | `(7081,)` | the model's internal wavelength grid (3600–9800 Å, log-spaced) |
| `spectrum_mask` | `(N, 273)` bool | which of the 273 token positions were masked |
| `center`, `scale` | `(N,)` | per-spectrum normalization stats applied internally |

---

## Reconstruction-benchmark mode

By default the model receives the full input spectrum and is asked to predict `z`. To benchmark **masked-region reconstruction** (deliverable 1b), pass `--mask-ratio` to randomly mask that fraction of the 273 token positions before the forward pass:

```bash
python3 -m desi_fm.predict \
    --checkpoint "$CKPT" \
    --input  YOUR_SPECTRA.npz \
    --output predictions.npz \
    --mask-ratio 0.5
```

(`$CKPT` is the checkpoint downloaded in the quick start above.)

`spectrum_mask` in the output tells you exactly which 26-pixel patches were hidden, so you can compute reconstruction error against ground truth on those regions only.

---

## Python API (optional)

```python
import numpy as np
from huggingface_hub import hf_hub_download
from desi_fm.predict import predict_spectrum, predict_spectra_batch

ckpt = hf_hub_download("jirustaroure/desi-spectra-fm", "checkpoint_last.pt")

# Single spectrum
result = predict_spectrum(
    flux=flux_1d,
    wavelength=wavelength_1d,
    ivar=optional_ivar,
    mask=optional_bad_pixel_mask,
    checkpoint_path=ckpt,
)
z = result["z_pred_map"]                        # official prediction (v2.1)
conf = result["z_confidence"]                   # posterior concentration [0, 1]
recon  = result["reconstruction_input_grid"]    # same shape as flux_1d

# Batch of N spectra
batch = predict_spectra_batch(
    fluxes=fluxes_2d,                # (N, P)
    wavelengths=wavelengths,         # (N, P) or (P,)
    checkpoint_path=ckpt,
)
batch["z_pred_map"]                  # (N,) official prediction
batch["reconstruction_input_grid"]   # (N, P)
```

---

## REST API (FastAPI)

**Public endpoint** — a free HF Space serving `desi_fm.api` (CPU inference; a request takes ~1-3 s, but the first one after idle may take ~30 s while the Space wakes up):

```bash
# health check
curl -s https://jirustaroure-desi-fm-api.hf.space/api/healthz

# redshifts for every spectrum in a .npz (flux + wavelength, optional ivar/mask)
curl -s -F "file=@spectrum.npz" \
  "https://jirustaroure-desi-fm-api.hf.space/api/predict?mask_ratio=0.0"
# → {"n": 1, "z_pred_map": [0.2267], "z_confidence": [0.6376], "z_pred": [0.2314]}

# a single spectrum as JSON lists
curl -s -X POST https://jirustaroure-desi-fm-api.hf.space/api/predict_json \
  -H "Content-Type: application/json" \
  -d '{"flux": [/* P floats */], "wavelength": [/* P Angstroms */]}'
```

Interactive Swagger docs: **<https://jirustaroure-desi-fm-api.hf.space/api/docs>**.
As everywhere in this project, the official prediction is `z_pred_map` (with its
`z_confidence`); `z_pred` is the secondary posterior mean. Limits: 32 spectra /
50 MB per request.

**Run it yourself:**

```bash
pip install -e ".[api]"
uvicorn desi_fm.api:app --port 7860      # checkpoint auto-downloads from the Hub

# or containerized:
docker build -t desi-fm-api .
docker run -p 7860:7860 desi-fm-api
```

Set `DESI_FM_CKPT=/path/to/checkpoint_last.pt` to serve a local checkpoint
instead of downloading. The deployed Space source is in [`api/`](api/) — it is
a ZeroGPU *Gradio* Space wrapping the same FastAPI app (Docker Spaces need a
PRO subscription; REST calls run on CPU and spend no ZeroGPU visitor quota).

---

## Repository layout

```
src/desi_fm/
  model.py            transformer encoder + reconstruction & redshift heads
  data.py             spectrum preprocessing (interpolation, normalization, patching)
  train.py            training loop
  evaluate.py         validation metrics on DESI streaming data
  predict.py          instrument-agnostic inference  ← use this for benchmarking
  api.py              FastAPI REST API (/predict, /predict_json, /healthz)
  inspect_schema.py   sanity-check utility for the MMU/DESI dataset
tests/                24 unit tests (shapes, no-leakage, split isolation, calibrated loss, MAP outputs, API)
demo/                 Gradio app deployed to the live HF Space (jirustaroure/desi-spectra-fm-demo)
api/                  API app deployed to the live HF Space (jirustaroure/desi-fm-api)
Dockerfile            CPU-only container image for the REST API
scripts/
  make_demo_examples.py   exports the demo's real held-out example spectra
  estimate_z_histogram.py estimates the training-label histogram (v2.1 class weights)
notebooks/
  evaluation.ipynb            ready-to-run evaluation notebook for v2.1 (metrics, plots, live demo)
  evaluation_v1_baseline.ipynb  the executed v1 "before" picture (bias/outlier diagnosis)
runs/desi_80k_classhead_v21/
  checkpoint_last.pt        the shipped model — v2.1 fine-tune of v1 (26 M parameters)
                            (not in git — download from hf.co/jirustaroure/desi-spectra-fm)
  config.json               model configuration
  training_args.json        exact training flags of the run
  metrics.jsonl             per-step training metrics
  predictions.csv           held-out predictions on 2000 DESI spectra (z_pred_map official)
  reconstructions.npz       held-out reconstructions on 50 DESI spectra
  comparison.json           v1 ↔ v2.1 release gates + promote decision
runs/desi_50k_big/
  checkpoint_last.pt        the v1 baseline (kept for comparison; local only, not in git)
DELIVERABLE.md        full deliverable documentation
README.md             this file (quick start for graders)
README.es.md          Spanish development walkthrough
RESUMEN.md            Spanish project summary
COMO_FUNCIONA.md      Spanish code walkthrough
```

---

## Achieved metrics

The shipped checkpoint is **v2.1**: a fine-tune of the v1 encoder with a new 100-bin
redshift classification head (official prediction: `z_pred_map`). Measured on the
**canonical held-out split** — 2,000 valid-label DESI spectra following the 80,000 used
for training, never seen by either model (full table, gates and progression in
[DELIVERABLE.md §5](DELIVERABLE.md); machine-readable decision in
`runs/desi_80k_classhead_v21/comparison.json`):

| metric (2,000 held-out spectra) | v1 baseline | **v2.1 (shipped)** |
|---|---|---|
| catastrophic outlier fraction η₀.₁₅ | 22.6 % | **14.95 %** |
| σ_NMAD | 0.083 | **0.030** |
| `redshift_mae_norm` = `mean(abs(z_pred - z) / (1 + z))` | 0.107 | **0.096** |
| η₀.₁₅ in z ∈ [1.5, 2.5) | 82.7 % | **23.5 %** |
| prediction ceiling (max z_pred) | 2.00 | **3.52** |
| `reconstruction_rmse_masked` (pixel-weighted, arcsinh space) | 0.819 | **0.817** |
| trainable parameters | 25,929,859 | 25,980,646 |

---

## How it handles non-DESI spectra (OOD)

Positional information inside the model is **a sinusoidal embedding of physical `log(λ)`**, not an arbitrary token index. So when a non-DESI spectrum arrives:

1. It is interpolated onto the model's internal `log(λ)` grid (3600–9800 Å, 7081 pixels).
2. Pixels outside the new instrument's wavelength coverage are marked `valid=0` so the model knows to ignore them.
3. The transformer reads the remaining valid tokens. Each token "knows" the physical wavelength it represents.
4. The reconstruction is interpolated back to the caller's grid in original flux units.

This is transparent to the caller — just pass any `(flux, wavelength)` arrays to `predict.py`.

---

## Design summary (one paragraph)

Encoder-only transformer (8 layers, `d_model=512`, 8 heads, 26 M parameters) trained with masked-token prediction on DESI EDR/SV3 spectra. Each spectrum is interpolated onto a log-λ grid, sliced into 273 continuous patches of 26 pixels, linearly projected to token embeddings, and augmented with a 274th always-masked redshift token. A sinusoidal log-λ positional embedding is added so the model can be applied to spectra from instruments with different wavelength coverage. Both redesign approaches from the specification are implemented: a lightweight MLP redshift head trained jointly with the encoder (Approach A), and the redshift token forcibly masked on every training example (Approach B). The v1 checkpoint (50 k spectra) used a scalar SmoothL1 regression on `log(1+z)`; the shipped **v2.1 checkpoint fine-tunes v1** on the first 80 k spectra × 3 epochs with a **100-bin classification head over `log(1+z)`** — cross-entropy normalized by `log(n_bins)`, sqrt-inverse class weights from the real training-label histogram, 1:1 loss weighting with reconstruction, and a leak-free train/held-out split — which removes the regression-to-the-mean collapse and the z ≈ 2 prediction ceiling diagnosed in v1. Full design rationale and oral-question answers are in [DELIVERABLE.md §4–5](DELIVERABLE.md).

---

## Troubleshooting

| symptom | fix |
|---|---|
| `RuntimeError: MPS backend …` | add `--device cpu` (slower but works on every machine) |
| `KeyError: 'flux'` | input `.npz` must contain at least `flux` and `wavelength` arrays |
| `RuntimeError: shape mismatch` | check that `flux` and `wavelength` have the same shape |
| inference seems slow | the batch API loops per spectrum; expect a few hundred ms per spectrum on MPS, ~1 s on CPU |

---

## References

- Project specification: PHYS303/CS486 final project (USF) — build and evaluate a self-supervised foundation model for DESI spectra with redshift prediction (course material, not distributed in this repo)
- Multimodal Universe dataset: <https://github.com/MultimodalUniverse/MultimodalUniverse>
- AION-1 (reference, not reproduced): <https://github.com/PolymathicAI/AION>
- AION-1 checkpoints (not used at inference time): <https://huggingface.co/polymathic-ai/aion-base>
