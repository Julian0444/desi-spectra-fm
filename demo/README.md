---
title: DESI Spectra Foundation Model
emoji: 🔭
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.22.0
app_file: app.py
pinned: false
license: mit
short_description: Redshift + masked reconstruction for galaxy spectra
models:
  - jirustaroure/desi-spectra-fm
---

# DESI Spectra Foundation Model — live demo

Interactive demo of a 26M-parameter foundation model for astronomical spectra:
**redshift prediction** (official output: `z_pred_map`, argmax of a 100-bin
posterior over `log(1+z)`, with a `z_confidence` score) and **masked-region
reconstruction** (the self-supervised task the model was trained on).

- Pick one of the four examples — **real DESI spectra from the held-out
  split**, never seen during training, shown with their pipeline `z_true` —
  or upload your own `.npz` with `flux` and `wavelength` (Å) arrays.
- Move the slider to hide a fraction of the input tokens and watch the model
  fill the shaded regions in.

Honest numbers (2,000 held-out DESI spectra): σ_NMAD = 0.030,
catastrophic-outlier rate η₀.₁₅ = 14.95 %. A course project trained on a
laptop — not production science.

Runs on ZeroGPU: anonymous visitors get a small daily allowance of runs —
if you hit "ZeroGPU quota exceeded", sign in to huggingface.co (free) for a
larger quota, or come back the next day.

Code, tests and evaluation notebook:
<https://github.com/Julian0444/desi-spectra-fm> ·
Model checkpoint: <https://huggingface.co/jirustaroure/desi-spectra-fm>
