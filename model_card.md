---
license: mit
tags:
  - astronomy
  - spectroscopy
  - transformer
  - masked-modeling
  - redshift
  - pytorch
  - self-supervised-learning
---

# DESI Spectra Foundation Model (26M)

**[▶ Live demo](https://huggingface.co/spaces/jirustaroure/desi-spectra-fm-demo)** —
try the model on real held-out DESI spectra (or your own) in the browser.

Encoder-only transformer (8 layers, `d_model=512`, 8 heads, ~26M parameters)
trained with masked-token prediction on DESI EDR/SV3 spectra
([Multimodal Universe](https://github.com/MultimodalUniverse/MultimodalUniverse)).
The redshift token is **always masked** and a prediction head is trained
**jointly** with reconstruction, so redshift enters the representation space
from step one — the redesign of AION-1's redshift handling that this course
project asked for.

**Current checkpoint: v2.1** — a **fine-tune of the v1 encoder** (not a from-scratch
model) with a new classification head over 100 `log(1+z)` bins: cross-entropy
normalized by `log(n_bins)`, sqrt-inverse class weights estimated from the real 80k
training labels, 1:1 loss weighting with reconstruction, and a leak-free
train/held-out split. It replaced v1 after passing every release gate of the project
plan (`comparison.json`, `decision: promote_v2_1`). The official redshift prediction
is **`z_pred_map`** (posterior argmax); `z_pred` (posterior expectation) is kept for
backward compatibility.

## Validation metrics (canonical held-out split)

Both models evaluated on the **same 2,000 valid-label held-out spectra** (the ones
following the 80,000 used for training — never seen by either model). v2.1 uses
`z_pred_map`:

| metric | v1 (50k, regression head) | **v2.1 (fine-tune, 100-bin classification)** |
|---|---|---|
| catastrophic outliers η₀.₁₅ | 22.6 % | **14.95 %** |
| σ_NMAD | 0.083 | **0.030** |
| MAE_norm ⟨\|Δz\|/(1+z)⟩ | 0.107 | **0.096** |
| η₀.₁₅ in z ∈ [1.5, 2.5) | 82.7 % | **23.5 %** |
| prediction ceiling (max z_pred) | 2.00 | **3.52** |
| reconstruction RMSE (masked, arcsinh space) | 0.819 | **0.817** |

The full bias/outlier analysis (with plots) is in the
[evaluation notebook](https://github.com/Julian0444/desi-spectra-fm/blob/main/notebooks/evaluation.ipynb);
the executed v1 "before" picture is preserved in
[`evaluation_v1_baseline.ipynb`](https://github.com/Julian0444/desi-spectra-fm/blob/main/notebooks/evaluation_v1_baseline.ipynb).

## Usage

```bash
pip install "desi-fm @ git+https://github.com/Julian0444/desi-spectra-fm"
pip install huggingface_hub
```

```python
from huggingface_hub import hf_hub_download
from desi_fm.predict import predict_spectrum

ckpt = hf_hub_download("jirustaroure/desi-spectra-fm", "checkpoint_last.pt")
result = predict_spectrum(flux=flux, wavelength=wavelength_angstrom,
                          checkpoint_path=ckpt)
result["z_pred_map"]                   # predicted redshift (official, posterior argmax)
result["z_confidence"]                 # posterior concentration in [0, 1]
result["reconstruction_input_grid"]    # reconstruction on your wavelength grid
```

Accepts spectra from **any instrument**: inputs are interpolated onto an
internal log-λ grid (3600–9800 Å, 7081 pixels) and the positional embedding is
a sinusoidal encoding of physical `log(λ)`, not a token index, so wavelength
coverage different from DESI's is handled transparently.

## Limitations

Trained on 80k spectra on a laptop (Apple MPS) — a course project, **not for
production science** (the DESI pipeline is ~3 orders of magnitude more accurate on z).
v2.1 removed v1's z ≈ 2 prediction ceiling and cut catastrophic outliers from 22.6 %
to 14.95 % globally (82.7 % → 23.5 % in z ∈ [1.5, 2.5)), but ~15 % of held-out spectra
are still catastrophic outliers — dominated by low-S/N spectra and line-misidentification
degeneracies — and predictions above z ≈ 3.5 remain unconstrained (few training
examples there). Use `z_confidence` to filter unreliable predictions. See the
[evaluation notebook](https://github.com/Julian0444/desi-spectra-fm/blob/main/notebooks/evaluation.ipynb)
for the quantified analysis.

## Files

| file | description |
|---|---|
| `checkpoint_last.pt` | model weights (state_dict) + training args |
| `config.json` | architecture configuration |
| `training_args.json` | exact training flags of the run |
| `metrics.jsonl` | per-step training/validation metrics |

## Links

- Live demo (HF Space): <https://huggingface.co/spaces/jirustaroure/desi-spectra-fm-demo>
- Code, tests, CI and evaluation notebook: <https://github.com/Julian0444/desi-spectra-fm>
