# Final Project Deliverable — A Unimodal Foundation Model for DESI Spectra

**Course:** PHYS303 / CS486 / CS686 — Deep Learning & Bayesian Learning
**Due:** 11 PM, Tuesday May 19, 2026
**Student:** Julian Irusta Roure

This document is the entry point for the instructor / TA. It describes the
trained model, how to load it, and how to run it on held-out spectra
(including non-DESI spectra) for the benchmark evaluation.

For the quick-start usage guide aimed at graders, see `README.md`.
For a longer Spanish-language walkthrough of the development process
(training pipeline, intermediate experiments, smoke tests), see
`README.es.md`.

---

## 1. Where the deliverables live


| artifact                                     | path                                    |
| -------------------------------------------- | --------------------------------------- |
| Trained checkpoint (recommended)             | `runs/desi_50k_big/checkpoint_last.pt`  |
| Model config (JSON)                          | `runs/desi_50k_big/config.json`         |
| Per-step training metrics                    | `runs/desi_50k_big/metrics.jsonl`       |
| Validation predictions (1000 DESI spectra)   | `runs/desi_50k_big/predictions.csv`     |
| Validation reconstructions (50 DESI spectra) | `runs/desi_50k_big/reconstructions.npz` |
| Inference entry point (Python + CLI)         | `src/desi_fm/predict.py`                |


Earlier checkpoints from intermediate experiments are kept under `runs/`
for the progression record.

---

## 2. Installation

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

Tested on macOS with Python 3.9, PyTorch 2.2 on MPS, and `datasets 2.18+`.
The inference path has no Hugging Face dependency; only training does.

---

## 3. Running inference on held-out spectra

The model exposes both a Python API and a CLI. Both are
**instrument-agnostic** — they accept arbitrary `(wavelength, flux)` arrays
in any units, resample them onto the model's internal log-lambda grid,
run the forward pass, and resample the reconstruction back onto the
caller's grid.

### 3a. Python API

```python
import numpy as np
from desi_fm.predict import predict_spectrum, predict_spectra_batch

result = predict_spectrum(
    flux=flux,                      # 1D NumPy, any flux units
    wavelength=wavelength,          # 1D NumPy, Angstroms
    ivar=ivar,                      # optional 1D inverse variance
    mask=bad_pixel_mask,            # optional 1D bool (True = bad)
    checkpoint_path="runs/desi_50k_big/checkpoint_last.pt",
)

z_pred                = result["z_pred"]                       # scalar
recon_on_input_grid   = result["reconstruction_input_grid"]    # same shape as flux
recon_on_model_grid   = result["reconstruction_model_grid"]    # (7081,)
model_wavelength_grid = result["model_wavelength"]             # (7081,)
```

For a batch:

```python
result = predict_spectra_batch(
    fluxes=spectra_2d,              # (N, P)
    wavelengths=wavelength_grid,    # (N, P) or (P,)
    checkpoint_path="runs/desi_50k_big/checkpoint_last.pt",
)
result["z_pred"]                       # (N,)
result["reconstruction_input_grid"]    # (N, P)
```

By default (`mask_ratio=0.0`) the model receives the full spectrum and is
asked to predict `z`. To benchmark masked-region reconstruction, set
`mask_ratio` to e.g. `0.35` or `0.5`; the function returns which token
positions were masked under `result["spectrum_mask"]`.

### 3b. CLI

```bash
python3 -m desi_fm.predict \
    --checkpoint runs/desi_50k_big/checkpoint_last.pt \
    --input  benchmark_spectra.npz \
    --output benchmark_predictions.npz
```

`benchmark_spectra.npz` must contain at minimum:


| key                 | dtype   | shape          | meaning                 |
| ------------------- | ------- | -------------- | ----------------------- |
| `flux`              | float32 | (N, P) or (P,) | observed flux           |
| `wavelength`        | float32 | (N, P) or (P,) | wavelength in Angstroms |
| `ivar` *(optional)* | float32 | (N, P)         | inverse variance        |
| `mask` *(optional)* | bool    | (N, P)         | True = bad pixel        |


The output `.npz` contains (see `src/desi_fm/predict.py` for full layout):


| key                         | shape     | meaning                                              |
| --------------------------- | --------- | ---------------------------------------------------- |
| `z_pred`                    | (N,)      | predicted redshift                                   |
| `reconstruction_input_grid` | (N, P)    | reconstruction on caller's grid, original flux units |
| `reconstruction_model_grid` | (N, 7081) | reconstruction on model's log-lambda grid            |
| `model_wavelength`          | (7081,)   | model's internal wavelength grid                     |
| `spectrum_mask`             | (N, 273)  | which token positions were masked                    |


---

## 4. Design decisions

### 4a. The redshift-handling redesign (the project's central technical contribution)

The assignment's central critique of AION-1 is that the redshift token is
treated identically to the 273 spectral tokens under uniform random masking,
which (i) under-weights it and (ii) keeps it out of the encoder's
representation space (AION-1 fits redshift with a head bolted onto a frozen
encoder).

This model combines **both** recommended fixes from the spec:

- **Approach A — Joint training with a lightweight predictor.** A small MLP
head reads the redshift token from the encoder output and is trained
jointly with the masked-token reconstruction objective. The encoder
weights are shaped by the redshift loss from step 1, never frozen.
See `DESIFoundationModel.redshift_head` in `src/desi_fm/model.py`.
- **Approach B — Always-mask the redshift token.** A learnable
`z_mask_token` is appended to every spectral sequence and the true `z`
is never fed to the encoder. The model must reconstruct `z` from the
spectral context on every training step. The unit test
`test_redshift_is_not_input_dependent_on_true_z`
(in `tests/test_model.py`) verifies that varying the `z` argument does
not change `z_pred`, ruling out information leakage through the input.

The two losses are combined as

```
loss = w_recon * MSE(masked patches) + w_z * SmoothL1(log(1+z_pred), log(1+z))
```

with `w_recon = 1.0`, `w_z = 10.0` in the shipped checkpoint. The
imbalance compensates for the fact that the reconstruction target has
~273 patches while the redshift target is one scalar.

### 4b. Tokenization (oral-question topic)

Each spectrum is interpolated onto a fixed 7,081-pixel grid spanning
3600-9800 Å. The grid is **logarithmic in wavelength**
(`np.geomspace`), because redshift acts as a translation in log-wavelength
coordinates, which is the natural symmetry for a transformer to learn.

The 7,081 pixels are then sliced into **273 contiguous patches** of 26
pixels each (padding to 273 × 26 = 7,098). Each patch is concatenated with
its per-pixel validity flag and projected to a `d_model`-dimensional
vector by a single learned linear layer after a LayerNorm. The result is
a sequence of 273 continuous spectral tokens. A 274th token, the always-
masked redshift slot, is appended to the sequence before the transformer.

This is a deliberate simplification of AION-1's separately-trained MaskGIT-
style discrete tokenizer. We chose continuous patches because they:

1. Avoid training a separate tokenizer (one fewer moving part).
2. Preserve flux magnitudes directly, which is useful for the
  reconstruction objective.
3. Make the relationship between patch index and physical wavelength
  exact, which matters for the wavelength positional embedding below.

The trade-off is that we cannot use cross-entropy over a discrete vocabulary
for the reconstruction loss — we use MSE on masked patches instead.

### 4c. Positional embedding (the OOD-generalization mechanism)

Each token receives a **fixed sinusoidal embedding indexed by its mean
log-wavelength**, normalized to [0, 1] over the model's 3600-9800 Å range.
An optional learned positional embedding is added on top. The redshift
token's wavelength embedding is exactly zero, so it cannot be confused
with any spectral position.

This is what gives the model a chance of generalizing to non-DESI spectra.
A token's representation is anchored to a physical wavelength, not to an
arbitrary index. A non-DESI spectrum covering, e.g., 4500-7500 Å will be
interpolated onto the model grid; the tokens whose physical wavelengths
fall inside that overlap region receive valid input, the others receive
zero input with `valid=0`, and the transformer is told via the embedding
what wavelength each token corresponds to.

---

## 5. Achieved metrics (validation, 2000 held-out DESI spectra)


| metric                                                       | value      |
| ------------------------------------------------------------ | ---------- |
| `redshift_mae`                                               | 0.222      |
| `redshift_mae_norm` = `mean(abs(z_pred - z) / (1 + z))`      | 0.124      |
| `reconstruction_loss` (MSE on masked patches, arcsinh space) | 0.747      |
| `reconstruction_rmse_masked` (pixel-weighted)                | 0.864      |
| trainable parameters                                         | 25,929,859 |


For comparison, the DESI pipeline reports `z` accuracies of order `1e-4`.
Our model is ~3 orders of magnitude worse on `z`, which is the expected
gap between a 26 M-parameter transformer trained on 50 k spectra for one
epoch on a laptop and the DESI production pipeline. The point of the
project is not to beat the pipeline but to show that a unimodal,
spectrum-only transformer with the redshift mechanism redesigned can
learn the task at all.

### Progression during the project


| run                                                   | data     | params   | `redshift_mae` | `redshift_mae_norm` | `recon_rmse_masked` |
| ----------------------------------------------------- | -------- | -------- | -------------- | ------------------- | ------------------- |
| smoke (500 ex, `w_z=2`, mask 0.35)                    | 500      | 11 M     | 0.537          | 0.282               | 0.968               |
| 10 k, `w_z=20`, mask 0.35                             | 10 k     | 11 M     | 0.259          | 0.150               | 0.957               |
| 10 k, `w_z=10`, mask 0.50                             | 10 k     | 11 M     | 0.259          | 0.146               | 0.950               |
| 50 k, `w_z=10`, mask 0.50                             | 50 k     | 11 M     | 0.220          | 0.130               | 0.862               |
| **50 k, `w_z=10`, mask 0.50, larger model** (shipped) | **50 k** | **26 M** | **0.222**      | **0.124**           | **0.864**           |


The trajectory confirms the architecture learns from data; the remaining
gap to the DESI pipeline is a model-capacity / compute issue, not a
structural one.

---

## 6. Scope adherence (what was *not* done, by design)

Per the spec:

- ❌ No imaging modalities (no Subaru/HSC, no Legacy Survey grz).
- ❌ No magnitude tokens, no per-band fluxes.
- ❌ No CNN-style direct redshift regressor.

Only DESI spectra and their redshifts were used.

The model is encoder-only with masked-token reconstruction; the redshift
prediction emerges from the encoder's representation, not from a
specialized regression network.

---

## 7. Reproducing the training run

The shipped checkpoint was produced by:

```bash
python3 -m desi_fm.train \
  --dataset MultimodalUniverse/desi \
  --data-dir edr_sv3 \
  --output-dir runs/desi_50k_big \
  --batch-size 4 \
  --max-train-examples 50000 \
  --val-examples 2000 \
  --epochs 1 \
  --redshift-loss-weight 10 \
  --mask-ratio 0.5 \
  --wavelength-grid log \
  --d-model 512 \
  --n-layers 8 \
  --n-heads 8 \
  --log-every-steps 50 \
  --save-every-steps 2500
```

Approximate wall time on an Apple M-series MPS: ~30 minutes including
validation. `metrics.jsonl` records the full loss trajectory.

---

## 8. References

- Project specification: PHYS303/CS486 final project (USF) — course material, not distributed in this repo
- Multimodal Universe: [https://github.com/MultimodalUniverse/MultimodalUniverse](https://github.com/MultimodalUniverse/MultimodalUniverse)
- AION-1 (reference, not reproduced): [https://github.com/PolymathicAI/AION](https://github.com/PolymathicAI/AION)
- AION-1 checkpoints: [https://huggingface.co/polymathic-ai/aion-base](https://huggingface.co/polymathic-ai/aion-base)
