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


| artifact                                        | path                                              |
| ----------------------------------------------- | ------------------------------------------------- |
| Trained checkpoint (recommended, **v2.1**)      | `runs/desi_80k_classhead_v21/checkpoint_last.pt`  |
| Model config (JSON)                             | `runs/desi_80k_classhead_v21/config.json`         |
| Training args + per-step metrics                | `runs/desi_80k_classhead_v21/training_args.json`, `metrics.jsonl` |
| Held-out predictions (2000 DESI spectra)        | `runs/desi_80k_classhead_v21/predictions.csv`     |
| Held-out reconstructions (50 DESI spectra)      | `runs/desi_80k_classhead_v21/reconstructions.npz` |
| v1 ↔ v2.1 gate comparison + release decision    | `runs/desi_80k_classhead_v21/comparison.json`     |
| v1 baseline checkpoint (kept for comparison)    | `runs/desi_50k_big/checkpoint_last.pt`            |
| Inference entry point (Python + CLI)            | `src/desi_fm/predict.py`                          |


v2.1 is a **fine-tune of the v1 encoder** with a new redshift classification head —
not a from-scratch model. Earlier checkpoints from intermediate experiments are kept
under `runs/` for the progression record.

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

## 5. Achieved metrics (canonical held-out split, 2000 DESI spectra)

The shipped checkpoint is **v2.1** (`runs/desi_80k_classhead_v21`): a **fine-tune of
the v1 encoder** with a new 100-bin redshift classification head over `log(1+z)`
(cross-entropy normalized by `log(n_bins)`, sqrt-inverse class weights from the real
80k-label histogram, leak-free `filter → skip → take → shuffle(train-only)` split).
Its official prediction is **`z_pred_map`** (posterior argmax). All numbers below are
measured on the **canonical held-out split** — the 2,000 valid-label spectra after the
80,000 used for training (`--skip-examples 80000`), never seen by v1 or v2.1.

| metric (held-out, `z_pred_map` for v2.1)                     | v1 baseline | **v2.1 (shipped)** |
| ------------------------------------------------------------ | ----------- | ------------------ |
| catastrophic outlier fraction η₀.₁₅                          | 22.6 %      | **15.0 %**         |
| σ_NMAD                                                       | 0.083       | **0.030**          |
| `redshift_mae_norm` = `mean(abs(z_pred - z) / (1 + z))`      | 0.107       | **0.096**          |
| η₀.₁₅ in z ∈ [1.5, 2.5)                                      | 82.7 %      | **23.5 %**         |
| prediction ceiling (max z_pred)                              | 2.00        | **3.52**           |
| `reconstruction_rmse_masked` (pixel-weighted, arcsinh space) | 0.819       | **0.817**          |
| trainable parameters                                         | 25,929,859  | 25,980,646         |

The machine-readable gate-by-gate comparison of both v2.1 checkpoints against the v1
baseline is in `runs/desi_80k_classhead_v21/comparison.json`
(`decision: promote_v2_1`).

For comparison, the DESI pipeline reports `z` accuracies of order `1e-4`.
Our model remains orders of magnitude worse on `z`, which is the expected
gap between a 26 M-parameter transformer trained on 80 k spectra on a laptop
and the DESI production pipeline. The point of the project is not to beat
the pipeline but to show that a unimodal, spectrum-only transformer with
the redshift mechanism redesigned can learn the task — and that the
redesign measurably removes the diagnosed failure modes.

### Progression during the project

Earlier rows were measured on differently-defined (partially seen) validation
windows and are kept as the historical record; the two final rows are on the
canonical held-out split described above.

| run                                                   | data     | params   | `redshift_mae_norm` | η₀.₁₅        | `recon_rmse_masked` |
| ----------------------------------------------------- | -------- | -------- | ------------------- | ------------ | ------------------- |
| smoke (500 ex, `w_z=2`, mask 0.35)                    | 500      | 11 M     | 0.282               | —            | 0.968               |
| 10 k, `w_z=20`, mask 0.35                             | 10 k     | 11 M     | 0.150               | —            | 0.957               |
| 10 k, `w_z=10`, mask 0.50                             | 10 k     | 11 M     | 0.146               | —            | 0.950               |
| 50 k, `w_z=10`, mask 0.50                             | 50 k     | 11 M     | 0.130               | —            | 0.862               |
| 50 k, `w_z=10`, mask 0.50, larger model (**v1**)      | 50 k     | 26 M     | 0.107 (held-out)    | 22.6 %       | 0.819               |
| v2.0 experiment: 200 bins, cap-10 rebalance, `w_z=10` | 80 k     | 26 M     | 0.182 (MAP)         | 31.1 % (MAP) | 0.877               |
| **v2.1 fine-tune: 100 bins, calibrated CE** (shipped) | **80 k** | **26 M** | **0.096 (MAP)**     | **15.0 %**   | **0.817**           |

The v2.0 experiment (kept in `runs/desi_150k_classhead/`, documented in
`plan/02-reentrenamiento-v2.md`) validated the classification head directionally but
mis-calibrated the loss (CE ≈ 98 % of the total) and over-rebalanced easy redshifts;
v2.1 fixed both and passed every release gate. The trajectory confirms the
architecture learns from data and that the remaining gap to the DESI pipeline is a
capacity / compute issue, not a structural one.

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
