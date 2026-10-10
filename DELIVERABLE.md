# Final Project Deliverable — A Unimodal Foundation Model for DESI Spectra

**Course:** PHYS303 / CS486 / CS686 — Deep Learning & Bayesian Learning
**Due:** 11 PM, Tuesday May 19, 2026
**Student:** Julian Irusta Roure

This document is the technical report for the instructor / TA. It describes
the shipped model (**v2.1**), how to load it, how to run it on held-out
spectra (resampling other grids does not validate their accuracy), and how each historical number was
obtained. The checkpoint submitted on the due date was the **v1** baseline;
the redshift-head redesign that produced v2.1 (§4a, §5) was carried out after
the submission, motivated by the bias and catastrophic-outlier behaviour
measured on v1, and this report was updated to describe what is shipped now.

For the quick-start usage guide, see `README.md`. For Spanish-language
readers, `README.es.md` is the Spanish version of `README.md` and
`COMO_FUNCIONA.md` is a file-by-file walkthrough of the code. The Hugging
Face model card is `model_card.md`.

---

## 1. Where the deliverables live

The artifacts are split across three places: the public repository
([github.com/Julian0444/desi-spectra-fm](https://github.com/Julian0444/desi-spectra-fm):
code, configs, metrics, predictions), the Hugging Face Hub
([jirustaroure/desi-spectra-fm](https://huggingface.co/jirustaroure/desi-spectra-fm):
weights, mirrored config / metrics, FAISS index), and local-only files that
are gitignored but can be regenerated with the scripts named below.
**Checkpoints (`*.pt`) are not in the public repository**: the shipped one is
downloaded from the Hub; the v1 baseline checkpoint exists only locally.

| artifact                                                | location                                                                                                                                    | public? |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Shipped checkpoint (**v2.1**, 103,964,688 bytes)        | Hub `jirustaroure/desi-spectra-fm` → `checkpoint_last.pt` (download with `hf_hub_download`, §3). Local: `runs/desi_80k_classhead_v21/checkpoint_last.pt` | Hub only; `runs/**/*.pt` is gitignored |
| Model config (JSON)                                     | `runs/desi_80k_classhead_v21/config.json` (also on the Hub)                                                                                 | repo    |
| Training args + per-step metrics                        | `runs/desi_80k_classhead_v21/training_args.json`, `metrics.jsonl` (also on the Hub)                                                         | repo    |
| Held-out metrics (2,000 DESI spectra)                   | `runs/desi_80k_classhead_v21/metrics_last.json` (shipped weights), `metrics_best.json`                                                      | repo    |
| Held-out predictions                                    | `runs/desi_80k_classhead_v21/predictions.csv` (identical to `predictions_last.csv`), `predictions_best.csv`; columns `z_true, z_pred, abs_dz, abs_dz_norm, z_pred_map, z_confidence` | repo |
| Held-out reconstructions (50 DESI spectra)              | `runs/desi_80k_classhead_v21/reconstructions.npz`                                                                                           | repo    |
| v1 ↔ v2.1 gate comparison + release decision            | `runs/desi_80k_classhead_v21/comparison.json` (`decision: promote_v2_1`)                                                                    | repo    |
| v1 baseline: config, training metrics, predictions      | `runs/desi_50k_big/config.json`, `metrics.jsonl`, `predictions.csv`, `reconstructions.npz`                                                  | repo    |
| v1 baseline checkpoint                                  | `runs/desi_50k_big/checkpoint_last.pt`                                                                                                      | local only (gitignored, not on the Hub) |
| v1 evaluated on the canonical held-out split            | `runs/calibration/metrics_v1_heldout_canonical.json`, `predictions_v1_heldout_canonical.csv`                                                | repo    |
| Class-weight histogram (100 bins, 80k training labels)  | `runs/calibration/z_hist_80k_100bins.npz`                                                                                                   | local only (gitignored); regenerate with `scripts/estimate_z_histogram.py` (§7) |
| Demo spectra (4 real held-out `.npz`)                   | `examples/heldout_z020.npz`, `heldout_z083.npz`, `heldout_z287.npz`, `heldout_lowconf_z157.npz`                                             | local only (gitignored); regenerate with `scripts/make_demo_examples.py` |
| Inference entry point (Python API + CLI)                | `src/desi_fm/predict.py`                                                                                                                    | repo    |
| Held-out evaluation protocol                            | `src/desi_fm/evaluate.py`                                                                                                                   | repo    |
| Training script                                         | `src/desi_fm/train.py`                                                                                                                      | repo    |
| Tests (26) + CI                                         | `tests/`, `.github/workflows/ci.yml` (Python 3.11, CPU PyTorch)                                                                             | repo    |
| Evaluation notebooks (pre-executed)                     | `notebooks/evaluation.ipynb` (v2.1), `notebooks/evaluation_v1_baseline.ipynb`                                                               | repo    |
| Live demo / HTTP API                                    | [Gradio Space](https://huggingface.co/spaces/jirustaroure/desi-spectra-fm-demo) · [FastAPI docs](https://jirustaroure-desi-fm-api.hf.space/api/docs) | hosted |

v2.1 is a **fine-tune of the v1 encoder** with a new redshift classification
head — not a from-scratch model (v1 was trained from scratch). The discarded
v2.0 experiment (`runs/desi_150k_classhead/`, §5) exists only on the
development machine; its local artifacts are now explicitly ignored and identified by hash.

---

## 2. Installation

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
python3 -m pytest -q          # → 26 passed
```

Python 3.9+, PyTorch 2.x. Developed on macOS (Python 3.9, MPS); CI runs the
same test suite on Python 3.11 with CPU-only PyTorch.

Dependency split:

- **Inference** needs only `numpy` and `torch` (the package's own
  dependencies in `pyproject.toml` are `numpy, torch, datasets, pyarrow, tqdm`).
- **`huggingface_hub`** (in `requirements.txt` and the `api` extra) is used
  only to download the shipped checkpoint from the Hub — a one-time
  ~104 MB download that is cached locally. If you already have
  `runs/desi_80k_classhead_v21/checkpoint_last.pt`, it is not needed.
- **`datasets` / `pyarrow`** are needed only to stream
  `MultimodalUniverse/desi` for training and for the held-out evaluation.
- **`fastapi` / `uvicorn`** are needed only for the HTTP API (`src/desi_fm/api.py`).

---

## 3. Running inference on held-out spectra

The model exposes both a Python API and a CLI. Both are
**grid-resampling interfaces** — they accept `(wavelength, flux)` arrays (wavelength in Å)
in any flux units, resample them onto the model's internal log-lambda grid,
run the forward pass, and resample the reconstruction back onto the caller's
grid. Accuracy outside DESI or with partial coverage has not been established.

The v2.1 checkpoint returns three redshift quantities:

- **`z_pred_map`** — the **official prediction**: `expm1` of the centre of
  the posterior's argmax bin. All metrics in §5 are computed on it.
- **`z_confidence`** — posterior concentration in `[0, 1]`, defined as
  `1 − H(p) / log(100)` (1 = all mass in one bin). On the held-out split,
  keeping only `z_confidence ≥ 0.5` (58.8 % coverage) gives 9 errors among 1,175 retained objects (0.77%, 58.75% coverage), retrospectively on masked, non-independent validation; not a calibrated probability or validated policy.
- **`z_pred`** — the posterior expectation of `log(1+z)`, decoded to `z`.
  Kept for compatibility with v1 checkpoints (whose scalar regression head
  produces only `z_pred`); it is *not* the official prediction, because the
  mean of a multimodal posterior falls between modes.

### 3a. Python API

```python
import numpy as np
from huggingface_hub import hf_hub_download
from desi_fm.predict import predict_spectrum, predict_spectra_batch

ckpt = hf_hub_download("jirustaroure/desi-spectra-fm", "checkpoint_last.pt")
# equivalently, if the local training output is present:
# ckpt = "runs/desi_80k_classhead_v21/checkpoint_last.pt"

result = predict_spectrum(
    flux=flux,                      # 1D NumPy, any flux units
    wavelength=wavelength,          # 1D NumPy, Angstroms
    ivar=ivar,                      # optional 1D inverse variance
    mask=bad_pixel_mask,            # optional 1D bool (True = bad)
    checkpoint_path=ckpt,
)

z_map  = result["z_pred_map"]                     # official prediction (float)
conf   = result["z_confidence"]                   # posterior concentration in [0, 1]
z_mean = result["z_pred"]                         # expm1(E[log(1+z)]) (compat)
recon_on_input_grid   = result["reconstruction_input_grid"]   # same shape as flux
recon_on_model_grid   = result["reconstruction_model_grid"]   # (7081,)
model_wavelength_grid = result["model_wavelength"]            # (7081,)
```

For a batch:

```python
result = predict_spectra_batch(
    fluxes=spectra_2d,              # (N, P)
    wavelengths=wavelength_grid,    # (N, P) or (P,)
    checkpoint_path=ckpt,
)
result["z_pred_map"]                   # (N,)
result["z_confidence"]                 # (N,)
result["reconstruction_input_grid"]    # (N, P)
```

By default (`mask_ratio=0.0`) the model receives the full spectrum and is
asked to predict `z`. To benchmark masked-region reconstruction, set
`mask_ratio` to e.g. `0.35` or `0.5`; the function returns which token
positions were masked under `result["spectrum_mask"]`. `predict.py` also
exposes `embed_spectrum(...)`, which returns the encoder representation used
by the similarity index in the companion project (§8).

Control value (the demo spectra in `examples/` are gitignored; regenerate
them with `scripts/make_demo_examples.py`): `examples/heldout_z020.npz` →
`z_pred_map` 0.2267, `z_confidence` 0.64, catalogue `z_true` 0.2036.

### 3b. CLI

```bash
CKPT=$(python3 -c "from huggingface_hub import hf_hub_download; \
  print(hf_hub_download('jirustaroure/desi-spectra-fm', 'checkpoint_last.pt'))")

python3 -m desi_fm.predict \
    --checkpoint "$CKPT" \
    --input  benchmark_spectra.npz \
    --output benchmark_predictions.npz
    # optional: --device {auto,cpu,mps,cuda}   --mask-ratio 0.5
```

`benchmark_spectra.npz` must contain at minimum:

| key                 | dtype   | shape          | meaning                 |
| ------------------- | ------- | -------------- | ----------------------- |
| `flux`              | float32 | (N, P) or (P,) | observed flux           |
| `wavelength`        | float32 | (N, P) or (P,) | wavelength in Angstroms |
| `ivar` *(optional)* | float32 | (N, P)         | inverse variance        |
| `mask` *(optional)* | bool    | (N, P)         | True = bad pixel        |

The output `.npz` contains (keys as written by `predict_spectra_batch` in
`src/desi_fm/predict.py`):

| key                          | shape     | meaning                                                          |
| ---------------------------- | --------- | ---------------------------------------------------------------- |
| `z_pred_map`                 | (N,)      | **official** predicted redshift (posterior argmax)               |
| `z_confidence`               | (N,)      | posterior concentration, `1 − H(p)/log(100)`                     |
| `z_pred`                     | (N,)      | expm1(E[log(1+z)]) (not the arithmetic mean of z) |
| `reconstruction_input_grid`  | (N, P)    | reconstruction on the caller's grid, original flux units         |
| `reconstruction_model_grid`  | (N, 7081) | reconstruction on the model's log-lambda grid, original units    |
| `normalized_reconstruction`  | (N, 7081) | reconstruction in the model's normalized (arcsinh) space         |
| `normalized_input`           | (N, 7081) | the normalized input the model actually saw                      |
| `model_wavelength`           | (7081,)   | model's internal wavelength grid (Å)                             |
| `spectrum_mask`              | (N, 273)  | which token positions were masked                                |
| `center`, `scale`            | (N,)      | per-spectrum normalization parameters                            |
| `mask_ratio_used`            | scalar    | the `--mask-ratio` that was applied                              |

`z_pred_map` and `z_confidence` are present only for checkpoints with the
classification head (v2.1); a v1 checkpoint yields `z_pred` alone. Note that
the JSON summary printed on the console reports `first_z_pred` (the
expectation); read `z_pred_map` from the `.npz`.

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
  not change the prediction, ruling out information leakage through the input.

Both fixes are shared by v1 and v2.1. What changed between them is the
*form* of the redshift target.

**v1 — scalar regression (replaced).** The head produced one scalar, trained
with

```
loss = 1.0 * MSE(masked patches) + 10.0 * SmoothL1(log(1+z_pred), log(1+z))
```

The weight 10 compensates for the reconstruction target having ~273 patches
against a single redshift scalar. Evaluated on the canonical held-out split
(§5), this design showed two structural failure modes rather than mere
noise: a regression-to-the-mean bias (σ_NMAD 0.083; the SmoothL1 optimum on
a heavily skewed `z` distribution is pulled toward the bulk of the training
labels, so line-confusion ambiguities are averaged instead of resolved) and
a hard prediction ceiling (max `z_pred` 2.00), which together made 82.7 % of
the spectra with `z ∈ [1.5, 2.5)` catastrophic outliers and gave η₀.₁₅ 22.6 %
overall.

**v2.1 — normalized classification loss (shipped).** The head now outputs 100
logits over bins uniform in `log(1+z)` on `[0, log(1+6)]` (`z_max = 6.0`), and
the redshift term is a class-weighted cross-entropy normalized by `log(n_bins)`:

```
loss = 1.0 * MSE(masked patches) + 1.0 * CE_w(logits, bin(log(1+z))) / log(100)

w_b  = clip( sqrt(mean_count / count_b), 0.5, 3.0 )    for bins with count_b > 0
```

- The bin counts come from the real histogram of the 80,000 training labels
  (`scripts/estimate_z_histogram.py` → `runs/calibration/z_hist_80k_100bins.npz`;
  91 of the 100 bins are populated). Sqrt-inverse weighting with the
  `[0.5, 3.0]` clamp lifts rare high-`z` bins without letting them dominate;
  bins with no training examples receive weight 0. Label smoothing is 0.
- Dividing the cross-entropy by `log(100)` puts it on the same scale as the
  reconstruction MSE (a uniform posterior scores exactly 1.0), which is what
  makes the 1 : 1 weighting between the two losses meaningful.
- The posterior `p` yields three outputs (§3): `z_pred_map` (argmax bin
  centre, official), `z_pred` (`expm1(E[log(1+z)])`, compatibility) and
  `z_confidence = 1 − H(p)/log(100)`.

Why classification: the posterior can be multimodal (e.g. a line
misidentification produces two candidate redshifts), so the argmax picks a
mode instead of averaging between them; the prediction range is set
explicitly by `z_max` instead of an implicit ceiling; and the posterior
entropy gives a confidence estimate at no extra cost. The measured effect is
in §5.

v2.1 is a **fine-tune**: training starts from
`runs/desi_50k_big/checkpoint_last.pt` (`--init-checkpoint`), the warm start
loads 115 of the 117 tensors and re-initialises only the final layer of the
redshift head (`redshift_head.4.*`, whose output size changed from 1 to 100;
recorded in `train.log`). This accounts for the parameter difference
25,929,859 → 25,980,646.

An intermediate **v2.0** experiment (200 bins, label smoothing 0.05,
un-normalized cross-entropy with weight 10 — so the redshift term was ≈ 98 %
of the total loss — and aggressive inverse-frequency rebalancing clamped to
[0.3, 10]) produced flat posteriors and η_MAP of 31.1% on a different window; this is
not directly comparable to canonical v1. It was not promoted (§5). That 31.1 % is diagnostic only: the
v2.0 run used the older split logic (shuffle before the cut), so part of the
2,000-spectrum window may have been seen in its training.

### 4b. Tokenization (oral-question topic)

Each spectrum is interpolated onto a fixed 7,081-pixel grid spanning
3600–9800 Å. The grid is **logarithmic in wavelength**
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
log-wavelength**, normalized to [0, 1] over the model's 3600–9800 Å range.
An optional learned positional embedding is added on top. The redshift
token's wavelength embedding is exactly zero, so it cannot be confused
with any spectral position.

This is what gives the model a chance of generalizing to non-DESI spectra.
A token's representation is anchored to a physical wavelength, not to an
arbitrary index. A non-DESI spectrum covering, e.g., 4500–7500 Å will be
interpolated onto the model grid; the tokens whose physical wavelengths
fall inside that overlap region receive valid input, the others receive
zero input with `valid=0`, and the transformer is told via the embedding
what wavelength each token corresponds to.

---

## 5. Achieved metrics (canonical held-out split, 2,000 DESI spectra)

**Protocol.** The `MultimodalUniverse/desi` (`edr_sv3`) stream is filtered to
examples with a valid redshift label; the first 80,000 are the training set,
the next 2,000 are the canonical held-out split
(`filter → skip(80000) → take(2000)`, shuffling applied to the training
window only). Both v1 and v2.1 are evaluated on exactly these spectra with
`desi_fm.evaluate` (`--skip-examples 80000 --max-examples 2000`), under the
model's training mask ratio of 0.5, so the reported reconstruction RMSE is on
masked patches. For v2.1 all redshift metrics use `z_pred_map`.

These objects were excluded from v2.1 fine-tuning. Historical v1 loader
reconstruction strongly supports prior exposure, and the same objects selected
checkpoints. Treat this as **non-independent historical validation**; the effect
of exposure has not been quantified. Best and last contain the same 117 tensors
at step 30,000; their 15.15% and 14.95% evaluations differ in random masks.
This is not a confidence interval or evidence of better last weights.
Several changes separate v1 and v2.1; there is no causal ablation here.

| metric (held-out, `z_pred_map` for v2.1)                     | v1 baseline | **v2.1 (shipped)** |
| ------------------------------------------------------------ | ----------- | ------------------ |
| catastrophic outlier fraction η₀.₁₅                          | 22.6 %      | **14.95 %**        |
| σ_NMAD                                                       | 0.083       | **0.030**          |
| `redshift_mae_norm` = `mean(abs(z_pred - z) / (1 + z))`      | 0.107       | **0.096**          |
| η₀.₁₅ in z ∈ [1.5, 2.5)                                      | 82.7 %      | **23.5 %**         |
| maximum observed prediction (configured z_max = 6)                              | 2.00        | **3.52**           |
| `reconstruction_rmse_masked` (pixel-weighted, arcsinh space) | 0.819       | 0.817          |
| mean `z_confidence`                                          | —           | 0.52               |
| trainable parameters                                         | 25,929,859  | 25,980,646         |

The machine-readable gate-by-gate comparison of both v2.1 checkpoints against
the v1 baseline (five gates: η₀.₁₅, σ_NMAD, MAE_norm, η₀.₁₅ at high `z`,
reconstruction RMSE) is in `runs/desi_80k_classhead_v21/comparison.json`
(`decision: promote_v2_1`). The v1 held-out run is in
`runs/calibration/metrics_v1_heldout_canonical.json`; older v1 figures
measured on partially seen validation windows are superseded by the numbers
above and should not be used as the baseline.

Retrospectively, `z_confidence ≥ 0.5` retains 1,175/2,000 objects with 9 errors
(58.75% coverage, 0.77% error; AUROC 0.9038). This is an entropy-based score,
not calibrated success probability or a validated service policy. Reconstruction
is about 0.82; comparison with a visible-only moving average on 50 stored
examples is exploratory, with widths selected on those examples. It establishes
neither superiority nor equivalence. [Evidence](docs/HISTORICAL_EVIDENCE.md).

Two evaluation protocols appear in the project's public material and are
**not comparable**: the 14.95 % η₀.₁₅ above (masked evaluation, 2,000
spectra) and the 92.7 % of predictions within 0.15 reported for the
companion `spectra-copilot` evaluation (150 held-out cases, full unmasked
spectra through the tools).

For comparison, the DESI pipeline reports `z` accuracies of order `1e-4`.
Our model remains orders of magnitude worse on `z`, which is the expected
gap between a 26 M-parameter transformer trained on 80 k spectra on a laptop
and the DESI production pipeline. The point of the project is not to beat
the pipeline but to show that a unimodal, spectrum-only transformer with
the redshift mechanism redesigned can learn the task — while retaining the documented limits of the historical evaluation.

### Progression during the project

The first row uses its own earlier validation window. Only the canonical v1
and v2.1 rows use the same 2,000 objects. The v2.0 CSV uses the older raw-skip
window; the CSV labels/order differ from the canonical window. It is diagnostic
and cannot be compared directly. Small smoke runs still exist locally and are
inventoried by hash; omission from this table does not mean their artifacts
were lost.

| run                                                                 | data     | params     | `redshift_mae_norm` | η₀.₁₅        | `recon_rmse_masked` |
| ------------------------------------------------------------------- | -------- | ---------- | ------------------- | ------------ | ------------------- |
| 50 k, `w_z=10`, mask 0.50, d_model 384 / 6 layers (`runs/desi_50k_zw10_mask50`) | 50 k | 11,081,603 | 0.130 (val window)  | —            | 0.862               |
| 50 k, `w_z=10`, mask 0.50, larger model (**v1**, `runs/desi_50k_big`) | 50 k   | 25,929,859 | 0.107 (held-out)    | 22.6 %       | 0.819               |
| v2.0 experiment: 200 bins, cap-10 rebalance, `w_z=10`, 2 epochs (diagnostic; see note above) | 80 k | 26,031,946 | 0.182 (MAP) | 31.1 % (MAP) | 0.877 |
| **v2.1 fine-tune: 100 bins, normalized CE, 3 epochs** (shipped)     | **80 k** | 25,980,646 | **0.096 (MAP)**     | **14.95 %**  | **0.817**           |

The v2.0 experiment lives in `runs/desi_150k_classhead/` despite using 80k
examples ×2 epochs. It completed 20,000 updates but was not promoted. Its
artifacts remain local and ignored; hashes are recorded. v2.1 met the later
promotion criteria, not the initial η<10% goal. The remaining error cannot be
attributed solely to capacity or compute from these experiments.

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

All flag names below are those of `src/desi_fm/train.py`,
`scripts/estimate_z_histogram.py` and `src/desi_fm/evaluate.py`; the values
match `runs/desi_80k_classhead_v21/training_args.json`. Because v2.1 is a
fine-tune, the v1 checkpoint (`runs/desi_50k_big/checkpoint_last.pt`, not
distributed) must exist first — see the historical baseline command at the
end of this section.

**Step 1 — class-weight histogram** (`runs/calibration/*.npz` is gitignored, so
regenerate it; streams the redshift labels of the first 80,000 valid-label
examples):

```bash
python3 scripts/estimate_z_histogram.py \
  --dataset MultimodalUniverse/desi \
  --data-dir edr_sv3 \
  --max-examples 80000 \
  --n-z-bins 100 \
  --z-max 6.0 \
  --output runs/calibration/z_hist_80k_100bins.npz
```

**Step 2 — v2.1 fine-tune** (the shipped checkpoint; 3 epochs × 80,000
examples at batch 8 = 30,000 steps; architecture flags are stated explicitly
because the script defaults are smaller than v1):

```bash
python3 -m desi_fm.train \
  --dataset MultimodalUniverse/desi \
  --data-dir edr_sv3 \
  --max-train-examples 80000 \
  --val-examples 2000 \
  --epochs 3 \
  --batch-size 8 \
  --lr 1e-4 \
  --lr-schedule cosine \
  --warmup-ratio 0.05 \
  --min-lr-ratio 0.1 \
  --weight-decay 0.05 \
  --grad-clip 1.0 \
  --d-model 512 \
  --n-layers 8 \
  --n-heads 8 \
  --mask-ratio 0.5 \
  --n-z-bins 100 \
  --z-max 6.0 \
  --z-label-smoothing 0 \
  --z-weighting sqrt_inverse \
  --z-histogram runs/calibration/z_hist_80k_100bins.npz \
  --z-weight-min 0.5 \
  --z-weight-cap 3.0 \
  --normalize-redshift-ce \
  --redshift-loss-weight 1.0 \
  --reconstruction-loss-weight 1.0 \
  --init-checkpoint runs/desi_50k_big/checkpoint_last.pt \
  --seed 42 \
  --log-every-steps 50 \
  --save-every-steps 5000 \
  --output-dir runs/desi_80k_classhead_v21
```

Approximate wall time on an Apple M-series MPS: ≈ 3.8 h including the three
validation passes. `metrics.jsonl` records the full loss trajectory.

**Step 3 — historical validation recipe** (random, unseeded 50% masking).
Use a new output directory. The saved `predictions.csv` equals last, but
`reconstructions.npz` came from the **best** evaluation under another mask;
the saved pair is not one forward pass. This command would create a new pair,
not reproduce the historical NPZ byte for byte:

```bash
python3 -m desi_fm.evaluate \
  --checkpoint runs/desi_80k_classhead_v21/checkpoint_last.pt \
  --dataset MultimodalUniverse/desi \
  --data-dir edr_sv3 \
  --skip-examples 80000 \
  --max-examples 2000 \
  --predictions-csv   runs/new_evaluation/predictions.csv \
  --reconstructions-npz runs/new_evaluation/reconstructions.npz \
  --num-reconstructions 50 \
  --metrics-json runs/new_evaluation/metrics.json
```

σ_NMAD, the high-`z` outlier fraction, the prediction ceiling and the release
gates in `comparison.json` are derived from the predictions CSVs in
`notebooks/evaluation.ipynb`.

**Baseline (historical) — v1**, trained from scratch (50,000 examples,
1 epoch at batch 4 = 12,500 steps; scalar regression head, `w_z = 10`):

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

Its canonical held-out evaluation (`runs/calibration/`) uses the same
`desi_fm.evaluate` invocation as Step 3 with `--checkpoint
runs/desi_50k_big/checkpoint_last.pt`.

---

## 8. References

- Project specification: PHYS303/CS486 final project (USF) — course material, not distributed in this repo
- Multimodal Universe: [https://github.com/MultimodalUniverse/MultimodalUniverse](https://github.com/MultimodalUniverse/MultimodalUniverse)
- AION-1 (reference, not reproduced): [https://github.com/PolymathicAI/AION](https://github.com/PolymathicAI/AION)
- AION-1 checkpoints: [https://huggingface.co/polymathic-ai/aion-base](https://huggingface.co/polymathic-ai/aion-base)
- This project — code: [https://github.com/Julian0444/desi-spectra-fm](https://github.com/Julian0444/desi-spectra-fm) · weights: [https://huggingface.co/jirustaroure/desi-spectra-fm](https://huggingface.co/jirustaroure/desi-spectra-fm) · demo: [https://huggingface.co/spaces/jirustaroure/desi-spectra-fm-demo](https://huggingface.co/spaces/jirustaroure/desi-spectra-fm-demo) · API: [https://jirustaroure-desi-fm-api.hf.space/api/docs](https://jirustaroure-desi-fm-api.hf.space/api/docs)
- Companion agent (tools, MCP server, similarity index, evaluations): [https://github.com/Julian0444/spectra-copilot](https://github.com/Julian0444/spectra-copilot)
