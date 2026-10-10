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
training labels, 1:1 loss weighting with reconstruction, and a split excluded from v2.1 fine-tuning but not independent of the v1 lineage. It replaced v1 after meeting the later promotion criteria (the initial target of η < 10% was not met) (`comparison.json`, `decision: promote_v2_1`). The official redshift prediction
is **`z_pred_map`** (posterior argmax); `z_pred` (`expm1(E[log(1+z)])`) is kept for
backward compatibility.

## Validation metrics (canonical held-out split)

Historical measurement on the same 2,000 valid-label DESI EDR/SV3 objects after
the first 80,000 valid rows. These objects were excluded from v2.1 fine-tuning,
but reconstructed v1 loader evidence strongly supports prior exposure. They also
served checkpoint selection; **this is not an independent test**.

Both evaluations randomly hide 50% of spectral tokens without a fixed seed.
v1 uses scalar z; v2.1 uses MAP from `checkpoint_last.pt` at step 30,000.
Best and last have identical tensors; 15.15% versus 14.95% comes from different
masks, not different model quality. This difference is not an uncertainty interval.
[Protocol, hashes and recalculation](docs/HISTORICAL_EVIDENCE.md)
are prepared locally for the next publication.

| metric | v1 (50k, regression head) | **v2.1 (fine-tune, 100-bin classification)** |
|---|---|---|
| catastrophic outliers η₀.₁₅ | 22.6 % | **14.95 %** |
| σ_NMAD | 0.083 | **0.030** |
| MAE_norm ⟨\|Δz\|/(1+z)⟩ | 0.107 | **0.096** |
| η₀.₁₅ in z ∈ [1.5, 2.5) | 82.7 % | **23.5 %** |
| maximum observed prediction (configured z_max = 6) | 2.00 | **3.52** |
| reconstruction RMSE (masked, arcsinh space) | 0.819 | 0.817 |

The full bias/outlier analysis (with plots) and the v1 → v2.1 comparison on this
split are in the
[evaluation notebook](https://github.com/Julian0444/desi-spectra-fm/blob/main/notebooks/evaluation.ipynb).

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

Other wavelength grids in Å can be resampled onto 3600–9800 Å (7,081 pixels).
This does not establish accuracy outside DESI or under partial coverage.

## Limitations

Historical, non-independent validation; multiple training changes prevent a
causal attribution to the classification head alone. Scientific use requires
independent validation. The maximum observed z ≈ 3.52 is not the configured
ceiling (z_max = 6; final bin center ≈ 5.93).

`z_confidence = 1 − H(p)/log(100)` is posterior concentration, not a calibrated
probability. Retrospectively, ≥0.5 keeps 1,175/2,000 objects and 9 outliers
(58.75% coverage, 0.77% error; AUROC 0.9038), on the same masked, non-independent
set. That threshold is not a validated service policy.

The full-spectrum Copilot baseline has 11/150 errors (7.33%) on a stratified
subset. Direct reweighting gives an estimate of 6.54%; a paired-difference
estimator gives 8.19%. Neither is a measurement of all 2,000 or independent
generalization. The agent reduced successes from 139/150 to 119/150.

Reconstruction RMSE remains about 0.82. On 50 stored examples the transformer
scored 0.81849 and a 301-pixel visible-only moving average 0.81688. Widths were
explored on those same examples, so this establishes neither equivalence nor
superiority. No physical denoising claim is supported.

## Files

| file | description |
|---|---|
| `checkpoint_last.pt` | model weights (state_dict) + training args |
| `config.json` | architecture configuration |
| `training_args.json` | exact training flags of the run |
| `metrics.jsonl` | per-step training/validation metrics |
| `faiss/spectra.faiss` | FAISS index (cosine) of 15k training-spectra embeddings from `embed_spectrum()` |
| `faiss/spectra_meta.npz` | catalog redshift per indexed row |

The `faiss/` pair powers the `find_similar_spectra` semantic-search tool in
[spectra-copilot](https://github.com/Julian0444/spectra-copilot#semantic-search-embeddings--faiss).

## Links

- Live demo (HF Space): <https://huggingface.co/spaces/jirustaroure/desi-spectra-fm-demo>
- REST API (FastAPI + Swagger docs): <https://jirustaroure-desi-fm-api.hf.space/api/docs>
- Code, tests, CI and evaluation notebook: <https://github.com/Julian0444/desi-spectra-fm>

## License and data attribution

Code: MIT. DESI data: CC BY 4.0, with EDR citation, modification notice and acknowledgments in [ATTRIBUTIONS.md](ATTRIBUTIONS.md). These corrections await publication.
