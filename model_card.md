---
license: mit
tags:
  - astronomy
  - spectroscopy
  - transformer
  - masked-modeling
  - redshift
  - pytorch
---

# DESI Spectra FM — multitask transformer (26M)

**[▶ Live demo](https://huggingface.co/spaces/jirustaroure/desi-spectra-fm-demo)** —
try the model on historical DESI examples (or your own) in the browser. The
historical examples are not an independent validation population.

Encoder-only transformer (8 layers, `d_model=512`, 8 heads, ~26M parameters)
trained with masked-token prediction on DESI EDR/SV3 spectra
([Multimodal Universe](https://github.com/MultimodalUniverse/MultimodalUniverse)).
The redshift token is **always masked** and a prediction head is trained
**jointly** with reconstruction, so supervised redshift learning is part of
the representation from the start. The historical repository name does not
establish foundation-model capabilities or transfer.

**Current checkpoint: v2.1** — a **fine-tune of the v1 encoder** (not a from-scratch
model) with a new classification head over 100 `log(1+z)` bins: cross-entropy
normalized by `log(n_bins)`, sqrt-inverse class weights estimated from the real 80k
training labels, 1:1 loss weighting with reconstruction, and a split excluded from v2.1 fine-tuning but not independent of the v1 lineage. It replaced v1 after meeting the later promotion criteria (the initial target of η < 10% was not met) (`comparison.json`, `decision: promote_v2_1`). The official redshift prediction
is **`z_pred_map`** (posterior argmax); `z_pred` (`expm1(E[log(1+z)])`) is kept for
backward compatibility.

## Historical validation metrics (not an independent test)

Historical measurement on the same 2,000 valid-label DESI EDR/SV3 objects after
the first 80,000 valid rows. These objects were excluded from v2.1 fine-tuning,
but reconstructed v1 loader evidence strongly supports prior exposure. They also
served checkpoint selection; **this is not an independent test**.

Both evaluations randomly hide 50% of spectral tokens without a fixed seed.
v1 uses scalar z; v2.1 uses MAP from `checkpoint_last.pt` at step 30,000.
Best and last have identical tensors; 15.15% versus 14.95% comes from different
masks, not different model quality. This difference is not an uncertainty interval.
[Evaluation](docs/EVALUATION.md) is the canonical results overview;
[historical evidence](docs/HISTORICAL_EVIDENCE.md) records hashes and recalculation.

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
# Source installation while 0.3.0 publication is pending.
pip install "desi-fm @ git+https://github.com/Julian0444/desi-spectra-fm.git"
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

Batch 3 measured **240/2,000 outliers (12.00%)** with full spectra and input
contract v2. This is a historical diagnostic on a non-independent population,
not a generalization estimate. The earlier stratified 150-case baseline had
11 errors (7.33%); 6.54% reweighting and 8.19% paired-difference estimates are
historical antecedents superseded by the complete measurement. The agent
reduced successes from 139/150 to 119/150. See [batch 3](docs/BATCH_3_RESULTS.md).
The 12% cannot be compared to CNN's 9.29% on the different batch-4B test;
v2.1 was not evaluated there. v2.1 remains the served predictor.

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
[spectra-copilot](https://github.com/Julian0444/spectra-copilot/blob/main/docs/DECISIONS.md).

## Links

- Live demo (HF Space): <https://huggingface.co/spaces/jirustaroure/desi-spectra-fm-demo>
- REST API (FastAPI + Swagger docs): <https://jirustaroure-desi-fm-api.hf.space/api/docs>
- Code, tests, CI and evaluation notebook: <https://github.com/Julian0444/desi-spectra-fm>

## License and data attribution

Code: MIT. DESI data: CC BY 4.0, with EDR citation, modification notice and acknowledgments in [ATTRIBUTIONS.md](ATTRIBUTIONS.md). These corrections await publication.


## Local package verification, 5C

Software version 0.3.0 packages the current input contract and local demo.
Installed-wheel checks cover imports/resources, CLI, API upload/JSON/rejection,
Gradio upload/predict and all five Copilot MCP tools. They do not change weights
or the validation populations above. The local demo requires an explicit local
checkpoint and performs no download. [Reproduction](docs/BATCH_5C_REPRODUCE.md).

A/B figures were regenerated from saved predictions, with no test inference:
A lowers outliers in all three frozen pairs, while normalized MAE worsens for
seeds 17 and 101. [Evidence and limits](docs/BATCH_5C_AB.md). The historical name
does not establish a foundation model; physical independence and CNN/v2.1
superiority remain unproven. Scientific acceptance and publication are separate
from local software checks; [current status](docs/BATCH_5C_REPORT.md).


## Evaluation and deployment status

The controlled A/B experiment met its aggregate criterion on the three frozen
checkpoint pairs, but reconstruction worsened normalized MAE in two pairs and
worsened the warning stratum. CNN and ridge are practical baselines; v2.1 was
not evaluated on that test. CNN-17 remains experimental. [Full protocol and
limits](docs/EVALUATION.md).

The completed 5C pilot selected low effort, with 11/11 valid reports passing
the operational gate. The unique low-effort acceptance is sealed: 128 requested
(80 real, 48 synthetic), 126 calls, zero retries, US$0.1086672 certified cost.
Mechanical checks passed: completion 127/128 (79/80 real, 48/48 synthetic),
model_z 122/122 eligible, prediction coverage 122/128 requested, evidence 909/909
and required abstention 18/18. One incomplete explanation remains in the result.
The corrected procedure and evaluated identity passed independent review with
zero P1/P2 before execution. Both new AI annotations are complete and were
hash-registered before either was read by the coordinator or labels opened.
Their 894 claim disagreements and 122 incomplete inventories leave all 128
cases semantically pending. The complete adjudication package has no human
decisions. Semantic review is by AI; Opus and Haiku share a provider. C_pair=0/0
is N/A, never 100% support. Procedure deviations and validator limits are
reported in [Evaluation](docs/EVALUATION.md).
Scientific accuracy and global acceptance remain pending human adjudication.
The earlier failed first call is preserved as an integration interruption with
no behavioral evidence. [Current 5C state](docs/BATCH_5C_REPORT.md).

Software 0.3.0 and the hosted apps are separate artifacts. This release edits
no Space app.py or requirements file. Both Spaces install desi-fm from GitHub's
unpinned default branch; after publication, a rebuild can install 0.3.0 under
the existing app.py, retain a cached dependency, or fail. A Space README commit
triggers rebuild/restart. No unchanged runtime or updated demo code is claimed.
The built-in demo wording predates this evaluation; its claims of unseen
examples and foundation-model capability are superseded by the limits here.

Development and audits used AI assistants (Codex and Claude), with decisions and
verifications recorded in docs. This does not describe personal contributions;
H-18 remains pending.
