# Historical evidence — consolidation batch 1

Local revision prepared on 2026-10-09 (America/Los_Angeles). No retraining,
dataset download, new scientific evaluation or paid LLM execution in this batch.
The historical name “Foundation Model” does not establish transfer. This is a
multitask transformer for DESI redshift and masked reconstruction.

## Model identity and protocol

The original checkout started at `dadc83e8f0b855723d7a3a81cf4f24f1ce9b9047`.
v2.1 is a fine-tune of v1, with no external pretrained weights. Its local
`runs/desi_80k_classhead_v21/checkpoint_last.pt` is 103,964,688 bytes, SHA-256
`40247cbe0e706a66084d001b36424081554c26d3eb75c1f8b0ebec89cf5afe5a`.
Best and last contain 117 equal tensors, step 30,000 / epoch 2. The final log
records `done step=30000 elapsed_sec=13818.9`; that is completion evidence,
not a new timing measurement. v2.0 completed 20,000 steps without promotion.

[Historical asset inventory](evidence/historical-assets.json) contains relative
paths, sizes and SHA-256 for 74 local artifacts, including all 29 checkpoints,
run configs/logs, CSVs, reconstruction NPZs, histogram and examples. Large
files remain where they were; hashes identify them but are not a backup.
Do not overwrite those paths with new runs. The initial uncommitted user
changes and index/working patches have a separate private local backup; the
execution table in the official consensus records its location.

## Stored results and their limits

| Evidence | Sample, protocol, checkpoint | Verified result / limit |
|---|---|---|
| `runs/calibration/predictions_v1_heldout_canonical.csv` | 2,000 valid-label objects after first 80k; v1 scalar output; random 50% masking | 452 outliers, 22.60% |
| `runs/desi_80k_classhead_v21/predictions_last.csv` | Same 2,000; v2.1 last MAP; random 50% masking, seed not fixed | 299 outliers, 14.95%; not the normal full-spectrum serving protocol |
| `predictions_best.csv` | Same objects/weights, different random mask | 303 outliers, 15.15%; does not show that last weights are better |
| Entropy score on last CSV | Retrospective threshold ≥0.5 on those 2,000 | 1,175 retained, 9 errors: 58.75% coverage, 0.77% error; AUROC 0.9038; not calibrated success probability or a validated policy |
| Copilot `eval/results_baseline.csv` / `results.csv` | Same 150 stratified historical objects; full spectra; v2.1 / historical Haiku + five tools and RAG | 139 vs 119 successes; 21 broken, 1 recovered; hybrid 135/150, below baseline |
| Full-spectrum estimates | Six strata, quotas 40/35/30/25/10/10, selection seed 7 | 11/150 errors (7.33%) measured on subset; direct reweighting 6.54% and paired-difference 8.19% are estimates for 2,000, not full measurements |
| `reconstructions.npz` | 50 stored normalized examples, best evaluation mask | Model RMSE 0.81849; visible-only 301-pixel moving average 0.81688; width explored on same data, so neither equivalence nor superiority is established |

Outlier convention for model CSVs: `abs(z_pred-z_true)/(1+z_true) > 0.15`.
Historical Copilot success convention: strict `< 0.15`; its runner retains
that convention. No stored row lies on the boundary in these comparisons.

The 2,000 objects were excluded from v2.1 fine-tuning but also selected its
checkpoints. Reconstruction of the historical v1 loader strongly supports
prior exposure. Treat them as **non-independent diagnostic validation**. The
fingerprint probe used synthetic inputs and is not a recovered training-ID
log; neither the sign nor size of the exposure effect has been measured.
Historical loader provenance is preserved in the audit evidence, not inferred
from current split tests. Multiple changes between v1/v2.1 prevent attributing
all improvement to classification or reconstruction.

`predictions.csv` is byte-identical to last CSV; `reconstructions.npz` is
byte-identical to `reconstructions_best.npz`, with z outputs matching the
first 50 best rows. They are not a single forward pass. v2.0 uses another
validation window; it is not directly comparable with canonical v1/v2.1.

## Software facts and demonstrative evidence

- `src/desi_fm/model.py` computes `z_pred = expm1(E[log(1+z)])`, MAP at the
  most likely bin center, and `z_confidence = 1 − H(p)/log(100)`. Configured
  z_max is 6, last center ≈5.93; observed maximum ≈3.52 is not a model ceiling.
- UMAP visualizes 15,000 training objects from an encoder jointly supervised
  on redshift. Resampling to 3600–9800 Å does not establish accuracy on other
  instruments or incomplete coverage.
- Copilot has three of the four real demo examples; it lacks `heldout_z083`.
  z≈2.87 has normalized error ≈0.110 and confidence ≈0.354: within the 0.15
  tolerance, not a high-confidence claim. z≈0.20 error ≈0.019 is also not
  catastrophic. Filenames reveal labels; the demos are not blind tests.
- The Opus trap transcript abstains; the later Haiku/RAG trap is falsely
  confident after arithmetically wrong alternatives. Toolsets differ, so
  this is not a controlled model comparison. Weak matcher confirmation is
  associated with failures; its causal role remains a hypothesis.
- Citation IDs matched retrieved documents in three saved reports (6/6).
  This does not measure whether each scientific assertion is supported.
- Historical cost of US$4.06 is documented in the bitácora, not reconstructed
  from a CSV that lacks input/output/cache breakdown or a tariff version.

## Recalculate without inference

With NumPy and the original assets available:

```bash
python scripts/check_historical_evidence.py --copilot-root /path/to/spectra-copilot --check-weights
```

`--check-weights` additionally requires PyTorch and local best/last files. The
script reads existing CSV/NPZ/checkpoints, checks preservation hashes and prints
JSON. It does not run model forwards, touch results, access the network or call
an LLM. [Saved result](evidence/historical-metrics.json).

Ten selected audit probes/outputs were preserved outside `/tmp`, with local
paths sanitized and source/preserved hashes recorded in their manifest. Their
location is in the consensus execution table. Some historical probes require
audit-specific inputs; current maintenance uses this recalculation script and
Copilot `tests/test_eval_persistence.py` for B11. No broader audit was repeated.

## Scope still open

Batch 1 repairs result persistence and presentation. Shared input contracts,
empty scientific-loader handling, prompt facts/label clues, package data,
MCP domain validation and the explanatory-agent policy remain later-batch work.
They are not silently certified by passing existing tests. External publication
and account revocation remain separate, unverified actions.
