# Batch 3: local snapshot and historical diagnostic evaluation

Work is sequential in the original repositories on `codex/lote-3`, based on the
batch-2 delivery HEADs. No training or new independent test is part of this work.

| Ticket | Acceptance criterion |
|---|---|
| R2-A | Resolved provenance path outside opaque destination; symlink regression; historical inputs unchanged. |
| R2-B | Immutable verified Hub identities across tools/API/demo/index-builder; simulated resolution/rejection; index provenance uncertainty explicit. |
| L3-01 | Only seven locked parquet; space check; raw sizes/hashes/schema/rows; complete immutable snapshot; lossless ID index; duplicates/label/input eligibility separately counted. |
| L3-02 | Explicit ordered local IDs, exact raw/preprocessing parity by identity, missing/duplicate rejection; worker membership checked; historical reconstruction matched to preserved audit IDs and exported cases. |
| L3-03 | Frozen checkpoint identities/selection; parameters/tolerances fixed; bounded reading/preprocessing/forward/writing profile with hardware/memory; measured budget before full evaluation. |
| L3-04 | Full spectra v1 scalar/v2.1 MAP and secondary on same requested historical IDs; exclusions by ID; metrics/strata/paired transitions/risk-coverage from persisted predictions; separate fixed-mask reconstruction and visible-only baselines. |
| L3-05 | New output per run, traceable manifest/CSV/NPZ/figures; complete offline repeat with declared tolerances; tests and all 243 historical hashes rechecked; consensus append and local commits. |

## Protocol fixed before full inference

Snapshot: `MultimodalUniverse/desi`, `edr_sv3`, revision
`933b9056b93b0f3e7790ee28c43412b49c39e232`, lexical shard order. Historical
membership: filter finite nonnegative Z, then eligible positions `[80000,82000)`.
The original 2,000 CSVs lack object IDs; agreement of their labels alone is
insufficient. Recovered audit metadata and 150 exported NPZ IDs/arrays are
additional identity evidence; residual limits must be stated.

All requested IDs remain in the results or ID/cause exclusion records. No
replacement objects or fixed fictitious denominator. ZWARN is preserved:
False means an original warning, not an established wrong label.

Redshift: `spectra-input-v2`, no artificial mask, frozen local v1 last and v2.1
last. v2.1 best/last tensor identity is checked and only one is evaluated. The
contract changes support/normalization relative to historical preprocessing;
new differences cannot all be attributed to weights or removing the mask.

`r=(z_pred-z_true)/(1+z_true)`, MAE_norm=`mean(abs(r))`,
eta15=`mean(abs(r)>0.15)`, NMAD=`1.4826*median(abs(r-median(r)))`,
bias=`mean(r)`. Strict historical accuracy `<0.15` is recorded separately.
Main v2.1 result is MAP; log-space expectation is secondary. Strata: z edges
`[0,.5,1,1.5,2,2.5,inf]`, ZWARN, valid-grid coverage `[0,.5,.9,.99,1.000001]`.
Risk versus coverage ranks entropy concentration in descending stable order,
with an ID tie break, and reports every retained-prefix denominator. It does
not choose a validated threshold.

Reconstruction: SHA-256 seed by object ID and fixed seed `20261009`, Bernoulli
0.5 token masks, persisted and shared between models. Input support follows v2.
Median/IQR (std fallback, epsilon 1e-6) normalization statistics use only valid
pixels in visible tokens; transformed hidden targets are used only for scoring.
Normalize using those visible statistics, then replace hidden tokens in the
model. This is a separately identified reconstruction protocol, different from
historical full-target normalization. Baselines fixed now: global visible mean,
linear interpolation with nearest endpoint extrapolation, and weighted moving
average width 301 with global visible mean fallback where the window has no
visible pixels. No baseline tuning on reported targets. RMSE is in clipped
arcsinh space, `sqrt(sum(hidden_valid_squared_error)/hidden_valid_pixels)`;
zero scored pixels gives null. Rejected visible support is recorded by ID/cause.

Tolerances declared: raw arrays, preprocessing, ID membership, masks and rejection
records exact; CPU predictions and reconstruction floats absolute 1e-6, relative
1e-6; regenerated metrics absolute 1e-8. If measured device variation requires a
change, record it before the full run. Device and batch settings remain fixed
between the run and complete offline repeat. Offline verification blocks socket
connections in addition to setting HF offline variables.

Datasets, weights, SQLite indexes and bulk arrays/results stay outside Git.
Versioned light manifests identify their hashes and locations. Each new run uses
an exclusive destination and starts as incomplete; complete requires every
requested object accounted for and all result files persisted. Publication,
credential revocation and lots 4/5 remain outside this delivery.
