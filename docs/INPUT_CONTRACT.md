# Spectrum input and inference contract v3 (batch 5A)

All preprocess, Python, CLI, REST, Gradio handlers and Copilot tools use
`desi_fm.inputs` and `preprocess_spectrum`. This changes validity/support,
not trained weights; historical metrics remain historical. Accuracy under
partial coverage and internal gaps has not been established.

## Arrays and units

- Real integer/floating arrays or rectangular numeric lists. Reject strings,
  objects, bool flux, complex numbers, scalars, ragged and empty containers.
- Single spectrum: `(P,)` or `(1,P)` at file/batch entry points; Python
  `predict_spectrum`, both Gradio handlers and tools require exactly one row.
  Multirow files use batch Python, REST upload or CLI. No silent first-row choice.
- Batch flux: `(N,P)`. Wavelength, ivar and mask must be exactly `(N,P)` or
  shared `(P,)`; only the latter broadcasts. `(1,P)` does not broadcast to N>1.
- `mask`: bool or integer flags, nonzero means invalid. `ivar`: inverse flux
  variance; only finite strictly positive values confer validity. It is a
  validity gate, not an extra model channel or statistical weight in serving.
- Legacy arrays/files without a unit declare Angstrom by this contract.
  Explicit `wavelength_unit="angstrom"` or `"nm"` is supported; nm converts by
  10. JSON has the same field; NPZ uses a scalar string. Units are never inferred
  from the values. Flux is normalized per spectrum and can use consistent units.

## Validity, grid and coverage

Validity is computed **before** replacing nonfinite flux: finite flux, finite
positive wavelength, finite positive ivar and no bad flag. Nonfinite wavelength
samples are unusable. Finite positive duplicate wavelengths are rejected.
Sorting permutes the grid and all associated fields together; equivalent grid
permutations preserve preprocessing and predictions.

Interpolate only between adjacent usable samples in that sorted input grid.
Exact usable samples remain usable. Intervals touching invalid samples and
outside support stay invalid; invalid values do not dilute neighboring signal.
A missing-coordinate interval larger than 1.5 times its smaller neighboring step
also remains invalid. This conservative local rule catches two gaps separated by one sample. It can
reject abruptly changing native sampling; uniformly sparse input alone cannot
reveal missing observations. This is not a physical coverage guarantee or an
instrument/unit guess.
Invalid model-grid flux is zero *with validity zero*, before robust median/IQR
and arcsinh normalization; the reconstruction loss excludes these pixels.

Require at least **8 usable input samples and 8 supported model-grid pixels**
for each spectrum. These are numerical prerequisites, **not calibrated scientific
quality thresholds**. Flat spectra can be valid inputs. Partial coverage is
accepted with explicit counts/fraction and min/max supported wavelength.
The min/max span alone does not imply that internal gaps are covered.
No usable coverage means a `SpectrumInputError`, without a prediction or neighbors.

Results include `coverage`, `valid_model_grid` (Python/CLI), and `model_identity`.
HTTP surfaces include coverage and identity; only domain input errors become
422. Upload limits remain 413. Checkpoint/internal errors propagate as server
failures. NPZ extras such as `z_true` are ignored by the shared scientific parser.
Gradio's historical human demo may display its label; the agent receives a
separate derived, opaque file containing only scientific inputs.

## Weights and inference mode

Checkpoint load uses `weights_only=True` and exact tensor keys/shapes. The only
allowed migration renames historical `pos_embed` to `learned_pos_embed` when the
latter is absent; strict loading checks all remaining keys/shapes. v1 and v2.1
local last checkpoints require no migration. Weights are not modified.

Predict/embed temporarily set **every module** to eval and restore each prior
training flag, including mixed train/eval submodules, even after failure.
Do not concurrently train and serve the same mutable model object. With
mask_ratio=0 dropout is disabled and predictions are deterministic; explicit
mask_ratio in [0,0.9] draws random token masks and is a distinct protocol.

Identity contains configuration, checkpoint SHA-256/step/migration, configured
redshift endpoints and decoded bin-center range. Caller-supplied models without
a checkpoint have explicit unknown checkpoint identity. Historical observed
maxima are separate evidence, never inferred from configured limits.
For classifiers `z_pred=expm1(E[log(1+z)])` from the same posterior as MAP;
`z_confidence=1-H(p)/log(n_bins)` measures concentration, not calibrated accuracy.

## Evaluation metrics

An empty evaluation fails before opening output artifacts. `n_examples` and
`n_batches` are explicit. Redshift errors/outlier fractions divide by examples;
losses retain historical equal-batch averaging (`loss_denominator=n_batches`).
`reconstruction_denominator` and `masked_valid_pixels` count actually hidden,
valid pixels. If zero, `reconstruction_rmse_masked=null` in JSON (Python `None`),
meaning not applicable, never perfect zero. Nonempty masked evaluations retain
the SSE/pixel RMSE formula. These changes do not perform a scientific evaluation.

## Acceptance protocol

Shared fixtures: `tests/contract_fixtures.py`; cross-interface checks in Copilot
`tests/test_interface_contract.py`. Seed 17, CPU, a small model with dropout 0.4,
mask_ratio=0; exact preprocessing, prediction absolute tolerance 1e-6. Tools and
demo text round to four decimals: tolerance 5.1e-5. Invalid inputs, gaps, masks,
partial coverage and reversed grids are checked through real handlers; UI host
layout/ZeroGPU allocation is simulated. MCP is separately checked over stdio.
Real-checkpoint smoke is bounded and distinct from these contract tests.

## Follow-up of batch-2 review (batch-3 branch)

All default checkpoint downloads (API/demo/Copilot/index-builder) use
`desi_fm.assets.download_asset`: immutable Hub revision
`40ff2ed09e5e1584303052e1a5dfe218a4a396b3`, size and SHA-256 checked before
loading. Published LFS identities were compared to the preserved local files.
The FAISS index and metadata resolve at that same revision, with both hashes
checked. This establishes release/byte identity; the historical index lacks a
checkpoint/preprocessing provenance record and is **not newly certified as
compatible with any encoder**. Local overrides remain explicit caller choices.
Batch 3 requires local weights with expected hashes, without remote fallback.
Old batch-2 wheels/manifests describe old bytes and must not be reused for this
source revision. No new wheel was built for this follow-up.

## Version boundary and historical reproduction (5A)

`spectra-input-v3` changes missing-coordinate gap support, including coordinates
discarded as nonfinite. It does not change mask/ivar semantics or weights.
The earlier `spectra-input-v2` results and `batch4-grid-v1` derivatives remain
immutable. Future materialization labels `batch4-grid-v2:unnormalized-spectra-input-v3`
and requires an index whose eligibility used v3. Current ExperimentData rejects
v1 derivatives before loading arrays. Raw ID lookup may still read a verified
v2 index: that does not certify eligibility for a new v3 experiment. No index
or derivative was rebuilt in 5A. No new scientific accuracy measurement exists
for v3, including with the unchanged default v2.1 weights.

Use the pinned historical source and environment described in [BATCH_5.md](BATCH_5.md)
for old runs, with original source-hash guards intact. Never rewrite run.json,
checkpoint identities or frozen manifests to accept current source.

Behavioral tests now use a genuinely absent-coordinate `gap` fixture, separate
from `masked`; double gaps, nonfinite coordinates, reversed/permuted inputs and
insufficient coverage are tested. The real demo/API/tools handlers are observed
at the model's validity input for mask-only and ivar-only cases. The Gradio mock
certifies handler propagation only; installed UI/hosting acceptance remains 5C.
