# Batch 4A — reproducible preparation, not the final experiment

Authority: consensus §7.5 and the 2026-10-10 objective. Only preparation and
train-only bounded profiles (at most 200 updates each of A, B, CNN) run here.
No final series, test predictions, paid services, remote data or agent changes.

| Ticket | Acceptance |
|---|---|
| 4A-01 | Verified original states and ancestry; immutable inputs; deterministic disjoint grouped splits; historical/demo IDs excluded from test; exclusions and rare strata accounted. |
| 4A-02 | Unnormalized local matrix outside Git; versioned IDs/config/hashes; parity with batch 3; truly random access timed. |
| 4A-03 | Paired A/B, fixed CNN <1M parameters and train-only ridge; explicit losses/recipe; full-spectrum validation and selection. |
| 4A-04 | Atomic compatible persistence, exact CPU interruption/resume, class weights/RNG/sampler retained; persisted metrics and paired analysis; explicit test gate. |
| 4A-05 | Bounded train profiles, historical preservation, suites, measured/extrapolated budget, local commits and runnable 4B commands; consensus appended. |

## Decisions registered before splits/materialization

Partition seed 20261009. Joint z bins [0,.5,1,1.5,2,2.5,infinity) and
ZWARN (False = original warning). All eligible warnings remain in train.
The 2,000 reconstructed historical IDs AND identifiable real demo IDs must be
in development (train or validation); propagate this rule to every member of
any observed relationship group. Synthetic demos do not represent snapshot IDs.
Target train/validation/test sizes 64,759/10,000/10,000. Group separation takes
priority. Deterministic largest-remainder quotas, stable ID order and one fixed
PCG64 seed; no reseeding to favor results. Rare strata are recorded, never dropped.

All seven schemas have object_id, Z/ZERR, ZWARN, EBV and photometric fluxes;
no sky coordinates, TARGETID, source or parent field. IDs are unique. Inspection
of exact photometric vectors found only two pairs with -99/0 fillers and matching
EBV; all four have negative labels. These are not established relationships.
No physical grouping key can be established from the available metadata. Use
object identity as the observed unit and explicitly retain this limitation;
singletons do not prove physical independence. The splitter supports supplied,
evidence-backed relationship groups and keeps mixed-stratum groups intact.

Local derived destination: ~/.local/share/spectra-data/batch4a-v1/. Immutable
raws remain in their existing directory. Store float32 unnormalized grid flux
and uint8 validity, 84,759 x 7,081: 3,000,892,395 payload bytes (~3.0 GB),
plus IDs/metadata (<50 MB). Train/validation/test have separate files, especially
labels; ordinary loaders cannot open test. One Parquet row group plus its grid
is expected below 1 GB working memory; random microbatches are <3 MB. Initial
materialization allowance 10 minutes (estimate, to replace with measurement).
Free disk before preparation ~70 GiB. Reserve 12 GB more for derivative,
checkpoint rotation and profile artefacts; future runs are budgeted separately.

Base preprocessing remains spectra-input-v2: log grid 3,600–9,800 Angstrom,
7,081 pixels, quality before sanitizing, no interpolation over invalid gaps.
Normalize each spectrum with median and IQR/1.349 (std then epsilon 1e-6 fallback),
arcsinh, clip [-8,8]. A/B statistics use ONLY valid visible target-grid pixels.
If fewer than eight visible pixels remain, deterministic center=0, scale=1;
record the fallback and retain the example. Hidden values never choose statistics.
CNN/ridge/full inference use all valid support. Dataset-level transforms and
class weights use train exclusively.

Transformer: historical-sized architecture (fresh weights), d=512, 8 layers,
8 heads, FF ratio 4, dropout .1, 273 tokens, 100 equal log(1+z) bins from 0 to
log(7). Learned and fixed wavelength positions, auxiliary head retained for B.
A/B differ only in reconstruction coefficient 1/0. Class labels bucketized at
interior edges (ties in the lower bin); out-of-range nonnegative labels enter
edge bins and are counted, never silently excluded. Evaluation uses original
labels. Negative/nonfinite labels excluded before partitioning.

Lz = sum(weight[target]*CE) / sum(weight[target]) / log(100), no smoothing.
Class weights = inverse sqrt(max(train histogram count,1)), normalized to mean 1.
Lrecon = SSE / number of artificially hidden valid pixels over the effective
batch; zero denominator contributes zero. A uses Lz+Lrecon, B Lz. Gradients
accumulate with effective-batch denominators, not an average of unequal means.

CNN: channels 3→32→64→128, kernel 9, stride 4, padding 4, GELU each; adaptive
average pool to 64, flatten and linear 8192→100. Input channels normalized flux,
validity and normalized log wavelength; full spectra in train and inference.
Ridge: 128 consecutive fixed grid bands, mean valid normalized flux (0 if none)
and valid fraction per band (256 features), standard scaling train-only,
intercept, log(1+z) target, alphas .1/1/10, at most three fits; validation eta15,
MAE then smaller alpha select. Predictions expm1 clipped below at zero.

Paired seeds 17/42/101. Stateless epoch permutations and masks keyed by seed,
absolute draw position and ID; Bernoulli .5 tokens, separate from model RNG.
AdamW lr 3e-4, betas (.9,.999), epsilon 1e-8, weight decay .01, clip norm 1.
Float32, microbatch 8, accumulation 4 (effective 32), workers 0, CPU threads 4.
Linear warmup 200 updates then cosine to .1 of peak at the fixed final update.
At most 30,000 updates/run; any recommended reduction is fixed before 4B.
Validation every 500 updates and at the final update on all validation IDs,
no artificial mask. Select lowest eta15 MAP, then MAE_norm, then earlier step.
Checkpoint every 500 updates and on interruption, retain last+previous+best.

Test is a separate command requiring a frozen nine-run + ridge manifest and
an explicit opening phrase. Ordinary train/profile/ridge reject test access.
Freeze after validation-only selection and record sensitivity assumptions
before final evaluation. No choosing a winning seed from test. Bootstrap
resamples observed groups/objects jointly across A/B and all three seeds;
CI is conditional on those seeds. All three Δη=etaA-etaB must be negative,
95% paired CI of their mean below 0, mean normalized MAE must not worsen.

Implementation evidence, measurements, limitations and exact commands are
added below after acceptance. This document never declares the whole lot 4 done.

## Acceptance and measured scope

Materialized counts: **64,759 / 10,000 / 10,000**, seed unchanged. All 84,759
eligible IDs occur once, without quality filtering. 15,241 negative-label
exclusions remain explicitly listed; four also lack sufficient input support.
All 2,000 historical IDs and the four distinct real demo IDs (already among the
2,000) are in development. Copilot's `trap_single_line` is synthetic according to
its existing generator and README, with no snapshot ID. No physical relations
could be established beyond identity; this remains a scientific limitation.

The allocation has no impossible/zero quotas (`rare_strata` in its allocation
report). **Statistical sparsity still matters:** warning objects at z [2,2.5)
number 12/2/2 and at z >=2.5 number 23/4/4 in train/validation/test. These counts
are separately flagged in `batch4-sparse-strata.json`; neither stratum was
merged or dropped. Test counts are metadata, not measured model errors.

The derivative occupies **3,012,425,410 bytes** including IDs and metadata.
Preparation took **52.623 s**, including **47.320 s** of materialization/hashing.
Every split has distinct flux, validity, ID and label/group files, with hashes.
Raw files and the original index were reused unchanged. The parity check loads
`data.py` directly from original batch-3 commit `008ce03`, avoiding a comparison
between two wrappers around only the new implementation. Sixty train objects
cover all seven shards, both quality values, worst coverage and high z plus a
random sample: full and masked preprocessing, masks/support and statistics
match exactly. Reading those IDs with the batch-3 reader took 5.742 s; derived
full preprocessing 0.0108 s. A separate 2,048-ID random gather took 0.0266 s for
arrays and 0.3621 s for reading + full normalization. The OS cache was warm;
these are real random accesses, not cold-disk guarantees.

A/B have **25,980,646** parameters; CNN **912,548**. CPU interruption/resume was
bit-exact for model tensors, optimizer, scheduler, class weights and losses.
The separate small synthetic CPU/MPS trajectories also had maximum tensor
difference 0, with identical sample/mask trace and non-unit class weights.
The predeclared numeric tolerance was atol 1e-6, rtol 1e-5. Only this small MPS
case was checked; full-size MPS training is not promised bit-exact. Tests cover
SIGINT, failed atomic writes, corrupted-last fallback, incompatible resumes,
weighted effective-batch accumulation, full inference, tie-breaks, paired
bootstrap accounting, train-only scalers and the final evaluator's test gate.
Historical `z_bin_weights` remains nonpersistent; only new experiment
checkpoints save and verify it explicitly, so old weights load unchanged.

### Profile and budget decision (before any 4B series)

Completed **32 train-only updates each for A, B and CNN**, seed 17, MPS, microbatch
8 × accumulation 4, float32, four CPU threads. Each evaluated 128 train objects
with full spectra solely to measure cost. No validation or test predictions
were made. Profile checkpoints are isolated from final runs and cannot resume
as training runs. Numerical train metrics in profile artifacts are smoke data,
not scientific results or a hyperparameter selection.

The initial CNN attempt stopped before the first optimizer update because MPS
cannot perform non-divisible adaptive pooling (111→64). The final implementation
uses the same floor/ceil average windows as a fixed nontrainable matrix, entirely
on-device. Forward and gradient parity with PyTorch's CPU operation pass at
atol/rtol 1e-6 for divisible, non-divisible and upsampling cases. The failed
output remains preserved. A/B profiles use `9d6adb6`; CNN uses pooling fix
`52e8d0d`. Transformer implementation and parameters did not change in that fix.

The first four updates are excluded only from the steady timing estimator;
all timings, startup, checkpoint writing/recovery and memory remain in manifests.
Device synchronization brackets measured compute. Source and config identities,
train class histogram/weights, exact draw/mask trace and environment are recorded.
Hardware: local Apple M4 Pro, 48 GiB unified memory, as recorded in acceptance.

**Recommend and preconfigure 10,000 updates for every neural run**, with all
three seeds and both baselines retained. This is 320,000 draws/run, about 4.94
passes over train; it is a budget choice, not a guarantee of convergence. The
profile retains its original 30,000-update scheduler horizon in
`configs/batch4-profile.json`; the future common 10,000-update cosine schedule is
in `configs/batch4.json`. Warmup 200 and validation every 500 remain fixed.

| Cost component, 10,000 updates/run | Estimate from profile |
|---|---:|
| Six transformer runs | 8.356 h |
| Three CNN runs | 9.85 min |
| Ridge train features + three fits | 13.0 s |
| Ridge validation | 2.0 s |
| 20 complete validations per neural run | 1.449 h |
| Checkpoint writes | 64.5 s |
| One final evaluation of nine checkpoints + ridge | 262.7 s |
| Persisted predictions | 53.6 s |
| Startup verification and final analysis allowance | 300 s |
| **Total estimated** | **10.16 h** |
| **Planning allowance, 50% margin** | **15.24 h** |

At the allowed 30,000-step ceiling, the same calculation is **30.16 h**, or
**45.24 h** with margin. These are extrapolations from short train profiles,
not measured full-series durations. Validation/test costs are estimated from
train only. Thermal throttling, competing load and long-run behavior are not
measured. Ridge cost uses a 1,024-train-object feature/three-fit probe; no
validation regularization choice was made here. Reserve another **12 GB** for
4B, including retained checkpoints, temporary atomic writes and prediction
artifacts. `batch4-budget.json` contains rates, component formulas, measured
memory and checkpoint sizes/times; regenerate with `scripts/budget_batch4.py`.

Exact start, interrupt, resume, ridge, freeze, final-test and analysis commands:
[BATCH_4B_COMMANDS.md](BATCH_4B_COMMANDS.md). Freeze recomputes selection from
persisted full-validation predictions at every prescribed point, verifies the
nine-run set and A/B traces, and records sensitivity assumptions using validation
before test opening. The final test command registers one output and requires
`OPEN_FROZEN_BATCH4_TEST_ONCE`; ordinary profile/train have no test option.
All three seed differences, their paired object/group interval (conditional on
these seeds) and mean MAE must meet the original criterion. No seed is selected
using test. Unknown physical relationships and sparse strata limit interpretation.

4A prepares that experiment. **4B and the scientific conclusion remain pending.**

### Measurement detail and checks

| Variant | Steady median update | Forward mean | Backward mean | Optimizer mean | Host RSS max | MPS driver max sampled | Checkpoint write / recover |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 0.5014 s | 0.1784 s | 0.2716 s | 0.0474 s | 0.811 GB | 3.454 GB | 0.553 / 0.295 s |
| B | 0.5013 s | 0.1784 s | 0.2718 s | 0.0473 s | 0.738 GB | 3.456 GB | 0.409 / 0.268 s |
| CNN | 0.01970 s | 0.00631 s | 0.00550 s | 0.00262 s | 0.547 GB | 0.0722 GB | 0.0157 / 0.0133 s |

Host/device memory are separate observations on unified memory, not quantities
to add into a claimed independent total. All profile normalization fallback
counts were zero; synthetic tests exercise the insufficient-visible-support
policy. Actual full transformer initial parameter fingerprints match A/B for
**each of 17, 42 and 101**, and the 32-step real A/B draw/mask trace is identical.
Train weights/histograms are equal and explicitly preserved in checkpoints.

Complete FM suites: **89 passed** on Python 3.12.13 and **89 passed** on 3.9.6.
Copilot: **85 passed**. The final interrupt-at-target fix then passed the
**15 training regressions on both Python versions**; it changes stop status,
not numerical training or profile cost. No test was skipped. Existing PyTorch,
matplotlib and urllib3 warnings remain. `pip check`, shell syntax and diff
whitespace checks pass. All **74 FM + 169 Copilot historical files** retain
size and SHA-256, including the prior untracked CSV. Raw snapshot, index and
every derivative file were rehashed; the fixed-seed splits were regenerated
from metadata and match exactly. Historical allocation: 1,528 train, 472
validation, **0 test**. The real test opening registration is absent.

Evidence: `docs/evidence/batch4-{acceptance,validation,derivative,splits,
partition-decision,data-parity,resume-check,budget,profile-A,profile-B,
profile-CNN,sparse-strata}.json` and `batch4-dependencies.txt`. Heavy artifacts
and logs are outside Git in `~/.local/state/spectra-consolidation/lote-4a-20261010/`.
The incomplete CNN attempt is retained. New derivatives live under
`~/.local/share/spectra-data/batch4a-v1/` and are identified by
manifest SHA-256 `4c412f03920cd12afb167ed6652fa6cb01a0f82d0c0d349e47a21590f9534547`.

Local implementation commits: `d7de1b3` splits/grid; `9d6adb6` paired models,
ridge, persistence and final guard; `52e8d0d` equivalent pooling for MPS;
`182761a` interruption at the final step and accumulated-loss regression.
Copilot requires no implementation change and remains at its previous commit,
on the requested `codex/lote-4a` branch. Final delivery/evidence commit and both
working-tree statuses are recorded in the consensus execution appendix.

What Julian should be able to explain: why an ID split is weaker than verified
physical independence; why historical/demo exclusion applies before sampling;
why normalization must follow the artificial mask; why B keeps its auxiliary
head and identical masks; how weighted CE and hidden-pixel MSE are normalized;
why CNN/ridge are practical controls whereas only A/B isolates reconstruction;
what optimizer/RNG/draw state resume restores; and why three seed predictions
on one object are not three independent bootstrap observations.
