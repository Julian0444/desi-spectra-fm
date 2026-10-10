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
