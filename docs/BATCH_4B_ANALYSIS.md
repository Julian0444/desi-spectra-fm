# Batch 4B analysis plan, registered before final test

Use persisted predictions only. No fitting, threshold selection or test-driven
method changes. Training/selection and paired estimands remain those of §7.5.

## Future-use candidate decision (validation only)

Compare A, B and CNN by the arithmetic mean of the three selected checkpoints'
validation eta15, then mean normalized MAE. Compare ridge's selected validation
metrics on the same scale. If both metrics tie exactly, use lexical method name
as a deterministic administrative tie-break, without a superiority claim.
If a neural family wins, nominate its middle checkpoint in validation order
(eta15, MAE, step, seed), retaining all three seeds in every scientific report.
This is a validation-based candidate for future batch-5 assessment; no deployment
or change of the served model is authorized. Selected validation scores are
optimistic and this decision is never revised after reading test.

## Report outputs

- All ten final candidates: object/outlier denominators, eta15 with fixed-model
  group-bootstrap 95% interval, normalized MAE, NMAD and bias. Original-warning
  and no-original-warning denominators/metrics/eta intervals remain visible.
- The pre-existing z and input-coverage strata, including empty/rare strata;
  no merging or removal. Secondary expected-log-redshift metrics are labeled
  separately from primary MAP/scalar metrics.
- Every individual seed; family arithmetic means are descriptive summaries,
  not a pooled sample of 30,000 independent objects.
- A/B paired mean, three individual Δη values, paired 95% interval and mean
  MAE difference. The existing bootstrap samples observed groups jointly across
  both variants and all seeds. Apply the unchanged three-part criterion.
- CNN/ridge observed results relative to both transformer families. These
  comparisons include architecture/input differences, so they do not isolate
  the reconstruction intervention. Similar point estimates are not equivalence.
- Risk–coverage from the fixed confidence score for each neural checkpoint,
  without installing a threshold. Ridge has no defined confidence score.
- Validation trajectories, test eta intervals, paired effects and risk–coverage
  figures. All are regenerated from identified persisted artifacts.
- Timed read/forward/backward/optimizer, validation and checkpoint components;
  series wall time; per-run host RSS maximum and sampled MPS driver memory.
  Host/device memory observations on unified memory must not be added together.
  Sampling does not establish an absolute device-memory peak.

Use existing metric formulas, 2,000 fixed-checkpoint eta bootstrap replicates,
5,000 paired replicates, seed 20261009, and fixed z/coverage bins. Intervals are
conditional on the trained checkpoints/three seeds. Physical independence of
the observed singleton IDs is unverified. Sparse warning strata limit precision;
percentile intervals for zero observed events can collapse and do not establish
zero population risk. No multiplicity-adjusted discovery claims by stratum.

## Verification before opening test

Verify all 20 full-validation prediction sets per neural run, all 10,000 updates,
selection tie-breaks, exact paired A/B draw/mask traces and class weights,
checkpoint hashes, common config/source/environment/device, three ridge fits
and train-only scaling. Persist sensitivity and the future-use decision, then
hash this plan and the report generator alongside the frozen manifest.
