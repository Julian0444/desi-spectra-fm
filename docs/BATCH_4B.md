# Batch 4B — fixed experiment execution

Authority: consensus §7.5 and the user's 2026-10-10 execution objective.
Scope: R4A-01, nine fresh neural runs, three ridge fits, validation selection,
one frozen final evaluation, paired analysis and local documentation. No batch 5,
publication, FAISS rebuild or served-model change.

| Ticket | Acceptance | Status before training |
|---|---|---|
| 4B-01 | Reject incompatible device/environment before test registration, output or access; retain valid resume; commit synthetic regressions | Complete |
| 4B-02 | Verified inputs/environment/space; A, B, CNN × 17/42/101, 10,000 updates each; ridge alphas .1/1/10 | Pending |
| 4B-03 | Verify all validation checkpoints, A/B traces, weights and identities; record sensitivity and future-use decision; freeze | Pending |
| 4B-04 | Authorized frozen test on the same 10,000 IDs; predefined paired analysis and all controls | Pending |
| 4B-05 | Reproducible report/figures, observed costs, historical hashes, local commits and consensus update | Pending |

## R4A-01

The common experiment identity now includes device, environment and validation
identity. Freeze rejects heterogeneous collections and a changed current
environment. Final evaluation checks every candidate and checkpoint identity,
the current environment, requested device and its availability before registering
the opening, creating its output directory or loading any test data. The same
preflight applies on resume, including candidates with persisted predictions.
Ridge uses NumPy on the host CPU; it does not inherit a neural MPS device.
Its recorded Python/platform/dependency environment must match the experiment.

Ten new synthetic cases cover mixed CPU/MPS runs, mixed environments, ridge
environment mismatch, requested-device mismatch, changed current environment,
tampered frozen candidate/checkpoint identities, and interruption after three
predictions followed by resumption without rewriting those predictions.
Before correction: 9 failed, 1 passed. Afterwards: 29 relevant tests passed;
the complete FM suite passed 99 tests on Python 3.12.13 and 99 on Python 3.9.6.
The original independent mixed-device reproduction now fails at freeze with
`Runs do not share the same recipe/data/code/device/environment`.
Only synthetic data were used for these regressions.

Private evidence and the initial Git-state record:
`/Users/jirustaroure/.local/state/spectra-consolidation/lote-4b-preparation-20261010/`.
Both originals started on the expected batch-4A commits and now use
`codex/lote-4b`. The sole pre-existing FM change was the untracked historical
CSV; Copilot was clean and needs no implementation change.

## Fixed design and execution

The question is whether auxiliary reconstruction improves redshift under this
specific recipe. A/B retain identical architecture, input masks, initialization,
draw order, class weights, optimizer and budget; only reconstruction coefficient
1 versus 0 differs. CNN is a practical alternative with architecture and input
differences; ridge is a train-scaled linear control. All eligible warnings stay
in training. Normalization is visible-only for masked A/B inputs.

Use `configs/batch4.json`, the existing `batch4a-v1` derivatives and the recorded
Python 3.12 environment with source imports through `PYTHONPATH`. Each neural
run uses fresh weights, 10,000 updates, microbatch 8 × accumulation 4, and full
validation every 500 updates. All three seeds are retained. Ridge uses exactly
three train-only fits, with alpha selected on validation and no train+validation
refit. Commands and recovery are in [BATCH_4B_COMMANDS.md](BATCH_4B_COMMANDS.md).

Select each checkpoint by validation eta15 MAP, normalized MAE, then earlier
step. Before test, verify selection, register validation-based sensitivity and
any future-use candidate, then freeze all candidates and analysis. The final
test is authorized only for that compatible frozen experiment.

Predeclared criterion: for all three seeds, eta_A − eta_B must be negative;
the 95% paired interval of their mean must lie below zero; mean normalized MAE
must not worsen. Bootstrap observed objects/groups jointly across A/B and all
seeds; seeds are not additional observations. The interval is conditional on
these seeds, and physical independence remains unestablished in this snapshot.
Report all methods, quality/z/coverage strata and defined risk–coverage curves,
including negative or mixed findings. Lack of significance is not equivalence.

The accepted estimate is 10.1623 hours, or 15.2435 hours with a 50% allowance;
actual elapsed time and resources will be reported separately. Completion
requires the full experiment and verified documentation, not process launch.
