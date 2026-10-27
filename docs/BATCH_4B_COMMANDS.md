# Batch 4B execution commands and recovery reference

These commands were prepared in 4A and executed in 4B on `codex/lote-4b`.
The fixed series and final test are complete; see [BATCH_4B.md](BATCH_4B.md)
for results and the command to regenerate reports without training or inference.
The commands below document execution and recovery, not a new authorized series.
Source imports intentionally override the old batch-2 installed wheels.
Python 3.12 is the recorded environment; no installation or network access is
needed. No profile weights were reused.

## Start the fixed nine-run series

```sh
cd "<fm-repo>"
export SPECTRA_PYTHON="<private-state>/lote-2-20261009/wheel-env/bin/python"
export PYTHONPATH="<fm-repo>/src"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export SPECTRA_BATCH4_DATA="<local-home>/.local/share/spectra-data/batch4a-v1"
export SPECTRA_BATCH4_RUNS="<private-state>/lote-4b-01"
sh scripts/run_batch4b.sh "$SPECTRA_BATCH4_RUNS"
```

The destination must be new. This executes A/B/CNN × 17/42/101 sequentially,
from independent fresh random initialization, with the common budget in
`configs/batch4.json`. There is no test evaluation or automatic ridge fit here.
Do not modify the configuration, source or dependencies midway through a series;
resume checks their identities and the train/validation files.

## Interrupt and resume

Press **Ctrl-C** in the running terminal. Once the current optimizer update
finishes, the runner persists model, explicit class weights, AdamW, scheduler,
step, draw position, Python/NumPy/CPU/MPS RNG and selection state, then returns
exit 130 so the shell stops before starting another run. SIGTERM behaves alike.
A forced termination loses updates since the last durable checkpoint (at most
500 after startup); resume reads the last intact generation or its predecessor.
Do not delete lock/pointer/checkpoint files to force a restart.

```sh
sh scripts/run_batch4b.sh "$SPECTRA_BATCH4_RUNS" --resume
```

A single known interrupted run can also be resumed explicitly:

```sh
"$SPECTRA_PYTHON" scripts/batch4.py train \
  --data "$SPECTRA_BATCH4_DATA" --config configs/batch4.json \
  --variant A --seed 17 --output "$SPECTRA_BATCH4_RUNS/A-17" \
  --device mps --authorize-series BATCH4B --resume
```

## Fit ridge and freeze using validation only

```sh
"$SPECTRA_PYTHON" scripts/batch4.py ridge \
  --data "$SPECTRA_BATCH4_DATA" --output "$SPECTRA_BATCH4_RUNS/ridge" \
  --authorize-series BATCH4B
"$SPECTRA_PYTHON" scripts/batch4.py freeze \
  --data "$SPECTRA_BATCH4_DATA" --runs "$SPECTRA_BATCH4_RUNS" \
  --ridge "$SPECTRA_BATCH4_RUNS/ridge" --output "$SPECTRA_BATCH4_RUNS/frozen.json"
```

Ridge computes train-only means/stds and exactly three alpha fits, then selects
on full validation. Its inexpensive command uses a new directory and does not
support partial-fit resume; an interrupted directory is preserved. There is no
fit on test and no retraining on train+validation. Freeze requires nine complete
neural runs, identical A/B example/mask traces, the three-fit ridge report, and
every common validation checkpoint score. It rechecks selection, source and
hashes, and records a sensitivity calculation from validation discordance,
with assumptions, before opening test. Inspect that frozen protocol in 4B.

## Open the final test once, only after freezing in 4B

```sh
"$SPECTRA_PYTHON" scripts/batch4.py final-test \
  --frozen "$SPECTRA_BATCH4_RUNS/frozen.json" \
  --output "$SPECTRA_BATCH4_RUNS/final-test" --device mps \
  --open-test OPEN_FROZEN_BATCH4_TEST_ONCE
```

This exact opening phrase and the frozen nine-model manifest are both required.
The data directory registers one final evaluation destination. If interrupted,
use the same arguments plus `--resume`; already verified predictions are reused.
Another destination, changed freeze, changed source or mismatched device is
rejected. This is an accidental-use guard, not an operating-system access sandbox.
There is no test argument in ordinary training, smoke or profiling commands.

## Analyze persisted predictions

```sh
"$SPECTRA_PYTHON" scripts/batch4.py analyze \
  --predictions "$SPECTRA_BATCH4_RUNS/final-test" \
  --output "$SPECTRA_BATCH4_RUNS/paired-analysis.json"
```

No inference or fitting occurs here. All three seeds must align on ID, group,
label and ordering. One bootstrap group draw is shared across A/B and all seeds;
the reported sampling interval is conditional on those three seeds. Global,
quality and z-stratum denominators remain visible, including very sparse strata.
Report all seeds and CNN/ridge regardless of whether reconstruction helps.

## Reproduce only the bounded infrastructure checks

Use new destinations and the profile configuration preserved with 4A evidence.
The original profiles used 32 updates per variant, not a tuning sweep. A command
rejects more than 200 profile updates and can only load train.

```sh
"$SPECTRA_PYTHON" scripts/batch4.py profile \
  --data "$SPECTRA_BATCH4_DATA" --config configs/batch4-profile.json \
  --variant A --seed 17 --output <private-state>/profile-A-new \
  --device mps --updates 32 --evaluation-objects 128
"$SPECTRA_PYTHON" scripts/check_batch4_resume.py \
  --output <private-state>/resume-check-new
```

The CPU/MPS resume check uses small synthetic arrays and does not open real
validation or test data. `scripts/check_batch4_data.py --help` documents the
random-access parity check against the fixed batch-3 commit.
