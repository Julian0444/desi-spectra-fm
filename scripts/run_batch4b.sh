#!/bin/sh
# FUTURE 4B ONLY. A new destination starts nine fresh runs; --resume continues it.
set -eu
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo 'Usage: sh scripts/run_batch4b.sh /absolute/new-run-root [--resume]' >&2
    exit 2
fi
spectra_batch4_root=$(cd "$(dirname "$0")/.." && pwd)
spectra_python=${SPECTRA_PYTHON:-/Users/jirustaroure/.local/state/spectra-consolidation/lote-2-20261009/wheel-env/bin/python}
spectra_data=${SPECTRA_BATCH4_DATA:-/Users/jirustaroure/.local/share/spectra-data/batch4a-v1}
spectra_output=$1
spectra_resume=${2:-}
if [ -n "$spectra_resume" ] && [ "$spectra_resume" != '--resume' ]; then
    echo 'Second argument must be --resume' >&2
    exit 2
fi
if [ -z "$spectra_resume" ]; then
    mkdir "$spectra_output"
else
    test -d "$spectra_output"
fi
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH="$spectra_batch4_root/src"
for spectra_variant in A B CNN; do
    for spectra_seed in 17 42 101; do
        spectra_run="$spectra_output/$spectra_variant-$spectra_seed"
        if [ -d "$spectra_run" ]; then
            "$spectra_python" "$spectra_batch4_root/scripts/batch4.py" train \
                --data "$spectra_data" --config "$spectra_batch4_root/configs/batch4.json" \
                --variant "$spectra_variant" --seed "$spectra_seed" --output "$spectra_run" \
                --device mps --authorize-series BATCH4B --resume
        else
            "$spectra_python" "$spectra_batch4_root/scripts/batch4.py" train \
                --data "$spectra_data" --config "$spectra_batch4_root/configs/batch4.json" \
                --variant "$spectra_variant" --seed "$spectra_seed" --output "$spectra_run" \
                --device mps --authorize-series BATCH4B
        fi
    done
done
# Ridge and the test are separate explicit commands. Ctrl-C stops this script
# after the active neural update is checkpointed; the runner returns code 130.
