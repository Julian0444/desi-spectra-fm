#!/bin/sh
# Local, offline scientific repetition. The output must be a NEW directory.
set -eu
if [ "$#" -ne 1 ]; then
    echo 'Usage: sh scripts/reproduce_batch3.sh /absolute/new-output-directory' >&2
    exit 2
fi
spectra_fm_root=$(cd "$(dirname "$0")/.." && pwd)
spectra_python_bin=${SPECTRA_PYTHON:-/Users/jirustaroure/.local/state/spectra-consolidation/lote-2-20261009/wheel-env/bin/python}
spectra_raw_root=${SPECTRA_DATA_ROOT:-/Users/jirustaroure/.local/share/spectra-data/933b9056b93b0f3e7790ee28c43412b49c39e232}
spectra_index_path=${SPECTRA_INDEX_PATH:-/Users/jirustaroure/.local/share/spectra-data/derived-933b9056/spectra-input-v2.sqlite}
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH="$spectra_fm_root/src" \
    "$spectra_python_bin" "$spectra_fm_root/scripts/evaluate_historical_local.py" \
    --root "$spectra_raw_root" --index "$spectra_index_path" \
    --ids "$spectra_fm_root/docs/evidence/batch3-historical-ids.json" \
    --checkpoints "$spectra_fm_root/docs/evidence/batch3-checkpoints.json" \
    --output "$1" --device "${SPECTRA_DEVICE:-mps}" --batch-size 8 --offline
