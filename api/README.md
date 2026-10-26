---
title: desi-fm API
emoji: 📡
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.22.0
app_file: app.py
pinned: false
license: mit
short_description: REST API (FastAPI) for galaxy-spectrum redshifts
models:
  - jirustaroure/desi-spectra-fm
---

# desi-fm API — REST endpoint

FastAPI app (`desi_fm.api`) serving the DESI multitask transformer v2.1
(encoder-only transformer, ~26M parameters, `checkpoint_last.pt` from
[jirustaroure/desi-spectra-fm](https://huggingface.co/jirustaroure/desi-spectra-fm)).
The official prediction is `z_pred_map` (argmax of a 100-bin posterior over
`log(1+z)`) with its `z_confidence` (posterior concentration, 0–1); `z_pred`
(`expm1(E[log(1+z)])`) is kept only for backwards compatibility.

## Paths: Space vs. local

On this Space the FastAPI app is mounted as a sub-app at `/api` inside the
Gradio server (`gradio_app.mount("/api", fastapi_app)` in `app.py`). Running
the same app yourself with `uvicorn` or Docker there is **no prefix**.

| Route | Space (`https://jirustaroure-desi-fm-api.hf.space`) | Local (`http://localhost:7860`) |
|---|---|---|
| Swagger UI | `/api/docs` | `/docs` |
| health check | `GET /api/healthz` | `GET /healthz` |
| `.npz` upload (1–32 spectra) | `POST /api/predict?mask_ratio=…` | `POST /predict?mask_ratio=…` |
| single spectrum as JSON | `POST /api/predict_json` | `POST /predict_json` |

`mask_ratio` (0.0–0.9, default 0.0) is a **query parameter** on `/predict`
and a **field of the JSON body** on `/predict_json`. 0 = plain redshift
prediction, deterministic; larger values hide a random fraction of the input
tokens before the forward pass.

```bash
BASE=https://jirustaroure-desi-fm-api.hf.space/api   # local: BASE=http://localhost:7860

curl -s $BASE/healthz
# → {"status":"ok","model_loaded":true}

curl -s -F "file=@spectrum.npz" "$BASE/predict?mask_ratio=0.0"
# → {"n": 1, "z_pred_map": [0.2267], "z_confidence": [0.6376], "z_pred": [0.2314]}
#   (examples/heldout_z020.npz, z_true 0.2036)

curl -s -X POST $BASE/predict_json -H "Content-Type: application/json" \
  -d '{"flux": [/* P floats */], "wavelength": [/* P Angstroms */],
       "ivar": [/* optional */], "mask": [/* optional bools */], "mask_ratio": 0.0}'
# → {"z_pred_map": 0.2267, "z_confidence": 0.6376, "z_pred": 0.2314}
```

## Input format and limits

- `.npz` (`/predict`): `flux` (P,) or (N, P) and `wavelength` (P,) or (N, P)
  in Ångströms; optional `ivar` (same shape as `flux`) and boolean `mask`.
  Both endpoints use `ivar`/`mask` when present.
- `/predict_json`: one spectrum as JSON lists (`flux`, `wavelength`, optional
  `ivar`, `mask`, plus `mask_ratio`).
- Other grids can be resampled; accuracy outside DESI is unvalidated. Spectra are interpolated onto the model's internal
  3600–9800 Å log-λ grid.
- Limits: 32 spectra and 50 MB per request; at least 8 usable input samples and 8 supported model-grid pixels (numerical minima, not quality thresholds).
  Oversize requests get HTTP 413. Only malformed/unusable scientific inputs become HTTP 422; checkpoint and internal inference failures remain server errors.

## Checkpoint loading (differs between Space and local)

- **Space:** `app.py` calls `get_model()` at import time — the checkpoint is
  downloaded and loaded on CPU while the Space starts, so `/api/healthz`
  reports `"model_loaded": true` from the first request and there is no
  cold-start penalty on `/api/predict` (the Space itself may take ~30 s to
  wake up after being idle).
- **uvicorn / Docker:** `desi_fm.api` loads lazily. `/healthz` returns
  `"model_loaded": false` until the first `/predict` or `/predict_json`
  call, which downloads `checkpoint_last.pt` (~104 MB) from the Hub and caches
  it (`HF_HOME=/tmp/hf` inside the container). Set
  `DESI_FM_CKPT=/path/to/checkpoint_last.pt` to skip the download and
  `DESI_FM_DEVICE` to override the default `cpu`.

```bash
pip install -e ".[api]"
uvicorn desi_fm.api:app --port 7860      # Swagger at http://localhost:7860/docs

docker build -t desi-fm-api .
docker run -p 7860:7860 desi-fm-api     # CPU-only image, python:3.11-slim
```

## Gradio tester panel (this page)

The panel above is a minimal tester wired to the same model. Notes:

- Its handler (`predict_ui`) carries `@spaces.GPU`, which zero-a10g hardware
  requires at startup, so UI clicks spend a little ZeroGPU quota. REST calls
  run on CPU and spend **none** — `curl` works anonymously.
- The tester only reads `flux` and `wavelength` (first spectrum of a multi-row
  file) and **ignores `ivar`/`mask`**, unlike the REST endpoints. For the same
  file the tester's `z_confidence`/`z_pred` can therefore differ slightly from
  the API's; `z_pred_map` matched on the control example.
- This Space uses the Gradio SDK on ZeroGPU with FastAPI mounted at `/api`;
  the repo's `Dockerfile` is the self-hosting path.

## Versions

- Space: `torch==2.11.0` pinned in `requirements.txt`, Gradio 6.22.0
  (`sdk_version` above), `desi-fm` installed from the GitHub `main` branch.
- Package (`pyproject.toml`): Python ≥ 3.9, `torch>=2.0` unpinned. CI and
  the Docker image install the current CPU wheel of torch unpinned.

## Model quality (for context)

The historical 2,000-object validation has σ_NMAD 0.030 and η₀.₁₅ 14.95%,
with v2.1 last and 50% of tokens randomly hidden without a fixed seed. REST
normally serves full spectra, so these numbers do not measure that protocol.
v1 exposure is strongly supported; the same objects selected checkpoints.
This is not an independent test. The confidence score is entropy concentration,
not a calibrated probability. Retrospectively ≥0.5 retains 1,175/2,000 with
9 errors (58.75% coverage; 0.77% error); that threshold is not a validated policy.
These local text corrections await publication to the Space.

Code, tests and evaluation notebook:
<https://github.com/Julian0444/desi-spectra-fm> ·
Model checkpoint: <https://huggingface.co/jirustaroure/desi-spectra-fm> ·
Interactive demo:
<https://huggingface.co/spaces/jirustaroure/desi-spectra-fm-demo>

Both the REST API and Gradio tester retain mask/ivar and follow [the shared input contract](../docs/INPUT_CONTRACT.md). `/healthz` identifies the loaded model with config and checkpoint SHA-256; prediction responses include this identity and coverage. JSON supports `wavelength_unit` (`angstrom` by default, or explicit `nm`).


## Installed 0.3.0 verification

The local installed package reports API version 0.3.0. The 5C checks passed
healthz, NPZ upload, JSON prediction, invalid-input 422 and version metadata
using verified local v2.1 assets and development inputs. Run
`python -I -B -m uvicorn desi_fm.api:app --host 127.0.0.1 --port 8000`
from the installed environment; [reproduction and limits](../docs/BATCH_5C_REPRODUCE.md).
These local checks did not deploy or verify the public Space.
