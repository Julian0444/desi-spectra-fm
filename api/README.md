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

FastAPI serving the DESI foundation model v2.1 (26M-parameter transformer,
official prediction `z_pred_map` + `z_confidence`). Swagger docs: append
[`/docs`](https://jirustaroure-desi-fm-api.hf.space/docs) to the Space URL.

```bash
curl -s https://jirustaroure-desi-fm-api.hf.space/healthz
curl -s -F "file=@spectrum.npz" \
  "https://jirustaroure-desi-fm-api.hf.space/predict?mask_ratio=0.0"
curl -s -X POST https://jirustaroure-desi-fm-api.hf.space/predict_json \
  -H "Content-Type: application/json" \
  -d '{"flux": [/* P floats */], "wavelength": [/* P Angstroms */]}'
```

The `.npz` needs `flux` (P,) or (N, P) and `wavelength` (P,) or (N, P) in
Ångströms; optional `ivar` / `mask`. Any instrument works (spectra are
interpolated onto the model's internal 3600–9800 Å log-λ grid). Limits: 32
spectra / 50 MB per request.

REST calls run on CPU — no ZeroGPU quota involved. Only the small Gradio
tester panel on this page uses ZeroGPU (anonymous visitors get a modest daily
allowance). Docker Spaces are PRO-only, hence this free ZeroGPU deploy; the
repo also ships a `Dockerfile` for self-hosting.

Code, tests and evaluation notebook:
<https://github.com/Julian0444/desi-spectra-fm> ·
Model checkpoint: <https://huggingface.co/jirustaroure/desi-spectra-fm> ·
Interactive demo:
<https://huggingface.co/spaces/jirustaroure/desi-spectra-fm-demo>
