---
title: DESI Spectra Foundation Model
emoji: 🔭
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.22.0
app_file: app.py
pinned: false
license: mit
short_description: Historical DESI redshift + masked reconstruction prototype
models:
  - jirustaroure/desi-spectra-fm
---

# DESI Spectra Foundation Model — live demo

Interactive demo of the v2.1 checkpoint (encoder-only transformer, ~26M
parameters, trained with masked-token prediction on DESI EDR/SV3 spectra with
a jointly trained redshift head): **redshift prediction** — official output
`z_pred_map`, argmax of a 100-bin posterior over `log(1+z)`, with a
`z_confidence` score (posterior concentration, 0–1) — and **masked-region
reconstruction**, the self-supervised task the model was trained on.

- Pick one of the four examples or upload your own `.npz` with `flux` and
  `wavelength` (Å) arrays; optional `ivar`, `mask` and `z_true` are used if
  present. Current single-spectrum tools reject multi-row files; use batch
  Python, CLI or REST for multiple spectra.
- Move the slider to hide a fraction of the input tokens before the forward
  pass and watch the model fill the shaded regions in.
- When `z_true` is in the file, the panel prints `|Δz|/(1+z)` and flags
  catastrophic outliers (> 0.15).

## The four examples

`heldout_z020`, `heldout_z083`, `heldout_z287`, `heldout_lowconf_z157` are
**real DESI spectra from the canonical held-out split** (the 2,000 examples
after the 80,000 training examples in the same filtered stream), stored with
their raw `flux/ivar/wavelength/mask` and the DESI pipeline `z_true`. The z≈2.87 example has normalized error ≈0.110 (within the 0.15 tolerance)
and confidence ≈0.354; within-tolerance does not imply high confidence. The
z≈1.57 example is a catastrophic outlier with low confidence. Control value: `heldout_z020` → `z_pred_map` 0.2267, `z_confidence`
0.64, `z_true` 0.2036.

Where the files live: `app.py` references them as `examples/heldout_*.npz`,
i.e. an `examples/` directory **next to `app.py` on this Space** (also passed
as `allowed_paths` so clicking an example keeps working after Gradio's upload
cache expires). In the GitHub repository `examples/` is **gitignored**;
regenerate it with `python3 scripts/make_demo_examples.py` (streams through
the 80k-example skip, 15 min – 2.5 h) and copy the four files into the Space.
The [spectra-copilot](https://github.com/Julian0444/spectra-copilot) repository
contains three of the four; `heldout_z083.npz` is absent there.

## Numbers and caveats

The stored evaluation uses 2,000 historical validation objects, a random 50%
token mask without a fixed seed, and v2.1 `checkpoint_last.pt`: σ_NMAD = 0.030,
η₀.₁₅ = 14.95%. The demo defaults to full spectrum, a different protocol.
The objects were excluded from v2.1 fine-tuning, but v1 exposure is strongly
supported and checkpoints were selected on them: no independent test.

The entropy score is not a calibrated probability. Retrospectively, ≥0.5 retains
1,175/2,000 with 9 errors (58.75% coverage; 0.77% error); not a validated policy.
These local text corrections await publication to the Space.

## Runtime

- ZeroGPU Space: the checkpoint is downloaded from the Hub and loaded at
  startup; `analyze` runs under `@spaces.GPU(duration=8)`, ~1–3 s per call.
  Anonymous visitors get a small daily allowance of runs — on "ZeroGPU quota
  exceeded", sign in to huggingface.co (free) for a larger quota or come back
  the next day.
- Versions: `torch==2.11.0` pinned in `requirements.txt`, Gradio 6.22.0
  (`sdk_version` above), `desi-fm` installed from the GitHub `main` branch;
  the package itself only requires `torch>=2.0`.
- Locally (`spaces` is a no-op off ZeroGPU): install `requirements.txt` and
  run `app.py` from a directory where `examples/` resolves — e.g. the
  repository root after regenerating it (`python3 demo/app.py`), or mirror the
  Space layout with `examples/` next to `app.py`.

Input format: `.npz` with `flux` (P,) or (N, P) and `wavelength` (P,) or
(N, P) in Ångströms. Spectra are interpolated onto the model's internal
3600–9800 Å log-λ grid. Accuracy outside DESI and under partial coverage has not been established.

Code, tests and evaluation notebook:
<https://github.com/Julian0444/desi-spectra-fm> ·
Model checkpoint: <https://huggingface.co/jirustaroure/desi-spectra-fm> ·
REST API: <https://jirustaroure-desi-fm-api.hf.space/api/docs>

Batch 2 input, validity, inference and evaluation semantics: [INPUT_CONTRACT.md](../docs/INPUT_CONTRACT.md). Single-spectrum files are required; multirow files use batch Python, CLI or REST.


## Local installed demo 0.3.0

The verified local entry point is `python -I -B -m desi_fm.demo`, with
`DESI_FM_CKPT` set to a verified local checkpoint. It binds 127.0.0.1, share=false,
requires a single spectrum and uses no Hub download or ZeroGPU dependency.
Upload/predict, masking 0/0.35 and invalid-support rejection were checked through
the real local Gradio HTTP API. Versions, hashes and commands are in the
[5C reproduction guide](../docs/BATCH_5C_REPRODUCE.md). The Space configuration
above is historical hosting configuration; this work did not redeploy it.
