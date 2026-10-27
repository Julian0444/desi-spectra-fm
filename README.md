# DESI Spectra FM

[![ci](https://github.com/Julian0444/desi-spectra-fm/actions/workflows/ci.yml/badge.svg)](https://github.com/Julian0444/desi-spectra-fm/actions/workflows/ci.yml)

A multitask transformer for DESI spectra, with redshift prediction and masked
reconstruction, a Python package, a local Gradio interface and a FastAPI service.
The historical repository name is retained; transfer or a foundation-model
capability has not been established. **v2.1 is the default predictor; CNN-17 is
experimental.** The official prediction is `z_pred_map`, called `model_z` by
[Spectra Copilot](https://github.com/Julian0444/spectra-copilot).

The principal controlled experiment compared reconstruction plus redshift loss
(A) with the same transformer and zero reconstruction coefficient (B). Across
three frozen checkpoint pairs, A reduced outliers by 1.21, 7.32 and 3.05 percentage
points on the same 10,000 test objects. It met the predefined aggregate criterion,
but worsened normalized MAE in two pairs and worsened the warning stratum.
The CNN was the better practical baseline in that experiment; it was not a
controlled comparison with historical v2.1. Read [Evaluation](docs/EVALUATION.md)
for protocols, denominators, the ridge baseline and the limits of these claims.

## Install and run locally

The commands below install the 0.3.0 source release.
Weights are separate from the Python package. Follow [REPRODUCE.md](docs/REPRODUCE.md)
for downloads fixed by revision and SHA-256 and a public example requiring no private files.

```bash
git clone --branch v0.3.0 https://github.com/Julian0444/desi-spectra-fm.git
cd desi-spectra-fm
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[api,ui,dev]'
export DESI_FM_CKPT=/path/to/checkpoint_last.pt
python -m desi_fm.predict --checkpoint "$DESI_FM_CKPT" --input spectrum.npz --output predictions.npz
python -m desi_fm.demo
```

The local demo requires that explicit checkpoint and does not download weights.
To run the API instead: `uvicorn desi_fm.api:app --host 127.0.0.1 --port 7860`;
its documentation is at `/docs`. The API can download the verified default
checkpoint when `DESI_FM_CKPT` is unset.

Inputs are NPZ arrays `flux` and `wavelength` in Å, with optional inverse variance
`ivar` and bad-pixel `mask`. See the [input contract](docs/INPUT_CONTRACT.md) for
shape, validity, gaps and rejection rules. Output includes the official MAP
redshift, posterior concentration, coverage, model identity and reconstruction.
The secondary `z_pred` is `expm1(E[log(1+z)])`, not the official MAP estimate.

## What the evidence supports

Historical v2.1 diagnostics used a non-independent validation population. Its
14.95% outlier rate under random 50% masking and 12.00% full-spectrum diagnostic
measure different protocols. They cannot establish superiority over the CNN
on another population. Posterior concentration is not calibrated correctness;
reconstruction stability and neighbors from the supervised encoder are not
independent confirmation. Accuracy on other instruments or low coverage has
not been established.

The historical agent reduced successes from 139/150 to 119/150 under its original
protocol. The later 5C pilot selected low effort and passed the operational gate
with 11/11 valid reports. The unique 128-case acceptance then passed mechanical
checks with 127/128 completed (79/80 real, 48/48 synthetic), at US$0.1086672
certified cost. Both AI annotations are complete; 894 claim disagreements leave
semantic review pending in 128/128 cases. Scientific accuracy and global
acceptance await human adjudication; no improved redshift accuracy is claimed. [Evaluation](docs/EVALUATION.md) distinguishes the
pilot, acceptance execution, semantic review and human adjudication.

## Documentation and services

- [Evaluation: canonical results and limitations](docs/EVALUATION.md)
- [Architecture and decisions](docs/DECISIONS.md)
- [Documentation index and process history](docs/README.md)
- [Model card](model_card.md), [data attribution](ATTRIBUTIONS.md), [license](LICENSE)
- [Public reproduction guide](docs/REPRODUCE.md) and [release notes](docs/RELEASE_0_3_0.md)
- [Hosted demo](https://huggingface.co/spaces/jirustaroure/desi-spectra-fm-demo) and [hosted API](https://jirustaroure-desi-fm-api.hf.space/api/docs)

The hosted demo and API application code is not updated by this release; their FM dependency is unpinned. Updates are separate work; see [deployment](docs/DEPLOYMENT.md).

Development and audits used AI coding assistants (Codex and Claude); decisions and verification records are in docs/.
