# Architecture and decisions

These are recorded design choices and their alternatives, not a retrospective
claim about personal authorship. Evidence and limits are in [Evaluation](EVALUATION.md).

| Decision | Implemented choice | Alternative and reason or limit |
|---|---|---|
| Representation | An encoder-only multitask transformer jointly learns masked reconstruction and redshift. | No external pretrained weights were used; foundation-model transfer has not been demonstrated. |
| Historical redshift head | v2.1 fine-tunes v1 with 100 log(1+z) classification bins; MAP is official. | v1 scalar regression showed the historical high-redshift failure. Several changes occurred together, so the head alone is not an isolated cause. |
| Predictor default | Keep v2.1; expose CNN-17 experimentally. | CNN was nominated by validation in another experiment, but v2.1 was not tested on that population. Its nomination does not authorize replacing the service. |
| Reconstruction test | A/B retains architecture, initialization, input/mask sequence and budget; B has zero reconstruction coefficient. | Removing the auxiliary head would change another factor. CNN and ridge are practical baselines, not equivalent causal controls. |
| Checkpoint selection | Validation outlier rate, then normalized MAE, then earlier step. | Test scores must not select a later checkpoint or threshold. The frozen selection is retained. |
| Input validity | Explicit flux/wavelength contract, inverse variance, masks, gaps, coverage and typed rejection. | Interpolation alone cannot justify treating missing coverage as evidence or establish other-instrument accuracy. |
| Confidence | Report posterior concentration with its definition. | Do not present it as a calibrated correctness probability or make a service policy from a retrospective threshold. |
| Deployment | Python inference, local Gradio and FastAPI share the package. | Hosted Space code is a separate deployment and is not changed by package publication. |
| Reproduction | Preserve frozen checkpoints, predictions, identities and hashes. | Rebuilding a wheel is not authorization to repeat a paid evaluation or reopen a test. |

The v2.1 path resamples to a log-wavelength grid, groups pixels into patches,
encodes them with the transformer, and applies reconstruction and redshift heads.
The redshift token is always masked during joint training. The official output
is `z_pred_map`; secondary `z_pred` is `expm1(E[log(1+z)])` from the same posterior.
Mean-pooled encoder vectors support descriptive neighbor retrieval in Copilot;
the shared supervised encoder does not supply independent confirmation.

The source implementation and experiment configuration retain the detailed
architecture. [Input contract](INPUT_CONTRACT.md), [batch-4B record](BATCH_4B.md),
[historical evidence](HISTORICAL_EVIDENCE.md), [model card](../model_card.md).
