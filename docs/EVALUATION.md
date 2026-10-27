# Evaluation

For a local functional demonstration with fixed public assets, use [REPRODUCE.md](REPRODUCE.md). It is not a blind evaluation.

This is the canonical overview of results for DESI Spectra FM and Spectra
Copilot. The historical technical records linked below retain their protocols,
artifacts and chronology. Software delivery, measured behavior, mechanical
acceptance, semantic review and human adjudication are separate states.

## Historical v2.1 diagnostics: not an independent test

The historical comparison used 2,000 DESI validation spectra, randomly masking
50% of tokens without a fixed seed. v1 used scalar regression; v2.1 fine-tuned
its encoder with a classification head over 100 log(1+z) bins. The objects were
excluded from v2.1 fine-tuning, but reconstructed loader evidence strongly
supports exposure in v1, and validation also selected checkpoints.

| Measurement | v1 | v2.1 |
|---|---:|---:|
| Outliers, normalized absolute error > 0.15 | 22.6% | 14.95% |
| NMAD | 0.083 | 0.030 |
| Mean normalized absolute error | 0.107 | 0.096 |
| Outliers in z ∈ [1.5, 2.5) | 82.7% | 23.5% |
| Masked reconstruction RMSE, arcsinh space | 0.819 | 0.817 |

Best and last v2.1 checkpoints have identical tensors at step 30,000. Their
15.15% and 14.95% measurements used different masks, not different model quality;
that difference is not an uncertainty interval. Multiple training changes
prevent attributing the difference solely to the classification head.

A later full-spectrum diagnostic under input contract v2 measured **240/2,000
outliers (12.00%)**. The different masking protocol explains why this is a
separate measurement, not a replacement estimate under the original protocol.
Neither measurement establishes independent generalization. Sources:
[historical evidence](HISTORICAL_EVIDENCE.md), [batch 3 results](BATCH_3_RESULTS.md).

Posterior concentration is not calibrated correctness. The retrospective 0.5
threshold retained 1,175/2,000 objects with nine errors on that same masked
population; it is not a validated service policy. Reconstruction and a
visible-only moving average were similar on 50 stored examples, with widths
explored on those examples. Neither equivalence nor superiority was established.

## Controlled A/B experiment and practical baselines

The batch-4B recipe used 64,759 training objects, 10,000 validation objects and
10,000 test objects. A used redshift plus reconstruction loss; B retained the
same architecture and auxiliary head with zero reconstruction coefficient.
For seeds 17, 42 and 101, initialization, sample order, masks, normalization and
budget were paired. Each run had 10,000 updates. Checkpoints were selected by
validation outlier rate, then normalized MAE, then earlier step.

| Seed | A outliers / 10,000 | B outliers / 10,000 | Difference A−B (pp) | Difference in normalized MAE |
|---|---:|---:|---:|---:|
| 17 | 1,835 | 1,956 | −1.21 | +0.038576 |
| 42 | 1,489 | 2,221 | −7.32 | −0.144347 |
| 101 | 1,997 | 2,302 | −3.05 | +0.026530 |

The mean outlier difference was −3.86 percentage points, with the historical
paired 95% interval [−4.3101, −3.3867]. A met the predefined aggregate criterion,
but its MAE worsened in seeds 17 and 101; seed 42 drove the mean MAE improvement.
In the 946 warning objects, A worsened outliers in all three pairs, with mean
difference +4.792 percentage points. Those cases remain in the results.

The CNN used full inputs and a different architecture; ridge used 128 bands,
valid fractions, train-only scaling and log(1+z) targets. Ridge selected alpha
0.1 from exactly three validation fits. Family mean test outlier rates were
A 17.737%, B 21.597%, CNN 9.290%, and ridge 38.930%. CNN-17, step 6,000, was
nominated by validation before test; its test count was 914/10,000. It remains
experimental and does not replace v2.1.

The CNN was the stronger practical baseline under this recipe; that comparison
changes input and architecture and does not isolate an architectural cause.
v2.1 was not evaluated on this test, so its historical 12% cannot be compared
with CNN's 9.29%. The three seeds are not 30,000 independent observations.
Intervals are conditional on selected checkpoint pairs; physical independence
was not established. No uniform benefit across seeds, strata or other
instruments follows.

The A/B criterion and budget preceded training. The analysis/nomination plan
was committed after training started and before the first validation. Test
opened automatically after freezing, without an intervening human inspection;
local timestamps are not an external immutable preregistration. These limits
are preserved in [batch 4B](BATCH_4B.md), [strata](BATCH_4B_STRATA.md) and
[the later sensitivity analysis](evidence/batch5a-analysis.md).
The [5C A/B figure](BATCH_5C_AB.md) was regenerated from saved predictions,
without another model evaluation or checkpoint selection.

## Historical agent: negative result

The original Haiku agent and v2.1 were evaluated on the same stratified
150-object sample from historical non-independent validation, using full
spectra. At normalized error below 0.15, the model achieved **139/150** successes
and the agent **119/150**. The agent broke 21 correct predictions and recovered
one of 11 outliers. Mean normalized errors were 0.061 and 0.104 respectively.
Historical recorded API cost was US$4.06; this is not a current price quote.

The confidence-based retrospective hybrid reached 90.0%, below the model alone.
This was not a controlled comparison between LLM providers or a demonstration
that model confidence was calibrated. It describes the historical policy;
current protected reports cannot inherit either its success or its failure
rate. [Protocol and saved results](https://github.com/Julian0444/spectra-copilot/blob/main/eval/README.md).

## 5C: pilot, acceptance and semantic review

The first pilot attempt was interrupted by an integration failure on its first
call, with no usable response and no behavioral evidence. Actual usage remained
uncertified; its US$0.0035706 reservation was debited as spent. Julian reported
an invalid credential and replaced it. A subsequent free token-count request
returned HTTP 200; no diagnostic generation was made.

The authorized repeat used the same 12 development cases for low and medium
effort, with 11 valid inputs and one invalid input each. All 24 rows were kept.
Low passed the operational gate with **11/11** valid reports; medium had **6/11**
and five incomplete explanations. The preregistered rule selected low. Repeat
cost was **US$0.0280614**, certified, for 29 calls. Including the earlier debited
reservation, pilot accounting was US$0.031632. This does not certify the first
attempt's bill. No new pilot is authorized or needed by the current corrections.

**Execution, 2026-10-27:** after the corrected review procedure and build-02
identity passed independent review with zero P1/P2, the single low-effort
acceptance run attempted all 128 cases (80 real and 48 synthetic). It made 126
calls with no retries, at certified cost **US$0.1086672** (US$0.0008489625 per
requested case), below its US$12.80 ceiling. Execution states were 72 ambiguous,
49 insufficient-data, six invalid-input and one incomplete explanation. No case
remained pending; no timeout or technical-error state was recorded. The incomplete
explanation is retained and will not be rerun. Wall latency p50/p95/maximum was
5.40/6.85/8.77 seconds, including all 128 cases.

Outputs and the external seal were registered before labels. After both new AI
annotations were saved and hash-registered, scoring passed the mechanical checks:

| Measurement | Global | Real | Synthetic |
|---|---:|---:|---:|
| Completed / requested | 127/128 | 79/80 | 48/48 |
| Official model_z preserved / eligible | 122/122 | 80/80 | 42/42 |
| Faithful template-evidence units | 909/909 | 621/621 | 288/288 |
| Tool-faithful valid-input cases | 122/122 | 80/80 | 42/42 |
| Required abstentions | 18/18 | N/A (0/0) | 18/18 |
| Valid emitted citation-ID mentions | 3/3 | 3/3 | N/A (0/0) |

Prediction coverage is 122/122 eligible inputs and **122/128 requested cases**.
Invalid-input rejections remain in N; the incomplete explanation is a completion
failure. The global and stratum latency/cost checks pass. Citation-ID membership
is not scientific support. [Aggregate derivative with all strata and intervals](evidence/batch5c-acceptance-score-public.json).

The two new blinded AI sessions, Astra and Opus, annotated equivalent packets
independently; each assembled per-claim judgments with its own scripts after
reading the cases. This is not independent human validation. Opus and the
evaluated Haiku share a provider. Both annotations
were saved and hash-registered before either was read by the coordinator or
labels opened. Their validated union has 2,194 provisional claims from 2,191
seeds and three additions. There are **894 claim disagreements**, incomplete
inventories in 122 cases, and semantic review pending in **128/128 cases**.
The private adjudication package includes every case, both original annotations,
all disputes and empty human decision fields. No human adjudication was made.

Scientific judgments provisionally count 783 agreed-correct claims against a
1,673 denominator; unsupported claims count six against 2,194. Both rates are
**null**, not scientific accuracy estimates or threshold passes. Six agreed-unmet
evidence obligations remain recorded; nine cases have obligation disagreements.
Zero agreed critical-error flags is not proof of no critical errors. C_pair=0/0
is N/A and leaves global acceptance pending. Excluding three quoted excerpts
from the factual denominator gives 6/2,191, also pending; this descriptive
sensitivity does not change the rubric or threshold.

Procedure limits remain: execution retained custody order and partial real/
synthetic blinding; session identities are declared by orchestration. Opus
posted content-bearing progress in its isolated annotation thread despite the
counts-only instruction. The coordinator read that thread only after both
hashes were registered; no exposure of the other annotator was observed. The
trace is retained, and annotations were not repeated. Validator limits are
disclosed in the remaining-work record and the human instructions.

Scientific accuracy and global acceptance remain pending human adjudication
and any protocol limits still applicable. Mechanical fidelity is not scientific
accuracy; public synthetic recipes are mechanism checks.

[5C report](BATCH_5C_REPORT.md), [semantic procedure](BATCH_5C_AI_REVIEW.md),
[remaining work](BATCH_5C_REMAINING.md), [human actions](BATCH_5C_HUMAN_ACTIONS.md).

## Reproduction and provenance

Versioned documents record protocols, source commits and hashes. Frozen
checkpoints, predictions and private custody artifacts are separate from source
packages. [Installed reproduction](BATCH_5C_REPRODUCE.md) and the documentation
[index](README.md) identify the technical records. Local preservation is not an
independent backup. A local reproduction bundle contains 84 payload files (2,352,766,193 bytes),
with manifests and hashes; total directory size is 2,352,794,036 bytes. It has not
been uploaded and excludes raw training data and private acceptance labels.
Development and audits used Codex and Claude; decisions and verifications are
recorded in the documentation. Personal contributions remain an unanswered H-18
item, not a conclusion inferred from these records.
