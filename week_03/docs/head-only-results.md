# Week 3: confidence improved; routing accuracy did not

This **real local run** completed the head-only ticket-routing exercise on
October 11, 2026. It used the generated 3,000-example dataset and the frozen
300-example test, not mock predictions. Fine-tuning left accuracy at **83.3%**.
Training plus temperature fitting improved the reliability of stated probabilities,
but did not fix the misrouted requests. This does not establish production quality.

## Frozen-test measurements

| Checkpoint | Correct routes | ECE ↓ | Predicted-class Brier ↓ | p50 latency | p95 latency | Inference errors |
| --- | --- | --- | --- | --- | --- | --- |
| Base Laya | 250/300 (83.3%) | 0.231 | 0.134 | 20.80 ms | 21.30 ms | 0 |
| Head-only, best validation epoch | 250/300 (83.3%) | 0.076 | 0.039 | 20.89 ms | 21.33 ms | 0 |
| Same head, fitted temperature | 250/300 (83.3%) | 0.074 | 0.036 | 21.14 ms | 21.51 ms | 0 |

Accuracy describes how often a customer reaches the correct team. All three
models made the same 50 routing mistakes: the held-out notification-configuration
family was sent to **billing** instead of **support**. The different request
references did not change those decisions. The other five held-out template
families were routed correctly.

ECE (expected calibration error) compares predicted-class probabilities with
observed correctness in probability bins; lower values indicate closer agreement.
Brier here is the squared error of predicted-class probability against whether the
choice was correct, not the full multiclass Brier score. Training mainly raised probability on correct routes: their average increased
from 0.689 to 0.995. The average probability on wrong routes changed from 0.466
to 0.469 after training, then fell to 0.450 after temperature fitting. The improved
separation can help identify tickets for human review, even when it does not reduce
the number of wrong routes. A review threshold needs separate validation on
representative customer tickets before use.

The small latency differences do not establish a meaningful speed change. Times
include local input-budget validation and inference, exclude model initialization
and one warmup, and come from one pass on one machine. They are not pure GPU
kernel timings. Each scenario contained one question.

## Existing bakeoff slice

The shared prompts contain **four** relevant `team` choice questions:
`billing-double-charge`, `password-reset`, `cancel-threat`, and `enterprise-sales`.
All three checkpoints answered all four correctly. ECE was 0.207 for base,
0.002 for head-only, and 0.005 after temperature fitting. Four questions are
insufficient to conclude that confidence is reliable in this broader domain.
Temperature fitting did not improve every calibration metric on every dataset.

Jev and D1 were **not exercised** in this run. The implemented evaluator supports
both with `--hosted jev d1`; it closes their clients after evaluation. No new
cross-vendor ranking or hosted pricing claim follows from this experiment.
Local hardware costs and output-token costs were not measured, and unknown
costs remain unknown. Local execution does not alone guarantee privacy.

## Training and calibration

The teacher uses single-intent ticket templates and the same three option
descriptions as the existing routing question. Splits contain 2,400 training,
150 stopping-validation, 150 calibration, and 300 test examples. Whole wording
families are disjoint; the test contains only **six independent template families**,
with 50 reference-number variants of each. The 100-example training spot check
covered 43 training families; the inspected intents agreed with the template
labels. This is a sanity check of synthetic labels, not an estimate of teacher
accuracy on real tickets. Actual customer intent and hand labels can be ambiguous.

The run froze the encoder and act/escalate head and trained 26,248,193 decision
parameters. It used CE, AdamW with learning rate `1e-4`, batch size 4,
accumulation 8, seed 42, fp16 CUDA autocast, and fp32 parameters/optimizer state.
Validation accuracy before training was 98.7%, and reached 100% in epoch 1.
Epochs 2 and 3 also scored 100%; the two-epoch accuracy plateau stopped training.
The first epoch remains the best checkpoint under the predefined accuracy-only
selection rule. Test results were examined only after selection and calibration.

Temperature fitting minimized NLL (negative log likelihood) on the separate
150-example calibration split, within the installed SDK's 0.5–5 bounds. It fitted
**1.174708**. Calibration-set NLL fell from 0.009911 to 0.009528, while ECE
increased slightly from 0.005343 to 0.005618. NLL fitting does not guarantee lower
ECE. Temperature scaling preserves the top choice, so it cannot repair the
50 notification mistakes. The frozen act/escalate head was not trained or
validated for this domain; use measured choice probabilities rather than assuming
its action score is trustworthy.

## Hardware, provenance, and limits

The observed GPU was an **NVIDIA GeForce RTX 2080 with approximately 8 GB**, rather
than the 11 GB 2080 Ti assumed in the plan. Head-only peak allocated GPU memory
was 1.934 GiB. Full tuning was implemented but **not successfully validated**:
isolated batch-size-1 smoke attempts ran out of memory, including after disabling
AdamW foreach temporaries and enabling encoder checkpointing. Fit on an 11 GB
card remains unverified. The experimental RLCD-style loss passed an offline finite
gradient check; it was not used in this completed head run.

Software: Laya 0.3.27, PyTorch 2.14.1+cu130, Python 3.13. The cached English model
snapshot was `7b928d828b7b0e022f929d9bd2e44165aa270148`, now the exercise default
revision. The base SDK emitted a warning that a checkpoint temperature bucket
was outside its allowed range and was clamped. This warning is recorded without
assuming which benchmark answers it affected.

Test-file SHA-256:
`411ac19194d664aab876d9053885d959940b3e235117084cdc2a6c0c13a43d50`.
The saved manifests record every split hash. Model selection, calibration, and
the mistake-mining exercise exclude the frozen test from training. The mistake-mining
smoke test added 12 teacher-labeled mistakes from 96 fresh requests drawn only
from original training families, preserving held-out files byte for byte. It was
not a second complete training experiment and establishes no improvement.

Vendor results on other benchmarks in the
[Laya model card](https://huggingface.co/convaiinnovations/laya) do not predict the
gain on these simple templates. Here, related variants, only six test families,
subjective real-world routing decisions, and one hardware/configuration limit
what can be generalized. There is no evidence of production readiness.

## Artifacts and next decision

The current workspace contains generated data in `week_03/data/tickets-v1`,
checkpoints and training history in `week_03/runs/ticket-head-v1`, and JSON/CSV/
Markdown reports plus reliability SVGs in `week_03/results/ticket-head-v1`.
These generated artifacts are ignored by Git; this write-up preserves the
reviewed measurements. See the [workflow guide](../README.md) to reproduce them
in new output directories.

The next useful experiment is more representative, varied ticket wording and
manually reviewed customer labels. Keep the current test frozen. Once its failure
family informs a new dataset design, freeze an additional unseen test for that
next experiment rather than claiming an unbiased gain on familiar examples.
Full tuning may be tried on hardware with enough memory, but these results do
not demonstrate that it would fix the error. Better confidence alone does not
remove the customer transfer caused by the incorrect route.
