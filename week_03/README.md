# Week 3: specialize Laya for support-ticket routing

Implement the [weekend plan](laya-finetuning-weekend-plan.md) with a single
three-option **choice** task: billing, sales, or support. It reuses the question
and option descriptions in four existing bakeoff scenarios. No existing benchmark
labels are included in training. Start with cross-entropy and a frozen encoder;
try full training only after reviewing validation results.

The [completed real head-only run](docs/head-only-results.md) kept frozen-test
accuracy at 83.3% and improved probability reliability. Its data, checkpoints, and
reports are present in this workspace as ignored generated artifacts. Use those
artifacts directly, or choose new output directories when repeating the commands
below; existing datasets and run outputs are intentionally protected.

## Start here

Use the shared environment described in the [repository README](../README.md).
From the repository root:

```bash
uv sync --locked
uv run jupyter lab week_03/notebooks/laya-finetuning.ipynb
```

Select **Python (system-one-models)** and run the cells in order. The notebook
defaults to `MOCK = True` for an offline walkthrough. Set `MOCK = False` to use
real checkpoints and run any missing training or calibration steps. Real mode
requires suitable hardware and may download the base model.

Read the [results write-up](docs/head-only-results.md) before starting another
experiment. The completed run improved confidence reliability but left all 50
notification-routing mistakes unchanged. Full tuning exceeded the observed
RTX 2080's approximately 8 GB capacity; fit on an 11 GB card remains unverified.

## Folder guide

| File or folder | Purpose |
| --- | --- |
| [Weekend plan](laya-finetuning-weekend-plan.md) | Original exercises and resource links. |
| [Exercise notebook](notebooks/laya-finetuning.ipynb) | Guided data, training, calibration, and evaluation workflow. |
| [prepare_data.py](prepare_data.py) / [ticket_data.py](ticket_data.py) | Generate deterministic, balanced data and freeze disjoint splits. |
| [train.py](train.py) | Head-only or full training, validation, and checkpoint export. |
| [calibrate.py](calibrate.py) | Fit temperature using the reserved calibration split. |
| [evaluate.py](evaluate.py) / [plots.py](plots.py) | Shared-harness comparisons, reports, and reliability diagrams. |
| [dagger.py](dagger.py) | Optional collection of fresh, teacher-labeled training mistakes. |
| [Results](docs/head-only-results.md) | Reviewed real measurements and their limitations. |
| `data/` | Generated datasets and split manifests; ignored by Git. |
| `runs/` | Generated checkpoints and training history; reviewed `run-summary.md` files may be tracked. |
| `results/` | Generated comparison reports and SVG plots; ignored by Git. |

Reusable helpers live in [finetuning.py](../src/system_one_models/finetuning.py),
and offline regression tests live in [tests/](../tests/).

## 1. Generate and review the data

Run from the repository root using the shared environment:

```bash
uv sync --locked
uv run python week_03/prepare_data.py --output week_03/data/tickets-v1 --count 3000
```

Each JSONL example contains `id`, `family`, `state`, `question`, a named
`options` mapping, and `label`. A deterministic teacher assigns the label from
the single-intent template family. References are random nuisance values.
There are 20 wording families per class, with 16/1/1/2 assigned to
train/validation/calibration/test. This gives 2,400/150/150/300 examples:
80% training, 10% validation (half reserved for calibration), 10% frozen test.
Classes are balanced. Entire families are disjoint across splits; variations of
the same wording cannot cross the boundary. A SHA-256 manifest detects edits.
Existing datasets cannot be overwritten; use a new directory for a new version.

Review the 100 training examples in `spot-check.jsonl` by hand before training.
Check that each request has one clear intent and that the option definitions
match the teacher. Record disagreements in your run notes. These synthetic
examples have consistent teacher labels but limited linguistic diversity; 3,000
rows do not mean 3,000 independent task patterns. Real ambiguous tickets will
need manually reviewed labels and a separately frozen real-world test set.

Open [the exercise notebook](notebooks/laya-finetuning.ipynb) and run it in order.
It defaults to `MOCK = True`, creates data and reports in a fresh `/tmp` directory,
and exercises the shared harness without credentials, weights, or training.
Its mock comparisons are identical simulations, not accuracy improvements.

## 2. Train the head first

```bash
uv run python week_03/train.py --data week_03/data/tickets-v1 --output week_03/runs/ticket-head-v1
```

Defaults: four epochs, micro-batch 4, accumulation 8 (effective batch 32),
AdamW head learning rate `1e-4`, seed 42, and fp16 CUDA autocast with gradient
scaling. Parameters and optimizer states remain fp32. The encoder and unsupervised
act/escalate head are frozen; the decision layers, scorer, and type embeddings train.
The script validates accuracy, ECE, and NLL every epoch, saves the best validation
accuracy checkpoint, and stops after two epochs without accuracy improvement.
It saves `best/`, `last/`, `history.json`, `run.json`, and optimizer/RNG state each
epoch. These files preserve progress; automatic optimizer resume is not implemented.
You can start a new run with `--model .../last`, which resets optimizer/epoch state.

The English base revision is pinned to
`7b928d828b7b0e022f929d9bd2e44165aa270148` (override `--revision` deliberately).
Loading may download weights and uses `HF_TOKEN` from the environment when needed.
Scripts do not read secret files. Inputs are checked with the actual tokenizer:
options over 48 tokens, truncated instructions, lost evidence, or collapsed
options cause errors. The assembled sequence is capped at 512 tokens.
GPU availability is checked to catch an initial SDK fallback. Memory fit and
training speed must be measured on your hardware; fp16 alone does not guarantee
that a full AdamW run fits 11 GB. Reduce micro-batch size if necessary.

For a full fine-tune after head-only review:

```bash
uv run python week_03/train.py --data week_03/data/tickets-v1 --output week_03/runs/ticket-full-v1 --model week_03/runs/ticket-head-v1/best --regime full --batch-size 1 --accumulation 32 --lr-encoder 2.5e-5 --lr-head 1e-4
```

Full mode disables AdamW foreach temporaries, enables encoder gradient checkpointing,
and trains encoder plus decision head. `--objective rlcd` enables an experimental
Gaussian-logit policy-gradient
objective using log and spherical proper scores, four exploration samples, a
leave-one-out baseline, and CE anchoring. This is **not** an exact reproduction of
the producer's GRPO notebook; keep CE as the starting point. Ordinal scoring and
act/escalate supervision are outside this single-choice experiment.

## 3. Fit confidence separately

```bash
uv run python week_03/calibrate.py --model week_03/runs/ticket-head-v1/best --data week_03/data/tickets-v1 --output week_03/runs/ticket-head-v1/calibrated
```

A scalar temperature within the installed SDK bounds (0.5–5) minimizes negative
log likelihood on the 150 calibration examples, never the test set. The calibrated copy keeps the original
weights and writes the temperature into checkpoint configuration. It preserves
the uncalibrated checkpoint. `calibration.json` reports before/after diagnostics
on the fitting set; independent calibration quality comes from the test report.
Temperature scaling preserves top choices and cannot repair routing mistakes.

## 4. Compare on the same frozen test and bakeoff slice

```bash
uv run python week_03/evaluate.py --data week_03/data/tickets-v1 --output /tmp/week03-real-v1 --real --checkpoint laya-head=week_03/runs/ticket-head-v1/best --checkpoint laya-calibrated=week_03/runs/ticket-head-v1/calibrated
```

Add `--hosted jev d1` to compare hosted models (requires `TYPESAFE_API_KEY` and
`LIQUID_API_KEY`; these incur service calls). Omit `--real` for offline mock
validation. Models run sequentially so base and tuned weights need not coexist
on the GPU. A direct single-checkpoint Laya adapter bypasses multilingual routing
and returns the existing harness answer format. Initialization and one warmup
request are outside the timer. Token-budget validation is included in the local
request timer; comparisons are approximate end-to-end workflow timings.

The report directory contains `comparison.md`, run metadata, and
`test/` / `bakeoff-slice/` folders with `results.md`, `results.csv`, and
`results.json`, plus a standalone `reliability.svg`. Inspect errors and
per-question answers as well as the table.
Choice accuracy, ECE, predicted-class Brier, reliability bins, and p50/p95 latency
are relevant; noul/score fields stay empty because this task has no such questions.
Prices remain unknown; local hardware and output-token costs are not estimated.

For product interpretation: accuracy represents correct routing and avoided
transfers; probability reliability informs which tickets require human review.
Do not assume an improvement until measured. The four-question legacy slice is
small; real customer tickets, subjective labels, network differences, model
versions, and hardware all limit comparisons. Local inference does not by itself
guarantee privacy, free operation, or production readiness.

## 5. Stretch: collect mistakes and train a new iteration

```bash
uv run python week_03/dagger.py --data week_03/data/tickets-v1 --model week_03/runs/ticket-head-v1/best --output week_03/data/tickets-v2 --seed 43
uv run python week_03/train.py --data week_03/data/tickets-v2 --model week_03/runs/ticket-head-v1/best --output week_03/runs/ticket-head-v2
```

This DAgger-inspired hard-example round labels fresh synthetic requests with the
same teacher and adds only mistaken answers from original **training families**.
All original validation, calibration, and test files are copied unchanged. It
fails on inference errors rather than treating them as mislabeled examples.
Recalibrate the new model and compare using the original frozen test. This is
hard-example mining, not interactive policy rollouts or evidence of measured gains.

## Validation and sources

Offline checks:

```bash
uv run python -c "import system_one_models; print(system_one_models.__file__)"
uv run python -m unittest discover -s tests
```

The [model card](https://huggingface.co/convaiinnovations/laya) describes the English
checkpoint, typed questions, context limits, and vendor benchmark results.
The [maintainer's repository](https://github.com/nandhakishorm/laya) provides the
reference training notebook. Those measurements use other data and do not predict
this exercise's improvement. The implementation follows the **installed SDK**
interfaces; it does not copy unverified memory-fit or expected-latency claims
from the plan. See the plan's resource list for further experiments.

Generated data and model/report artifacts are ignored by Git. Preserve reviewed
notes in a `run-summary.md` or a separate Markdown findings file. Keep future
week-specific orchestration here and reusable validation/calibration helpers in
[`finetuning.py`](../src/system_one_models/finetuning.py).
