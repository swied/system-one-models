# Laya practice datasets

These four JSONL files contain **handwritten synthetic support tickets**, not real customer records or independently measured outcomes. They are ready to use with [finetune_laya_head.py](../finetune_laya_head.py) and
[calibrate_laya_head.py](../calibrate_laya_head.py).

| File | Cases | Question sequences | Purpose |
| --- | ---: | ---: | --- |
| `laya_train.jsonl` | 36 | 108 | Fit the trainable head |
| `laya_calibration.jsonl` | 24 | 72 | Fit temperatures |
| `laya_validation.jsonl` | 24 | 72 | Choose epochs, schema, and routing thresholds |
| `laya_test.jsonl` | 24 | 72 | Evaluate the frozen final configuration |

Each case has three questions: `queue`, `urgency`, and `cancel_threat`. Every split includes all three queues, all three urgency levels, and both boolean outcomes. In each split, queue and urgency labels are balanced, and half the cases contain explicit cancellation threats. These proportions are artificial and do not represent normal support traffic.

## Record format and label order

Each line is one JSON object containing:

- `case_id`: unique across all four files.
- `schema_version`: `triage-v1`, matching the guide's example.
- `source`: `synthetic_handwritten`.
- `split`: the file's intended use.
- `state`: the ticket text supplied to the model.
- `questions`: the same question definitions in every record.
- `targets`: one-hot target distributions, in the following order.

| Question | Target order |
| --- | --- |
| `queue` | `[billing, technical, other]` |
| `urgency` | `[routine, time sensitive, production blocked]` |
| `cancel_threat` | `[false, true]` |

For example, `"queue": [1, 0, 0]` labels a billing ticket, and `"cancel_threat": [0, 1]` labels an explicit cancellation threat. These are training labels, not confidence estimates obtained from Laya.

## Annotation conventions

- **Billing:** charges, receipts, invoices, payments, credits, refunds, and account suspensions caused by billing records.
- **Technical:** software bugs, product malfunction, server errors, and technical outages.
- **Other:** administrative contacts, onboarding arrangements, procurement documents, or manual legal/compliance authorizations. Some examples intentionally describe administrative production holds; their routing cause is administrative even though their operational impact is severe.
- **Routine:** no active production blockage and no concrete near-term deadline.
- **Time sensitive:** a stated upcoming deadline, with production still available.
- **Production blocked:** live operations are explicitly stopped or unavailable.
- **Cancellation threat:** an explicit conditional or unconditional statement that the customer will cancel. A future renewal threat can be routine when there is no immediate deadline or production impact; churn and urgency are separate labels. Explicit denials and canceled meetings/jobs are negative examples, even when they contain cancellation-related words.

## Run the training example

From the repository root, use the shared uv environment. File arguments are
relative to your current working directory:

```bash
uv run python week_01/finetune_laya_head.py \
  --data week_01/data/laya_train.jsonl \
  --output week_01/runs/laya-head-v1 \
  --max-length 256 --epochs 2 --accumulation 16
```

Wait for training to finish successfully: it saves the checkpoint only after all epochs complete. The first-prediction script does not create this checkpoint. Use the included calibration script with the same directory for `--model` as training's `--output`:

```bash
uv run python week_01/calibrate_laya_head.py \
  --model week_01/runs/laya-head-v1 \
  --calibration week_01/data/laya_calibration.jsonl \
  --evaluate week_01/data/laya_validation.jsonl
```

Once choices are frozen, evaluate using `week_01/data/laya_test.jsonl` instead of the validation file. Keep calibration data separate from both training and evaluation. Each calibration question type has 24 items, enough to enter the guide script's fitting branch, but far too few to establish robust calibration for real use.

## What these files can demonstrate

Use them to practice parsing, tokenization, native head training, checkpoint export, calibration, and evaluation. IDs and literal ticket texts are disjoint across the files, but the same synthetic scenario patterns and vocabulary recur. Good test performance therefore demonstrates behavior on this small practice distribution; it does not establish real-world generalization.

Replace or expand these records with reviewed examples from your actual task before drawing conclusions about accuracy, calibration, or automation thresholds. Split related real tickets/accounts before deriving question items, and measure the class prevalence of your real traffic.

See the [Laya setup and fine-tuning guide](../laya-ubuntu-rtx2080-usage-and-fine-tuning-guide.md) for the complete workflow.
