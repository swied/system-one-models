# Week 1: Laya inference, training, and calibration

This week explores local Laya and trains its decision head on synthetic support
tickets. See the [repository README](../README.md) for shared installation,
kernel registration, and credentials. Use the root `.venv` for this week.

- [Laya setup and fine-tuning guide](laya-ubuntu-rtx2080-usage-and-fine-tuning-guide.md)
- [SalesRLAgent and confidence routing research guide](salesrlagent-confidence-routing-fine-tuning-guide.md)
- [Practice datasets and label conventions](data/README.md)
- [Saved training and calibration run](runs/laya-head-v1/run-summary.md)
- [Smoke test notebook](smoke-test-laya.ipynb)

Run scripts from the repository root:

```bash
uv run python week_01/smoke-test-laya.py
uv run python week_01/laya_first_prediction.py
uv run python week_01/finetune_laya_head.py \
  --data week_01/data/laya_train.jsonl \
  --output week_01/runs/laya-head-v1 \
  --max-length 256 --epochs 2 --accumulation 16
uv run python week_01/calibrate_laya_head.py \
  --model week_01/runs/laya-head-v1 \
  --calibration week_01/data/laya_calibration.jsonl \
  --evaluate week_01/data/laya_validation.jsonl
```

File arguments resolve against the current working directory. To follow guide
examples that use `data/` and `runs/`, change to `week_01/` and invoke scripts
with `uv run --project .. python <script>.py`. The calibration script imports
its sibling training module; invoking it as a script keeps that import working.

These commands perform live inference or training and may download weights.
The training and calibration scripts default to CUDA and require suitable NVIDIA
hardware. Calibration requires a completed checkpoint; the saved run summary
alone is not a checkpoint. The synthetic data demonstrates the workflow, not
performance on real customer traffic. Keep generated weights out of commits.

The PDFs and saved run summary preserve the earlier exploration. Their recorded
environment and commands describe that run; the commands above use the current
consolidated repository. Optional research examples may need extra dependencies.
