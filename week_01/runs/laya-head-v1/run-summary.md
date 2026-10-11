# Laya head training and calibration results

Completed on September 27, 2026 using the NVIDIA GeForce RTX 2080 and the Python environment at `/home/scott/anaconda3/envs/uv/bin/python`. Training and calibration both exited successfully.

## Commands

Historical commands, run from `/home/scott/github/jev-weekend` with the `uv`
environment active. These files now live in `week_01/`; see the
[current week 1 commands](../../README.md) for the consolidated environment:

```bash
python finetune_laya_head.py \
  --data data/laya_train.jsonl \
  --output runs/laya-head-v1 \
  --max-length 256 --epochs 2 --accumulation 16
```

```bash
python calibrate_laya_head.py \
  --model runs/laya-head-v1 \
  --calibration data/laya_calibration.jsonl \
  --evaluate data/laya_validation.jsonl
```

## Training

- Base checkpoint revision: `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`.
- Training set: 36 cases, 108 question sequences.
- Trainable parameters: 26,248,193; encoder and action head frozen.
- Epoch 1 mean loss: 0.49981.
- Epoch 2 mean loss: 0.47149.
- Peak PyTorch allocated GPU memory: 1.979 GiB. This excludes CUDA context and other allocations outside PyTorch's allocator.
- Export: the training script generated full weights, encoder configuration, tokenizer, and `rl_agent_config.json`. These generated checkpoint files are not included in this repository.

The base checkpoint emitted the previously discussed invalid `choice:11+` temperature warning during loading. Training reset inherited temperatures and removed bucket overrides; calibration then fitted the temperatures below. Calibration completed without that warning.

## Calibration and validation

Calibration used 24 synthetic cases / 72 questions, with 24 questions per type. Serving inference used CUDA with `torch.float16` autocast.

| Type | Fitted temperature | Validation accuracy |
| --- | ---: | ---: |
| choice / queue | 1.244217 | 75.0% (18/24) |
| score / urgency | 0.574795 | 79.2% (19/24) |
| noul / cancel_threat | 1.065334 | 91.7% (22/24) |

Validation across 72 questions:

| Metric | Value |
| --- | ---: |
| Top-label accuracy | 0.819444 (59/72) |
| Soft-target negative log likelihood | 0.384868 |
| Mean multiclass Brier sum | 0.236730 |
| ECE using maximum answer probability | 0.058349 |

These are small synthetic practice datasets with recurring scenario patterns. Results demonstrate the training/calibration workflow, not real-world performance. The test split was left unused for evaluation after model and routing choices are frozen.

The checkpoint configuration includes inherited base-model `training` metadata; the `guide_training` and `guide_calibration` entries describe this local run.

## Load the calibrated checkpoint

The checkpoint files are absent from this checkout. Run the training and calibration commands above to recreate them before loading the model.

```python
import laya

agent = laya.load("runs/laya-head-v1", device="cuda")
```
