import argparse
import json
from pathlib import Path

import numpy as np
import torch
import laya
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from laya.common import QTYPE_NAMES, collate_items, ece_score
from finetune_laya_head import read_items

@torch.inference_mode()
def collect(agent, filename):
    rows = []
    for item in read_items(agent, filename):
        batch = collate_items([[item]], agent.tok.pad_token_id)
        logits, _ = agent._infer(batch)
        k = len(item["target"])
        rows.append((
            item["qtype"], logits[0, :k].float().cpu().numpy().astype(float),
            np.asarray(item["target"], dtype=float),
        ))
    return rows

def nll(rows, temperature):
    losses = []
    for _, logits, target in rows:
        z = logits / temperature
        losses.append(-float(target @ (z - logsumexp(z))))
    return float(np.mean(losses))

def report(rows, temperatures):
    correct, confidence, brier, losses = [], [], [], []
    for qt, logits, target in rows:
        p = softmax(logits / temperatures[qt])
        correct.append(float(p.argmax() == target.argmax()))
        confidence.append(float(p.max()))
        brier.append(float(np.square(p - target).sum()))
        losses.append(-float(target @ np.log(np.maximum(p, 1e-12))))
    return {
        "questions": len(rows), "top_label_accuracy": float(np.mean(correct)),
        "soft_target_nll": float(np.mean(losses)),
        "multiclass_brier_sum": float(np.mean(brier)),
        "ece_max_probability": ece_score(np.asarray(confidence), np.asarray(correct)),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--evaluate", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    model_dir = Path(args.model)
    if not (model_dir / "rl_agent_config.json").is_file():
        parser.error(
            f"No trained checkpoint found: {(model_dir / 'rl_agent_config.json').resolve()}\n"
            "Run finetune_laya_head.py to completion first, with --output set to "
            "the same directory as this command's --model. If training saved elsewhere, "
            "set --model to that output directory."
        )
    agent = laya.load(str(model_dir.resolve()), device=args.device)
    if agent.device.type != torch.device(args.device).type:
        raise RuntimeError("Requested device unavailable")
    rows = collect(agent, args.calibration)
    temperatures = [1.0, 1.0, 1.0]
    for qt in range(3):
        selected = [row for row in rows if row[0] == qt]
        if len(selected) < 20:
            print("Too few calibration items for", QTYPE_NAMES[qt], "; using T=1")
            continue
        fitted = minimize_scalar(
            lambda log_t: nll(selected, float(np.exp(log_t))),
            bounds=(np.log(0.5), np.log(5.0)), method="bounded",
        )
        if not fitted.success:
            raise RuntimeError("Temperature optimization failed")
        temperatures[qt] = float(np.exp(fitted.x))
    agent.cfg["temperature"] = temperatures
    agent.cfg.pop("temperature_by_options", None)
    agent.cfg["guide_calibration"] = {
        "method": "per_type_bounded_temperature", "items": len(rows),
        "runtime_dtype": str(agent.dtype), "device": str(agent.device),
    }
    with open(model_dir / "rl_agent_config.json", "w", encoding="utf-8") as handle:
        json.dump(agent.cfg, handle, indent=2)
    print("Temperatures:", temperatures)
    evaluation = collect(agent, args.evaluate)
    print("Evaluation:", json.dumps(report(evaluation, temperatures), indent=2))
    for qt in range(3):
        selected = [row for row in evaluation if row[0] == qt]
        if selected:
            print(QTYPE_NAMES[qt], report(selected, temperatures))

if __name__ == "__main__":
    main()
