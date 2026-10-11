import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import laya
from laya.agent import Agent
from laya.common import QTYPES, build_sequence, collate_items, render_options
from safetensors.torch import save_file
from huggingface_hub import snapshot_download

MODEL_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"

def read_items(agent, filename):
    items = []
    with open(filename, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for qid, definition in row["questions"].items():
                Agent._check_question(qid, definition)
                q = Agent._to_internal(definition)
                options = render_options(q)
                target = np.asarray(row["targets"][qid], dtype=np.float32)
                if (target.shape != (len(options),) or
                    not np.isfinite(target).all() or (target < 0).any() or
                    not np.isclose(target.sum(), 1.0, atol=1e-5)):
                    raise ValueError(f"Invalid target on line {line_number}, {qid}")
                ids, markers = build_sequence(
                    agent.tok, row["state"], q,
                    agent.cfg["max_len"], agent.cfg["head_max_len"],
                    truncate_left=isinstance(row["state"], list),
                )
                if len(markers) != len(options):
                    raise ValueError(f"Options truncated on line {line_number}, {qid}")
                items.append({
                    "ids": ids, "markers": markers, "qtype": QTYPES[q["t"]],
                    "target": target.tolist(), "label": int(target.argmax()),
                    "case_id": row.get("case_id", str(line_number)), "qid": qid,
                })
    if not items:
        raise ValueError("Dataset contains no question items")
    return items

def model_inputs(batch, device):
    return {key: batch[key].to(device) for key in (
        "input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype",
    )}

def save_checkpoint(agent, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    weights = {key: value.detach().cpu().contiguous()
               for key, value in agent.model.state_dict().items()}
    save_file(weights, str(output / "model.safetensors"))
    agent.model.encoder.config.save_pretrained(output / "encoder")
    agent.tok.save_pretrained(output / "tokenizer")
    with open(output / "rl_agent_config.json", "w", encoding="utf-8") as handle:
        json.dump(agent.cfg, handle, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="convaiinnovations/laya")
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    if args.accumulation < 1 or args.epochs < 1 or args.max_length < 32:
        raise ValueError("Invalid epoch, accumulation, or length setting")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    model_path = args.model
    if not Path(model_path).is_dir():
        model_path = snapshot_download(
            args.model, revision=args.revision,
            allow_patterns=[
                "rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*",
            ],
        )
    agent = laya.load(model_path, device=args.device)
    if agent.device.type != torch.device(args.device).type:
        raise RuntimeError("Requested device unavailable; inspect SDK fallback")
    agent.cfg = copy.deepcopy(agent.cfg)
    agent.cfg["max_len"] = min(agent.cfg["max_len"], args.max_length)
    agent.cfg["temperature"] = [1.0, 1.0, 1.0]
    agent.cfg.pop("temperature_by_options", None)
    agent.cfg["fine_tuned"] = True
    agent.cfg["model_name"] = "laya-domain-head"
    agent.cfg["guide_training"] = {
        "objective": "soft_cross_entropy", "encoder_frozen": True,
        "epochs": args.epochs, "seed": 42, "source_revision": args.revision,
    }
    model = agent.model.float()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            not name.startswith("encoder.") and not name.startswith("act_head.")
        )
    model.head_checkpointing = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    print("Trainable parameters:", sum(p.numel() for p in trainable))
    items = read_items(agent, args.data)
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=0.01,
    )
    for epoch in range(args.epochs):
        model.train()
        model.encoder.eval()
        model.act_head.eval()
        random.shuffle(items)
        total_loss = 0.0
        for start in range(0, len(items), args.accumulation):
            group = items[start:start + args.accumulation]
            optimizer.zero_grad(set_to_none=True)
            for item in group:
                batch = collate_items([[item]], agent.tok.pad_token_id)
                logits, _ = model(**model_inputs(batch, agent.device))
                target = batch["target"].to(agent.device)
                loss = -(target * logits.log_softmax(-1)).sum(-1).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                (loss / len(group)).backward()
                total_loss += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
        print(f"Epoch {epoch + 1}: mean loss {total_loss / len(items):.5f}")
    model.eval()
    save_checkpoint(agent, args.output)
    if agent.device.type == "cuda":
        print("Peak allocated GiB:", torch.cuda.max_memory_allocated() / 2**30)

if __name__ == "__main__":
    main()
