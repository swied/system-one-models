import json
import torch
import laya
from huggingface_hub import snapshot_download

checkpoint_dir = snapshot_download(
    "convaiinnovations/laya",
    revision="55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
    allow_patterns=[
        "rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*",
    ],
)
agent = laya.load(checkpoint_dir, device="cuda", fast=False)
assert agent.device.type == "cuda", "SDK fell back to CPU; inspect its warning"
print("Loaded:", agent.device, "autocast:", agent.dtype)

questions = {
    "queue": {
        "type": "choice", "instructions": "Which queue should handle this ticket?",
        "criteria": {
            "billing": "charges, invoices, payments, or refunds",
            "technical": "bugs, outages, or product malfunction",
            "other": "neither billing nor technical",
        },
    },
    "urgency": {
        "type": "score", "instructions": "Rate operational urgency.",
        "criteria": ["routine", "time sensitive", "production blocked"],
    },
    "cancel_threat": {
        "type": "noul",
        "instructions": "Does the customer explicitly threaten to cancel?",
    },
}
state = "We were charged twice. Refund the duplicate today or we will cancel."
result = agent.predict(state, questions)
print(json.dumps(result, indent=2))
print("Device after prediction:", agent.device)
free, total = torch.cuda.mem_get_info()
print("Free/total GiB:", round(free / 2**30, 2), round(total / 2**30, 2))
