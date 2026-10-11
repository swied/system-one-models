# Laya Fine-Tuning Weekend — Plan

**Goal:** take Laya — the fastest backend in your bake-off, but the worst zero-shot — and specialize it for *one* type of decision question, then measure the before/after gap with your existing harness.

**Why this is the right project:** Laya's own model card is blunt: zero-shot typed-decision accuracy is **0.362** (near the 0.318 random baseline), while the checkpoint fine-tuned on a benchmark's training split hits **0.766** — above Jev's 0.727. The card's line: *"Laya is a fast base to specialise, not a zero-shot decision engine."* Independent projects confirm the jump is real and cheap: one Snake-game fine-tune went **27.0% → 81.4%** held-out accuracy with 3 epochs on 3,000 examples, trained entirely on a single Apple Silicon Mac. Your 2080 Ti has far more headroom.

## 1. Pick your task (Saturday morning, ~1 hr)

Fine-tuning works when the task is **one question type, labelable, and repeatable**. Good candidates:

- **One of your existing bake-off scenarios.** The 33 questions in `prompts.json` are already typed and labeled — e.g. the WaterPlum phish lures as a `choice`/`noul` phishing classifier. Fastest path: expand one scenario family into a few thousand labeled variants.
- **A new domain you care about:** email triage (inbox / spam / phishing), ticket routing, content-moderation verdicts. The model card names invoice processing, ticket routing, moderation, and guardrails as target workloads.

Task-design rules (from Laya's hard limits):

- Keep choice sets **small** (under ~20 options; fewer is better for calibration).
- Each **option ≤ 48 tokens**; each **question ≤ 512 tokens** total (question + options + state).
- One question type per fine-tune (`choice`, `noul`, or `score`) — mixing types is possible but start pure.

**Decision to make before training:** which single question type. My recommendation: start with a `choice` question with 3–6 options — it's the easiest to label and evaluate.

## 2. How Laya learns (read this, don't skip)

- **Architecture:** ModernBERT-large backbone (395M) + a decision head trained from scratch (2 transformer layers, an option-marker scorer, act/escalate head) = **421M total**. Each option is scored at its own `[MASK]` marker; softmax over the question's options.
- **Training method (RLCD):** the policy reports a probability distribution; training adds Gaussian noise to the logits (exploration) and rewards **proper scoring rules** — log score + spherical score (plus ranked probability score for ordinal `score` questions). Maximizing expected reward forces *honest* probabilities, which is why the confidence numbers are thresholdable.
- **Two regimes — do them in this order:**
  1. **Head-only tuning (start here):** freeze the encoder, train only the decision head. One Tetris project peaked at **11.0 GB** doing this — comfortably inside your 2080 Ti. This alone may get you most of the gain.
  2. **Full fine-tune (if head-only plateaus):** encoder + head with AdamW, differential learning rates (encoder 2.5e-5, head 1e-4).

## 3. Dataset (Saturday, the main work)

Format per example: `{state, question, options[], label}` — exactly the shape your `prompts.json` already uses. Targets:

- **Head-only:** 1,000–5,000 labeled examples.
- **Full fine-tune:** 5,000–30,000 (the reference Kaggle run used ~30k questions, 4 epochs).

Labeling strategies, cheapest first:

1. **Rule/teacher program** (best when the task is computable): write a deterministic teacher like the Snake project's BFS solver (shortest path to food ⇒ label). Zero labeling cost, perfectly consistent labels.
2. **LLM-as-teacher:** run a strong model over unlabeled inputs and take its decisions as labels — this is how AIMultiple built their browser-task dataset (606 records from GLM-5.3-Flash runs). Spot-check ~100 by hand to estimate teacher error.
3. **Hand labeling:** slowest, but fine for a few hundred seed examples to validate the other two.

Splits: **80 / 10 / 10** train/val/test. **Freeze the test set before training starts** and never train on it — it's your before/after scoreboard. Val drives early stopping.

## 4. Training setup on the 2080 Ti (Sunday)

**Precision:** fp16. (Turing has no native bf16 — use fp16, which the reference T4 configs also use.)

**Memory (11 GB budget):**

| Regime | Fit | Settings hint |
|---|---|---|
| Head-only | Easy | micro-batch 4–8, no tricks needed |
| Full fine-tune, fp16 | Fits with care | micro-batch 1–2, gradient accumulation 16–64, keep sequence length ≤ 1024 |

**Two starting points (pick one):**

- **A. The reference notebook** — `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb` in [nandhakishorm/laya](https://github.com/nandhakishorm/laya). Does the whole loop: dataset → RLCD training (proper-scoring rewards, GRPO-style policy gradient) → temperature fitting → eval → Hub push. Written for Kaggle 2×T4 (~4–5 hrs for 4 epochs over 30k questions); adapt the GPU/paths cells to your local box.
- **B. The script route** — [pixelzx0/xeron](https://github.com/pixelzx0/xeron): `scripts/preprocess.py` → `torchrun scripts/train_ddp.py` with `configs/finetune.json`. Hyperparameters as env vars: `EPOCHS=4`, `LR_ENCODER=2.5e-5`, `LR_HEAD=1e-4`, `GROUP_SIZE=4` (GRPO samples), `SIGMA_START=0.4`/`SIGMA_END=0.1` (exploration noise), `CHECKPOINT_EVERY=1` (survive interruptions). Single-GPU works — DDP with one process is fine.

**Reference recipe** (from the producer's own fine-tune): 4 epochs, AdamW (encoder 2.5e-5, head 1e-4), seed 42, proper-scoring policy gradient + soft cross-entropy — then one fitted temperature per question type (see §5).

**Watch while training:** val accuracy every epoch + val ECE. Stop when val accuracy plateaus for 2 epochs — with small datasets Laya overfits fast.

## 5. Calibration (Sunday, ~30 min, don't skip)

Training improves accuracy; **temperature fitting** fixes the confidence numbers. The model card reports fitting one temperature per question type moves mean ECE **0.466 → 0.081**. Procedure: freeze the trained model, hold out a few hundred val examples, fit a single scalar temperature per question type minimizing NLL (or ECE). Your harness already computes ECE/Brier and draws reliability diagrams — run it before and after.

## 6. Evaluation — reuse your harness

Point your existing `LayaBackend` at the tuned checkpoint and re-run `bakeoff.py` on:

1. **Your frozen test set** (the one from §3) — the honest number.
2. **The relevant slice of `prompts.json`** — the before/after comparison against base Laya, Jev, and d1.

Report the same table you're used to: choice/noul accuracy, score MAE + Spearman, ECE, Brier, p50/p95 latency. Expected story: accuracy jumps toward the fine-tuned-checkpoint regime while latency stays ~30–40 ms.

## 7. Weekend schedule

**Saturday — data day**
- Morning: pick the one question type (§1); freeze the test set first.
- Afternoon: build the labeler (teacher program or LLM-as-teacher), generate 3–10k examples, spot-check quality.
- Evening: kick off head-only training; let it run overnight.

**Sunday — train & measure day**
- Morning: evaluate head-only vs base. If the gap is small, launch the full fine-tune.
- Afternoon: fit temperatures (§5), run the full harness eval (§6), write up the before/after table.
- Stretch: **DAgger round** — collect the tuned model's mistakes on fresh inputs, label them, retrain. This is the highest-ROI second iteration.

## 8. Gotchas

- **48-token option cap / 512-token question budget** are hard — truncate or rephrase, don't silently overflow.
- **Label quality beats quantity.** 3,000 clean labels beat 30,000 noisy ones (the Snake result used perfectly-labeled synthetic data).
- **Don't train on your test set.** Obvious, easy to violate when you're iterating on data generation.
- **Start simple:** head-only + cross-entropy first. Add the full RLCD recipe only if head-only plateaus.
- **Calibration is a separate step**, not a training hyperparameter — fit temperatures after, not during.

## 9. Resources

- Model: [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) (Apache-2.0)
- Reference fine-tune notebook: [nandhakishorm/laya](https://github.com/nandhakishorm/laya) — `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`
- Script-based training (DDP, env-var hyperparams): [pixelzx0/xeron](https://github.com/pixelzx0/xeron)
- Head-only tuning example (encoder frozen): [llamadart — laya_tetris training](https://github.com/leehack/llamadart/blob/HEAD/example/laya_tetris/training/README.md)
- Colab T4 fine-tune pipeline (dataset gen + DAgger): [okbatti/snakegame_laya](https://github.com/okbatti/snakegame_laya/blob/HEAD/README.md)
- Worked before/after (27% → 81.4%, Apple Silicon): [yliuai/laya-eval](https://github.com/yliuai/laya-eval)
- RLCD recipe detail (producer's notebook, 4 epochs, AdamW 2.5e-5/1e-4): [AIMultiple/laya-typed-decisions-browser-ft](https://huggingface.co/AIMultiple/laya-typed-decisions-browser-ft)
