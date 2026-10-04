# Jev-like "System 1" Decision Models — Weekend Learning Plan
**Sat Oct 3 – Sun Oct 4, 2026** · Home PC (RTX 2080 Ti, 11GB VRAM)

## The big idea in one paragraph

A new category of model appeared in September 2026: **decision models** (TypeSafe calls
them "System One" models, after Kahneman's fast/intuitive System 1). Instead of
generating text token-by-token like an LLM, they take a *state* (a ticket, an email, a
command) plus *typed questions* — pick-one-of-N, score-on-a-scale, yes/no — and return
**calibrated probabilities in a single forward pass**. No streaming, no parsing prose
into labels, no hallucinated categories. That makes them ~100–500ms fast, nearly free
per call, and shaped exactly like the `if` conditions inside real software. The three
names to know: **Jev** (TypeSafe's hosted API, closed weights), **Laya** (open-weight,
runs locally), and **D1** (Liquid's hosted decision model).

## The three players

| | Jev (TypeSafe) | Laya | D1 (Liquid) |
|---|---|---|---|
| Execution | Hosted API | Local Router | Hosted API |
| Repository adapter | `jev` | `laya` | `d1` |
| Credentials | `TYPESAFE_API_KEY` | `HF_TOKEN` when needed | `LIQUID_API_KEY` |
| Quickstart | `quickstart-jev.ipynb` | `quickstart-laya.ipynb` | `quickstart-d1.ipynb` |
| Interface | TypeSafe SDK | Python `Router.predict()` | TypeSafe SDK with Liquid base URL |

---

## Friday Oct 2 — prep evening (~1 hour)

**1. Python environment** (this repository targets Python 3.13)
```bash
uv sync --locked
uv run jupyter lab notebooks/
```

**2. Get a Jev API key.** Sign up at typesafe.ai — the waitlist was dropped Sep 20, and
there's a free tier. (Backup route: OpenRouter also serves Jev with just an OpenRouter
key.) Save the key as an env var; you'll use it Saturday:
```
export TYPESAFE_API_KEY="..."
```

**3. Prepare D1 access.** Save `LIQUID_API_KEY` in
`~/.config/liquid/secrets.env`, following the README credential instructions.
Run `notebooks/quickstart-d1.ipynb` to check hosted inference.

**4. Smoke test Laya** (should answer in well under a second on CPU):
```python
from laya import Router
router = Router(preload=True)
result = router.predict(
    state="The build is red on a Friday evening.",
    questions={"deploy": {
        "type": "choice",
        "instructions": "Deploy now or wait?",
        "criteria": {"deploy": "Ship now", "wait": "Hold the deployment"}}},
)
print(result)
```

**5. Check the bakeoff offline.** Open `notebooks/model-bakeoff.ipynb`,
set `MOCK = True`, and run all cells before attempting real inference.

---

## Saturday Oct 3 — foundations + the hosted API

### Morning: how decision models work (~2.5 hrs)

- **System 1 vs System 2.** Kahneman's distinction: fast pattern-matching judgment vs
  slow deliberative reasoning. Jev-like models are AI's System 1 — the reflex layer,
  not the thinker. (This is also why they pair with LLMs rather than replacing them.)
- **Typed outputs.** The three question types and when each fits:
  - `choice` — one of up to 255 options you define (department, intent, action)
  - `score` — position on an ordered scale you describe (severity, urgency, quality)
  - `noul` — yes/no probability for a proposition ("the customer asks for a refund")
- **One forward pass, zero tokens.** An LLM writes thousands of tokens to produce a
  one-word label; a decision model scores your whole answer space in parallel and
  emits no text. That's the entire reason for the 100–500ms latency and the
  "output too cheap to meter" pricing.
- **Calibration — the core concept.** A *calibrated* 0.8 means "I'm right about 80%
  of the time I say 0.8." Jev was explicitly trained for this (TypeSafe calls the
  method RLCD — reinforcement learning for calibrated decisions). Choice/score
  answers also carry a `confidence` derived from how concentrated the distribution
  is. This is what makes confidence-gated automation possible: act above a
  threshold, escalate below it.
- **The routing framework** — for every decision point in a system, in order:
  1. Exactly computable (counts, dates, regex)? → keep it in code.
  2. Open-ended output or multi-step reasoning? → LLM.
  3. Fixed options / scale / yes-no, judgeable in seconds? → decision model.
  4. Wrong answer expensive? → threshold tuned on your data; below it goes to a human.
  5. Someone needs to know *why*? → log the decision; have an LLM explain afterward.
- **"Can't hallucinate," honestly.** The guarantee is schema compliance — the answer
  is always one of your options — not correctness. A ticket can still be misrouted
  with high confidence. Independent tests confirm the speed/cost wins but show the
  probabilities need verification on *your* data. That's Sunday's bake-off.

📖 Reading: the laozhang.ai Jev guide (best single overview, with the five-checks
framework and cost arithmetic), the GoPenAI "Jev, Explained" piece.

### Afternoon: hands-on (~3 hrs)

**Lab 1 — Jev API quickstart.** Same state, three question types, one request:
```bash
curl -s https://api.typesafe.ai/v1/systemone \
  -H "Authorization: Bearer $TYPESAFE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jev-latest",
    "state": "Customer writes: my card was charged twice for order #4812, please fix this today.",
    "questions": {
      "team":    {"type": "choice", "instructions": "Which team handles this?",
                  "criteria": {"billing": "payment and charge issues",
                               "support": "general help",
                               "sales": "new purchases"}},
      "urgency": {"type": "score", "instructions": "How urgent is this?",
                  "criteria": ["not urgent", "normal", "urgent", "critical"]},
      "refund":  {"type": "noul", "instructions": "Does the customer ask for a refund?"}
    }}'
```
Read the response: `choice` + full `probabilities` + `confidence` per question, all
from one parallel pass.

**Lab 2 — Laya, same questions.** Run the identical state/questions through
`Router.predict`. Compare the output shapes — this is the open/hosted interchange
in miniature.

**Lab 3 — build a confidence-gated classifier.** Collect 20–30 short texts (support
tickets, emails — or phishing lures, given your inbox project: a `noul` question
"is this a job-scam phish?" is a perfect fit). Label them yourself, run them
through Jev, and sweep the confidence threshold: what fraction auto-resolves at
0.9? At 0.7? How many errors slip through? You've just built the core loop of
production decision automation.

### Saturday evening (optional reading)
- Emil Lindfors' independent Jev-vs-LLM test: the finding that matters is that
  Jev's ≥0.9 probabilities cleanly separated right from wrong answers while the
  LLM's self-reported confidence didn't.

---

## Sunday Oct 4 — self-hosting + the bake-off

### Morning: run the open weights (~2 hrs)

Run `notebooks/quickstart-laya.ipynb` and inspect the local Router response.
Record the hardware used, weight-loading time, and inference latency separately.
Then run `notebooks/quickstart-d1.ipynb` and compare its hosted response with Jev.

Compare the practical tradeoffs:
- Local Laya inference versus hosted Jev and D1 requests.
- Local hardware requirements versus hosted account limits.
- Where input data is processed and how credentials are loaded.
- Which configuration gives useful accuracy and calibration on your scenarios.

### Afternoon: the bake-off (~3 hrs) — the centerpiece

Label 30–50 items Saturday (or reuse Lab 3's set). Run all three backends over the
identical set: **Jev API, Liquid D1, and Laya**.

1. **Accuracy** per backend. Compare the labeled results without assuming a ranking.
2. **Latency**: p50 per backend. Feel the architecture difference, don't just read it.
3. **Cost per 1,000 scenario requests**: use applicable hosted input-token pricing
   and reported usage; account for local hardware costs separately.
4. **Calibration — the data-scientist experiment.** Bin each model's predicted
   probabilities (0.5–0.6, 0.6–0.7, …) and plot empirical accuracy per bin: the
   **reliability diagram**. Which model is honest about its uncertainty? Where does
   each model's confidence stop meaning anything?
5. **Threshold sweep.** For each backend, plot the coverage-vs-accuracy curve as you
   move the auto-act threshold. This curve *is* the production decision.

**Adaptation, conceptually.** Explore whether better question instructions,
criteria, or task-specific examples improve results before considering changes
to local weights. Validate each change on held-out labeled scenarios.

### Capstone — pick one

1. **Local guardrail microservice** (pi-warden style): an allow/hold gate in front
   of shell commands — `choice: [allow, hold]` with a confidence threshold, risky
   patterns escalated. The canonical System 1 production pattern.
2. **Confidence-gated ticket router**: Lab 3 hardened — review queue for
   below-threshold items, logged decisions, your selected backend behind it.
3. **Bake-off write-up**: the three-backend comparison (accuracy, calibration plots,
   latency, cost) as a short report. The most data-scientist-shaped artifact of
   the three.

---

## Resources

- TypeSafe Jev docs — https://docs.typesafe.ai (API reference, question-type guide)
- "Jev AI Model: What It Is, How It Compares, and How to Use It" — laozhang.ai
  (Sep 24, 2026; best single overview)
- "Jev, Explained: The AI That Refuses to Write a Word" — GoPenAI (Sep 2026)
- Laya repo — https://github.com/NandhaKishorM/laya · weights: `convaiinnovations/laya`
- "The Laya Model: Python, Local APIs, and How It Compares with Jev AI" — dev.to
- `laya-server` (community FastAPI `/v1/systemone` wrapper) —
  https://github.com/noahbclarkson/laya-server

## How this weekend differs from the LLM one

No 9GB downloads of chat models, no quantization ladders, no sampling sweeps.
Everything here is small, fast, and CPU-friendly — the 2080 Ti is overkill for
most of it, which is itself the lesson: this is the *reflex* layer of AI systems,
and it runs at the edge. The math you'll lean on isn't attention — it's
**calibration**: what a probability means, when to trust it, and where to set the
threshold. That's pure data science.
