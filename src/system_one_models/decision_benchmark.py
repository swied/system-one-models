"""Notebook-friendly decision model adapters and benchmark scoring.

Importing this module never initializes clients or downloads model weights.
See notebooks/model-bakeoff.ipynb for an executable comparison.
"""

import csv
import hashlib
import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

DEFAULT_PROMPTS = Path(__file__).with_name("prompts.json")


# ---------------------------------------------------------------- backends

class Backend:
    """One decision-model backend behind a common interface."""
    name = "?"

    def predict(self, state: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Return per-question answer dictionaries. May raise."""
        raise NotImplementedError


class JevBackend(Backend):
    """Hosted Jev using the same SDK as quickstart-jev.ipynb."""

    name = "jev"

    def __init__(self, *, api_key: str | None = None,
                 model: str | None = None, base_url: str | None = None):
        from typesafe_sdk import TypeSafeClient

        key = api_key if api_key is not None else os.environ.get("TYPESAFE_API_KEY")
        if not key:
            raise ValueError("Set TYPESAFE_API_KEY or pass api_key")
        self.client = TypeSafeClient(api_key=key, model=model, base_url=base_url)
        self._last_tokens = None

    def predict(self, state: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        from typesafe_sdk import Choice, Noul, Score

        types = {"choice": Choice, "noul": Noul, "score": Score}
        typed = {name: types[q["type"]](**{k: v for k, v in q.items() if k != "type"})
                 for name, q in questions.items()}
        self._last_tokens = None
        response = self.client.system_one(state=state, questions=typed)
        self._last_tokens = response.usage.input_tokens
        return {name: answer.model_dump() for name, answer in response.answers.items()}

    def close(self) -> None:
        self.client.close()


class D1Backend(JevBackend):
    """Liquid D1; quickstart-d1.ipynb calls this hosted model."""

    name = "d1"

    def __init__(self, *, api_key: str | None = None, model: str = "d1:free",
                 base_url: str = "https://api.liquid.ai/decisions"):
        key = api_key if api_key is not None else os.environ.get("LIQUID_API_KEY")
        if not key:
            raise ValueError("Set LIQUID_API_KEY or pass api_key")
        super().__init__(api_key=key, model=model, base_url=base_url)


class LayaBackend(Backend):
    """Local Laya router. Initialization may download model weights."""

    name = "laya"

    def __init__(self, *, token: str | None = None, preload: bool = True,
                 **router_options: Any):
        from laya import Router

        self.router = Router(preload=preload,
                             token=token if token is not None else os.environ.get("HF_TOKEN"),
                             **router_options)
        self._last_tokens = None

    def predict(self, state: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        self._last_tokens = None
        response = self.router.predict(state=state, questions=questions)
        self._last_tokens = (response.get("usage") or {}).get("input_tokens")
        return response["answers"]


def create_backend(name: str, *, mock: bool = False, **options: Any) -> Backend:
    """Create jev, d1, or laya; mock=True needs no credentials or weights."""
    backends = {"jev": JevBackend, "d1": D1Backend, "laya": LayaBackend}
    if name not in backends:
        raise ValueError(f"Unknown backend {name!r}; choose jev, d1, or laya")
    return MockBackend(name, **options) if mock else backends[name](**options)


class MockBackend(Backend):
    """Deterministic stand-in for dry runs: `p_correct` of answers right,
    plausibly-shaped responses, seeded so runs are reproducible."""

    def __init__(self, name, p_correct=0.85, seed=7):
        self.name = name
        self.p_correct = p_correct
        self.seed = seed
        self._last_tokens = 0.0

    def _rng(self, *parts):
        h = hashlib.sha256("|".join(map(str, (self.seed,) + parts)).encode()).hexdigest()
        return random.Random(int(h[:16], 16))

    def predict(self, state, questions):
        out = {}
        labels = getattr(self, "_labels", {})
        # fake input-token estimate so the cost column is exercised in dry runs
        self._last_tokens = len(json.dumps({"state": state,
                                            "questions": questions})) / 4.0
        for qname, qdef in questions.items():
            qtype = qdef["type"]
            label = labels.get(qname)
            rng = self._rng(self.name, qname, state[:60])
            correct = rng.random() < self.p_correct
            if qtype == "choice":
                options = list(qdef["criteria"].keys())
                pred = label if (correct and label in options) else rng.choice(
                    [o for o in options if o != label] or options)
                probs = {o: (0.02 + 0.05 * rng.random()) for o in options}
                p = 0.65 + 0.30 * rng.random() if pred == label else 0.45 + 0.25 * rng.random()
                probs[pred] = p
                s = sum(probs.values())
                probs = {k: v / s for k, v in probs.items()}
                out[qname] = {"choice": pred, "probabilities": probs,
                              "confidence": p, "_mock_latency_ms": 25 + 60 * rng.random()}
            elif qtype == "noul":
                pred = label if (correct and label in ("yes", "no")) else (
                    "no" if label == "yes" else "yes")
                p = 0.65 + 0.30 * rng.random() if pred == label else 0.50 + 0.25 * rng.random()
                p_yes = p if pred == "yes" else 1 - p
                out[qname] = {"probability": p_yes, "confidence": p,
                              "_mock_latency_ms": 25 + 60 * rng.random()}
            elif qtype == "score":
                levels = list(qdef["criteria"])
                li = levels.index(label) if label in levels else 0
                if correct:
                    pi = li
                    p = 0.60 + 0.30 * rng.random()
                else:
                    pi = min(len(levels) - 1, max(0, li + rng.choice([-1, 1])))
                    p = 0.45 + 0.25 * rng.random()
                probs = {lv: 0.05 + 0.05 * rng.random() for lv in levels}
                probs[levels[pi]] = p
                s = sum(probs.values())
                probs = {k: v / s for k, v in probs.items()}
                out[qname] = {"score": levels[pi], "probabilities": probs,
                              "confidence": p, "_mock_latency_ms": 25 + 60 * rng.random()}
        return out


# ------------------------------------------------------- response shaping

def _as_float(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _extract_yes_prob(raw):
    """Probability that a `noul` answer is 'yes', from several plausible shapes."""
    for key in ("noul", "probability", "prob", "p_yes", "yes"):
        if key in raw:
            return _as_float(raw[key])
    probs = raw.get("probabilities") or {}
    for key in ("yes", "Yes", "true", "True", "1"):
        if key in probs:
            return _as_float(probs[key])
    # last resort: a boolean-ish answer
    ans = str(raw.get("choice", raw.get("answer", ""))).lower()
    if ans in ("yes", "true", "1"):
        return 0.99
    if ans in ("no", "false", "0"):
        return 0.01
    raise ValueError(f"cannot find yes-probability in {str(raw)[:120]}")


def _extract_score(raw, levels):
    """Return (pred_index, prob_of_pred) for a `score` question."""
    probs = raw.get("probabilities") or {}
    # normalize prob keys to level indexes where possible
    idx_probs = {}
    for k, v in probs.items():
        ks = str(k)
        if ks in levels:
            idx_probs[levels.index(ks)] = _as_float(v)
        else:
            try:
                i = int(ks)
                if 0 <= i < len(levels):
                    idx_probs[i] = _as_float(v)
            except ValueError:
                pass
    val = raw.get("score", raw.get("choice", raw.get("prediction")))
    pred_idx = None
    if isinstance(val, bool):
        pass
    elif isinstance(val, (int, float)) and math.isfinite(val) and 0 <= val <= len(levels) - 1:
        pred_idx = max(idx_probs, key=idx_probs.get) if idx_probs else int(val + 0.5)
    elif isinstance(val, str) and val in levels:
        pred_idx = levels.index(val)
    if pred_idx is None and idx_probs:
        pred_idx = max(idx_probs, key=idx_probs.get)
    if pred_idx is None:
        raise ValueError(f"cannot find score prediction in {str(raw)[:120]}")
    prob = idx_probs.get(pred_idx, _as_float(raw.get("confidence"), 0.0))
    return pred_idx, prob


def normalize(qname, qdef, raw, latency_ms):
    """Map one raw backend answer to a common record."""
    qtype = qdef.get("type")
    rec = {"question": qname, "type": qtype, "latency_ms": latency_ms,
           "ok": True, "error": "", "confidence": None}
    try:
        if qtype == "choice":
            pred = raw.get("choice", raw.get("prediction", raw.get("answer")))
            if pred is None:
                raise ValueError(f"no choice in {str(raw)[:120]}")
            probs = raw.get("probabilities", raw.get("probs")) or {}
            probs = {str(k): _as_float(v) for k, v in probs.items()}
            conf = raw.get("confidence")
            prob = probs.get(str(pred), _as_float(conf, 0.0))
            rec.update(pred=str(pred), prob=prob, probs=probs,
                       confidence=_as_float(conf) if conf is not None else None)
        elif qtype == "noul":
            p_yes = _extract_yes_prob(raw)
            conf = raw.get("confidence")
            pred = "yes" if p_yes >= 0.5 else "no"
            # calibration is over the probability of the PREDICTED class
            prob = p_yes if pred == "yes" else 1.0 - p_yes
            rec.update(pred=pred, prob=prob, p_yes=p_yes,
                       confidence=_as_float(conf) if conf is not None else None)
        elif qtype == "score":
            levels = list(qdef["criteria"])
            pred_idx, prob = _extract_score(raw, levels)
            conf = raw.get("confidence")
            rec.update(pred_idx=pred_idx, pred=levels[pred_idx], prob=prob,
                       score_value=raw.get("score") if isinstance(raw.get("score"), (int, float)) else pred_idx,
                       confidence=_as_float(conf) if conf is not None else None)
        else:
            raise ValueError(f"unknown question type {qtype!r}")
        if not math.isfinite(rec["prob"]) or not 0 <= rec["prob"] <= 1:
            raise ValueError("Answer probability must be finite and between zero and one")
        if qtype == "choice" and rec["pred"] not in qdef["criteria"]:
            raise ValueError("Answer choice is not in the question criteria")
    except Exception as e:  # never let one bad answer kill the run
        rec["ok"] = False
        rec["error"] = str(e)
    return rec

# ---------------------------------------------------------------- scoring

def _ranks(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    r = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(xs, ys):
    """Spearman rank correlation (pure stdlib, tie-corrected ranks)."""
    n = len(xs)
    if n < 2:
        return float("nan")
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


ECE_EDGES = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def calibration_stats(pairs):
    """pairs: list of (prob_of_prediction, correct_bool).
    Returns (ece, brier, per-bin rows)."""
    if not pairs:
        return float("nan"), float("nan"), []
    brier = sum((p - (1.0 if c else 0.0)) ** 2 for p, c in pairs) / len(pairs)
    ece, rows = 0.0, []
    for lo, hi in zip(ECE_EDGES[:-1], ECE_EDGES[1:]):
        b = [(p, c) for p, c in pairs if lo < p <= hi or (lo == 0.0 and p == 0.0)]
        if not b:
            rows.append((lo, hi, 0, float("nan"), float("nan")))
            continue
        acc = sum(1 for _, c in b if c) / len(b)
        conf = sum(p for p, _ in b) / len(b)
        ece += abs(acc - conf) * (len(b) / len(pairs))
        rows.append((lo, hi, len(b), conf, acc))
    return ece, brier, rows


def score_backend(name, records, avg_input_tokens):
    """Aggregate one backend's per-question records into metrics."""
    ok = [r for r in records if r["ok"]]
    lat = sorted(r["latency_ms"] for r in ok)
    m = {"backend": name, "items": len(ok), "errors": len(records) - len(ok),
         "p50_ms": statistics.median(lat) if lat else float("nan"),
         "p95_ms": lat[int(0.95 * len(lat))] if lat else float("nan"),
         "avg_input_tokens": avg_input_tokens}

    ch = [r for r in ok if r["type"] == "choice"]
    m["choice_n"] = len(ch)
    m["choice_acc"] = (sum(1 for r in ch if r["pred"] == r["label"]) / len(ch)
                       if ch else float("nan"))

    yn = [r for r in ok if r["type"] == "noul"]
    m["noul_n"] = len(yn)
    m["noul_acc"] = (sum(1 for r in yn if r["pred"] == r["label"]) / len(yn)
                     if yn else float("nan"))

    sc = [r for r in ok if r["type"] == "score" and r.get("pred_idx") is not None
          and r.get("label_idx") is not None]
    m["score_n"] = len(sc)
    if sc:
        m["score_mae"] = sum(abs(r["score_value"] - r["label_idx"]) for r in sc) / len(sc)
        m["score_rho"] = spearman([r["score_value"] for r in sc],
                                  [r["label_idx"] for r in sc])
    else:
        m["score_mae"] = float("nan")
        m["score_rho"] = float("nan")

    pairs = []
    for r in ok:
        if r["type"] == "score":
            if r.get("pred_idx") is not None and r.get("label_idx") is not None:
                pairs.append((r["prob"], r["pred_idx"] == r["label_idx"]))
        else:
            pairs.append((r["prob"], r["pred"] == r["label"]))
    m["cal_n"] = len(pairs)
    m["ece"], m["brier"], m["bins"] = calibration_stats(pairs)

    # Pricing is caller-supplied; unknown pricing must not look like free inference.
    m["est_usd_per_1k"] = None
    return m


# ---------------------------------------------------------------- rendering

def _fmt(x, digits=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


METRIC_COLS = [
    ("backend", "backend", 0), ("items", "items", 0),
    ("choice_acc", "choice acc", 3), ("noul_acc", "noul acc", 3),
    ("score_mae", "score MAE", 3), ("score_rho", "score rho", 3),
    ("ece", "ECE", 3), ("brier", "Brier", 3),
    ("p50_ms", "p50 ms", 1), ("est_usd_per_1k", "est $/1k", 4),
]


def render_markdown(metrics_list):
    lines = ["# Bake-off results", ""]
    header = "| " + " | ".join(c[1] for c in METRIC_COLS) + " |"
    lines += [header, "|" + "|".join(["---"] * len(METRIC_COLS)) + "|"]
    for m in metrics_list:
        row = "| " + " | ".join(_fmt(m[c[0]], c[2]) for c in METRIC_COLS) + " |"
        lines.append(row)
    lines += ["",
              "_choice acc / noul acc_: fraction of exact-label matches. "
              "_score MAE_: mean |expected score - true level| (lower is better). "
              "_score rho_: Spearman rank correlation of predicted vs true levels. "
              "_ECE_: expected calibration error over probability bins (lower is better; "
              "values depend on the dataset). "
              "_Brier_: mean squared error of top-label probabilities versus correctness (lower is better). "
              "_est $/1k_: estimated USD per 1,000 scenario requests, using supplied input-token pricing._",
              ""]
    for m in metrics_list:
        lines += [f"## {m['backend']} — reliability diagram",
                  f"_{m['cal_n']} scored items, {m['errors']} errors_",
                  "",
                  "| prob bin | n | avg conf | emp acc |",
                  "|---|---|---|---|"]
        for lo, hi, n, conf, acc in m["bins"]:
            lines.append(f"| {lo:.1f}-{hi:.1f} | {n} | {_fmt(conf)} | {_fmt(acc)} |")
        lines.append("")
    return "\n".join(lines)


def write_csv(metrics_list, path):
    cols = [c[0] for c in METRIC_COLS] + ["choice_n", "noul_n", "score_n",
                                          "cal_n", "errors", "p95_ms",
                                          "avg_input_tokens"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for m in metrics_list:
            w.writerow([m.get(c) if not isinstance(m.get(c), float) or not math.isnan(m.get(c))
                        else "" for c in cols])


# ---------------------------------------------------------------- driver

def load_prompts(path: str | Path = DEFAULT_PROMPTS) -> list[dict[str, Any]]:
    """Load labeled scenarios, validating question types and score labels."""
    with open(path, encoding="utf-8") as f:
        scenarios = json.load(f)
    for scenario in scenarios:
        for name, question in scenario["questions"].items():
            if question["type"] not in {"choice", "score", "noul"}:
                raise ValueError(f"Unknown question type for {name}")
            label = scenario["labels"][name]
            if question["type"] == "score":
                scenario.setdefault("_label_idx", {})[name] = list(question["criteria"]).index(label)
    return scenarios


@dataclass
class BenchmarkResult:
    """Metrics and individual answers, ready to inspect in a notebook."""

    metrics: list[dict[str, Any]]
    records: list[dict[str, Any]]

    def markdown(self) -> str:
        return render_markdown(self.metrics)

    def write(self, directory: str | Path) -> None:
        """Write Markdown, CSV, and JSON reports to directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "results.md").write_text(self.markdown() + "\n", encoding="utf-8")
        write_csv(self.metrics, directory / "results.csv")

        def clean(value):
            if isinstance(value, float) and not math.isfinite(value):
                return None
            if isinstance(value, dict):
                return {k: clean(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [clean(v) for v in value]
            return value

        (directory / "results.json").write_text(
            json.dumps(clean({"metrics": self.metrics, "records": self.records}),
                       indent=2, allow_nan=False), encoding="utf-8")


def run_benchmark(backends: Iterable[Backend],
                  scenarios: Iterable[dict[str, Any]] | None = None, *,
                  input_prices_per_million: dict[str, float] | None = None) -> BenchmarkResult:
    """Run each backend on the same scenarios; preserve per-question failures.

    Latency is measured per scenario request (repeated on its question records).
    Score MAE uses expected numeric scores; calibration uses the modal level.
    Missing pricing/token usage is reported as unknown. No files are written.
    """
    backends = list(backends)
    if not backends or len({b.name for b in backends}) != len(backends):
        raise ValueError("Supply at least one backend, with unique names")
    scenarios = list(load_prompts() if scenarios is None else scenarios)
    if not scenarios:
        raise ValueError("Supply at least one scenario")
    records, metrics = [], []
    for backend in backends:
        recs, tokens = [], []
        for scenario in scenarios:
            if isinstance(backend, MockBackend):
                backend._labels = scenario["labels"]
            start = time.perf_counter()
            try:
                raw = backend.predict(scenario["state"], scenario["questions"])
                error = None
            except Exception as exc:
                raw, error = {}, str(exc)
            elapsed = (time.perf_counter() - start) * 1000
            tokens.append(getattr(backend, "_last_tokens", None) if error is None else None)
            for name, question in scenario["questions"].items():
                if error is not None:
                    rec = dict(question=name, type=question["type"], latency_ms=elapsed,
                               ok=False, error=error)
                else:
                    answer = raw.get(name, {})
                    rec = normalize(name, question, answer,
                                    answer.get("_mock_latency_ms", elapsed)
                                    if isinstance(answer, dict) else elapsed)
                label = scenario["labels"][name]
                rec.update(backend=backend.name, scenario=scenario["id"], label=label,
                           label_idx=list(question["criteria"]).index(label)
                           if question["type"] == "score" else None)
                recs.append(rec)
        avg_tokens = sum(tokens) / len(tokens) if all(t is not None for t in tokens) else None
        metric = score_backend(backend.name, recs, avg_tokens)
        price = (input_prices_per_million or {}).get(backend.name)
        if price is not None and avg_tokens is not None:
            metric["est_usd_per_1k"] = avg_tokens * price / 1000
        metrics.append(metric)
        records.extend(recs)
    return BenchmarkResult(metrics, records)
