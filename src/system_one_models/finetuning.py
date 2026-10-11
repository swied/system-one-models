"""Reusable choice-data validation, strict Laya encoding, and calibration.

Model loading is explicit; imports never download weights or read secrets.
"""

import hashlib
import json
from pathlib import Path
from typing import Any


def read_examples(path: str | Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Empty dataset: {path}")
    ids = set()
    for row in rows:
        if row['id'] in ids:
            raise ValueError('Duplicate example id')
        ids.add(row['id'])
        if not isinstance(row['state'], str) or not row['state'].strip():
            raise ValueError('State must be nonempty text')
        if not isinstance(row['question'], str) or not row['question'].strip():
            raise ValueError('Question must be nonempty text')
        options = row['options']
        if not isinstance(options, dict) or not 2 <= len(options) < 20:
            raise ValueError('Use 2–19 named choice options')
        if any(not isinstance(k, str) or not isinstance(v, str) or not k or not v
               for k, v in options.items()):
            raise ValueError('Options must have nonempty names and descriptions')
        if row['label'] not in options:
            raise ValueError('Label must name an option')
    return rows


def scenarios(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert the exercise format to the existing benchmark schema."""
    return [{'id': r['id'], 'state': r['state'],
             'questions': {'team': {'type': 'choice', 'instructions': r['question'],
                                    'criteria': r['options']}},
             'labels': {'team': r['label']}} for r in rows]


def file_digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_dataset(directory: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Reject changed frozen splits, shared examples, or shared template families."""
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    splits = {}
    seen_ids, seen_states, seen_families = set(), set(), set()
    for split, digest in manifest['sha256'].items():
        path = directory / f'{split}.jsonl'
        if file_digest(path) != digest:
            raise ValueError(f'Frozen split changed: {split}')
        rows = read_examples(path)
        ids = {r['id'] for r in rows}
        states = {r['state'] for r in rows}
        families = {r['family'] for r in rows}
        if seen_ids & ids or seen_states & states or seen_families & families:
            raise ValueError('Split leakage detected')
        if len(states) != len(rows):
            raise ValueError('Duplicate states within split')
        seen_ids |= ids
        seen_states |= states
        seen_families |= families
        splits[split] = rows
    if set(splits) != {'train', 'validation', 'calibration', 'test'}:
        raise ValueError('Missing required split')
    return splits


def encode_examples(agent: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reject truncation of instructions, options, or evidence before training."""
    from laya.agent import Agent
    from laya.common import QTYPES, build_sequence, render_options

    items = []
    for row in rows:
        question = {'type': 'choice', 'instructions': row['question'], 'criteria': row['options']}
        Agent._check_question('team', question)
        q = Agent._to_internal(question)
        opts = render_options(q)
        lengths = [len(agent.tok.encode(' ' + o, add_special_tokens=False)) for o in opts]
        if max(lengths) > 48:
            raise ValueError(f"{row['id']}: option exceeds 48 tokens")
        instruction_length = len(agent.tok.encode('choice question: ' + q['ins'], add_special_tokens=False))
        # build_head reserves one marker per option and at least 16 instruction tokens.
        if sum(n + 1 for n in lengths) + max(16, instruction_length) > agent.cfg['head_max_len']:
            raise ValueError(f"{row['id']}: question head would truncate")
        ids, markers, stats, state_stats = build_sequence(
            agent.tok, row['state'], q, min(512, agent.cfg['max_len']),
            agent.cfg['head_max_len'], return_stats=True, return_truncation_stats=True)
        if (state_stats['truncated'] or stats['tokens_per_option'] is not None
                or len(markers) != len(opts) or stats['options_distinct'] != len(opts)):
            raise ValueError(f"{row['id']}: input would truncate or collapse options")
        label = list(row['options']).index(row['label'])
        items.append({'ids': ids, 'markers': markers, 'qtype': QTYPES['choice'],
                      'target': [float(i == label) for i in range(len(opts))],
                      'label': label, 'id': row['id']})
    return items


def fit_temperature(logits: Any, labels: Any) -> float:
    """Fit calibration-set NLL within the installed Laya SDK bounds (0.5–5)."""
    import numpy as np
    from scipy.optimize import minimize_scalar
    from scipy.special import logsumexp

    logits, labels = np.asarray(logits, dtype=float), np.asarray(labels, dtype=int)
    if (logits.ndim != 2 or not len(logits) or labels.shape != (len(logits),)
            or not np.isfinite(logits).all() or (labels < 0).any()
            or (labels >= logits.shape[1]).any()):
        raise ValueError('Invalid calibration logits/labels')
    def objective(log_temperature):
        z = logits / np.exp(log_temperature)
        return float(np.mean(logsumexp(z, axis=1) - z[np.arange(len(z)), labels]))
    result = minimize_scalar(objective, bounds=(np.log(0.5), np.log(5.0)), method='bounded')
    if not result.success:
        raise RuntimeError('Temperature optimization failed')
    return float(np.exp(result.x))


class CheckpointBackend:
    """Explicit single-checkpoint adapter for base/tuned comparisons.

    Uses Agent rather than multilingual routing so every row reaches this checkpoint.
    """
    def __init__(self, name: str, model: str, *, device: str = 'cuda', revision: str | None = None):
        import laya
        self.name = name
        self.agent = laya.load(model, device=device, revision=revision, fast=False, compile=False)
        if self.agent.device.type != device.split(':')[0]:
            raise RuntimeError('Laya fell back from requested device')
        self._last_tokens = None

    def predict(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
        # Apply the same strict budget checks during evaluation.
        for qid, q in questions.items():
            if q['type'] != 'choice':
                raise ValueError('This experiment supports choice questions only')
            encode_examples(self.agent, [{'id': qid, 'state': state,
                'question': q['instructions'], 'options': q['criteria'],
                'label': next(iter(q['criteria']))}])
        requested_device = self.agent.device.type
        response = self.agent.system_one(state=state, questions=questions)
        if self.agent.device.type != requested_device:
            raise RuntimeError('Laya changed device during inference')
        return response['answers']
