"""Stretch exercise: collect teacher-labeled mistakes from fresh training families.

Never mines validation/calibration/test. Retrain a new run on the exported dataset.
"""
import argparse
import json
import tempfile
from pathlib import Path

from ticket_data import generate
from system_one_models.decision_benchmark import run_benchmark
from system_one_models.finetuning import (
    CheckpointBackend, file_digest, scenarios, verify_dataset,
)


def collect(data, model, output, count=3000, seed=43, device='cuda'):
    original = verify_dataset(data)
    manifest = json.loads((Path(data) / 'manifest.json').read_text())
    if seed == manifest['seed']:
        raise ValueError('Use a fresh generator seed')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new dataset directory')
    with tempfile.TemporaryDirectory() as temp:
        generate(temp, count, seed)
        pool = [r for rows in verify_dataset(temp).values() for r in rows
                if r['family'] in {x['family'] for x in original['train']}]
    existing_states = {r['state'] for rows in original.values() for r in rows}
    pool = [r for r in pool if r['state'] not in existing_states]
    backend = CheckpointBackend('dagger', model, device=device)
    result = run_benchmark([backend], scenarios(pool))
    if any(not r['ok'] for r in result.records):
        raise RuntimeError('Resolve inference errors before collecting mistakes')
    mistakes = {r['scenario'] for r in result.records if r['pred'] != r['label']}
    additions = [r for r in pool if r['id'] in mistakes]
    if not additions:
        raise ValueError('No mistakes found; no dataset written')
    output.mkdir(parents=True, exist_ok=True)
    for split, rows in original.items():
        if split == 'train':
            rows = rows + additions
            payload = ''.join(json.dumps(r) + '\n' for r in rows)
        else:
            payload = (Path(data) / f'{split}.jsonl').read_text()
        (output / f'{split}.jsonl').write_text(payload)
    manifest['sha256'] = {s: file_digest(output / f'{s}.jsonl') for s in original}
    manifest['dagger'] = {'seed': seed, 'model': model, 'pool': len(pool), 'mistakes': len(additions)}
    from collections import Counter
    manifest['counts'] = {s: dict(Counter(r['label'] for r in rows))
                          for s, rows in verify_dataset(data).items()}
    manifest['counts']['train'] = dict(Counter(r['label'] for r in original['train'] + additions))
    manifest['examples'] = sum(len(r) for r in original.values()) + len(additions)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    verify_dataset(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--count', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    collect(args.data, args.model, args.output, args.count, args.seed, args.device)
