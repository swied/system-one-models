"""Compare sequentially through the shared harness, avoiding multiple GPU models."""
import argparse
import gc
import json
from pathlib import Path

from system_one_models.decision_benchmark import BenchmarkResult, create_backend, load_prompts, run_benchmark
from system_one_models.finetuning import CheckpointBackend, scenarios, verify_dataset
from train import MODEL_REVISION
from plots import reliability_svg


def relevant_prompts():
    selected = []
    for row in load_prompts():
        if 'team' in row['questions'] and row['questions']['team']['type'] == 'choice':
            selected.append({'id': row['id'], 'state': row['state'],
                             'questions': {'team': row['questions']['team']},
                             'labels': {'team': row['labels']['team']}})
    return selected


def evaluate(data, output, *, checkpoints=None, hosted=(), mock=True, device='cuda', base_model='convaiinnovations/laya', base_revision=MODEL_REVISION):
    sets = {'test': scenarios(verify_dataset(data)['test']), 'bakeoff-slice': relevant_prompts()}
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new report directory')
    checkpoints = checkpoints or {}
    if 'laya-base' in checkpoints or set(checkpoints) & set(hosted):
        raise ValueError('Checkpoint names must be unique and distinct from base/hosted names')
    if len(set(hosted)) != len(hosted):
        raise ValueError('Hosted names must be unique')
    specs = [('laya-base', base_model), *checkpoints.items()]
    if mock and not checkpoints:
        specs += [('laya-head', 'mock'), ('laya-calibrated', 'mock')]
    all_results = {key: BenchmarkResult([], []) for key in sets}
    metadata = {'mock': mock, 'device': device, 'base_revision': base_revision,
                'checkpoints': dict(specs), 'hosted': list(hosted),
                'question_counts': {k: len(v) for k, v in sets.items()}}
    metadata['models'] = {}
    for name, model in [*specs, *((name, None) for name in hosted)]:
        if mock:
            backend = create_backend('laya' if model is not None else name, mock=True)
            backend.name = name
        elif model is None:
            backend = create_backend(name)
        else:
            if name != 'laya-base' and not (Path(model) / 'rl_agent_config.json').is_file():
                raise ValueError(f'Missing checkpoint: {model}')
            backend = CheckpointBackend(name, str(model), device=device,
                                        revision=base_revision if name == 'laya-base' and not Path(model).is_dir() else None)
        if not mock and hasattr(backend, 'agent'):
            import torch
            import laya
            metadata['models'][name] = {
                'torch': torch.__version__, 'laya': getattr(laya, '__version__', 'unknown'),
                'device': str(backend.agent.device),
                'hardware': torch.cuda.get_device_name(backend.agent.device)
                            if backend.agent.device.type == 'cuda' else 'CPU',
                'training': backend.agent.cfg.get('week03_training'),
                'calibration': backend.agent.cfg.get('week03_calibration')}
        try:
            if not mock:
                # Initialization and one warmup are excluded from reported request latency.
                first = sets['test'][0]
                backend.predict(first['state'], first['questions'])
            for key, rows in sets.items():
                result = run_benchmark([backend], rows)
                all_results[key].metrics.extend(result.metrics)
                all_results[key].records.extend(result.records)
        finally:
            if hasattr(backend, 'close'):
                backend.close()
            del backend
            gc.collect()
            if not mock:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    output.mkdir(parents=True)
    for key, result in all_results.items():
        result.write(output / key)
        (output / key / 'reliability.svg').write_text(reliability_svg(result.records))
    (output / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    report = ['# Week 3 ticket-routing comparison',
              '**MOCK: workflow validation only; no model measurements.**' if mock
              else '**Real inference on the specified checkpoints.**',
              '## Frozen synthetic test', all_results['test'].markdown(),
              '## Existing bakeoff ticket-routing slice', all_results['bakeoff-slice'].markdown(),
              '## Interpretation',
              'Accuracy measures correctly routed tickets. Errors mean a customer may reach the wrong '
              'team. ECE compares stated confidence with observed correctness; Brier is the squared '
              'error of predicted-class confidence against correctness, not multiclass Brier. '
              'Use calibration bins in each report as a reliability diagram. A fitted temperature '
              'changes confidence, not the top choice. p50/p95 describe typical and slower requests. '
              'Inspect errors before interpreting accuracy. Prices remain unknown without input pricing.',
              'The synthetic teacher labels single-intent templates, not real customer tickets. '
              'Held-out template families reduce leakage but do not establish real-world quality. '
              'The four-question bakeoff slice is small and labels are subjective. Hosted calls '
              'include network time; local calls do not. Initialization and warmup are excluded. '
              'Local hardware and output-token costs are excluded. Local execution does not '
              'guarantee privacy. No tuning improvement or production readiness is assumed.',
              'Before deciding to deploy, compare accuracy and errors on the same frozen test, '
              'then use confidence bins to choose which tickets need human review. Evaluate on '
              'manually labeled real tickets before making an operating-cost or quality decision.']
    (output / 'comparison.md').write_text('\n\n'.join(report) + '\n')
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint', action='append', default=[], metavar='NAME=PATH')
    parser.add_argument('--hosted', nargs='*', choices=['jev', 'd1'], default=[])
    parser.add_argument('--real', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--base-model', default='convaiinnovations/laya')
    parser.add_argument('--base-revision', default=MODEL_REVISION)
    args = parser.parse_args()
    checkpoints = dict(value.split('=', 1) for value in args.checkpoint)
    if args.real and not checkpoints:
        parser.error('Provide at least one tuned --checkpoint for a real before/after run')
    evaluate(args.data, args.output, checkpoints=checkpoints, hosted=args.hosted,
             mock=not args.real, device=args.device, base_model=args.base_model, base_revision=args.base_revision)
