"""Fit on calibration data only; save a separate calibrated checkpoint."""
import argparse
import copy
import json
from pathlib import Path

from system_one_models.finetuning import (
    CheckpointBackend, encode_examples, fit_temperature, verify_dataset,
)
from train import save_checkpoint


def calibrate(model, data, output, device='cuda'):
    import numpy as np
    import torch
    from laya.common import QTYPES, collate_items, ece_score
    from scipy.special import softmax, logsumexp

    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new calibrated checkpoint directory')
    if not (Path(model) / 'rl_agent_config.json').is_file():
        raise ValueError('Train a local checkpoint first')
    rows = verify_dataset(data)['calibration']
    agent = CheckpointBackend('calibration', str(model), device=device).agent
    agent.model.eval()
    logits, labels = [], []
    with torch.inference_mode():
        for item in encode_examples(agent, rows):
            batch = collate_items([[item]], agent.tok.pad_token_id)
            # Raw model forward bypasses all previously fitted calibration.
            z, _ = agent._infer(batch)
            if agent.device.type != device.split(':')[0]:
                raise RuntimeError('Calibration device fallback')
            logits.append(z[0, :len(item['target'])].float().cpu().numpy())
            labels.append(item['label'])
    logits, labels = np.asarray(logits), np.asarray(labels)
    temperature = fit_temperature(logits, labels)
    def metrics(t):
        z = logits / t
        p = softmax(z, axis=1)
        correct = p.argmax(1) == labels
        return {'accuracy': float(correct.mean()),
                'nll': float(np.mean(logsumexp(z, axis=1) - z[np.arange(len(z)), labels])),
                'ece': float(ece_score(p.max(1), correct)),
                'brier_predicted_class': float(np.mean((p.max(1) - correct) ** 2))}
    agent.cfg = copy.deepcopy(agent.cfg)
    agent.cfg['temperature'] = [1.0, 1.0, 1.0]
    agent.cfg['temperature'][QTYPES['choice']] = temperature
    for key in ('temperature_by_options', 'lang_temperatures', 'binning_map'):
        agent.cfg.pop(key, None)
    report = {'temperature': temperature, 'examples': len(rows),
              'before': metrics(1.0), 'after': metrics(temperature),
              'note': 'Calibration-set diagnostics; independent quality measured on frozen test.',
              'manifest': json.loads((Path(data) / 'manifest.json').read_text())}
    agent.cfg['week03_calibration'] = report
    save_checkpoint(agent, output)
    (output / 'calibration.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    calibrate(args.model, args.data, args.output, args.device)
